from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.run_portfolio_entry_score_feature_audit import (
    FEATURE_DEFINITIONS,
    build_feature_quintiles,
    build_feature_summary,
    build_feature_windows,
    build_screen,
    build_ticker_leave_one_out,
    causal_atr14_percent,
    extract_opportunities,
    feature_values,
    normalize_horizons,
    rank_correlation,
    save_entry_score_feature_audit,
    verify_slot_source,
)


def market_frame(rows: int = 50) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(range(100, 100 + rows), index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 2,
            "Low": close - 1,
            "Close": close,
            "EMA20": close - 2,
            "EMA50": close - 4,
            "RSI14": 57.5,
            "MACD": 1.0,
        },
        index=index,
    )


def event(timestamp: str = "2024-02-10") -> dict[str, object]:
    return {
        "event_id": "W01:FIXED_BASELINE:0001:AAPL",
        "window_id": "W01",
        "stop_model": "FIXED_BASELINE",
        "timestamp": timestamp,
        "candidate_ticker": "AAPL",
        "candidate_asset_class": "EQUITY",
        "candidate_score": 80.0,
        "candidate_minus_mean_held_score": 5.0,
        "candidate_minus_mean_held_5_bars_percent": 1.0,
        "candidate_minus_mean_held_10_bars_percent": 2.0,
        "candidate_minus_mean_held_20_bars_percent": 3.0,
    }


def test_horizons_are_unique_sorted_and_positive():
    assert normalize_horizons([20, 5, 5, 10]) == (5, 10, 20)
    with pytest.raises(ValueError, match="positive"):
        normalize_horizons([0])


def test_features_use_strictly_previous_signal_bar():
    data = market_frame()
    values = feature_values(event(), data)
    assert values["signal_timestamp"] == pd.Timestamp("2024-02-09")
    expected = (float(data.loc["2024-02-09", "Close"]) / 119.0 - 1) * 100
    assert values["MOMENTUM_20_PERCENT"] == pytest.approx(expected)
    assert values["CURRENT_SCORE"] == 80.0
    assert values["SCORE_PREMIUM"] == 5.0


def test_atr_is_causal_and_uses_only_past_rows():
    data = market_frame()
    before = causal_atr14_percent(data, 20)
    changed = data.copy()
    changed.iloc[21:, changed.columns.get_loc("High")] = 10000
    assert causal_atr14_percent(changed, 20) == before
    assert before == pytest.approx(3 / 120 * 100)


def test_rank_correlation_handles_constants_without_warning():
    assert rank_correlation(pd.Series([1, 2, 3]), pd.Series([3, 2, 1])) == -1
    assert rank_correlation(pd.Series([1, 1, 1]), pd.Series([1, 2, 3])) == 0


def test_extract_opportunities_keeps_targets():
    output = extract_opportunities(
        pd.DataFrame([event()]), {"AAPL": market_frame()}
    )
    assert len(output) == 1
    assert output.iloc[0]["opportunity_gap_20_bars_percent"] == 3.0
    assert set(FEATURE_DEFINITIONS).issubset(output.columns)


def analytical_frame() -> pd.DataFrame:
    rows = []
    for stop in ("FIXED_BASELINE", "FIXED_MAX_RETURN"):
        for window in range(1, 14):
            for index in range(10):
                score = float(window * 10 + index)
                rows.append(
                    {
                        "event_id": f"{stop}:{window}:{index}",
                        "window_id": f"W{window:02d}",
                        "stop_model": stop,
                        "candidate_ticker": f"T{index % 3}",
                        "CURRENT_SCORE": score,
                        "opportunity_gap_20_bars_percent": score / 10,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_and_screen_require_both_stop_strata():
    data = analytical_frame()
    windows = build_feature_windows(
        data, horizons=[20], features=["CURRENT_SCORE"]
    )
    quintiles = build_feature_quintiles(
        data, horizons=[20], features=["CURRENT_SCORE"]
    )
    leave = build_ticker_leave_one_out(
        data, horizons=[20], features=["CURRENT_SCORE"]
    )
    summary = build_feature_summary(
        data,
        windows,
        quintiles,
        leave,
        horizons=[20],
        features=["CURRENT_SCORE"],
    )
    screen = build_screen(summary)
    assert screen["stratum_pass"].all()
    assert screen["robust_feature_pass"].all()
    damaged = summary.copy()
    damaged.loc[
        damaged["stop_model"] == "FIXED_MAX_RETURN", "pooled_spearman_rho"
    ] = -0.2
    assert not build_screen(damaged)["robust_feature_pass"].any()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_slot_source(tmp_path: Path, robust: bool = False):
    directory = tmp_path / "slot"
    directory.mkdir()
    stamp = "20240101_000000"
    prefix = "portfolio_slot_opportunity_audit_"
    names = (
        "events",
        "holdings",
        "runs",
        "score_quintiles",
        "screen",
        "summary",
        "tickers",
        "windows",
    )
    paths = {}
    for name in names:
        path = directory / f"{prefix}{name}_{stamp}.csv"
        frame = (
            pd.DataFrame(
                [{"robust_replacement_audit_pass": robust}]
            )
            if name == "screen"
            else pd.DataFrame([{"value": 1}])
        )
        frame.to_csv(path, index=False)
        paths[name] = path
    json_path = directory / f"{prefix}{stamp}.json"
    write_json(json_path, {"snapshot_fingerprint": "fp"})
    paths["json"] = json_path
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"fingerprint": "fp"})
    provenance = {
        "audit_stamp": stamp,
        "snapshot_fingerprint": "fp",
        "snapshot_manifest_path": str(manifest),
        "snapshot_manifest_sha256": digest(manifest),
        "source_files": {},
        "result_files": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in paths.items()
        },
    }
    provenance_path = directory / f"{prefix}provenance_{stamp}.json"
    write_json(provenance_path, provenance)
    return directory, stamp


def test_predecessor_pass_routes_to_replacement_ablation(tmp_path: Path):
    directory, stamp = minimal_slot_source(tmp_path, robust=True)
    with pytest.raises(ValueError, match="replacement ablation"):
        verify_slot_source(
            directory=directory,
            stamp=stamp,
            snapshot_manifest={"fingerprint": "fp"},
        )


def test_source_hash_mismatch_is_rejected(tmp_path: Path):
    directory, stamp = minimal_slot_source(tmp_path)
    path = directory / f"portfolio_slot_opportunity_audit_events_{stamp}.csv"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_slot_source(
            directory=directory,
            stamp=stamp,
            snapshot_manifest={"fingerprint": "fp"},
        )


def test_save_writes_complete_hashed_provenance(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "source.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    frames = {
        name: pd.DataFrame([{"value": 1}])
        for name in (
            "opportunities",
            "feature_windows",
            "feature_quintiles",
            "ticker_leave_one_out",
            "summary",
        )
    }
    frames["screen"] = pd.DataFrame(
        [{"feature": "CURRENT_SCORE", "robust_feature_pass": False}]
    )
    bundle = {
        **frames,
        "snapshot_id": "snapshot",
        "snapshot_fingerprint": "fingerprint",
        "snapshot_manifest_path": manifest,
        "slot_stamp": "stamp",
        "slot_paths": {"json": source},
        "horizons": [5, 10, 20],
        "features": FEATURE_DEFINITIONS,
    }
    paths = save_entry_score_feature_audit(
        bundle, output_directory=tmp_path / "out"
    )
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["snapshot_fingerprint"] == "fingerprint"
    assert len(provenance["result_files"]) == 7
