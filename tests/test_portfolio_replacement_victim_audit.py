from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.run_portfolio_replacement_victim_audit import (
    RULE_DEFINITIONS,
    build_pairs,
    build_rule_summary,
    build_rule_windows,
    build_screen,
    build_ticker_leave_one_out,
    held_current_features,
    normalize_horizons,
    save_replacement_victim_audit,
    select_victims,
    top_tail_trimmed_mean,
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


def test_horizons_are_unique_sorted_and_positive():
    assert normalize_horizons([20, 5, 10, 5]) == (5, 10, 20)
    with pytest.raises(ValueError, match="positive"):
        normalize_horizons([0])


def test_held_features_use_strictly_prior_close():
    data = market_frame()
    result = held_current_features(data, "2024-02-10")
    assert result["held_state_timestamp"] == pd.Timestamp("2024-02-09")
    assert result["held_current_rsi_quality"] == 15.0


def source_frames():
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "window_id": "W01",
                "stop_model": "FIXED_BASELINE",
                "timestamp": "2024-02-10",
                "candidate_ticker": "AAPL",
                "candidate_asset_class": "EQUITY",
                "candidate_score": 80.0,
                "candidate_forward_return_5_bars_percent": 5.0,
                "candidate_forward_return_10_bars_percent": 6.0,
                "candidate_forward_return_20_bars_percent": 7.0,
                "candidate_minus_mean_held_5_bars_percent": 2.0,
                "candidate_minus_mean_held_10_bars_percent": 3.0,
                "candidate_minus_mean_held_20_bars_percent": 4.0,
            }
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "held_ticker": ticker,
                "held_asset_class": "EQUITY",
                "held_score": score,
                "holding_age_bars": age,
                "held_unrealized_return_percent": unrealized,
                "forward_return_5_bars_percent": 1.0,
                "forward_return_10_bars_percent": 2.0,
                "forward_return_20_bars_percent": 3.0,
            }
            for ticker, score, age, unrealized in (
                ("MSFT", 70, 10, -2),
                ("GOOGL", 60, 20, 1),
            )
        ]
    )
    opportunities = pd.DataFrame([{"event_id": "E1", "RSI_QUALITY": 14.0}])
    return events, holdings, opportunities


def test_pairs_calculate_candidate_minus_each_holding():
    events, holdings, opportunities = source_frames()
    result = build_pairs(
        events=events,
        holdings=holdings,
        opportunities=opportunities,
        data_by_ticker={"MSFT": market_frame(), "GOOGL": market_frame()},
    )
    assert len(result) == 2
    assert (result["candidate_minus_held_20_bars_percent"] == 4.0).all()
    assert (result["candidate_rsi_quality"] == 14.0).all()


def test_victim_selection_is_deterministic_for_min_and_max_rules():
    rows = []
    for ticker, value, age in (("A", -2, 10), ("B", -2, 20), ("C", 1, 30)):
        rows.append(
            {
                "event_id": "E1",
                "held_ticker": ticker,
                "held_unrealized_return_percent": value,
                "holding_age_bars": age,
            }
        )
    rules = {
        "WORST": ("held_unrealized_return_percent", "min", ""),
        "OLDEST": ("holding_age_bars", "max", ""),
    }
    result = select_victims(pd.DataFrame(rows), rules=rules)
    assert result.loc[result["victim_rule"] == "WORST", "held_ticker"].iloc[0] == "A"
    assert result.loc[result["victim_rule"] == "OLDEST", "held_ticker"].iloc[0] == "C"


def test_trimmed_mean_removes_largest_positive_tail():
    values = pd.Series([1.0] * 19 + [101.0])
    assert top_tail_trimmed_mean(values) == 1.0


def analytical_selections() -> pd.DataFrame:
    rows = []
    for stop in ("FIXED_BASELINE", "FIXED_MAX_RETURN"):
        for window in range(1, 14):
            for index in range(10):
                rows.append(
                    {
                        "event_id": f"{stop}:{window}:{index}",
                        "window_id": f"W{window:02d}",
                        "stop_model": stop,
                        "victim_rule": "WORST_UNREALIZED_RETURN",
                        "candidate_ticker": f"T{index % 3}",
                        "candidate_minus_held_20_bars_percent": 2.0,
                        "candidate_minus_mean_held_20_bars_percent": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_and_screen_require_both_stop_strata():
    selections = analytical_selections()
    windows = build_rule_windows(selections, horizons=[20])
    leave = build_ticker_leave_one_out(selections, horizons=[20])
    summary = build_rule_summary(
        selections, windows, leave, horizons=[20]
    )
    screen = build_screen(summary)
    assert screen["stratum_pass"].all()
    assert screen["robust_victim_rule_pass"].all()
    damaged = summary.copy()
    damaged.loc[
        damaged["stop_model"] == "FIXED_MAX_RETURN",
        "median_candidate_minus_victim_percent",
    ] = -1
    assert not build_screen(damaged)["robust_victim_rule_pass"].any()


def test_rule_catalog_is_pre_registered_and_complete():
    assert "WORST_UNREALIZED_RETURN" in RULE_DEFINITIONS
    assert len(RULE_DEFINITIONS) == 6
    assert all(value[1] in ("min", "max") for value in RULE_DEFINITIONS.values())


def test_save_writes_hashed_provenance(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "source.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    frames = {
        name: pd.DataFrame([{"value": 1}])
        for name in (
            "pairs",
            "selections",
            "rule_windows",
            "ticker_leave_one_out",
            "summary",
        )
    }
    frames["screen"] = pd.DataFrame(
        [
            {
                "victim_rule": "WORST_UNREALIZED_RETURN",
                "robust_victim_rule_pass": False,
            }
        ]
    )
    bundle = {
        **frames,
        "snapshot_id": "snapshot",
        "snapshot_fingerprint": "fingerprint",
        "snapshot_manifest_path": manifest,
        "slot_stamp": "slot",
        "feature_stamp": "feature",
        "source_paths": {"source": source},
        "horizons": [5, 10, 20],
    }
    paths = save_replacement_victim_audit(
        bundle, output_directory=tmp_path / "out"
    )
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["snapshot_fingerprint"] == "fingerprint"
    assert len(provenance["result_files"]) == 7
