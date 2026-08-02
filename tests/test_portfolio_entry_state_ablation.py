from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backtest import portfolio_backtest_engine as engine
from src.backtest.run_portfolio_entry_state_ablation import (
    ENTRY_POLICIES,
    EntryPolicy,
    _exact_sign_test_two_sided,
    build_factor_effects,
    build_policy_entry_signal,
    build_screen,
    entry_policy_context,
    normalize_policy_names,
    save_entry_state_ablation,
)


def market_frame(valid: list[bool]) -> pd.DataFrame:
    rows = []
    for index, is_valid in enumerate(valid):
        rows.append(
            {
                "Open": 100 + index,
                "High": 102 + index,
                "Low": 99 + index,
                "Close": 101 + index,
                "EMA20": 100 if is_valid else 99,
                "EMA50": 99.5 if is_valid else 100,
                "RSI14": 55 if is_valid else 40,
                "RegimeAllowed": True,
            }
        )
    return pd.DataFrame(
        rows, index=pd.date_range("2024-01-01", periods=len(rows), freq="D")
    )


def policy(name: str) -> EntryPolicy:
    return next(item for item in ENTRY_POLICIES if item.name == name)


def test_policy_validation_and_control_requirement():
    assert normalize_policy_names(["edge_only", "boundary_edge"]) == (
        "EDGE_ONLY",
        "BOUNDARY_EDGE",
    )
    with pytest.raises(ValueError, match="EDGE_ONLY"):
        normalize_policy_names(["BOUNDARY_EDGE"])
    with pytest.raises(ValueError, match="Unknown"):
        normalize_policy_names(["EDGE_ONLY", "NOT_A_POLICY"])
    with pytest.raises(ValueError, match="at least one"):
        EntryPolicy("bad", False, 0)


def test_edge_only_matches_newly_valid_semantics():
    data = market_frame([False, True, True])
    assert build_policy_entry_signal("AAPL", data, 1, policy("EDGE_ONLY"))
    assert build_policy_entry_signal("AAPL", data, 2, policy("EDGE_ONLY")) is None


def test_boundary_treatment_is_causal_on_first_close():
    data = market_frame([True, True, True])
    signal = build_policy_entry_signal("AAPL", data, 0, policy("BOUNDARY_EDGE"))
    assert signal is not None
    assert "boundary-valid" in signal.reason
    assert build_policy_entry_signal("AAPL", data, 1, policy("BOUNDARY_EDGE")) is None
    assert build_policy_entry_signal("AAPL", data, 2, policy("BOUNDARY_EDGE")) is None
    assert build_policy_entry_signal("AAPL", data, 0, policy("EDGE_ONLY")) is None
    assert build_policy_entry_signal("AAPL", data, 1, policy("EDGE_ONLY")) is None


def test_persistence_is_limited_to_five_attempt_bars():
    data = market_frame([False, True, True, True, True, True, True])
    active = policy("EDGE_PERSIST_5")
    assert all(
        build_policy_entry_signal("AAPL", data, index, active) is not None
        for index in range(1, 6)
    )
    assert build_policy_entry_signal("AAPL", data, 6, active) is None


def test_policy_context_restores_engine_builder_after_error():
    original = engine._build_entry_signal
    with pytest.raises(RuntimeError):
        with entry_policy_context(policy("BOUNDARY_EDGE")):
            assert engine._build_entry_signal is not original
            raise RuntimeError("stop")
    assert engine._build_entry_signal is original


def test_exact_sign_test_is_two_sided():
    assert _exact_sign_test_two_sided(13, 0) == pytest.approx(0.000244)
    assert _exact_sign_test_two_sided(7, 6) == pytest.approx(1.0)
    assert _exact_sign_test_two_sided(0, 0) == 1.0


def test_factor_effects_use_paired_window_differences():
    rows = []
    values = {
        "EDGE_ONLY": [1.0, 2.0],
        "BOUNDARY_EDGE": [2.0, 1.0],
        "EDGE_PERSIST_5": [3.0, 4.0],
        "BOUNDARY_PERSIST_5": [4.0, 5.0],
    }
    for name, returns in values.items():
        for index, value in enumerate(returns, start=1):
            rows.append(
                {
                    "window_id": f"W{index:02d}",
                    "stop_model": "FIXED_BASELINE",
                    "entry_policy": name,
                    "total_return_percent": value,
                }
            )
    result = build_factor_effects(pd.DataFrame(rows))
    row = result.loc[
        result["contrast"] == "PERSISTENCE_WITHOUT_BOUNDARY"
    ].iloc[0]
    assert row["window_wins"] == 2
    assert row["average_window_return_delta_percent"] == 2.0


def test_screen_requires_every_stop_stratum():
    effects = pd.DataFrame(
        [
            {
                "candidate_policy": "BOUNDARY_EDGE",
                "compounded_return_delta_percent": 5.0,
                "return_drawdown_ratio_delta": 0.5,
                "window_wins": 8,
                "window_losses": 5,
                "drawdown_advantage_percent": 0.2,
            },
            {
                "candidate_policy": "BOUNDARY_EDGE",
                "compounded_return_delta_percent": -1.0,
                "return_drawdown_ratio_delta": -0.1,
                "window_wins": 6,
                "window_losses": 7,
                "drawdown_advantage_percent": 0.1,
            },
        ]
    )
    result = build_screen(effects)
    assert bool(result.iloc[0]["robust_screen_pass"]) is False
    assert result.iloc[0]["strata_passing_all_criteria"] == 1


def test_save_writes_hashed_provenance(tmp_path: Path):
    source_json = tmp_path / "source.json"
    source_windows = tmp_path / "windows.csv"
    source_runs = tmp_path / "runs.csv"
    source_provenance = tmp_path / "source_provenance.json"
    manifest = tmp_path / "manifest.json"
    for path in (
        source_json,
        source_windows,
        source_runs,
        source_provenance,
        manifest,
    ):
        path.write_text("{}\n", encoding="utf-8")
    frames = {
        name: pd.DataFrame([{"value": 1}])
        for name in (
            "runs",
            "aggregate",
            "paired_effects",
            "factor_effects",
            "screen",
            "tickers",
            "rejections",
            "equity",
        )
    }
    bundle = {
        **frames,
        "snapshot_id": "snapshot-1",
        "snapshot_fingerprint": "fingerprint-1",
        "snapshot_manifest_path": manifest,
        "source_stamp": "stamp-1",
        "source_paths": {
            "json": source_json,
            "windows": source_windows,
            "test_runs": source_runs,
        },
        "source_provenance_path": source_provenance,
        "stop_models": ["FIXED_BASELINE"],
        "entry_policies": [policy("EDGE_ONLY").to_dict()],
        "base_config": {},
    }
    paths = save_entry_state_ablation(bundle, output_directory=tmp_path / "out")
    assert all(path.exists() for path in paths.values())
    provenance = pd.read_json(paths["provenance"], typ="series")
    assert provenance["snapshot_fingerprint"] == "fingerprint-1"
    assert len(provenance["result_files"]) == 9
