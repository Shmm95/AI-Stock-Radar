from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import portfolio_backtest_engine as engine
from src.backtest.run_portfolio_rsi_replacement_ablation import (
    CONTROL_POLICY,
    REPLACEMENT_POLICIES,
    ReplacementCollector,
    ReplacementPolicy,
    build_comparisons,
    build_screen,
    candidate_rsi_quality,
    normalize_policy_names,
    replacement_policy_context,
    rsi_quality,
    save_rsi_replacement_ablation,
    select_executable_victim,
    validate_victim_authorization,
)


def market_frame(start: str = "2024-01-01", rows: int = 5) -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0 + index for index in range(rows)],
            "High": [102.0 + index for index in range(rows)],
            "Low": [99.0 + index for index in range(rows)],
            "Close": [101.0 + index for index in range(rows)],
            "EMA20": [100.0 + index for index in range(rows)],
            "EMA50": [99.0 + index for index in range(rows)],
            "RSI14": [50.0, 55.0, 57.5, 60.0, 65.0],
            "MACD": 1.0,
        },
        index=index,
    )


def test_policy_validation_and_control_requirement():
    assert normalize_policy_names([CONTROL_POLICY, "rsi_q9_any"]) == (
        CONTROL_POLICY,
        "RSI_Q9_ANY",
    )
    with pytest.raises(ValueError, match=CONTROL_POLICY):
        normalize_policy_names(["RSI_Q9_ANY"])
    with pytest.raises(ValueError, match="threshold"):
        ReplacementPolicy("BAD", True, None, False)


def test_rsi_quality_and_signal_timestamp_are_causal():
    assert rsi_quality(57.5) == 15.0
    assert rsi_quality(55.0) == 12.0
    timestamp, quality = candidate_rsi_quality(market_frame(), "2024-01-03")
    assert timestamp == pd.Timestamp("2024-01-03")
    assert quality == 15.0
    with pytest.raises(ValueError, match="absent"):
        candidate_rsi_quality(market_frame(), "2024-01-03 12:00")


def position(entry_price: float, entry_bar: int):
    return SimpleNamespace(
        entry_price=entry_price,
        entry_portfolio_bar_index=entry_bar,
    )


def test_victim_requires_current_bar_and_excludes_same_bar_entries():
    state = SimpleNamespace(
        positions={
            "A": position(110.0, 1),
            "B": position(100.0, 1),
            "C": position(200.0, 4),
        }
    )
    active = market_frame()
    stale = market_frame(start="2023-12-01")
    result = select_executable_victim(
        state=state,
        data_by_ticker={"A": active, "B": stale, "C": active},
        timestamp="2024-01-05",
        portfolio_bar_index=4,
    )
    assert result["victim_ticker"] == "A"
    assert result["victim_raw_open"] == 104.0


def test_control_context_restores_engine_function_after_error():
    original = engine._attempt_open_position
    policy = next(item for item in REPLACEMENT_POLICIES if not item.enabled)
    collector = ReplacementCollector("FIXED_BASELINE", policy, [])
    with pytest.raises(RuntimeError):
        with replacement_policy_context(
            policy=policy,
            stop_model="FIXED_BASELINE",
            data_by_ticker={"A": market_frame()},
            collector=collector,
        ):
            assert engine._attempt_open_position is not original
            raise RuntimeError("stop")
    assert engine._attempt_open_position is original


def summary_rows() -> pd.DataFrame:
    rows = []
    for stop in ("FIXED_BASELINE", "FIXED_MAX_RETURN"):
        rows.extend(
            [
                {
                    "stop_model": stop,
                    "replacement_policy": CONTROL_POLICY,
                    "total_return_percent": 10.0,
                    "maximum_drawdown_percent": 5.0,
                    "return_drawdown_ratio": 2.0,
                    "profit_factor": 1.5,
                    "excess_return_vs_matched_percent": 1.0,
                    "completed_trades": 20,
                    "total_fees": 100.0,
                    "executed_replacements": 0,
                },
                {
                    "stop_model": stop,
                    "replacement_policy": "RSI_Q12_ANY",
                    "total_return_percent": 15.0,
                    "maximum_drawdown_percent": 5.5,
                    "return_drawdown_ratio": 2.7,
                    "profit_factor": 1.6,
                    "excess_return_vs_matched_percent": 4.0,
                    "completed_trades": 35,
                    "total_fees": 140.0,
                    "executed_replacements": 15,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_comparison_and_screen_require_both_stop_strata():
    compared = build_comparisons(summary_rows())
    screen = build_screen(compared)
    assert screen["stratum_pass"].all()
    assert screen["robust_policy_pass"].all()
    damaged = compared.copy()
    damaged.loc[
        (damaged["stop_model"] == "FIXED_MAX_RETURN")
        & (damaged["replacement_policy"] == "RSI_Q12_ANY"),
        "profit_factor_delta_vs_control",
    ] = -0.1
    assert not build_screen(damaged)["robust_policy_pass"].any()


def test_victim_authorization_must_be_exact():
    payload = {
        "authorized_candidate_feature": "RSI_QUALITY",
        "robust_victim_rules_authorized_for_separate_ablation": [
            "WORST_UNREALIZED_RETURN"
        ],
    }
    screen = pd.DataFrame(
        [
            {
                "victim_rule": "WORST_UNREALIZED_RETURN",
                "robust_victim_rule_pass": True,
            }
        ]
    )
    validate_victim_authorization(payload, screen)
    payload["authorized_candidate_feature"] = "CURRENT_SCORE"
    with pytest.raises(ValueError, match="RSI_QUALITY"):
        validate_victim_authorization(payload, screen)


def test_save_writes_hashed_provenance(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "source.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    frames = {
        name: pd.DataFrame([{"value": 1}])
        for name in (
            "summary",
            "events",
            "tickers",
            "annual",
            "rejections",
            "equity",
            "trades",
        )
    }
    frames["screen"] = pd.DataFrame(
        [{"replacement_policy": "RSI_Q12_ANY", "robust_policy_pass": False}]
    )
    bundle = {
        **frames,
        "snapshot_id": "snapshot",
        "snapshot_fingerprint": "fingerprint",
        "snapshot_manifest_path": manifest,
        "victim_stamp": "victim",
        "victim_paths": {"source": source},
        "policies": [REPLACEMENT_POLICIES[0].to_dict()],
        "stop_profiles": {"FIXED_BASELINE": (5.0, 5.0)},
        "base_config": {},
    }
    paths = save_rsi_replacement_ablation(
        bundle, output_directory=tmp_path / "out"
    )
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["snapshot_fingerprint"] == "fingerprint"
    assert len(provenance["result_files"]) == 9
