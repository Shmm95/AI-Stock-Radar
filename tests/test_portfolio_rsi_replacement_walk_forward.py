from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.run_portfolio_rsi_replacement_walk_forward import (
    CONTROL_POLICY,
    MODEL_CONTROL,
    MODEL_DYNAMIC,
    MODEL_REPLACEMENT,
    MODELS,
    REPLACEMENT_POLICY,
    STOP_PROFILES,
    aggregate_model,
    build_parser,
    build_research_screen,
    save_rsi_replacement_walk_forward,
    training_decision,
    validate_ablation_authorization,
)


def summary(
    policy: str,
    *,
    total_return: float,
    drawdown: float,
    ratio: float,
    profit_factor: float,
    excess: float,
    replacements: int,
) -> dict[str, object]:
    return {
        "replacement_policy": policy,
        "total_return_percent": total_return,
        "maximum_drawdown_percent": drawdown,
        "return_drawdown_ratio": ratio,
        "profit_factor": profit_factor,
        "excess_return_vs_matched_percent": excess,
        "executed_replacements": replacements,
    }


def test_training_decision_selects_only_when_every_gate_passes():
    control = summary(
        CONTROL_POLICY,
        total_return=10,
        drawdown=5,
        ratio=2,
        profit_factor=1.5,
        excess=1,
        replacements=0,
    )
    candidate = summary(
        REPLACEMENT_POLICY,
        total_return=15,
        drawdown=6,
        ratio=2.5,
        profit_factor=1.6,
        excess=3,
        replacements=2,
    )
    decision = training_decision(control, candidate)
    assert decision["selected_policy"] == REPLACEMENT_POLICY
    assert decision["train_criteria_passed"] == 6

    candidate["profit_factor"] = 1.4
    decision = training_decision(control, candidate)
    assert decision["selected_policy"] == CONTROL_POLICY
    assert not decision["nonnegative_train_profit_factor_delta"]


def test_training_decision_defaults_to_control_for_sparse_or_tied_candidate():
    control = summary(
        CONTROL_POLICY,
        total_return=10,
        drawdown=5,
        ratio=2,
        profit_factor=1.5,
        excess=1,
        replacements=0,
    )
    candidate = summary(
        REPLACEMENT_POLICY,
        total_return=10,
        drawdown=5,
        ratio=2,
        profit_factor=1.5,
        excess=1,
        replacements=1,
    )
    decision = training_decision(control, candidate)
    assert decision["selected_policy"] == CONTROL_POLICY
    assert not decision["at_least_2_train_replacements"]
    assert not decision["positive_train_return_delta"]


def test_ablation_authorization_must_cover_exact_policy_and_both_strata():
    payload = {
        "robust_policies_authorized_for_walk_forward": [REPLACEMENT_POLICY]
    }
    screen = pd.DataFrame(
        [
            {
                "replacement_policy": REPLACEMENT_POLICY,
                "stop_model": stop_model,
                "stratum_pass": True,
                "robust_policy_pass": True,
            }
            for stop_model in STOP_PROFILES
        ]
    )
    validate_ablation_authorization(payload, screen)
    damaged = screen.copy()
    damaged.loc[0, "stratum_pass"] = False
    with pytest.raises(ValueError, match="every fixed stop stratum"):
        validate_ablation_authorization(payload, damaged)


def test_aggregate_uses_compounding_stitched_drawdown_and_pooled_pf():
    runs = pd.DataFrame(
        [
            {
                "stop_model": "FIXED_BASELINE",
                "model": MODEL_DYNAMIC,
                "window_id": "W01",
                "total_return_percent": 10.0,
                "matched_benchmark_return_percent": 5.0,
                "scaled_gross_profit": 200.0,
                "scaled_gross_loss": 100.0,
                "scaled_total_fees": 10.0,
                "actual_policy": REPLACEMENT_POLICY,
                "excess_return_vs_matched_percent": 5.0,
                "completed_trades": 3,
                "executed_replacements": 2,
                "average_exposure_percent": 50.0,
            },
            {
                "stop_model": "FIXED_BASELINE",
                "model": MODEL_DYNAMIC,
                "window_id": "W02",
                "total_return_percent": 10.0,
                "matched_benchmark_return_percent": 5.0,
                "scaled_gross_profit": 300.0,
                "scaled_gross_loss": 100.0,
                "scaled_total_fees": 12.0,
                "actual_policy": CONTROL_POLICY,
                "excess_return_vs_matched_percent": 5.0,
                "completed_trades": 4,
                "executed_replacements": 0,
                "average_exposure_percent": 60.0,
            },
        ]
    )
    equity = pd.DataFrame(
        [
            {
                "stop_model": "FIXED_BASELINE",
                "model": MODEL_DYNAMIC,
                "timestamp": "2024-01-01",
                "total_equity": 10_000,
            },
            {
                "stop_model": "FIXED_BASELINE",
                "model": MODEL_DYNAMIC,
                "timestamp": "2024-03-01",
                "total_equity": 9_000,
            },
            {
                "stop_model": "FIXED_BASELINE",
                "model": MODEL_DYNAMIC,
                "timestamp": "2024-12-31",
                "total_equity": 12_100,
            },
        ]
    )
    result = aggregate_model(
        runs,
        equity,
        stop_model="FIXED_BASELINE",
        model=MODEL_DYNAMIC,
        initial_cash=10_000,
        first_test_start=pd.Timestamp("2024-01-01"),
        last_test_end_exclusive=pd.Timestamp("2025-01-01"),
    )
    assert result["ending_equity"] == 12_100
    assert result["compounded_return_percent"] == 21.0
    assert result["maximum_drawdown_percent"] == 10.0
    assert result["pooled_profit_factor"] == 2.5
    assert result["replacement_policy_window_count"] == 1


def screen_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    test_rows = []
    for stop_model in STOP_PROFILES:
        for model in MODELS:
            is_dynamic = model == MODEL_DYNAMIC
            aggregate_rows.append(
                {
                    "stop_model": stop_model,
                    "model": model,
                    "compounded_return_percent": 20.0 if is_dynamic else 10.0,
                    "maximum_drawdown_percent": 5.0,
                    "return_drawdown_ratio": 4.0 if is_dynamic else 2.0,
                    "pooled_profit_factor": 1.7 if is_dynamic else 1.5,
                    "excess_return_vs_matched_percent": 3.0 if is_dynamic else 1.0,
                    "executed_replacements": 5 if is_dynamic else 0,
                    "replacement_policy_window_count": 3 if is_dynamic else 0,
                }
            )
        for index in range(3):
            for model, value in (
                (MODEL_DYNAMIC, 2.0),
                (MODEL_CONTROL, 1.0),
                (MODEL_REPLACEMENT, 1.5),
            ):
                test_rows.append(
                    {
                        "stop_model": stop_model,
                        "window_id": f"W{index + 1:02d}",
                        "model": model,
                        "total_return_percent": value,
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(test_rows)


def test_research_screen_requires_both_strata_and_complete_windows():
    aggregate, tests = screen_inputs()
    result = build_research_screen(
        aggregate, tests, complete_window_set=True
    )
    assert result["stratum_pass"].all()
    assert result["robust_walk_forward_pass"].all()
    assert not result["production_authorized"].any()

    incomplete = build_research_screen(
        aggregate, tests, complete_window_set=False
    )
    assert not incomplete["stratum_pass"].any()
    assert not incomplete["robust_walk_forward_pass"].any()


def test_save_writes_all_results_and_hashed_provenance(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "source.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    frames = {
        name: pd.DataFrame([{"value": 1}])
        for name in (
            "windows",
            "training",
            "test_runs",
            "aggregate",
            "events",
            "tickers",
            "rejections",
            "trades",
            "equity",
        )
    }
    frames["screen"] = pd.DataFrame(
        [
            {
                "robust_walk_forward_pass": False,
                "production_authorized": False,
            }
        ]
    )
    bundle = {
        **frames,
        "snapshot_id": "snapshot",
        "snapshot_fingerprint": "fingerprint",
        "snapshot_manifest_path": manifest,
        "ablation_stamp": "ablation",
        "ablation_paths": {"source": source},
        "base_config": {},
        "window_definitions": [],
        "expected_window_count": 13,
        "actual_window_count": 1,
        "complete_window_set": False,
        "selection_summary": {},
    }
    paths = save_rsi_replacement_walk_forward(
        bundle, output_directory=tmp_path / "out"
    )
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["snapshot_fingerprint"] == "fingerprint"
    assert len(provenance["result_files"]) == 11
    assert provenance["source_files"]["source"]["sha256"]


def test_cli_does_not_expose_policy_stop_or_gate_tuning():
    actions = {action.dest for action in build_parser()._actions}
    assert "policies" not in actions
    assert "stops" not in actions
    assert "minimum_train_replacements" not in actions
    assert {"snapshot", "ablation_stamp", "max_windows"}.issubset(actions)
