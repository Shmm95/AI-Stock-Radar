from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.run_portfolio_replacement_stop_interaction_audit import (
    MODEL_CONTROL,
    MODEL_DYNAMIC,
    MODEL_REPLACEMENT,
    OUTPUT_FRAME_NAMES,
    REPLACEMENT_POLICY,
    STOP_BASELINE,
    STOP_MAX_RETURN,
    build_event_outcomes,
    build_interaction_screen,
    build_leave_one_window_out,
    build_parser,
    build_ticker_deltas,
    save_interaction_audit,
    validate_source_decision,
)


def source_screen() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stop_model": STOP_BASELINE,
                "stratum_pass": True,
                "robust_walk_forward_pass": False,
                "production_authorized": False,
            },
            {
                "stop_model": STOP_MAX_RETURN,
                "stratum_pass": False,
                "robust_walk_forward_pass": False,
                "production_authorized": False,
            },
        ]
    )


def test_source_decision_must_be_complete_failed_and_nonproduction():
    payload = {
        "authorized_policy": REPLACEMENT_POLICY,
        "complete_window_set": True,
        "robust_walk_forward_pass": False,
        "production_authorized": False,
    }
    validate_source_decision(payload, source_screen())
    damaged = dict(payload, robust_walk_forward_pass=True)
    with pytest.raises(ValueError, match="failure audit is inapplicable"):
        validate_source_decision(damaged, source_screen())


def test_leave_one_window_out_compounds_candidate_and_control():
    rows = []
    for window, control, dynamic, fixed in (
        ("W01", 10.0, 20.0, 15.0),
        ("W02", 0.0, 10.0, -5.0),
    ):
        for model, value in (
            (MODEL_CONTROL, control),
            (MODEL_DYNAMIC, dynamic),
            (MODEL_REPLACEMENT, fixed),
        ):
            rows.append(
                {
                    "stop_model": STOP_BASELINE,
                    "window_id": window,
                    "model": model,
                    "total_return_percent": value,
                }
            )
    result = build_leave_one_window_out(pd.DataFrame(rows))
    dynamic = result.loc[result["model"] == MODEL_DYNAMIC].set_index(
        "excluded_window_id"
    )
    assert dynamic.loc["W01", "return_delta_vs_control_percent"] == 10.0
    assert dynamic.loc["W02", "return_delta_vs_control_percent"] == 10.0
    assert dynamic["positive_after_exclusion"].all()


def test_ticker_deltas_aggregate_scaled_pnl_against_control():
    frame = pd.DataFrame(
        [
            {
                "stop_model": STOP_BASELINE,
                "model": model,
                "ticker": "AAPL",
                "asset_class": "EQUITY",
                "completed_trades": 1,
                "scaled_net_pnl": pnl,
                "scaled_gross_profit": max(pnl, 0),
                "scaled_gross_loss": abs(min(pnl, 0)),
                "scaled_total_fees": fee,
            }
            for model, pnl, fee in (
                (MODEL_CONTROL, 100.0, 1.0),
                (MODEL_DYNAMIC, 150.0, 2.0),
                (MODEL_REPLACEMENT, 90.0, 3.0),
            )
        ]
    )
    result = build_ticker_deltas(frame).set_index("model")
    assert result.loc[MODEL_DYNAMIC, "scaled_net_pnl_delta"] == 50.0
    assert result.loc[MODEL_REPLACEMENT, "scaled_net_pnl_delta"] == -10.0


def test_event_outcome_matches_candidate_victim_and_control_continuation():
    events = pd.DataFrame(
        [
            {
                "window_id": "W01",
                "stop_model": STOP_BASELINE,
                "model": MODEL_DYNAMIC,
                "status": "EXECUTED",
                "timestamp": "2024-02-01T00:00:00",
                "candidate_ticker": "NEW",
                "victim_ticker": "OLD",
            }
        ]
    )
    common = {
        "window_id": "W01",
        "stop_model": STOP_BASELINE,
        "entry_portfolio_bar_index": 1,
        "exit_portfolio_bar_index": 2,
        "scaled_net_pnl": 0.0,
    }
    trades = pd.DataFrame(
        [
            {
                **common,
                "model": MODEL_DYNAMIC,
                "ticker": "NEW",
                "entry_timestamp": "2024-02-01T00:00:00",
                "exit_timestamp": "2024-03-01T00:00:00",
                "exit_reason": "EXIT_SIGNAL_NEXT_OPEN",
                "holding_period_bars": 20,
                "return_percent": 10.0,
                "net_pnl": 100.0,
            },
            {
                **common,
                "model": MODEL_DYNAMIC,
                "ticker": "OLD",
                "entry_timestamp": "2024-01-01T00:00:00",
                "exit_timestamp": "2024-02-01T00:00:00",
                "exit_reason": "REPLACEMENT_RSI_Q12_LOSER_ONLY",
                "holding_period_bars": 10,
                "return_percent": -2.0,
                "net_pnl": -20.0,
            },
            {
                **common,
                "model": MODEL_CONTROL,
                "ticker": "OLD",
                "entry_timestamp": "2024-01-01T00:00:00",
                "exit_timestamp": "2024-04-01T00:00:00",
                "exit_reason": "STOP_LOSS",
                "holding_period_bars": 30,
                "return_percent": -5.0,
                "net_pnl": -50.0,
            },
        ]
    )
    result = build_event_outcomes(events, trades).iloc[0]
    assert result["candidate_trade_matched"]
    assert result["victim_trade_matched"]
    assert result["control_victim_continuation_matched"]
    assert result["local_pair_delta_vs_control_victim"] == 130.0


def interaction_summary() -> pd.DataFrame:
    rows = []
    for stop_model in (STOP_BASELINE, STOP_MAX_RETURN):
        for model in (MODEL_DYNAMIC, MODEL_CONTROL, MODEL_REPLACEMENT):
            if model == MODEL_CONTROL:
                delta = 0.0
            elif model == MODEL_DYNAMIC:
                delta = 60.0 if stop_model == STOP_BASELINE else 2.0
            else:
                delta = 40.0 if stop_model == STOP_BASELINE else -8.0
            rows.append(
                {
                    "stop_model": stop_model,
                    "model": model,
                    "return_delta_vs_control_percent": delta,
                    "leave_one_window_out_min_return_delta_percent": (
                        10.0 if stop_model == STOP_BASELINE else -15.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_interaction_screen_supports_mechanism_but_never_authorizes_policy():
    pairs = pd.DataFrame(
        [
            {"present_in_both_stops": True, "same_victim": index % 2 == 0}
            for index in range(6)
        ]
    )
    result = build_interaction_screen(
        interaction_summary(), source_screen(), pairs
    ).iloc[0]
    assert result["stop_interaction_supported"]
    assert not result["conditional_policy_authorized"]
    assert not result["production_authorized"]


def test_save_writes_hashed_provenance_and_all_frames(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "source.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    bundle = {
        name: pd.DataFrame([{"value": 1}]) for name in OUTPUT_FRAME_NAMES
    }
    bundle["screen"] = pd.DataFrame(
        [
            {
                "stop_interaction_supported": True,
                "recommended_next_stage": "FORWARD_ONLY",
            }
        ]
    )
    bundle.update(
        {
            "source_stamp": "source",
            "source_paths": {"source": source},
            "snapshot_id": "snapshot",
            "snapshot_fingerprint": "fingerprint",
            "snapshot_manifest_path": manifest,
        }
    )
    paths = save_interaction_audit(
        bundle, output_directory=tmp_path / "out"
    )
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["snapshot_fingerprint"] == "fingerprint"
    assert len(provenance["result_files"]) == 11


def test_cli_exposes_source_stamp_but_no_strategy_tuning():
    actions = {action.dest for action in build_parser()._actions}
    assert "walk_forward_stamp" in actions
    assert "stops" not in actions
    assert "policies" not in actions
    assert "threshold" not in actions
