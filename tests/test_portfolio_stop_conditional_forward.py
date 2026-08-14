from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.run_portfolio_stop_conditional_forward import (
    EXPECTED_BLOCK_COUNT,
    FORWARD_END_EXCLUSIVE,
    FORWARD_START,
    MINIMUM_CROSS_STOP_EVENT_PAIRS,
    MODEL_CONTROL,
    MODEL_REPLACEMENT,
    OBSERVED_CUTOFF,
    PROTOCOL_NAME,
    STOP_BASELINE,
    STOP_MAX_RETURN,
    build_block_leave_one_out,
    build_forward_event_pairs,
    build_forward_screen,
    build_parser,
    build_strata_comparison,
    fixed_blocks,
    save_preregistration,
    sha256_file,
    validate_future_manifest,
    validate_interaction_decision,
    validate_preregistration,
    verify_preregistration,
    verify_replay_anchors,
)


def interaction_payload() -> dict[str, object]:
    return {
        "stop_interaction_supported": True,
        "conditional_policy_authorized": False,
        "production_authorized": False,
        "recommended_next_stage": PROTOCOL_NAME,
    }


def interaction_screen() -> pd.DataFrame:
    return pd.DataFrame([interaction_payload()])


def test_interaction_decision_requires_supported_but_unauthorized_source():
    validate_interaction_decision(interaction_payload(), interaction_screen())
    damaged = interaction_screen()
    damaged.loc[0, "conditional_policy_authorized"] = True
    with pytest.raises(ValueError, match="unexpectedly authorized"):
        validate_interaction_decision(interaction_payload(), damaged)


def registration_payload() -> dict[str, object]:
    core = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "source": {
            "interaction_snapshot_fingerprint": "old-fingerprint",
            "snapshot_manifest_path": "/tmp/source/manifest.json",
            "snapshot_manifest_sha256": "manifest-hash",
            "config_sha256": "config-hash",
            "tickers": ["AAPL", "BTC-USD"],
            "code_files": {"engine.py": "code-hash"},
            "observed_cutoff": OBSERVED_CUTOFF.isoformat(),
        },
        "evaluation_period": {
            "start_inclusive": FORWARD_START.isoformat(),
            "end_exclusive": FORWARD_END_EXCLUSIVE.isoformat(),
        },
        "success_gates": {
            "minimum_cross_stop_event_pairs": MINIMUM_CROSS_STOP_EVENT_PAIRS,
            "maximum_baseline_drawdown_worsening_percent": 2.5,
        },
        "authorization": {
            "baseline_change_authorized": False,
            "paper_trading_authorized": False,
            "shadow_trading_authorized": False,
            "production_authorized": False,
        },
    }
    from src.backtest.run_research_data_snapshot import sha256_json

    protocol_hash = sha256_json(core)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "registration_id": f"TEST_{protocol_hash[:12]}",
        "protocol_hash": protocol_hash,
        **core,
    }


def test_preregistration_hash_and_authority_are_immutable():
    registration = registration_payload()
    validate_preregistration(registration)
    damaged = json.loads(json.dumps(registration))
    damaged["authorization"]["paper_trading_authorized"] = True
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        validate_preregistration(damaged)


def test_save_and_verify_registration_hash_chain(tmp_path: Path):
    registration = registration_payload()
    source_file = tmp_path / "source.json"
    source_file.write_text("{}\n", encoding="utf-8")
    source = {"paths": {"json": source_file}}
    paths = save_preregistration(
        registration, source=source, output_directory=tmp_path / "out"
    )
    result = verify_preregistration(
        registration_path=paths["registration"],
        provenance_path=paths["provenance"],
    )
    assert result["registration"]["protocol_hash"] == registration["protocol_hash"]
    assert sha256_file(paths["registration"]) == result["provenance"][
        "result_files"
    ]["registration"]["sha256"]


def test_future_manifest_rejects_config_universe_or_code_drift():
    registration = registration_payload()
    source = {
        "code_files": {"engine.py": "code-hash"},
    }
    future = {
        "fingerprint": "new-fingerprint",
        "created_at": (
            pd.Timestamp(registration["created_at"]) + pd.Timedelta(days=1)
        ).isoformat(),
        "config": {"sha256": "config-hash"},
        "code_files": {"engine.py": "code-hash"},
        "market_files": {
            "AAPL": {"end": "2026-08-04T00:00:00"},
            "BTC-USD": {"end": "2026-08-04T00:00:00"},
        },
    }
    validate_future_manifest(registration, source, future)
    bad = json.loads(json.dumps(future))
    bad["config"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="config hash drifted"):
        validate_future_manifest(registration, source, bad)


def test_replay_anchor_allows_tiny_float_noise_but_rejects_drift():
    index = pd.date_range("2025-01-01", periods=205, freq="D")
    source = {
        "AAPL": pd.DataFrame(
            {"Close": [float(i) for i in range(205)], "Allowed": [True] * 205},
            index=index,
        )
    }
    future_frame = source["AAPL"].copy()
    future_frame.loc[index[-1], "Close"] += 1e-12
    verify_replay_anchors(source, {"AAPL": future_frame}, bars=200)
    future_frame.loc[index[-1], "Close"] += 1e-3
    with pytest.raises(ValueError, match="replay-anchor drift"):
        verify_replay_anchors(source, {"AAPL": future_frame}, bars=200)


def arm_summary(
    stop_model: str,
    model: str,
    *,
    total_return: float,
    drawdown: float,
    ratio: float,
    profit_factor: float,
    excess: float,
    replacements: int,
) -> dict[str, object]:
    return {
        "stop_model": stop_model,
        "model": model,
        "total_return_percent": total_return,
        "maximum_drawdown_percent": drawdown,
        "return_drawdown_ratio": ratio,
        "profit_factor": profit_factor,
        "excess_return_vs_matched_percent": excess,
        "completed_trades": 10,
        "executed_replacements": replacements,
    }


def passing_strata() -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            arm_summary(
                STOP_BASELINE,
                MODEL_CONTROL,
                total_return=10,
                drawdown=5,
                ratio=2,
                profit_factor=1.5,
                excess=1,
                replacements=0,
            ),
            arm_summary(
                STOP_BASELINE,
                MODEL_REPLACEMENT,
                total_return=15,
                drawdown=6,
                ratio=2.5,
                profit_factor=1.6,
                excess=3,
                replacements=6,
            ),
            arm_summary(
                STOP_MAX_RETURN,
                MODEL_CONTROL,
                total_return=12,
                drawdown=4,
                ratio=3,
                profit_factor=1.6,
                excess=2,
                replacements=0,
            ),
            arm_summary(
                STOP_MAX_RETURN,
                MODEL_REPLACEMENT,
                total_return=13,
                drawdown=4.5,
                ratio=2.8,
                profit_factor=1.55,
                excess=1.5,
                replacements=6,
            ),
        ]
    )
    return build_strata_comparison(summary)


def test_strata_comparison_computes_primary_interaction():
    strata = passing_strata().set_index("stop_model")
    assert strata.loc[STOP_BASELINE, "return_delta_vs_control_percent"] == 5
    assert strata.loc[STOP_MAX_RETURN, "return_delta_vs_control_percent"] == 1
    assert (
        strata.loc[
            STOP_BASELINE,
            "interaction_spread_baseline_minus_max_return_percent",
        ]
        == 4
    )


def block_rows() -> pd.DataFrame:
    rows = []
    for block_id, _, _ in fixed_blocks():
        for stop_model in (STOP_BASELINE, STOP_MAX_RETURN):
            for model, value in ((MODEL_CONTROL, 1.0), (MODEL_REPLACEMENT, 2.0)):
                rows.append(
                    {
                        "period_id": block_id,
                        "stop_model": stop_model,
                        "model": model,
                        "total_return_percent": value,
                    }
                )
    return pd.DataFrame(rows)


def test_fixed_blocks_and_leave_one_block_out_are_complete():
    blocks = fixed_blocks()
    assert len(blocks) == EXPECTED_BLOCK_COUNT
    assert blocks[0][1] == FORWARD_START
    assert blocks[-1][2] == FORWARD_END_EXCLUSIVE
    result = build_block_leave_one_out(block_rows())
    assert len(result) == EXPECTED_BLOCK_COUNT * 2
    assert result["positive_after_exclusion"].all()


def test_event_pairs_count_only_common_executed_replacements():
    events = pd.DataFrame(
        [
            {
                "stop_model": stop_model,
                "model": MODEL_REPLACEMENT,
                "status": "EXECUTED",
                "timestamp": "2027-01-01",
                "candidate_ticker": "NEW",
                "victim_ticker": victim,
            }
            for stop_model, victim in (
                (STOP_BASELINE, "OLD"),
                (STOP_MAX_RETURN, "OLD"),
            )
        ]
        + [
            {
                "stop_model": STOP_BASELINE,
                "model": MODEL_REPLACEMENT,
                "status": "EXECUTED",
                "timestamp": "2027-02-01",
                "candidate_ticker": "ONLY_BASELINE",
                "victim_ticker": "OLD",
            }
        ]
    )
    result = build_forward_event_pairs(events)
    assert len(result) == 2
    assert result["present_in_both_stops"].sum() == 1
    assert result["same_victim"].sum() == 1


def passing_event_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"present_in_both_stops": True, "same_victim": True}
            for _ in range(MINIMUM_CROSS_STOP_EVENT_PAIRS)
        ]
    )


def ticker_loo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "excluded_ticker": "AAPL",
                "stop_model": STOP_BASELINE,
                "return_delta_vs_control_percent": 1.0,
            }
        ]
    )


def test_screen_never_authorizes_and_separates_monitoring_from_success():
    monitoring = build_forward_screen(
        strata=passing_strata(),
        block_leave_one_out=build_block_leave_one_out(block_rows()),
        ticker_leave_one_out=ticker_loo(),
        event_pairs=passing_event_pairs(),
        horizon_complete=False,
    ).iloc[0]
    assert monitoring["decision_status"] == "MONITORING"
    assert not monitoring["research_hypothesis_passed"]
    complete = build_forward_screen(
        strata=passing_strata(),
        block_leave_one_out=build_block_leave_one_out(block_rows()),
        ticker_leave_one_out=ticker_loo(),
        event_pairs=passing_event_pairs(),
        horizon_complete=True,
    ).iloc[0]
    assert complete["decision_status"] == (
        "HYPOTHESIS_SUPPORTED_READY_FOR_HUMAN_REVIEW"
    )
    assert complete["research_hypothesis_passed"]
    assert not complete["conditional_policy_authorized"]
    assert not complete["paper_trading_authorized"]
    assert not complete["production_authorized"]


def test_complete_but_sparse_result_is_inconclusive():
    sparse = build_forward_screen(
        strata=passing_strata(),
        block_leave_one_out=build_block_leave_one_out(block_rows()),
        ticker_leave_one_out=ticker_loo(),
        event_pairs=passing_event_pairs().iloc[:2],
        horizon_complete=True,
    ).iloc[0]
    assert sparse["decision_status"] == "INCONCLUSIVE_INSUFFICIENT_EVENTS"
    assert not sparse["research_hypothesis_passed"]


def test_cli_exposes_sources_but_no_tunable_protocol_parameters():
    parser = build_parser()
    root_actions = {action.dest for action in parser._actions}
    assert "command" in root_actions
    subparsers_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    all_destinations = set()
    for choice in subparsers_action.choices.values():
        all_destinations.update(action.dest for action in choice._actions)
    assert {"interaction_stamp", "registration", "snapshot"}.issubset(
        all_destinations
    )
    assert "cutoff" not in all_destinations
    assert "horizon_months" not in all_destinations
    assert "minimum_events" not in all_destinations
    assert "drawdown_threshold" not in all_destinations

