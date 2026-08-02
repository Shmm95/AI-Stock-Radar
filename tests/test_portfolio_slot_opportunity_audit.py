from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import portfolio_backtest_engine as engine
from src.backtest.run_portfolio_slot_opportunity_audit import (
    SlotOpportunityCollector,
    _rank_correlation,
    _verify_entry_predecessor,
    build_score_quintiles,
    build_screen,
    forward_from_open,
    save_slot_opportunity_audit,
    slot_opportunity_context,
    summarize_events,
)


def market(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": values,
            "High": [value + 1 for value in values],
            "Low": [value - 1 for value in values],
            "Close": [value + 0.5 for value in values],
        },
        index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
    )


def test_forward_from_open_is_causal_and_half_open():
    data = market([100, 101, 102, 103, 104, 105])
    result = forward_from_open(
        data, timestamp="2024-01-01", reference_open=100, horizons=(2, 5)
    )
    assert result["forward_return_2_bars_percent"] == 2.5
    assert result["max_forward_return_2_bars_percent"] == 2.5
    assert result["forward_return_5_bars_percent"] == 5.5


def test_forward_returns_none_when_horizon_is_unavailable():
    result = forward_from_open(
        market([100, 101]), timestamp="2024-01-01", horizons=(5,)
    )
    assert result["forward_return_5_bars_percent"] is None


def test_rank_correlation_needs_no_optional_statistics_library():
    assert _rank_correlation(pd.Series([1, 2, 3]), pd.Series([10, 20, 30])) == 1
    assert _rank_correlation(pd.Series([1, 2, 3]), pd.Series([30, 20, 10])) == -1


def test_collector_ignores_nonfull_portfolio():
    data = {"AAPL": market([100, 101, 102])}
    collector = SlotOpportunityCollector(
        data, "W01", "FIXED_BASELINE", (1,), [], []
    )
    state = SimpleNamespace(positions={})
    pending = SimpleNamespace(
        signal=SimpleNamespace(ticker="AAPL", score=80.0)
    )
    collector.capture(
        state=state,
        pending=pending,
        timestamp="2024-01-01",
        portfolio_bar_index=0,
        raw_open_price=100,
        config=SimpleNamespace(maximum_open_positions=4),
    )
    assert collector.events == []


def test_collector_builds_candidate_vs_holding_event():
    data = {
        "AAPL": market([100, 101, 102, 103]),
        "MSFT": market([200, 201, 202, 203]),
    }
    collector = SlotOpportunityCollector(
        data, "W01", "FIXED_BASELINE", (1,), [], []
    )
    held = SimpleNamespace(
        entry_timestamp="2024-01-01",
        entry_price=200.0,
        signal_score=70.0,
    )
    state = SimpleNamespace(positions={"MSFT": held})
    pending = SimpleNamespace(
        signal=SimpleNamespace(ticker="AAPL", score=80.0)
    )
    collector.capture(
        state=state,
        pending=pending,
        timestamp="2024-01-01",
        portfolio_bar_index=0,
        raw_open_price=100,
        config=SimpleNamespace(maximum_open_positions=1),
    )
    assert len(collector.events) == 1
    assert len(collector.holdings) == 1
    assert collector.events[0]["candidate_minus_mean_held_score"] == 10
    assert collector.events[0]["candidate_beats_mean_held_1_bars"] is True


def test_context_restores_instrumented_engine_function():
    original = engine._attempt_open_position
    collector = SlotOpportunityCollector({}, "W01", "X", (5,), [], [])
    with pytest.raises(RuntimeError):
        with slot_opportunity_context(collector):
            assert engine._attempt_open_position is not original
            raise RuntimeError("stop")
    assert engine._attempt_open_position is original


def event_frame() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append(
            {
                "stop_model": "FIXED_BASELINE",
                "window_id": f"W{index % 2 + 1:02d}",
                "candidate_ticker": "AAPL",
                "candidate_score": 60 + index,
                "candidate_minus_mean_held_score": index - 5,
                "candidate_forward_return_5_bars_percent": index,
                "mean_held_forward_return_5_bars_percent": index - 1,
                "candidate_minus_mean_held_5_bars_percent": 1.0,
                "candidate_beats_mean_held_5_bars": True,
                "candidate_beats_worst_held_5_bars": True,
                "candidate_beats_all_held_5_bars": index % 2 == 0,
                "candidate_forward_return_10_bars_percent": index,
                "mean_held_forward_return_10_bars_percent": index - 1,
                "candidate_minus_mean_held_10_bars_percent": 1.0,
                "candidate_beats_mean_held_10_bars": True,
                "candidate_beats_worst_held_10_bars": True,
                "candidate_beats_all_held_10_bars": True,
                "candidate_forward_return_20_bars_percent": index,
                "mean_held_forward_return_20_bars_percent": index - 1,
                "candidate_minus_mean_held_20_bars_percent": 1.0,
                "candidate_beats_mean_held_20_bars": True,
                "candidate_beats_worst_held_20_bars": True,
                "candidate_beats_all_held_20_bars": True,
            }
        )
    return pd.DataFrame(rows)


def test_summaries_and_score_quintiles_are_auditable():
    events = event_frame()
    summary = summarize_events(
        events, group_columns=("stop_model",), horizons=(5, 10, 20)
    )
    assert summary.iloc[0]["average_opportunity_gap_20_bars_percent"] == 1
    quintiles = build_score_quintiles(events, (5, 10, 20))
    assert quintiles["score_premium_quintile"].nunique() == 5


def test_screen_requires_score_predictiveness_and_both_strata():
    summary = pd.DataFrame(
        [
            {
                "stop_model": "FIXED_BASELINE",
                "average_opportunity_gap_10_bars_percent": 1,
                "average_opportunity_gap_20_bars_percent": 1,
                "candidate_beat_mean_rate_20_bars_percent": 60,
                "score_premium_spearman_to_gap_20_bars": 0.05,
            }
        ]
    )
    windows = pd.DataFrame(
        {
            "stop_model": ["FIXED_BASELINE"] * 13,
            "average_opportunity_gap_20_bars_percent": [1] * 13,
        }
    )
    result = build_screen(summary, windows)
    assert bool(result.iloc[0]["stratum_pass"]) is False
    assert bool(result.iloc[0]["robust_replacement_audit_pass"]) is False


def test_predecessor_rejects_a_previously_passing_policy(tmp_path: Path):
    result_file = tmp_path / "result.csv"
    result_file.write_text("x\n1\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(result_file.read_bytes()).hexdigest()
    provenance = {
        "snapshot_fingerprint": "fp",
        "source_stamp": "stop",
        "result_files": {
            "result": {"path": str(result_file), "sha256": digest}
        },
    }
    payload = {
        "screen": {"results": [{"robust_screen_pass": True}]}
    }
    (tmp_path / "portfolio_entry_state_ablation_provenance_entry.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (tmp_path / "portfolio_entry_state_ablation_entry.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="passing policy"):
        _verify_entry_predecessor(
            directory=tmp_path,
            stamp="entry",
            snapshot_fingerprint="fp",
            source_stamp="stop",
        )


def test_save_writes_result_hashes(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source_files = []
    for name in ("stop.json", "windows.csv", "runs.csv", "stop_prov.json", "entry.json", "entry_prov.json"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        source_files.append(path)
    frames = {
        name: pd.DataFrame([{"value": 1}])
        for name in (
            "events",
            "holdings",
            "runs",
            "summary",
            "windows",
            "tickers",
            "score_quintiles",
            "screen",
        )
    }
    bundle = {
        **frames,
        "snapshot_id": "snap",
        "snapshot_fingerprint": "fp",
        "snapshot_manifest_path": manifest,
        "stop_stamp": "stop",
        "stop_paths": {
            "json": source_files[0],
            "windows": source_files[1],
            "test_runs": source_files[2],
        },
        "stop_provenance_path": source_files[3],
        "entry_stamp": "entry",
        "entry_json_path": source_files[4],
        "entry_provenance_path": source_files[5],
        "stop_models": ["FIXED_BASELINE"],
        "horizons": [5, 10, 20],
    }
    paths = save_slot_opportunity_audit(bundle, output_directory=tmp_path / "out")
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text())
    assert len(provenance["result_files"]) == 9
