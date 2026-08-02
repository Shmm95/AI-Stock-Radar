"""Tests for rolling walk-forward validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.run_walk_forward import (
    CandidateWindowSummary,
    WalkForwardRunRow,
    build_oos_aggregate,
    generate_walk_forward_windows,
    normalize_candidate_values,
    rank_candidate_summaries,
)


def _candidate_summary(
    *,
    trailing: float,
    median_return: float,
    average_return: float,
    profit_factor: float,
    drawdown: float,
    profitable_tickers: int = 8,
) -> CandidateWindowSummary:
    """Create one synthetic candidate summary."""

    return CandidateWindowSummary(
        window_id="W01",
        phase="TRAIN",
        rank=0,
        trailing_close_percent=trailing,
        successful_tickers=9,
        failed_tickers=0,
        total_trades=100,
        total_winning_trades=30,
        total_losing_trades=70,
        profitable_tickers=(
            profitable_tickers
        ),
        profitable_tickers_percent=(
            profitable_tickers
            / 9
            * 100
        ),
        outperformed_count=3,
        outperformed_percent=(
            3
            / 9
            * 100
        ),
        average_strategy_return_percent=(
            average_return
        ),
        median_strategy_return_percent=(
            median_return
        ),
        best_ticker_return_percent=20.0,
        worst_ticker_return_percent=-2.0,
        average_benchmark_return_percent=10.0,
        average_excess_return_percent=(
            average_return
            - 10.0
        ),
        median_excess_return_percent=(
            median_return
            - 10.0
        ),
        average_max_drawdown_percent=(
            drawdown
        ),
        average_benchmark_drawdown_percent=15.0,
        average_profit_factor=(
            profit_factor
        ),
        average_win_rate_percent=30.0,
    )


def _run_row(
    *,
    window_id: str,
    ticker: str,
    trailing: float,
    selected: bool,
    strategy_return: float,
    benchmark_return: float = 2.0,
) -> WalkForwardRunRow:
    """Create one synthetic TEST result."""

    excess_return = (
        strategy_return
        - benchmark_return
    )

    return WalkForwardRunRow(
        window_id=window_id,
        phase="TEST",
        ticker=ticker,
        asset_class="EQUITY",
        trailing_close_percent=trailing,
        selected_for_window=selected,
        success=True,
        error="",
        period_start="2025-01-01",
        period_end="2025-06-30",
        prepared_bars=100,
        completed_trades=5,
        winning_trades=2,
        losing_trades=3,
        win_rate_percent=40.0,
        profit_factor=1.5,
        average_trade=10.0,
        total_fees=10.0,
        strategy_return_amount=(
            strategy_return
            * 100
        ),
        strategy_return_percent=(
            strategy_return
        ),
        strategy_max_drawdown_percent=5.0,
        benchmark_return_amount=(
            benchmark_return
            * 100
        ),
        benchmark_return_percent=(
            benchmark_return
        ),
        benchmark_max_drawdown_percent=8.0,
        excess_return_amount=(
            excess_return
            * 100
        ),
        excess_return_percent=(
            excess_return
        ),
        profitable=(
            strategy_return > 0
        ),
        strategy_outperformed=(
            excess_return > 0
        ),
    )


def test_normalize_candidate_values_sorts_and_deduplicates() -> None:
    """Candidates must be unique and sorted."""

    result = normalize_candidate_values(
        [
            15.0,
            7.5,
            15.0,
        ]
    )

    assert result == [
        7.5,
        15.0,
    ]


def test_normalize_candidate_values_requires_two_candidates() -> None:
    """Walk-forward selection requires at least two candidates."""

    with pytest.raises(
        ValueError
    ):
        normalize_candidate_values(
            [
                7.5,
            ]
        )


def test_generate_walk_forward_windows_uses_future_test_period() -> None:
    """Every test period must begin after its training period."""

    windows = generate_walk_forward_windows(
        common_start=pd.Timestamp(
            "2021-01-01"
        ),
        common_end=pd.Timestamp(
            "2024-12-31"
        ),
        train_months=12,
        test_months=6,
        step_months=6,
    )

    assert len(windows) > 1

    first = windows[0]

    assert first.window_id == "W01"

    assert first.train_start == pd.Timestamp(
        "2021-01-01"
    )

    assert first.train_end == pd.Timestamp(
        "2022-01-01"
    )

    assert first.test_start == pd.Timestamp(
        "2022-01-01"
    )

    assert first.test_end == pd.Timestamp(
        "2022-07-01"
    )

    for window in windows:
        assert (
            window.train_end
            == window.test_start
        )

        assert (
            window.test_end
            > window.test_start
        )


def test_training_ranking_prioritizes_median_return() -> None:
    """Median return is the primary performance criterion."""

    summaries = [
        _candidate_summary(
            trailing=7.5,
            median_return=5.0,
            average_return=6.0,
            profit_factor=1.5,
            drawdown=5.0,
        ),
        _candidate_summary(
            trailing=15.0,
            median_return=7.0,
            average_return=5.0,
            profit_factor=1.2,
            drawdown=6.0,
        ),
    ]

    ranked = rank_candidate_summaries(
        summaries
    )

    assert (
        ranked[0]
        .trailing_close_percent
        == pytest.approx(15.0)
    )

    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_dynamic_aggregate_uses_only_selected_rows() -> None:
    """Dynamic aggregation must ignore unselected test candidates."""

    rows = [
        _run_row(
            window_id="W01",
            ticker="AAPL",
            trailing=7.5,
            selected=True,
            strategy_return=4.0,
        ),
        _run_row(
            window_id="W01",
            ticker="AAPL",
            trailing=15.0,
            selected=False,
            strategy_return=20.0,
        ),
        _run_row(
            window_id="W02",
            ticker="AAPL",
            trailing=7.5,
            selected=False,
            strategy_return=-10.0,
        ),
        _run_row(
            window_id="W02",
            ticker="AAPL",
            trailing=15.0,
            selected=True,
            strategy_return=6.0,
        ),
    ]

    summary = build_oos_aggregate(
        rows=rows,
        label="DYNAMIC",
        selected_only=True,
    )

    assert summary.successful_runs == 2
    assert summary.windows == 2

    assert (
        summary.average_strategy_return_percent
        == pytest.approx(5.0)
    )

    assert (
        summary.median_strategy_return_percent
        == pytest.approx(5.0)
    )

    assert summary.total_trades == 10