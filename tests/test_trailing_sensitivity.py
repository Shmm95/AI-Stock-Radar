"""Tests for trailing sensitivity aggregation."""

from __future__ import annotations

import pytest

from src.backtest.run_trailing_sensitivity import (
    TrailingSensitivityRow,
    normalize_trailing_values,
    summarize_sensitivity,
)


def _row(
    *,
    ticker: str,
    asset_class: str,
    trailing: float,
    strategy_return: float,
    drawdown: float,
    profit_factor: float,
    benchmark_return: float = 5.0,
) -> TrailingSensitivityRow:
    """Create a compact successful test row."""

    excess_return = (
        strategy_return
        - benchmark_return
    )

    return TrailingSensitivityRow(
        ticker=ticker,
        asset_class=asset_class,
        trailing_close_percent=trailing,
        success=True,
        error="",
        prepared_bars=100,
        completed_trades=10,
        winning_trades=4,
        losing_trades=6,
        win_rate_percent=40.0,
        profit_factor=profit_factor,
        average_trade=10.0,
        total_fees=20.0,
        strategy_return_amount=(
            strategy_return
            * 100
        ),
        strategy_return_percent=(
            strategy_return
        ),
        strategy_max_drawdown_percent=(
            drawdown
        ),
        benchmark_return_amount=(
            benchmark_return
            * 100
        ),
        benchmark_return_percent=(
            benchmark_return
        ),
        benchmark_max_drawdown_percent=10.0,
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


def test_normalize_trailing_values_sorts_and_deduplicates() -> None:
    """Distances must be ordered and unique."""

    result = normalize_trailing_values(
        [
            15.0,
            10.0,
            7.5,
            10.0,
            12.5,
        ]
    )

    assert result == [
        7.5,
        10.0,
        12.5,
        15.0,
    ]


def test_normalize_trailing_values_rejects_invalid_values() -> None:
    """Zero, negative, and 100% values are invalid."""

    with pytest.raises(
        ValueError
    ):
        normalize_trailing_values(
            [
                0.0,
            ]
        )

    with pytest.raises(
        ValueError
    ):
        normalize_trailing_values(
            [
                -5.0,
            ]
        )

    with pytest.raises(
        ValueError
    ):
        normalize_trailing_values(
            [
                100.0,
            ]
        )


def test_summaries_are_split_by_asset_class() -> None:
    """ALL, EQUITY, and CRYPTO summaries must be separate."""

    rows = [
        _row(
            ticker="AAPL",
            asset_class="EQUITY",
            trailing=10.0,
            strategy_return=8.0,
            drawdown=5.0,
            profit_factor=1.4,
        ),
        _row(
            ticker="BTC-USD",
            asset_class="CRYPTO",
            trailing=10.0,
            strategy_return=4.0,
            drawdown=9.0,
            profit_factor=1.2,
        ),
    ]

    summaries = summarize_sensitivity(
        rows
    )

    all_summary = next(
        summary
        for summary in summaries
        if (
            summary.scope == "ALL"
            and summary.trailing_close_percent
            == 10.0
        )
    )

    equity_summary = next(
        summary
        for summary in summaries
        if (
            summary.scope == "EQUITY"
            and summary.trailing_close_percent
            == 10.0
        )
    )

    crypto_summary = next(
        summary
        for summary in summaries
        if (
            summary.scope == "CRYPTO"
            and summary.trailing_close_percent
            == 10.0
        )
    )

    assert (
        all_summary.successful_tickers
        == 2
    )

    assert (
        equity_summary.successful_tickers
        == 1
    )

    assert (
        crypto_summary.successful_tickers
        == 1
    )

    assert (
        all_summary.average_strategy_return_percent
        == pytest.approx(6.0)
    )


def test_ranking_prioritizes_median_return() -> None:
    """The robust ranking should prioritize median return."""

    rows = [
        _row(
            ticker="AAPL",
            asset_class="EQUITY",
            trailing=7.5,
            strategy_return=4.0,
            drawdown=5.0,
            profit_factor=1.2,
        ),
        _row(
            ticker="MSFT",
            asset_class="EQUITY",
            trailing=7.5,
            strategy_return=4.0,
            drawdown=5.0,
            profit_factor=1.2,
        ),
        _row(
            ticker="AAPL",
            asset_class="EQUITY",
            trailing=10.0,
            strategy_return=3.0,
            drawdown=4.0,
            profit_factor=2.0,
        ),
        _row(
            ticker="MSFT",
            asset_class="EQUITY",
            trailing=10.0,
            strategy_return=20.0,
            drawdown=4.0,
            profit_factor=2.0,
        ),
    ]

    summaries = summarize_sensitivity(
        rows
    )

    all_ranked = sorted(
        [
            summary
            for summary in summaries
            if summary.scope == "ALL"
        ],
        key=lambda summary: (
            summary.rank
        ),
    )

    assert (
        all_ranked[0]
        .trailing_close_percent
        == pytest.approx(10.0)
    )

    assert all_ranked[0].rank == 1
    assert all_ranked[1].rank == 2