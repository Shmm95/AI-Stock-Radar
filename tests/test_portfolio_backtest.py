"""Regression tests for shared-cash portfolio backtesting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig


def _data(rows: list[dict[str, float | bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        index=pd.date_range("2026-01-01", periods=len(rows), freq="D"),
    )


def _config(**overrides) -> PortfolioBacktestConfig:
    defaults = dict(
        initial_cash=10_000.0,
        risk_per_trade_percent=1.0,
        maximum_position_percent=100.0,
        maximum_total_open_risk_percent=10.0,
        maximum_crypto_allocation_percent=100.0,
        maximum_open_positions=4,
        stock_stop_loss_percent=5.0,
        crypto_stop_loss_percent=5.0,
        stock_trailing_close_percent=50.0,
        crypto_trailing_close_percent=50.0,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        allow_fractional_stocks=False,
        allow_fractional_crypto=True,
        force_close_at_end=False,
    )
    defaults.update(overrides)
    return PortfolioBacktestConfig(**defaults)


def test_buy_signal_executes_at_next_open() -> None:
    data = _data(
        [
            {"Open": 100, "High": 101, "Low": 99, "Close": 100, "EMA20": 99, "EMA50": 100, "RSI14": 50},
            {"Open": 100, "High": 103, "Low": 99, "Close": 102, "EMA20": 101, "EMA50": 100, "RSI14": 55},
            {"Open": 104, "High": 106, "Low": 103, "Close": 105, "EMA20": 102, "EMA50": 100, "RSI14": 56},
        ]
    )

    result = run_portfolio_backtest(
        data_by_ticker={"AAA": data},
        config=_config(),
        include_benchmark=False,
    )

    assert len(result.open_positions) == 1
    position = result.open_positions[0]
    assert position.entry_timestamp.startswith("2026-01-03")
    assert position.entry_price == pytest.approx(104.0)


def test_same_day_candidate_ranking_controls_single_slot() -> None:
    weak = _data(
        [
            {"Open": 100, "High": 101, "Low": 99, "Close": 100, "EMA20": 99, "EMA50": 100, "RSI14": 50},
            {"Open": 100, "High": 103, "Low": 99, "Close": 102, "EMA20": 101, "EMA50": 100, "RSI14": 45},
            {"Open": 103, "High": 105, "Low": 102, "Close": 104, "EMA20": 102, "EMA50": 100, "RSI14": 50},
        ]
    )
    strong = _data(
        [
            {"Open": 100, "High": 101, "Low": 99, "Close": 100, "EMA20": 99, "EMA50": 100, "RSI14": 50},
            {"Open": 100, "High": 110, "Low": 99, "Close": 109, "EMA20": 104, "EMA50": 100, "RSI14": 57.5},
            {"Open": 110, "High": 112, "Low": 109, "Close": 111, "EMA20": 105, "EMA50": 100, "RSI14": 58},
        ]
    )

    result = run_portfolio_backtest(
        data_by_ticker={"WEAK": weak, "STRONG": strong},
        config=_config(maximum_open_positions=1),
        include_benchmark=False,
    )

    assert len(result.open_positions) == 1
    assert result.open_positions[0].ticker == "STRONG"
    assert any(
        item.ticker == "WEAK" and item.reason_code == "MAX_OPEN_POSITIONS"
        for item in result.rejections
    )


def test_gap_stop_uses_actual_open() -> None:
    data = _data(
        [
            {"Open": 100, "High": 101, "Low": 99, "Close": 100, "EMA20": 99, "EMA50": 100, "RSI14": 50},
            {"Open": 100, "High": 103, "Low": 99, "Close": 102, "EMA20": 101, "EMA50": 100, "RSI14": 55},
            {"Open": 100, "High": 102, "Low": 99, "Close": 101, "EMA20": 101, "EMA50": 100, "RSI14": 55},
            {"Open": 90, "High": 92, "Low": 88, "Close": 91, "EMA20": 99, "EMA50": 100, "RSI14": 40},
        ]
    )

    result = run_portfolio_backtest(
        data_by_ticker={"AAA": data},
        config=_config(),
        include_benchmark=False,
    )

    assert result.completed_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "GAP_STOP_LOSS"
    assert trade.exit_price == pytest.approx(90.0)


def test_total_open_risk_limit_rejects_second_position() -> None:
    rows = [
        {"Open": 100, "High": 101, "Low": 99, "Close": 100, "EMA20": 99, "EMA50": 100, "RSI14": 50},
        {"Open": 100, "High": 103, "Low": 99, "Close": 102, "EMA20": 101, "EMA50": 100, "RSI14": 55},
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "EMA20": 101, "EMA50": 100, "RSI14": 55},
    ]

    result = run_portfolio_backtest(
        data_by_ticker={"AAA": _data(rows), "BBB": _data(rows)},
        config=_config(
            maximum_open_positions=2,
            maximum_total_open_risk_percent=1.0,
        ),
        include_benchmark=False,
    )

    assert len(result.open_positions) == 1
    assert any(
        item.reason_code == "MAX_TOTAL_OPEN_RISK"
        for item in result.rejections
    )


def test_crypto_regime_blocks_entry_until_gate_opens() -> None:
    data = _data(
        [
            {"Open": 100, "High": 101, "Low": 99, "Close": 100, "EMA20": 99, "EMA50": 100, "RSI14": 50, "RegimeAllowed": False},
            {"Open": 100, "High": 103, "Low": 99, "Close": 102, "EMA20": 101, "EMA50": 100, "RSI14": 55, "RegimeAllowed": False},
            {"Open": 102, "High": 104, "Low": 101, "Close": 103, "EMA20": 102, "EMA50": 100, "RSI14": 55, "RegimeAllowed": True},
            {"Open": 104, "High": 106, "Low": 103, "Close": 105, "EMA20": 103, "EMA50": 100, "RSI14": 56, "RegimeAllowed": True},
        ]
    )

    result = run_portfolio_backtest(
        data_by_ticker={"BTC-USD": data},
        config=_config(),
        include_benchmark=False,
    )

    assert len(result.open_positions) == 1
    assert result.open_positions[0].entry_timestamp.startswith("2026-01-04")
    assert result.open_positions[0].entry_price == pytest.approx(104.0)



def test_fractional_crypto_affordability_never_loops_or_overdraws_cash() -> None:
    data = _data(
        [
            {
                "Open": 13,
                "High": 13,
                "Low": 13,
                "Close": 13,
                "EMA20": 12,
                "EMA50": 13,
                "RSI14": 50,
                "RegimeAllowed": True,
            },
            {
                "Open": 13,
                "High": 14,
                "Low": 13,
                "Close": 14,
                "EMA20": 13.5,
                "EMA50": 13,
                "RSI14": 55,
                "RegimeAllowed": True,
            },
            {
                "Open": 13,
                "High": 14,
                "Low": 13,
                "Close": 13.5,
                "EMA20": 13.4,
                "EMA50": 13,
                "RSI14": 55,
                "RegimeAllowed": True,
            },
        ]
    )

    result = run_portfolio_backtest(
        data_by_ticker={"BTC-USD": data},
        config=_config(
            initial_cash=100.0,
            risk_per_trade_percent=100.0,
            maximum_position_percent=100.0,
            maximum_total_open_risk_percent=100.0,
            minimum_fee=1.0,
        ),
        include_benchmark=False,
    )

    assert len(result.open_positions) == 1

    position = result.open_positions[0]

    assert position.quantity == pytest.approx(7.615384)
    assert result.ending_cash >= 0.0
    assert (
        position.quantity * position.entry_price
        + position.entry_fee
        <= 100.0 + 1e-9
    )
