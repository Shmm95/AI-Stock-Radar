from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    annual_returns_from_equity,
    build_variant_config,
    concentration_metrics,
    normalize_position_limits,
    rejection_counts,
    run_position_ablation,
    ticker_contributions,
)


def _equity_point(timestamp: str, equity: float) -> SimpleNamespace:
    return SimpleNamespace(timestamp=timestamp, total_equity=equity)


def _trade(ticker: str, pnl: float, fee: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        asset_class="EQUITY",
        net_pnl=pnl,
        total_fees=fee,
    )


def _synthetic_market_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    result: dict[str, pd.DataFrame] = {}

    for offset, ticker in enumerate(tickers):
        base = 100.0 + offset
        result[ticker] = pd.DataFrame(
            {
                "Open": [base, base, base + 2, base + 3, base + 4, base + 5],
                "High": [base + 1, base + 2, base + 4, base + 5, base + 6, base + 7],
                "Low": [base - 1, base - 1, base + 1, base + 2, base + 3, base + 4],
                "Close": [base, base + 2, base + 3, base + 4, base + 5, base + 6],
                "EMA20": [base, base + 1, base + 1.5, base + 2, base + 3, base + 4],
                "EMA50": [base + 1, base, base, base, base + 1, base + 2],
                "RSI14": [40.0, 57.5, 58.0, 59.0, 60.0, 61.0],
                "RegimeAllowed": [True] * 6,
            },
            index=dates,
        )

    return result


def test_normalize_position_limits() -> None:
    assert normalize_position_limits([5, 3, 4, 4]) == (3, 4, 5)


def test_invalid_position_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalize_position_limits([0, 3])


def test_build_variant_config_changes_only_limit() -> None:
    base = PortfolioBacktestConfig(maximum_open_positions=4)
    variant = build_variant_config(base, 5)
    assert variant.maximum_open_positions == 5
    assert variant.initial_cash == base.initial_cash
    assert base.maximum_open_positions == 4


def test_annual_returns_are_chained() -> None:
    curve = [
        _equity_point("2023-01-01", 10_000),
        _equity_point("2023-12-31", 11_000),
        _equity_point("2024-12-31", 12_100),
    ]
    annual = annual_returns_from_equity(curve, initial_cash=10_000)
    assert annual.loc[0, "return_percent"] == pytest.approx(10.0)
    assert annual.loc[1, "return_percent"] == pytest.approx(10.0)


def test_ticker_contribution_and_concentration() -> None:
    result = SimpleNamespace(
        tickers=("AAA", "BBB", "CCC"),
        trades=[
            _trade("AAA", 100),
            _trade("AAA", -20),
            _trade("BBB", 50),
            _trade("CCC", -40),
        ],
    )
    frame = ticker_contributions(result)
    metrics = concentration_metrics(frame)
    assert frame.loc[frame["ticker"] == "AAA", "net_pnl"].iloc[0] == 80
    assert metrics["profitable_ticker_count"] == 2
    assert metrics["losing_ticker_count"] == 1
    assert metrics["largest_positive_ticker"] == "AAA"


def test_rejection_counts() -> None:
    result = SimpleNamespace(
        rejections=[
            SimpleNamespace(reason_code="MAX_OPEN_POSITIONS"),
            SimpleNamespace(reason_code="MAX_OPEN_POSITIONS"),
            SimpleNamespace(reason_code="MAX_CRYPTO_ALLOCATION"),
        ]
    )
    assert rejection_counts(result) == {
        "MAX_OPEN_POSITIONS": 2,
        "MAX_CRYPTO_ALLOCATION": 1,
    }


def test_integration_obeys_each_position_limit() -> None:
    data = _synthetic_market_data(
        ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    )
    base = PortfolioBacktestConfig(
        initial_cash=10_000,
        risk_per_trade_percent=1.0,
        maximum_position_percent=20.0,
        maximum_total_open_risk_percent=10.0,
        maximum_crypto_allocation_percent=100.0,
        maximum_open_positions=5,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        force_close_at_end=True,
    )

    bundle = run_position_ablation(
        data_by_ticker=data,
        base_config=base,
        position_limits=[3, 4, 5],
    )
    summary = bundle["summary"].set_index("maximum_open_positions")

    assert summary.loc[3, "maximum_open_positions_observed"] == 3
    assert summary.loc[4, "maximum_open_positions_observed"] == 4
    assert summary.loc[5, "maximum_open_positions_observed"] == 5
    assert summary.loc[3, "max_open_position_rejections"] > 0
