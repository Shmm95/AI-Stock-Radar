from __future__ import annotations

import math

import pandas as pd
import pytest

from src.backtest.run_portfolio_benchmark_analysis import (
    annual_returns,
    build_analysis,
    calculate_cagr,
    return_drawdown_ratio,
    run_benchmark,
)


def _frame(values: list[float]) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-12-31"])
    return pd.DataFrame({"Open": values, "Close": values}, index=dates)


def _data() -> dict[str, pd.DataFrame]:
    return {"AAPL": _frame([100, 80, 120]), "BTC-USD": _frame([100, 80, 120])}


def _config() -> dict:
    return {
        "commission_rate": 0.0,
        "minimum_fee": 0.0,
        "slippage_bps": 0.0,
        "allow_fractional_stocks": True,
        "allow_fractional_crypto": True,
    }


def _payload() -> dict:
    return {
        "config": _config(),
        "tickers": ["AAPL", "BTC-USD"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_cash": 10_000.0,
        "ending_equity": 11_500.0,
        "total_return_percent": 15.0,
        "maximum_drawdown_percent": 8.0,
        "average_exposure_percent": 50.0,
        "benchmark": {
            "total_return_percent": 20.0,
            "maximum_drawdown_percent": 20.0,
        },
        "equity_curve": [
            {"timestamp": "2024-01-01", "total_equity": 10_000.0},
            {"timestamp": "2024-01-02", "total_equity": 9_200.0},
            {"timestamp": "2024-12-31", "total_equity": 11_500.0},
        ],
    }


def test_matched_exposure_scales_return_and_drawdown() -> None:
    full, _ = run_benchmark(
        _data(), initial_cash=10_000, exposure_percent=100,
        config=_config(), label="FULL"
    )
    matched, _ = run_benchmark(
        _data(), initial_cash=10_000, exposure_percent=50,
        config=_config(), label="MATCHED"
    )
    assert full["total_return_percent"] == pytest.approx(20.0)
    assert full["maximum_drawdown_percent"] == pytest.approx(20.0)
    assert matched["total_return_percent"] == pytest.approx(10.0)
    assert matched["maximum_drawdown_percent"] == pytest.approx(10.0)


def test_analysis_uses_average_exposure() -> None:
    result = build_analysis(
        _payload(), source_json="synthetic.json", data_by_ticker=_data()
    )["summary"]
    assert result["matched_benchmark"]["target_exposure_percent"] == 50.0
    assert result["matched_benchmark"]["total_return_percent"] == 10.0
    assert result["comparison"]["strategy_excess_return_vs_matched_percent"] == 5.0
    assert result["reconciliation"]["return_difference_vs_source"] == 0.0


def test_target_exposure_override() -> None:
    result = build_analysis(
        _payload(), source_json="synthetic.json", data_by_ticker=_data(),
        target_exposure=25.0
    )["summary"]["matched_benchmark"]
    assert result["total_return_percent"] == 5.0
    assert result["maximum_drawdown_percent"] == 5.0


def test_invalid_exposure() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        run_benchmark(
            _data(), initial_cash=10_000, exposure_percent=0,
            config=_config(), label="INVALID"
        )


def test_annual_returns_are_chained() -> None:
    dates = pd.to_datetime(["2023-01-01", "2023-12-31", "2024-12-31"])
    strategy = pd.DataFrame({"timestamp": dates, "total_equity": [10_000, 11_000, 12_100]})
    full = pd.DataFrame({"timestamp": dates, "total_equity": [10_000, 12_000, 12_600]})
    matched = pd.DataFrame({"timestamp": dates, "total_equity": [10_000, 10_500, 11_025]})
    rows = annual_returns(strategy, full, matched, initial_cash=10_000)
    assert rows.loc[1, "strategy_return_percent"] == pytest.approx(10.0)
    assert rows.loc[1, "matched_benchmark_return_percent"] == pytest.approx(5.0)


def test_metric_helpers() -> None:
    cagr = calculate_cagr(
        10_000, 12_100, pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")
    )
    assert cagr == pytest.approx(10.0, abs=0.02)
    assert return_drawdown_ratio(20, 10) == 2.0
    assert math.isinf(return_drawdown_ratio(20, 0))
