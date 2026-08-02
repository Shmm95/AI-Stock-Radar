from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import run_matched_benchmark
from src.backtest.run_portfolio_walk_forward import (
    build_walk_forward_windows,
    compound_returns,
    maximum_drawdown_percent,
    rank_training_candidates,
    run_portfolio_walk_forward,
    slice_prepared_data,
)


def _prepared_frame(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="B")
    sequence = np.arange(len(index), dtype=float)
    close = 100 + sequence * 0.05 + np.sin(sequence / 8) * 2
    open_price = close * 0.999
    high = np.maximum(open_price, close) * 1.01
    low = np.minimum(open_price, close) * 0.99

    rsi = np.full(len(index), 55.0)
    rsi[::20] = 40.0

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "EMA20": close * 0.99,
            "EMA50": close * 0.98,
            "RSI14": rsi,
            "RegimeAllowed": True,
        },
        index=index,
    )


def test_build_walk_forward_windows_uses_only_complete_test_windows() -> None:
    data = {"AAA": _prepared_frame("2020-01-01", "2022-12-30")}

    windows = build_walk_forward_windows(
        data,
        train_months=12,
        test_months=6,
        step_months=6,
    )

    assert [window.window_id for window in windows] == ["W01", "W02", "W03"]
    assert windows[0].train_start == pd.Timestamp("2020-01-01")
    assert windows[0].test_start == pd.Timestamp("2021-01-01")
    assert windows[-1].test_end_exclusive == pd.Timestamp("2022-07-01")


def test_slice_prepared_data_uses_half_open_interval() -> None:
    frame = _prepared_frame("2022-01-01", "2022-03-31")
    sliced = slice_prepared_data(
        {"AAA": frame},
        start=pd.Timestamp("2022-02-01"),
        end_exclusive=pd.Timestamp("2022-03-01"),
    )["AAA"]

    assert sliced.index.min() >= pd.Timestamp("2022-02-01")
    assert sliced.index.max() < pd.Timestamp("2022-03-01")


def test_rank_training_candidates_uses_rank_aggregation() -> None:
    summary = pd.DataFrame(
        [
            {
                "maximum_open_positions": 3,
                "return_drawdown_ratio": 1.0,
                "excess_return_vs_matched_percent": 1.0,
                "profit_factor": 1.1,
                "total_return_percent": 5.0,
                "maximum_drawdown_percent": 5.0,
            },
            {
                "maximum_open_positions": 4,
                "return_drawdown_ratio": 2.0,
                "excess_return_vs_matched_percent": 3.0,
                "profit_factor": 1.5,
                "total_return_percent": 8.0,
                "maximum_drawdown_percent": 4.0,
            },
            {
                "maximum_open_positions": 5,
                "return_drawdown_ratio": 1.5,
                "excess_return_vs_matched_percent": 2.0,
                "profit_factor": 1.3,
                "total_return_percent": 7.0,
                "maximum_drawdown_percent": 6.0,
            },
        ]
    )

    ranked = rank_training_candidates(summary, baseline_position_limit=4)

    assert int(ranked.iloc[0]["maximum_open_positions"]) == 4
    assert bool(ranked.iloc[0]["selected_for_test"])


def test_rank_training_candidates_breaks_complete_tie_toward_baseline() -> None:
    rows = []
    for limit in (3, 4, 5):
        rows.append(
            {
                "maximum_open_positions": limit,
                "return_drawdown_ratio": 1.0,
                "excess_return_vs_matched_percent": 1.0,
                "profit_factor": 1.0,
                "total_return_percent": 1.0,
                "maximum_drawdown_percent": 1.0,
            }
        )

    ranked = rank_training_candidates(
        pd.DataFrame(rows),
        baseline_position_limit=4,
    )

    assert int(ranked.iloc[0]["maximum_open_positions"]) == 4


def test_maximum_drawdown_percent() -> None:
    assert maximum_drawdown_percent([100, 120, 90, 110]) == pytest.approx(25.0)


def test_compound_returns() -> None:
    ending = compound_returns([10.0, -10.0], initial_cash=10_000.0)
    assert ending == pytest.approx(9_900.0)


def test_walk_forward_identical_candidates_select_fixed_baseline() -> None:
    data = {"AAA": _prepared_frame("2020-01-01", "2023-12-29")}
    config = PortfolioBacktestConfig(
        initial_cash=10_000.0,
        maximum_open_positions=5,
        minimum_fee=0.0,
    )

    bundle = run_portfolio_walk_forward(
        data_by_ticker=data,
        base_config=config,
        position_limits=(3, 4, 5),
        baseline_position_limit=4,
        train_months=12,
        test_months=6,
        step_months=6,
    )

    assert not bundle["windows"].empty
    assert set(bundle["windows"]["selected_position_limit"]) == {4}
    assert bundle["comparison"]["dynamic_minus_fixed_compounded_return_percent"] == pytest.approx(0.0)
    assert bundle["comparison"]["dynamic_test_window_loss_count"] == 0


def test_matched_benchmark_fractional_quantity_does_not_loop() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    data = {
        "BTC-USD": pd.DataFrame(
            {"Open": [13.0, 13.0], "Close": [13.0, 13.0]},
            index=index,
        )
    }
    config = PortfolioBacktestConfig(
        initial_cash=100.0,
        commission_rate=0.0005,
        minimum_fee=1.0,
        slippage_bps=0.0,
        allow_fractional_crypto=True,
    )

    summary, curve = run_matched_benchmark(
        data,
        initial_cash=100.0,
        exposure_percent=100.0,
        config=config,
        label="TEST",
    )

    assert summary["invested_tickers"] == 1
    assert not curve.empty
    assert summary["ending_equity"] <= 100.0
