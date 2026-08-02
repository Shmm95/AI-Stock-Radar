from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_risk_ablation import (
    build_risk_grid,
    build_risk_variant_config,
    normalize_maximum_total_risk_values,
    normalize_risk_per_trade_values,
    rank_risk_candidates,
    run_portfolio_risk_ablation,
)


def _synthetic_market_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    output: dict[str, pd.DataFrame] = {}

    for offset, ticker in enumerate(tickers):
        base = 100.0 + offset
        output[ticker] = pd.DataFrame(
            {
                "Open": [
                    base,
                    base,
                    base + 2,
                    base + 3,
                    base + 4,
                    base + 5,
                    base + 6,
                    base + 7,
                    base + 8,
                    base + 9,
                ],
                "High": [base + index + 2 for index in range(10)],
                "Low": [base + max(index - 1, 0) for index in range(10)],
                "Close": [base + index for index in range(10)],
                "EMA20": [
                    base,
                    base + 1,
                    base + 1.5,
                    base + 2,
                    base + 3,
                    base + 4,
                    base + 5,
                    base + 6,
                    base + 7,
                    base + 8,
                ],
                "EMA50": [
                    base + 1,
                    base,
                    base,
                    base,
                    base + 1,
                    base + 2,
                    base + 3,
                    base + 4,
                    base + 5,
                    base + 6,
                ],
                "RSI14": [40.0, 57.5, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0],
                "RegimeAllowed": [True] * 10,
            },
            index=dates,
        )
    return output


def test_normalize_risk_values() -> None:
    assert normalize_risk_per_trade_values([1.0, 0.5, 1.0, 0.75]) == (
        0.5,
        0.75,
        1.0,
    )
    assert normalize_maximum_total_risk_values([5, 3, 4, 4]) == (
        3.0,
        4.0,
        5.0,
    )


def test_invalid_risk_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalize_risk_per_trade_values([0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        normalize_maximum_total_risk_values([-1.0, 4.0])


def test_build_risk_grid_is_cartesian_and_deterministic() -> None:
    grid = build_risk_grid(
        risk_per_trade_values=[1.0, 0.5],
        maximum_total_risk_values=[4.0, 3.0],
    )
    assert [row["variant_id"] for row in grid] == [
        "R0p50_T3p00",
        "R0p50_T4p00",
        "R1p00_T3p00",
        "R1p00_T4p00",
    ]


def test_build_risk_variant_config_keeps_other_fields() -> None:
    base = PortfolioBacktestConfig(
        initial_cash=12_345.0,
        risk_per_trade_percent=1.0,
        maximum_total_open_risk_percent=4.0,
        maximum_open_positions=4,
    )
    variant = build_risk_variant_config(
        base,
        risk_per_trade_percent=0.75,
        maximum_total_open_risk_percent=3.0,
        maximum_open_positions=4,
    )
    assert variant.risk_per_trade_percent == pytest.approx(0.75)
    assert variant.maximum_total_open_risk_percent == pytest.approx(3.0)
    assert variant.maximum_open_positions == 4
    assert variant.initial_cash == pytest.approx(12_345.0)
    assert base.risk_per_trade_percent == pytest.approx(1.0)


def test_rank_risk_candidates_breaks_complete_tie_toward_baseline() -> None:
    rows = []
    for risk, total in ((0.75, 3.0), (1.0, 4.0), (1.0, 5.0)):
        rows.append(
            {
                "variant_id": f"{risk}-{total}",
                "risk_per_trade_percent": risk,
                "maximum_total_open_risk_percent": total,
                "return_drawdown_ratio": 1.0,
                "excess_return_vs_matched_percent": 1.0,
                "profit_factor": 1.0,
                "total_return_percent": 1.0,
                "maximum_drawdown_percent": 1.0,
            }
        )

    ranked = rank_risk_candidates(pd.DataFrame(rows))
    assert ranked.iloc[0]["variant_id"] == "1.0-4.0"
    assert bool(ranked.iloc[0]["selected_candidate"])


def test_integration_runs_complete_risk_grid() -> None:
    data = _synthetic_market_data(["AAA", "BBB", "CCC", "DDD", "EEE"])
    base = PortfolioBacktestConfig(
        initial_cash=10_000.0,
        risk_per_trade_percent=1.0,
        maximum_position_percent=25.0,
        maximum_total_open_risk_percent=4.0,
        maximum_crypto_allocation_percent=100.0,
        maximum_open_positions=4,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        force_close_at_end=True,
    )

    bundle = run_portfolio_risk_ablation(
        data_by_ticker=data,
        base_config=base,
        risk_per_trade_values=[0.5, 1.0],
        maximum_total_risk_values=[3.0, 4.0],
        maximum_open_positions=4,
    )

    summary = bundle["summary"]
    assert len(summary) == 4
    assert set(summary["risk_per_trade_percent"]) == {0.5, 1.0}
    assert set(summary["maximum_total_open_risk_percent"]) == {3.0, 4.0}
    assert set(summary["configured_maximum_open_positions"]) == {4}
    assert summary["rank"].tolist() == [1, 2, 3, 4]
