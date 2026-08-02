from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_risk_walk_forward import (
    MODEL_BASELINE,
    MODEL_BALANCED,
    MODEL_AGGRESSIVE,
    MODEL_DYNAMIC,
    RiskProfile,
    _comparison_counts,
    _profile_from_row,
    run_portfolio_risk_walk_forward,
)


def _prepared_frame(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="B")
    sequence = np.arange(len(index), dtype=float)
    close = 100 + sequence * 0.04 + np.sin(sequence / 9) * 2.5
    open_price = close * 0.999
    high = np.maximum(open_price, close) * 1.01
    low = np.minimum(open_price, close) * 0.99
    rsi = np.full(len(index), 56.0)
    rsi[::18] = 40.0

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


def test_risk_profile_normalizes_and_builds_variant_id() -> None:
    profile = RiskProfile(0.7500000001, 4)

    assert profile.risk_per_trade_percent == pytest.approx(0.75)
    assert profile.maximum_total_open_risk_percent == pytest.approx(4.0)
    assert profile.variant_id == "R0p75_T4p00"


def test_risk_profile_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        RiskProfile(0.0, 4.0)
    with pytest.raises(ValueError, match="positive"):
        RiskProfile(0.75, -1.0)


def test_profile_from_row() -> None:
    profile = _profile_from_row(
        {
            "risk_per_trade_percent": 1.0,
            "maximum_total_open_risk_percent": 5.0,
        }
    )
    assert profile == RiskProfile(1.0, 5.0)


def test_comparison_counts_uses_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "advantage": [1.0, -1.0, 0.0, 1e-12],
        }
    )
    counts = _comparison_counts(frame, "advantage")
    assert counts == {"win_count": 1, "tie_count": 2, "loss_count": 1}


def test_walk_forward_requires_fixed_profiles_inside_candidate_grid() -> None:
    data = {"AAA": _prepared_frame("2020-01-01", "2023-12-29")}
    config = PortfolioBacktestConfig(
        initial_cash=10_000.0,
        maximum_open_positions=4,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
    )

    with pytest.raises(ValueError, match="Balanced risk"):
        run_portfolio_risk_walk_forward(
            data_by_ticker=data,
            base_config=config,
            risk_per_trade_values=(1.0,),
            maximum_total_risk_values=(4.0,),
            baseline_profile=RiskProfile(1.0, 4.0),
            balanced_profile=RiskProfile(0.75, 4.0),
            aggressive_profile=RiskProfile(1.0, 4.0),
            train_months=12,
            test_months=6,
            step_months=12,
        )


def test_integration_produces_all_comparison_models() -> None:
    data = {"AAA": _prepared_frame("2020-01-01", "2023-12-29")}
    config = PortfolioBacktestConfig(
        initial_cash=10_000.0,
        risk_per_trade_percent=1.0,
        maximum_position_percent=100.0,
        maximum_total_open_risk_percent=4.0,
        maximum_crypto_allocation_percent=100.0,
        maximum_open_positions=4,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        force_close_at_end=True,
    )

    bundle = run_portfolio_risk_walk_forward(
        data_by_ticker=data,
        base_config=config,
        risk_per_trade_values=(0.75, 1.0),
        maximum_total_risk_values=(4.0,),
        maximum_open_positions=4,
        baseline_profile=RiskProfile(1.0, 4.0),
        balanced_profile=RiskProfile(0.75, 4.0),
        aggressive_profile=RiskProfile(1.0, 4.0),
        train_months=12,
        test_months=6,
        step_months=12,
    )

    assert not bundle["windows"].empty
    assert set(bundle["test_runs"]["model"]) == {
        MODEL_DYNAMIC,
        MODEL_BASELINE,
        MODEL_BALANCED,
        MODEL_AGGRESSIVE,
    }
    assert set(bundle["aggregate"]["model"]) == {
        MODEL_DYNAMIC,
        MODEL_BASELINE,
        MODEL_BALANCED,
        MODEL_AGGRESSIVE,
    }
    assert len(bundle["training_candidates"]) == (
        len(bundle["windows"]) * 2
    )
    assert set(bundle["windows"]["selected_variant_id"]).issubset(
        {"R0p75_T4p00", "R1p00_T4p00"}
    )
    assert bundle["aggregate"]["window_count"].eq(
        len(bundle["windows"])
    ).all()
    assert "balanced_minus_baseline_compounded_return_percent" in bundle[
        "comparison"
    ]
    assert "aggressive_minus_baseline_compounded_return_percent" in bundle[
        "comparison"
    ]
