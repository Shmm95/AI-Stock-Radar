from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_stop_walk_forward import (
    MODEL_BASELINE, MODEL_DYNAMIC, MODEL_LOW_DRAWDOWN, MODEL_MAX_RETURN,
    MODEL_RANK_WINNER, StopProfile, _profile_from_row,
    run_portfolio_stop_walk_forward,
)


def _frame(start="2020-01-01", end="2024-12-31"):
    index = pd.date_range(start, end, freq="B")
    x = np.arange(len(index), dtype=float)
    close = 100 + x * .035 + np.sin(x / 9) * 3.0
    return pd.DataFrame({"Open":close*.999, "High":close*1.012,
        "Low":close*.988, "Close":close, "EMA20":close*.99,
        "EMA50":close*.98, "RSI14":np.where(x%19==0,40,56),
        "RegimeAllowed":True}, index=index)


def _config():
    return PortfolioBacktestConfig(initial_cash=10_000,
        maximum_position_percent=100, maximum_crypto_allocation_percent=100,
        commission_rate=0, minimum_fee=0, slippage_bps=0)


def test_stop_profile_normalizes_and_builds_id():
    profile = StopProfile(5.0000000001, 3.5)
    assert profile.stock_stop_loss_percent == pytest.approx(5.0)
    assert profile.variant_id == "S5p00_C3p50"


def test_stop_profile_rejects_invalid_values():
    with pytest.raises(ValueError): StopProfile(0, 3.5)
    with pytest.raises(ValueError): StopProfile(5, 101)


def test_profile_from_row():
    assert _profile_from_row({"stock_stop_loss_percent":5,
        "crypto_stop_loss_percent":3.5}) == StopProfile(5,3.5)


def test_requires_controls_in_candidate_grid():
    with pytest.raises(ValueError, match="Low drawdown stock"):
        run_portfolio_stop_walk_forward(data_by_ticker={"AAA":_frame()},
            base_config=_config(), stock_stops=(3.5,5), crypto_stops=(3.5,5),
            train_months=12, test_months=6, step_months=12)


def test_integration_produces_all_models_and_oos_windows():
    bundle = run_portfolio_stop_walk_forward(
        data_by_ticker={"AAA":_frame()}, base_config=_config(),
        stock_stops=(3.5,5,7.5), crypto_stops=(3.5,5),
        baseline_profile=StopProfile(5,5), rank_winner_profile=StopProfile(5,3.5),
        max_return_profile=StopProfile(3.5,3.5),
        low_drawdown_profile=StopProfile(7.5,3.5),
        train_months=12, test_months=6, step_months=12)
    models = {MODEL_DYNAMIC, MODEL_BASELINE, MODEL_RANK_WINNER,
              MODEL_MAX_RETURN, MODEL_LOW_DRAWDOWN}
    assert set(bundle["test_runs"].model) == models
    assert set(bundle["aggregate"].model) == models
    assert len(bundle["training_candidates"]) == len(bundle["windows"]) * 6
    assert bundle["aggregate"].window_count.eq(len(bundle["windows"])).all()
    assert "dynamic_minus_baseline_compounded_return_percent" in bundle["comparison"]


def test_duplicate_control_profiles_remain_separate_models():
    same = StopProfile(5,5)
    bundle = run_portfolio_stop_walk_forward(
        data_by_ticker={"AAA":_frame("2020-01-01","2023-12-29")},
        base_config=_config(), stock_stops=(5,), crypto_stops=(5,),
        baseline_profile=same, rank_winner_profile=same,
        max_return_profile=same, low_drawdown_profile=same,
        train_months=12, test_months=6, step_months=12)
    assert bundle["aggregate"].compounded_return_percent.nunique() == 1
    assert len(bundle["aggregate"]) == 5
