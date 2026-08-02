from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_stop_ablation import (
    build_stop_config, build_stop_grid, normalize_stops,
    rank_stop_candidates, run_portfolio_stop_ablation, stop_variant_id,
)


def frame():
    index = pd.date_range("2022-01-03", "2023-12-29", freq="B")
    x = np.arange(len(index), dtype=float); close = 100 + x*.04 + np.sin(x/8)*3
    return pd.DataFrame({"Open": close*.999, "High": close*1.01,
        "Low": close*.99, "Close": close, "EMA20": close*.99,
        "EMA50": close*.98, "RSI14": np.where(x%18==0, 40, 56),
        "RegimeAllowed": True}, index=index)


def test_normalization_and_id():
    assert normalize_stops([5, 3.5, 5]) == (3.5, 5.0)
    assert stop_variant_id(3.5, 7.5) == "S3p50_C7p50"


def test_invalid_stops():
    with pytest.raises(ValueError): normalize_stops([])
    with pytest.raises(ValueError): normalize_stops([0])


def test_grid_is_cartesian():
    grid = build_stop_grid([3.5, 5], [5, 7.5])
    assert len(grid) == 4


def test_config_changes_only_stops():
    base = PortfolioBacktestConfig()
    changed = build_stop_config(base, 3.5, 7.5)
    assert changed.stock_stop_loss_percent == 3.5
    assert changed.crypto_stop_loss_percent == 7.5
    assert changed.risk_per_trade_percent == base.risk_per_trade_percent


def test_ranking_prefers_baseline_on_complete_tie():
    rows=[]
    for s,c in ((3.5,3.5),(5.,5.)):
        rows.append(dict(variant_id=stop_variant_id(s,c), stock_stop_loss_percent=s,
            crypto_stop_loss_percent=c, return_drawdown_ratio=2.,
            excess_return_vs_matched_percent=1., profit_factor=1.5,
            total_return_percent=10., maximum_drawdown_percent=5.))
    ranked=rank_stop_candidates(pd.DataFrame(rows))
    assert ranked.iloc[0].variant_id == "S5p00_C5p00"


def test_integration_produces_all_variants():
    base=PortfolioBacktestConfig(initial_cash=10000, maximum_position_percent=100,
        maximum_crypto_allocation_percent=100, commission_rate=0,
        minimum_fee=0, slippage_bps=0)
    bundle=run_portfolio_stop_ablation(data_by_ticker={"AAA":frame()},
        base_config=base, stock_stops=[3.5,5], crypto_stops=[5])
    assert len(bundle["summary"]) == 2
    assert set(bundle["summary"].variant_id) == {"S3p50_C5p00","S5p00_C5p00"}
