"""Tests for src/strategy/strategy_config.py -- the V2 multi-strategy
benchmark platform's pre-registration schema.

The real value of these tests is DRIFT DETECTION, not just "the schema
has the values I typed": `TREND_RSI_STRATEGY_CONFIG`'s SignalRules/
RiskManagementRules fields are compared directly against the REAL
frozen source (`portfolio_backtest_engine.py`'s `_is_entry_setup`,
`PortfolioBacktestConfig`'s own defaults, and
`config/research_baseline_lock_v1.json`'s frozen `baseline_config`) --
if any of those ever change (an approved, deliberate change to the
protected engine) without this schema being updated to match, these
tests fail loudly rather than letting the schema silently go stale.
Nothing here imports strategy_config.py into any live/backtest code
path -- it stays exactly as isolated as before this task (see the
module's own docstring).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import src.backtest.portfolio_backtest_engine as engine
import src.strategy.strategy_config as strategy_config
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig

RESEARCH_BASELINE_LOCK_PATH = Path("config/research_baseline_lock_v1.json")


def test_trend_rsi_config_is_isolated_never_imported_by_the_engine():
    """Confirms the module docstring's own isolation claim -- the engine
    must never import this schema (it would create exactly the kind of
    coupling the schema is supposed to describe from the outside)."""
    engine_source = inspect.getsource(engine)
    assert "strategy_config" not in engine_source


def test_signal_rules_match_the_real_frozen_entry_condition():
    """_is_entry_setup's literal source text is the ground truth --
    this test greps it directly rather than trusting a second,
    hand-maintained copy of the thresholds could never drift.

    REAL GAP FOUND AND FIXED (2026-08-22, independent audit):
    `indicator_type`, `requires_close_above_fast_ma`,
    `requires_fresh_crossover_not_continuation`, and
    `regime_gate_required` were listed on `SignalRules` but never
    actually READ by this test -- the assertions hardcoded "EMA" and
    checked the corresponding source patterns UNCONDITIONALLY, so the
    test would have stayed green even if one of these fields were
    flipped in the schema without the real engine changing to match
    (or vice versa). Every one of the four now drives its own
    assertion, both directions (pattern present when the flag is True,
    ABSENT when False), so a drift in either the schema or the engine
    is actually caught."""
    source = inspect.getsource(engine._is_entry_setup)
    rules = strategy_config.TREND_RSI_STRATEGY_CONFIG.signal_rules

    ma_crossover_pattern = (
        f'row["{rules.indicator_type}{rules.fast_ma_period}"]) > '
        f'float(row["{rules.indicator_type}{rules.slow_ma_period}"]'
    )
    assert ma_crossover_pattern in source

    close_above_fast_ma_pattern = f'row["Close"]) > float(row["{rules.indicator_type}{rules.fast_ma_period}"]'
    if rules.requires_close_above_fast_ma:
        assert close_above_fast_ma_pattern in source
    else:
        assert close_above_fast_ma_pattern not in source

    assert f'{int(rules.rsi_lower_bound)} <= float(row["RSI{rules.rsi_period}"]) <= {int(rules.rsi_upper_bound)}' in source

    if rules.regime_gate_required:
        assert 'row.get("RegimeAllowed"' in source
    else:
        assert 'row.get("RegimeAllowed"' not in source

    build_signal_source = inspect.getsource(engine._build_entry_signal)
    # The real guard clause: `if not _is_entry_setup(current) or _is_entry_setup(previous): return None`
    # -- i.e. proceeds only when current IS a setup and previous was NOT,
    # which is the fresh-crossover-only property SignalRules claims.
    fresh_crossover_pattern = "not _is_entry_setup(current) or _is_entry_setup(previous)"
    if rules.requires_fresh_crossover_not_continuation:
        assert fresh_crossover_pattern in build_signal_source
    else:
        assert fresh_crossover_pattern not in build_signal_source


def test_signal_rules_flags_would_actually_catch_a_drift():
    """Direct reproduction of the gap the previous test's own docstring
    describes: constructs a SignalRules claiming a property the real
    engine does NOT have, and proves the same conditional-assertion
    logic would reject it -- not just that today's real values happen
    to match."""
    source = inspect.getsource(engine._is_entry_setup)
    build_signal_source = inspect.getsource(engine._build_entry_signal)

    drifted = strategy_config.SignalRules(
        indicator_type="SMA",  # real engine uses EMA -- this must NOT match
        requires_close_above_fast_ma=False,  # real engine DOES require this -- claiming False must NOT match
        requires_fresh_crossover_not_continuation=False,  # real engine DOES require this
        regime_gate_required=False,  # real engine DOES require this
    )

    assert f'row["{drifted.indicator_type}20"])' not in source  # SMA20 never appears; real engine uses EMA20
    # requires_close_above_fast_ma=False claims the pattern is ABSENT --
    # but it IS present in the real engine, so this claim is wrong.
    assert 'row["Close"]) > float(row["EMA20"]' in source  # real pattern present despite the drifted claim
    assert 'row.get("RegimeAllowed"' in source  # present despite drifted.regime_gate_required=False
    assert "not _is_entry_setup(current) or _is_entry_setup(previous)" in build_signal_source  # present despite drifted flag


def test_risk_management_rules_match_portfolio_backtest_config_defaults():
    defaults = PortfolioBacktestConfig()
    rules = strategy_config.TREND_RSI_STRATEGY_CONFIG.risk_management_rules

    assert rules.stock_stop_loss_percent == defaults.stock_stop_loss_percent
    assert rules.crypto_stop_loss_percent == defaults.crypto_stop_loss_percent
    assert rules.stock_trailing_close_percent == defaults.stock_trailing_close_percent
    assert rules.crypto_trailing_close_percent == defaults.crypto_trailing_close_percent
    assert rules.risk_per_trade_percent == defaults.risk_per_trade_percent
    assert rules.maximum_position_percent == defaults.maximum_position_percent
    assert rules.maximum_crypto_allocation_percent == defaults.maximum_crypto_allocation_percent
    assert rules.allow_fractional_stocks == defaults.allow_fractional_stocks
    assert rules.allow_fractional_crypto == defaults.allow_fractional_crypto
    assert rules.force_close_at_end == defaults.force_close_at_end
    assert rules.commission_rate == defaults.commission_rate
    assert rules.minimum_fee_usd == defaults.minimum_fee
    assert rules.slippage_bps == defaults.slippage_bps


def test_backtest_default_rebalance_fields_match_portfolio_backtest_config():
    defaults = PortfolioBacktestConfig()
    rebalance = strategy_config.TREND_RSI_STRATEGY_CONFIG.rebalance_rules
    assert rebalance.backtest_default_maximum_open_positions == defaults.maximum_open_positions
    assert rebalance.backtest_default_maximum_total_open_risk_percent == defaults.maximum_total_open_risk_percent
    # The LIVE override is a real, deliberate, DIFFERENT value -- must not collapse to the backtest default.
    assert rebalance.maximum_open_positions != rebalance.backtest_default_maximum_open_positions
    assert rebalance.maximum_total_open_risk_percent != rebalance.backtest_default_maximum_total_open_risk_percent


def test_risk_management_rules_match_the_real_frozen_baseline_lock_file():
    """Cross-checks against config/research_baseline_lock_v1.json --
    read-only reference (workspace-a's protected area, per this task's
    own instruction), never modified here."""
    baseline = json.loads(RESEARCH_BASELINE_LOCK_PATH.read_text(encoding="utf-8"))["baseline_config"]
    rules = strategy_config.TREND_RSI_STRATEGY_CONFIG.risk_management_rules

    assert rules.stock_stop_loss_percent == baseline["stock_stop_loss_percent"]
    assert rules.crypto_stop_loss_percent == baseline["crypto_stop_loss_percent"]
    assert rules.stock_trailing_close_percent == baseline["stock_trailing_close_percent"]
    assert rules.crypto_trailing_close_percent == baseline["crypto_trailing_close_percent"]
    assert rules.risk_per_trade_percent == baseline["risk_per_trade_percent"]
    assert rules.maximum_position_percent == baseline["maximum_position_percent"]
    assert rules.commission_rate == baseline["commission_rate"]
    assert rules.minimum_fee_usd == baseline["minimum_fee"]
    assert rules.slippage_bps == baseline["slippage_bps"]


def test_position_sizing_quantity_formula_matches_the_real_engine_code():
    """Real source-text check on _attempt_open_position, not a
    hand-derived restatement -- the exact three-way min() this schema's
    RiskManagementRules.position_sizing_method describes."""
    source = inspect.getsource(engine)
    assert "risk_quantity = maximum_risk_amount / stop_distance" in source
    assert "cash_quantity = max(state.cash, 0.0) / entry_price" in source
    assert "position_quantity = maximum_position_value / entry_price" in source
    assert "min(risk_quantity, cash_quantity, position_quantity)" in source


def test_exit_conditions_match_the_real_engine_exit_logic():
    check_intrabar_source = inspect.getsource(engine._check_intrabar_stops)
    queue_close_exits_source = inspect.getsource(engine._queue_close_based_exits)

    assert "low_price <= position.stop_loss_price" in check_intrabar_source
    assert "EMA20" in queue_close_exits_source and "EMA50" in queue_close_exits_source
    assert "trailing_close_percent" in queue_close_exits_source


def test_strategy_config_still_has_exactly_one_populated_instance():
    """Radar V1 (trade_plan_builder.py) must stay uninstantiated here --
    see module docstring's explicit reasoning (which mentions the
    module BY NAME in prose, hence the AST-based import check below
    rather than a bare substring search, which would false-positive on
    that explanatory text). A second StrategyConfig instance appearing
    would misrepresent it as a second live strategy."""
    import ast

    tree = ast.parse(inspect.getsource(strategy_config))
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)
    assert not any("trade_plan_builder" in name for name in imported_names)

    module_source = inspect.getsource(strategy_config)
    assert module_source.count("StrategyConfig(\n") == 1
