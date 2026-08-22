"""StrategyConfig schema — V2 multi-strategy benchmark platform, Phase 0
(pre-registration). Pure schema + documentation artifact.

NOT INTEGRATED: this module is not imported by, and does not modify,
`portfolio_backtest_engine.py`, `run_daily_decision.py`, or any live
path. Nothing here changes runtime behavior. Its purpose is to give the
"universe must come from methodology, not from results" principle a
concrete, typed shape to pre-register against before any V2 backtest is
run.

Only ONE instance is populated below: `TREND_RSI_STRATEGY_CONFIG`, the
single currently-live strategy (`docs/BASELINE.md:8`: "Strategy:
TREND_RSI"; confirmed live via `run_daily_decision.py`'s direct calls
into `portfolio_backtest_engine.py`'s `_is_entry_setup`/step functions).

"Radar V1" (`src/strategy/trade_plan_builder.py`, score/confidence/RR
based) is deliberately NOT instantiated here. Confirmed NOT live by
direct code read: `run_daily_decision.py` never imports it, and
`docs/PROJECT_CONTEXT.md:25` states its presence "does not authorize
paper-trading work, does not advance the current stage." Adding a
config instance for it now would misrepresent it as a second active
strategy for this benchmark platform — this was an explicit instruction
for this task, not an oversight.

`deployment_universes` is plural (a tuple), not the singular
`deployment_universe` this task's own instruction named. This is a
deliberate generalization, not a drift from the request — see
`docs/UNIVERSE_MECHANISMS_MAP.md` for the reasoning: TREND_RSI already
has two real, concrete deployment universes today (the live 83-ticker
universe and the 44-ticker MidCap control arm), so a schema that could
only hold one would already be wrong for the one strategy this file
actually populates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"
    EQUITY_AND_CRYPTO = "EQUITY_AND_CRYPTO"


@dataclass(frozen=True)
class LiquidityRules:
    """Minimum tradability bar a ticker must clear. Fields left `None`
    are gates that were checked for and confirmed NOT to exist in the
    current codebase (see this module's own docstring / the mapping
    doc for what was actually verified vs. assumed) -- `None` here
    means "verified absent," not "not yet filled in."""

    minimum_average_dollar_volume: float | None = None
    minimum_price: float | None = None
    minimum_history_days: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class EligibilityRules:
    """Non-liquidity inclusion/exclusion criteria."""

    excluded_tickers: tuple[str, ...] = ()
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    requires_broker_tradability_check: bool = False
    redundancy_correlation_threshold: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class RebalanceRules:
    """How/when the strategy re-evaluates its portfolio.

    `maximum_open_positions`/`maximum_total_open_risk_percent` are the
    LIVE-ONLY override (see this class's own field values below and
    TREND_RSI_STRATEGY_CONFIG's notes); `backtest_default_*` are the
    SEPARATE, unmodified `PortfolioBacktestConfig` defaults
    (portfolio_backtest_models.py) that apply to every backtest/research
    run that does not explicitly override them -- these two pairs
    genuinely differ (4/4.0% vs 6/6.0%) and conflating them would
    misrepresent which number applies to which context."""

    signal_frequency: str = ""
    execution_timing: str = ""
    maximum_open_positions: int | None = None
    maximum_total_open_risk_percent: float | None = None
    backtest_default_maximum_open_positions: int | None = None
    backtest_default_maximum_total_open_risk_percent: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class SignalRules:
    """TREND_RSI's entry-signal parameters, extracted verbatim from
    `portfolio_backtest_engine.py`'s `_is_entry_setup`/`_build_entry_signal`
    (frozen, read-only reference -- never edited by this schema). NOT a
    universal shape every future strategy must reuse: a differently
    structured strategy (e.g. a pure momentum ranking, no MA/RSI
    concept at all) would define its OWN parallel dataclass with the
    same PURPOSE (typed, config-able entry parameters) but a different
    FIELD set -- see StrategyConfig's own docstring note on
    `signal_rules`'s type."""

    indicator_type: str = "EMA"  # portfolio_backtest_engine.py: EMA20/EMA50 columns, not SMA
    fast_ma_period: int = 20
    slow_ma_period: int = 50
    rsi_period: int = 14
    rsi_lower_bound: float = 45.0
    rsi_upper_bound: float = 70.0
    requires_close_above_fast_ma: bool = True
    requires_fresh_crossover_not_continuation: bool = True
    regime_gate_required: bool = True
    notes: str = (
        "_is_entry_setup: EMA20>EMA50 and Close>EMA20 and 45<=RSI14<=70 "
        "and RegimeAllowed. _build_entry_signal additionally requires "
        "`not _is_entry_setup(previous)` -- a signal only fires on the "
        "bar the setup FIRST becomes true, never on a bar where it was "
        "already true the previous bar (fresh crossover only, not a "
        "continuation)."
    )


@dataclass(frozen=True)
class RiskManagementRules:
    """Stop-loss/trailing/position-sizing parameters, extracted
    verbatim from `PortfolioBacktestConfig`'s dataclass defaults
    (portfolio_backtest_models.py) -- these exact values are also what
    `config/research_baseline_lock_v1.json`'s own `baseline_config`
    freezes (confirmed by direct read, byte-for-byte match, not
    assumed). Referenced here read-only; this schema never edits either
    source."""

    stock_stop_loss_percent: float = 5.0
    crypto_stop_loss_percent: float = 5.0
    stock_trailing_close_percent: float = 7.5
    crypto_trailing_close_percent: float = 7.5
    risk_per_trade_percent: float = 1.0
    maximum_position_percent: float = 25.0
    maximum_crypto_allocation_percent: float = 25.0
    position_sizing_method: str = (
        "quantity = min(risk_quantity, cash_quantity, position_quantity), where "
        "risk_quantity = (equity * risk_per_trade_percent/100) / stop_distance, "
        "cash_quantity = cash / entry_price, "
        "position_quantity = (equity * maximum_position_percent/100) / entry_price"
    )
    exit_conditions: tuple[str, ...] = (
        "HARD_STOP_LOSS: Low <= stop_loss_price (intrabar) or Open <= stop_loss_price (gap)",
        "TREND_REVERSAL: EMA20 < EMA50",
        "TRAILING_STOP: Close <= highest_close_since_entry * (1 - trailing_close_percent/100)",
    )
    allow_fractional_stocks: bool = False
    allow_fractional_crypto: bool = True
    force_close_at_end: bool = True
    commission_rate: float = 0.0005
    minimum_fee_usd: float = 1.0
    slippage_bps: float = 5.0
    notes: str = (
        "stock/crypto stop-loss and trailing values are currently identical "
        "(5.0%/7.5% for both asset classes) -- separate fields exist in the "
        "engine's own config, not because they differ today, but because "
        "the engine already supports them differing."
    )


@dataclass(frozen=True)
class StrategyConfig:
    strategy: str
    asset_class: AssetClass

    # None is a valid, deliberate value here -- see
    # `not_applicable_reason` and this module's docstring / the "OUR-V1
    # native_universe decision" section of the mapping doc for why.
    native_universe: str | None
    not_applicable_reason: str | None

    deployment_universes: tuple[str, ...]

    liquidity_rules: LiquidityRules
    eligibility_rules: EligibilityRules
    rebalance_rules: RebalanceRules

    # Typed `object`, deliberately, not `SignalRules`/`RiskManagementRules`
    # directly: those two classes are TREND_RSI's own concrete parameter
    # shape (see their own docstrings). A future strategy variant with a
    # fundamentally different signal concept (e.g. the still-frozen RSI
    # Replacement group, or a pure cross-sectional ranking with no
    # MA/RSI/stop-loss concept at all) defines and populates its OWN
    # parallel dataclass here -- the FRAMEWORK (this outer StrategyConfig
    # shape: universe/liquidity/eligibility/rebalance/signal/risk) stays
    # constant across strategies; only the inner shape of these two
    # fields varies per strategy, which is the honest, correct design
    # since different strategies genuinely have different parameter
    # shapes -- forcing them into one identical dataclass would either
    # need every field to be Optional (silently allowing nonsense
    # partial configs) or misrepresent a strategy's real parameters.
    signal_rules: object
    risk_management_rules: object

    notes: str = ""


# ---------------------------------------------------------------------
# TREND_RSI -- the one currently-live strategy. See docs/BASELINE.md:8
# and portfolio_backtest_engine.py's _is_entry_setup for the frozen
# entry condition this describes (EMA20>EMA50, Close>EMA20,
# 45<=RSI14<=70, RegimeAllowed) -- this file does not restate or alter
# that logic, only classifies it.
# ---------------------------------------------------------------------

TREND_RSI_STRATEGY_CONFIG = StrategyConfig(
    strategy="TREND_RSI",
    asset_class=AssetClass.EQUITY_AND_CRYPTO,
    native_universe=None,
    not_applicable_reason=(
        "TREND_RSI was not sourced from published literature validated on "
        "a specific external universe (e.g. a paper's own S&P 500 or "
        "Russell sample) -- there is no 'native_universe' in that "
        "traditional sense to record. It WAS originally designed/tuned "
        "against this project's own 'controlled research basket' "
        "(docs/PROJECT_CONTEXT.md: AAPL, AMZN, GOOGL, META, MSFT, NVDA, "
        "TSLA, BTC-USD, ETH-USD -- 9 tickers), but that basket was this "
        "project's own dev/research sandbox, not an externally-validated "
        "'native' universe the strategy inherits authority from -- see "
        "docs/UNIVERSE_MECHANISMS_MAP.md's 'OUR-V1 native_universe "
        "decision' section for the full reasoning."
    ),
    deployment_universes=(
        "live_universe_v1",     # src/live/live_universe.py, 83 tickers -- see mapping doc
        "control_universe_v1",  # src/live/control_universe.py, 44 tickers -- see mapping doc
    ),
    liquidity_rules=LiquidityRules(
        minimum_average_dollar_volume=None,
        minimum_price=None,
        minimum_history_days=150,  # data_preparer.py: MINIMUM_DELIVERED_BARS, verified by direct read
        notes=(
            "No dollar-volume or price-floor liquidity gate exists anywhere "
            "in portfolio_backtest_engine.py or the live data pipeline -- "
            "confirmed by direct read of _validate_market_data (only OHLC "
            "sanity: positive prices, valid High/Low bounds, >=2 rows). "
            "minimum_history_days=150 is a data-quality/indicator-warm-up "
            "gate (data_preparer.py's prepare_live_market_data), not a "
            "traditional liquidity screen, but it is the closest real "
            "analog that exists in code today."
        ),
    ),
    eligibility_rules=EligibilityRules(
        excluded_tickers=(),  # no engine-level exclusion list; per-universe exclusions live in each universe's own definition
        exclusion_reasons={},
        requires_broker_tradability_check=True,  # both live_universe.py's and control_universe.py's own selection histories cite a real Alpaca get_all_assets() check
        redundancy_correlation_threshold=0.75,  # live_universe.py's own widening methodology, verified by direct docstring read
        notes=(
            "This threshold (0.75) and the tradability-check requirement "
            "describe the ONE-TIME SELECTION methodology used to build "
            "live_universe.py's ticker list, not an ongoing per-run "
            "eligibility check the engine re-evaluates -- confirmed by "
            "reading portfolio_backtest_engine.py, which has no "
            "correlation or tradability logic of its own."
        ),
    ),
    rebalance_rules=RebalanceRules(
        signal_frequency="one bar (daily) close per run",
        execution_timing="next available Open (next-Open rule; a signal generated on today's Close never executes today)",
        maximum_open_positions=6,   # run_daily_decision.py: LIVE_MAXIMUM_OPEN_POSITIONS, verified by direct read
        maximum_total_open_risk_percent=6.0,  # run_daily_decision.py: LIVE_MAXIMUM_TOTAL_OPEN_RISK_PERCENT
        backtest_default_maximum_open_positions=4,  # PortfolioBacktestConfig.maximum_open_positions default, matches research_baseline_lock_v1.json's baseline_config byte-for-byte
        backtest_default_maximum_total_open_risk_percent=4.0,  # PortfolioBacktestConfig.maximum_total_open_risk_percent default, same match
        notes=(
            "6/6.0% is a LIVE-ONLY override on top of PortfolioBacktestConfig's "
            "own hash-locked defaults (research_baseline_lock_v1.json) -- "
            "verified by reading run_daily_decision.py's own comment on "
            "these two constants. Raising the cap to 8/8.0% was tried "
            "during live_universe.py's stage-3 widening and explicitly "
            "abandoned (backtest evidence showed it made results worse)."
        ),
    ),
    signal_rules=SignalRules(),  # every field default already matches the frozen engine verbatim -- see SignalRules' own docstring
    risk_management_rules=RiskManagementRules(),  # same -- every default matches PortfolioBacktestConfig / research_baseline_lock_v1.json verbatim
    notes=(
        "Frozen strategy per CLAUDE.md's protected-areas policy -- entry "
        "rules, exit rules, risk management, execution order, fees, and "
        "slippage in portfolio_backtest_engine.py are never edited without "
        "explicit owner approval in the main planning chat."
    ),
)
