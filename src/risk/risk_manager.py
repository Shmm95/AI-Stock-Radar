"""Portfolio-level risk validation for AI-Stock-Radar."""

from dataclasses import dataclass, field

from src.risk.position_sizer import PositionSizeResult
from src.strategy.trade_plan import TradePlan


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Minimum portfolio state required by the risk manager."""

    account_value: float
    available_cash: float

    open_position_count: int = 0
    open_tickers: tuple[str, ...] = ()

    current_open_risk_amount: float = 0.0
    current_crypto_value: float = 0.0

    daily_realized_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskRules:
    """Configurable portfolio risk limits."""

    trading_enabled: bool = True

    maximum_open_positions: int = 8

    maximum_position_percent: float = 25.0
    maximum_total_open_risk_percent: float = 5.0
    maximum_crypto_allocation_percent: float = 20.0
    maximum_daily_loss_percent: float = 2.0

    minimum_risk_reward: float = 1.5
    minimum_confidence: int = 50
    minimum_overall_score: int = 60

    allow_duplicate_ticker: bool = False


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Final portfolio-level approval decision."""

    ticker: str
    approved: bool

    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    projected_position_count: int = 0

    projected_open_risk_amount: float = 0.0
    projected_open_risk_percent: float = 0.0

    projected_crypto_value: float = 0.0
    projected_crypto_percent: float = 0.0

    projected_cash: float = 0.0

    checks: dict[str, bool] = field(
        default_factory=dict,
    )


def _validate_portfolio_snapshot(
    portfolio: PortfolioSnapshot,
) -> None:
    """Validate incoming portfolio values."""

    if portfolio.account_value <= 0:
        raise ValueError(
            "portfolio.account_value must be greater than zero."
        )

    if portfolio.available_cash < 0:
        raise ValueError(
            "portfolio.available_cash cannot be negative."
        )

    if portfolio.open_position_count < 0:
        raise ValueError(
            "portfolio.open_position_count cannot be negative."
        )

    if portfolio.current_open_risk_amount < 0:
        raise ValueError(
            "portfolio.current_open_risk_amount cannot be negative."
        )

    if portfolio.current_crypto_value < 0:
        raise ValueError(
            "portfolio.current_crypto_value cannot be negative."
        )


def _validate_risk_rules(
    rules: RiskRules,
) -> None:
    """Validate configured risk limits."""

    if rules.maximum_open_positions < 1:
        raise ValueError(
            "maximum_open_positions must be at least 1."
        )

    percentage_fields = {
        "maximum_position_percent": rules.maximum_position_percent,
        "maximum_total_open_risk_percent": (
            rules.maximum_total_open_risk_percent
        ),
        "maximum_crypto_allocation_percent": (
            rules.maximum_crypto_allocation_percent
        ),
        "maximum_daily_loss_percent": rules.maximum_daily_loss_percent,
    }

    for field_name, value in percentage_fields.items():
        if value <= 0 or value > 100:
            raise ValueError(
                f"{field_name} must be greater than 0 and at most 100."
            )

    if rules.minimum_risk_reward <= 0:
        raise ValueError(
            "minimum_risk_reward must be greater than zero."
        )

    if not 0 <= rules.minimum_confidence <= 100:
        raise ValueError(
            "minimum_confidence must be between 0 and 100."
        )

    if not 0 <= rules.minimum_overall_score <= 100:
        raise ValueError(
            "minimum_overall_score must be between 0 and 100."
        )


def evaluate_trade_risk(
    trade_plan: TradePlan,
    position_size: PositionSizeResult,
    portfolio: PortfolioSnapshot,
    rules: RiskRules | None = None,
) -> RiskDecision:
    """Evaluate whether a position may be opened."""

    if rules is None:
        rules = RiskRules()

    _validate_portfolio_snapshot(portfolio)
    _validate_risk_rules(rules)

    rejection_reasons: list[str] = []
    warnings: list[str] = []

    checks: dict[str, bool] = {}

    ticker = trade_plan.ticker.upper()
    existing_tickers = {
        symbol.upper()
        for symbol in portfolio.open_tickers
    }

    # ---------------------------------------------------------
    # Kill switch
    # ---------------------------------------------------------
    checks["trading_enabled"] = rules.trading_enabled

    if not rules.trading_enabled:
        rejection_reasons.append(
            "Trading kill switch is disabled."
        )

    # ---------------------------------------------------------
    # Action validation
    # ---------------------------------------------------------
    action_supported = trade_plan.action.upper() == "BUY"
    checks["supported_action"] = action_supported

    if not action_supported:
        rejection_reasons.append(
            "Paper Trading V1 supports BUY trade plans only."
        )

    # ---------------------------------------------------------
    # Position sizing result
    # ---------------------------------------------------------
    checks["position_size_approved"] = position_size.approved

    if not position_size.approved:
        rejection_reasons.extend(
            position_size.rejection_reasons
        )

    # ---------------------------------------------------------
    # Duplicate ticker
    # ---------------------------------------------------------
    duplicate_allowed = (
        rules.allow_duplicate_ticker
        or ticker not in existing_tickers
    )

    checks["duplicate_ticker_allowed"] = duplicate_allowed

    if not duplicate_allowed:
        rejection_reasons.append(
            f"An open position already exists for {ticker}."
        )

    # ---------------------------------------------------------
    # Maximum position count
    # ---------------------------------------------------------
    projected_position_count = (
        portfolio.open_position_count + 1
    )

    position_count_allowed = (
        projected_position_count
        <= rules.maximum_open_positions
    )

    checks["position_count_allowed"] = position_count_allowed

    if not position_count_allowed:
        rejection_reasons.append(
            "Maximum open-position limit would be exceeded."
        )

    # ---------------------------------------------------------
    # Cash validation
    # ---------------------------------------------------------
    projected_cash = round(
        portfolio.available_cash
        - position_size.position_value,
        2,
    )

    cash_sufficient = projected_cash >= 0
    checks["cash_sufficient"] = cash_sufficient

    if not cash_sufficient:
        rejection_reasons.append(
            "Available cash is insufficient for the position."
        )

    # ---------------------------------------------------------
    # Single position allocation
    # ---------------------------------------------------------
    position_percent = (
        position_size.position_value
        / portfolio.account_value
        * 100
    )

    position_allocation_allowed = (
        position_percent
        <= rules.maximum_position_percent
    )

    checks["position_allocation_allowed"] = (
        position_allocation_allowed
    )

    if not position_allocation_allowed:
        rejection_reasons.append(
            "Position exceeds the maximum portfolio allocation."
        )

    # ---------------------------------------------------------
    # Total open risk
    # ---------------------------------------------------------
    projected_open_risk_amount = round(
        portfolio.current_open_risk_amount
        + position_size.actual_risk_amount,
        2,
    )

    projected_open_risk_percent = round(
        projected_open_risk_amount
        / portfolio.account_value
        * 100,
        4,
    )

    total_risk_allowed = (
        projected_open_risk_percent
        <= rules.maximum_total_open_risk_percent
    )

    checks["total_open_risk_allowed"] = total_risk_allowed

    if not total_risk_allowed:
        rejection_reasons.append(
            "Maximum total open-risk limit would be exceeded."
        )

    # ---------------------------------------------------------
    # Crypto allocation
    # ---------------------------------------------------------
    projected_crypto_value = portfolio.current_crypto_value

    if trade_plan.asset_type.lower() == "crypto":
        projected_crypto_value += position_size.position_value

    projected_crypto_value = round(
        projected_crypto_value,
        2,
    )

    projected_crypto_percent = round(
        projected_crypto_value
        / portfolio.account_value
        * 100,
        4,
    )

    crypto_allocation_allowed = (
        projected_crypto_percent
        <= rules.maximum_crypto_allocation_percent
    )

    checks["crypto_allocation_allowed"] = (
        crypto_allocation_allowed
    )

    if not crypto_allocation_allowed:
        rejection_reasons.append(
            "Maximum cryptocurrency allocation would be exceeded."
        )

    # ---------------------------------------------------------
    # Daily loss guard
    # ---------------------------------------------------------
    daily_loss_limit_amount = (
        portfolio.account_value
        * rules.maximum_daily_loss_percent
        / 100
    )

    daily_loss_allowed = (
        portfolio.daily_realized_pnl
        > -daily_loss_limit_amount
    )

    checks["daily_loss_allowed"] = daily_loss_allowed

    if not daily_loss_allowed:
        rejection_reasons.append(
            "Daily loss limit has been reached."
        )

    # ---------------------------------------------------------
    # Strategy quality filters
    # ---------------------------------------------------------
    risk_reward_allowed = (
        trade_plan.risk_reward
        >= rules.minimum_risk_reward
    )

    checks["risk_reward_allowed"] = risk_reward_allowed

    if not risk_reward_allowed:
        rejection_reasons.append(
            "Trade risk/reward ratio is below the configured minimum."
        )

    confidence_allowed = (
        trade_plan.confidence
        >= rules.minimum_confidence
    )

    checks["confidence_allowed"] = confidence_allowed

    if not confidence_allowed:
        rejection_reasons.append(
            "Trade confidence is below the configured minimum."
        )

    overall_score_allowed = (
        trade_plan.overall_score
        >= rules.minimum_overall_score
    )

    checks["overall_score_allowed"] = (
        overall_score_allowed
    )

    if not overall_score_allowed:
        rejection_reasons.append(
            "Overall score is below the configured minimum."
        )

    # ---------------------------------------------------------
    # Non-blocking warnings
    # ---------------------------------------------------------
    if position_percent >= (
        rules.maximum_position_percent * 0.8
    ):
        warnings.append(
            "Position is close to the maximum allocation limit."
        )

    if projected_open_risk_percent >= (
        rules.maximum_total_open_risk_percent * 0.8
    ):
        warnings.append(
            "Portfolio open risk is close to the configured limit."
        )

    if (
        trade_plan.asset_type.lower() == "crypto"
        and projected_crypto_percent
        >= rules.maximum_crypto_allocation_percent * 0.8
    ):
        warnings.append(
            "Crypto allocation is close to the configured limit."
        )

    approved = not rejection_reasons

    return RiskDecision(
        ticker=ticker,
        approved=approved,
        rejection_reasons=tuple(
            dict.fromkeys(rejection_reasons)
        ),
        warnings=tuple(
            dict.fromkeys(warnings)
        ),
        projected_position_count=projected_position_count,
        projected_open_risk_amount=projected_open_risk_amount,
        projected_open_risk_percent=projected_open_risk_percent,
        projected_crypto_value=projected_crypto_value,
        projected_crypto_percent=projected_crypto_percent,
        projected_cash=projected_cash,
        checks=checks,
    )


def print_risk_decision(
    decision: RiskDecision,
) -> None:
    """Print a readable risk-manager report."""

    print()
    print("=" * 76)
    print("RISK MANAGER DECISION")
    print("=" * 76)

    print(f"Ticker:                     {decision.ticker}")
    print(
        f"Approved:                   "
        f"{'YES' if decision.approved else 'NO'}"
    )

    print()
    print(
        f"Projected open positions:   "
        f"{decision.projected_position_count}"
    )
    print(
        f"Projected available cash:   "
        f"€{decision.projected_cash:,.2f}"
    )
    print(
        f"Projected open risk:        "
        f"€{decision.projected_open_risk_amount:,.2f}"
    )
    print(
        f"Projected open risk %:      "
        f"{decision.projected_open_risk_percent:.4f}%"
    )
    print(
        f"Projected crypto value:     "
        f"€{decision.projected_crypto_value:,.2f}"
    )
    print(
        f"Projected crypto %:         "
        f"{decision.projected_crypto_percent:.4f}%"
    )

    print()
    print("Checks")

    for check_name, passed in decision.checks.items():
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check_name}")

    if decision.warnings:
        print()
        print("Warnings")

        for warning in decision.warnings:
            print(f"- {warning}")

    if decision.rejection_reasons:
        print()
        print("Rejection reasons")

        for reason in decision.rejection_reasons:
            print(f"- {reason}")

    print("=" * 76)
