"""Execute pipeline candidates through the local paper-trading system.

Flow:
    DeepResult
    -> TradePlanBuilder
    -> PositionSizer
    -> RiskManager
    -> PaperBroker

This module is paper-only. It cannot send real broker orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.paper.paper_broker import (
    PaperBroker,
    print_open_positions,
)
from src.paper.paper_models import PaperPosition
from src.pipeline.pipeline_models import DeepResult
from src.risk.position_sizer import (
    PositionSizeResult,
    calculate_position_size,
)
from src.risk.risk_manager import (
    RiskDecision,
    RiskRules,
    evaluate_trade_risk,
)
from src.strategy.trade_plan import TradePlan
from src.strategy.trade_plan_builder import (
    TradePlanBuildResult,
    TradePlanRules,
    build_trade_plan,
)


@dataclass(frozen=True, slots=True)
class PaperExecutionResult:
    """Final execution result for one pipeline candidate."""

    ticker: str
    approved: bool
    executed: bool
    dry_run: bool

    stage: str
    message: str

    trade_plan: TradePlan | None = None
    build_result: TradePlanBuildResult | None = None
    position_size: PositionSizeResult | None = None
    risk_decision: RiskDecision | None = None
    opened_position: PaperPosition | None = None

    warnings: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PaperExecutionBatch:
    """Summary of one multi-candidate paper execution run."""

    results: tuple[PaperExecutionResult, ...]

    starting_cash: float
    ending_cash: float

    starting_account_value: float
    ending_account_value: float

    opened_tickers: tuple[str, ...] = ()
    rejected_tickers: tuple[str, ...] = ()
    skipped_tickers: tuple[str, ...] = ()
    failed_tickers: tuple[str, ...] = ()

    @property
    def candidate_count(self) -> int:
        """Return the number of processed candidates."""

        return len(self.results)

    @property
    def executed_count(self) -> int:
        """Return the number of opened paper positions."""

        return sum(
            1
            for result in self.results
            if result.executed
        )

    @property
    def approved_count(self) -> int:
        """Return the number approved before execution."""

        return sum(
            1
            for result in self.results
            if result.approved
        )


@dataclass(frozen=True, slots=True)
class PaperRunnerRules:
    """Configuration for the paper-trade runner."""

    maximum_position_percent: float = 25.0
    allow_fractional_stocks: bool = False
    allow_fractional_crypto: bool = True

    stop_after_first_execution_failure: bool = False
    skip_existing_positions: bool = True

    trade_plan_rules: TradePlanRules = field(
        default_factory=TradePlanRules,
    )

    risk_rules: RiskRules = field(
        default_factory=RiskRules,
    )


def _validate_runner_rules(
    rules: PaperRunnerRules,
) -> None:
    """Validate runner-level configuration."""

    if (
        rules.maximum_position_percent <= 0
        or rules.maximum_position_percent > 100
    ):
        raise ValueError(
            "maximum_position_percent must be greater than "
            "zero and at most 100."
        )


def _normalize_prices(
    prices: dict[str, float],
) -> dict[str, float]:
    """Normalize and validate supplied market prices."""

    normalized: dict[str, float] = {}

    for ticker, price in prices.items():
        normalized_ticker = ticker.strip().upper()
        numeric_price = float(price)

        if not normalized_ticker:
            raise ValueError(
                "Price ticker cannot be empty."
            )

        if numeric_price <= 0:
            raise ValueError(
                f"Price for {normalized_ticker} must be "
                "greater than zero."
            )

        normalized[normalized_ticker] = numeric_price

    return normalized


def _build_rejected_result(
    *,
    ticker: str,
    stage: str,
    message: str,
    dry_run: bool,
    build_result: TradePlanBuildResult | None = None,
    trade_plan: TradePlan | None = None,
    position_size: PositionSizeResult | None = None,
    risk_decision: RiskDecision | None = None,
    warnings: tuple[str, ...] = (),
    rejection_reasons: tuple[str, ...] = (),
) -> PaperExecutionResult:
    """Create a standardized rejected execution result."""

    return PaperExecutionResult(
        ticker=ticker,
        approved=False,
        executed=False,
        dry_run=dry_run,
        stage=stage,
        message=message,
        trade_plan=trade_plan,
        build_result=build_result,
        position_size=position_size,
        risk_decision=risk_decision,
        opened_position=None,
        warnings=warnings,
        rejection_reasons=rejection_reasons,
    )


def execute_candidate(
    *,
    candidate: DeepResult,
    entry_price: float,
    broker: PaperBroker,
    rules: PaperRunnerRules | None = None,
    dry_run: bool = False,
) -> PaperExecutionResult:
    """Process one candidate through all paper-trading checks."""

    if rules is None:
        rules = PaperRunnerRules()

    _validate_runner_rules(rules)

    ticker = candidate.ticker.strip().upper()

    if entry_price <= 0:
        return _build_rejected_result(
            ticker=ticker,
            stage="PRICE_VALIDATION",
            message="Entry price is invalid.",
            dry_run=dry_run,
            rejection_reasons=(
                "entry_price must be greater than zero.",
            ),
        )

    # =========================================================
    # Existing-position guard
    # =========================================================
    existing_position = broker.find_position(ticker)

    if (
        existing_position is not None
        and rules.skip_existing_positions
    ):
        return _build_rejected_result(
            ticker=ticker,
            stage="EXISTING_POSITION",
            message=(
                f"Skipped because an open position already "
                f"exists for {ticker}."
            ),
            dry_run=dry_run,
            rejection_reasons=(
                "An open paper position already exists.",
            ),
        )

    # =========================================================
    # Trade-plan builder
    # =========================================================
    try:
        build_result = build_trade_plan(
            candidate=candidate,
            entry_price=entry_price,
            rules=rules.trade_plan_rules,
        )

    except Exception as error:
        return _build_rejected_result(
            ticker=ticker,
            stage="TRADE_PLAN_BUILD",
            message=(
                f"Trade-plan creation failed: {error}"
            ),
            dry_run=dry_run,
            rejection_reasons=(
                str(error),
            ),
        )

    if (
        not build_result.approved
        or build_result.trade_plan is None
    ):
        return _build_rejected_result(
            ticker=ticker,
            stage="TRADE_PLAN_REJECTED",
            message=(
                "Trade Plan Builder rejected the candidate."
            ),
            dry_run=dry_run,
            build_result=build_result,
            warnings=build_result.warnings,
            rejection_reasons=(
                build_result.rejection_reasons
            ),
        )

    trade_plan = build_result.trade_plan

    # =========================================================
    # Current portfolio snapshot
    # =========================================================
    try:
        snapshot = broker.create_portfolio_snapshot()

    except Exception as error:
        return _build_rejected_result(
            ticker=ticker,
            stage="PORTFOLIO_SNAPSHOT",
            message=(
                f"Could not create portfolio snapshot: "
                f"{error}"
            ),
            dry_run=dry_run,
            build_result=build_result,
            trade_plan=trade_plan,
            warnings=build_result.warnings,
            rejection_reasons=(
                str(error),
            ),
        )

    # =========================================================
    # Position sizing
    # =========================================================
    allow_fractional = (
        rules.allow_fractional_crypto
        if trade_plan.asset_type == "crypto"
        else rules.allow_fractional_stocks
    )

    try:
        position_size = calculate_position_size(
            trade_plan=trade_plan,
            account_value=snapshot.account_value,
            available_cash=snapshot.available_cash,
            risk_percent=trade_plan.max_risk_percent,
            maximum_position_percent=(
                rules.maximum_position_percent
            ),
            allow_fractional=allow_fractional,
        )

    except Exception as error:
        return _build_rejected_result(
            ticker=ticker,
            stage="POSITION_SIZING",
            message=(
                f"Position sizing failed: {error}"
            ),
            dry_run=dry_run,
            build_result=build_result,
            trade_plan=trade_plan,
            warnings=build_result.warnings,
            rejection_reasons=(
                str(error),
            ),
        )

    if not position_size.approved:
        return _build_rejected_result(
            ticker=ticker,
            stage="POSITION_SIZE_REJECTED",
            message=(
                "Position Sizer rejected the trade."
            ),
            dry_run=dry_run,
            build_result=build_result,
            trade_plan=trade_plan,
            position_size=position_size,
            warnings=build_result.warnings,
            rejection_reasons=(
                position_size.rejection_reasons
            ),
        )

    # =========================================================
    # Portfolio-level risk decision
    # =========================================================
    try:
        risk_decision = evaluate_trade_risk(
            trade_plan=trade_plan,
            position_size=position_size,
            portfolio=snapshot,
            rules=rules.risk_rules,
        )

    except Exception as error:
        return _build_rejected_result(
            ticker=ticker,
            stage="RISK_MANAGER",
            message=(
                f"Risk evaluation failed: {error}"
            ),
            dry_run=dry_run,
            build_result=build_result,
            trade_plan=trade_plan,
            position_size=position_size,
            warnings=build_result.warnings,
            rejection_reasons=(
                str(error),
            ),
        )

    combined_warnings = tuple(
        dict.fromkeys(
            (
                *build_result.warnings,
                *risk_decision.warnings,
            )
        )
    )

    if not risk_decision.approved:
        return _build_rejected_result(
            ticker=ticker,
            stage="RISK_REJECTED",
            message=(
                "Risk Manager rejected the trade."
            ),
            dry_run=dry_run,
            build_result=build_result,
            trade_plan=trade_plan,
            position_size=position_size,
            risk_decision=risk_decision,
            warnings=combined_warnings,
            rejection_reasons=(
                risk_decision.rejection_reasons
            ),
        )

    # =========================================================
    # Dry-run mode
    # =========================================================
    if dry_run:
        return PaperExecutionResult(
            ticker=ticker,
            approved=True,
            executed=False,
            dry_run=True,
            stage="DRY_RUN_APPROVED",
            message=(
                "Trade passed all checks. "
                "No paper order was created."
            ),
            trade_plan=trade_plan,
            build_result=build_result,
            position_size=position_size,
            risk_decision=risk_decision,
            opened_position=None,
            warnings=combined_warnings,
            rejection_reasons=(),
        )

    # =========================================================
    # Paper Broker execution
    # =========================================================
    try:
        opened_position = broker.open_position(
            trade_plan=trade_plan,
            position_size=position_size,
            risk_decision=risk_decision,
        )

    except Exception as error:
        return _build_rejected_result(
            ticker=ticker,
            stage="BROKER_EXECUTION",
            message=(
                f"Paper Broker execution failed: {error}"
            ),
            dry_run=False,
            build_result=build_result,
            trade_plan=trade_plan,
            position_size=position_size,
            risk_decision=risk_decision,
            warnings=combined_warnings,
            rejection_reasons=(
                str(error),
            ),
        )

    return PaperExecutionResult(
        ticker=ticker,
        approved=True,
        executed=True,
        dry_run=False,
        stage="POSITION_OPENED",
        message=(
            f"Paper position opened for {ticker}."
        ),
        trade_plan=trade_plan,
        build_result=build_result,
        position_size=position_size,
        risk_decision=risk_decision,
        opened_position=opened_position,
        warnings=combined_warnings,
        rejection_reasons=(),
    )


def execute_candidates(
    *,
    candidates: list[DeepResult],
    prices: dict[str, float],
    broker: PaperBroker | None = None,
    rules: PaperRunnerRules | None = None,
    dry_run: bool = False,
) -> PaperExecutionBatch:
    """Process multiple pipeline candidates in ranking order."""

    if broker is None:
        broker = PaperBroker()

    if rules is None:
        rules = PaperRunnerRules()

    _validate_runner_rules(rules)

    normalized_prices = _normalize_prices(prices)

    starting_account = broker.account_store.load()
    starting_snapshot = broker.create_portfolio_snapshot()

    results: list[PaperExecutionResult] = []

    opened_tickers: list[str] = []
    rejected_tickers: list[str] = []
    skipped_tickers: list[str] = []
    failed_tickers: list[str] = []

    for candidate in candidates:
        ticker = candidate.ticker.strip().upper()
        entry_price = normalized_prices.get(ticker)

        if entry_price is None:
            result = _build_rejected_result(
                ticker=ticker,
                stage="MISSING_PRICE",
                message=(
                    "Candidate skipped because no current "
                    "price was supplied."
                ),
                dry_run=dry_run,
                rejection_reasons=(
                    "No current price was supplied.",
                ),
            )

            results.append(result)
            skipped_tickers.append(ticker)
            continue

        result = execute_candidate(
            candidate=candidate,
            entry_price=entry_price,
            broker=broker,
            rules=rules,
            dry_run=dry_run,
        )

        results.append(result)

        if result.executed:
            opened_tickers.append(ticker)

        elif result.stage in {
            "MISSING_PRICE",
            "EXISTING_POSITION",
        }:
            skipped_tickers.append(ticker)

        elif result.stage == "BROKER_EXECUTION":
            failed_tickers.append(ticker)

            if rules.stop_after_first_execution_failure:
                break

        else:
            rejected_tickers.append(ticker)

    ending_account = broker.account_store.load()
    ending_snapshot = broker.create_portfolio_snapshot()

    return PaperExecutionBatch(
        results=tuple(results),
        starting_cash=starting_account.cash,
        ending_cash=ending_account.cash,
        starting_account_value=(
            starting_snapshot.account_value
        ),
        ending_account_value=(
            ending_snapshot.account_value
        ),
        opened_tickers=tuple(opened_tickers),
        rejected_tickers=tuple(
            dict.fromkeys(rejected_tickers)
        ),
        skipped_tickers=tuple(
            dict.fromkeys(skipped_tickers)
        ),
        failed_tickers=tuple(
            dict.fromkeys(failed_tickers)
        ),
    )


def print_execution_result(
    result: PaperExecutionResult,
) -> None:
    """Print one paper execution result."""

    print()
    print("=" * 90)
    print("PAPER TRADE EXECUTION")
    print("=" * 90)

    print(f"Ticker:       {result.ticker}")
    print(f"Stage:        {result.stage}")
    print(
        f"Approved:     "
        f"{'YES' if result.approved else 'NO'}"
    )
    print(
        f"Executed:     "
        f"{'YES' if result.executed else 'NO'}"
    )
    print(
        f"Dry run:      "
        f"{'YES' if result.dry_run else 'NO'}"
    )
    print(f"Message:      {result.message}")

    if result.trade_plan is not None:
        plan = result.trade_plan

        print()
        print("Trade plan")
        print(f"- Action:     {plan.action}")
        print(f"- Entry:      {plan.entry_price:.6f}")
        print(f"- Stop:       {plan.stop_loss:.6f}")
        print(f"- Target:     {plan.take_profit:.6f}")
        print(f"- Risk/Reward:{plan.risk_reward:.2f}")

    if result.position_size is not None:
        size = result.position_size

        print()
        print("Position size")
        print(f"- Quantity:   {size.quantity}")
        print(
            f"- Value:      "
            f"{size.position_value:,.2f}"
        )
        print(
            f"- Risk:       "
            f"{size.actual_risk_amount:,.2f}"
        )

    if result.opened_position is not None:
        position = result.opened_position

        print()
        print("Opened position")
        print(
            f"- Fill price: "
            f"{position.entry_price:.6f}"
        )
        print(
            f"- Market value: "
            f"{position.market_value:,.2f}"
        )

    if result.warnings:
        print()
        print("Warnings")

        for warning in result.warnings:
            print(f"- {warning}")

    if result.rejection_reasons:
        print()
        print("Rejection reasons")

        for reason in result.rejection_reasons:
            print(f"- {reason}")

    print("=" * 90)


def print_execution_batch(
    batch: PaperExecutionBatch,
    *,
    broker: PaperBroker | None = None,
) -> None:
    """Print a compact batch execution report."""

    print()
    print("=" * 100)
    print("PAPER TRADE RUNNER — BATCH SUMMARY")
    print("=" * 100)

    print(
        f"Candidates processed: "
        f"{batch.candidate_count}"
    )
    print(
        f"Approved candidates:  "
        f"{batch.approved_count}"
    )
    print(
        f"Positions executed:   "
        f"{batch.executed_count}"
    )

    print()
    print(
        f"Starting cash:        "
        f"{batch.starting_cash:,.2f}"
    )
    print(
        f"Ending cash:          "
        f"{batch.ending_cash:,.2f}"
    )
    print(
        f"Starting value:       "
        f"{batch.starting_account_value:,.2f}"
    )
    print(
        f"Ending value:         "
        f"{batch.ending_account_value:,.2f}"
    )

    print()
    print(
        "Opened:   "
        + (
            ", ".join(batch.opened_tickers)
            if batch.opened_tickers
            else "None"
        )
    )
    print(
        "Rejected: "
        + (
            ", ".join(batch.rejected_tickers)
            if batch.rejected_tickers
            else "None"
        )
    )
    print(
        "Skipped:  "
        + (
            ", ".join(batch.skipped_tickers)
            if batch.skipped_tickers
            else "None"
        )
    )
    print(
        "Failed:   "
        + (
            ", ".join(batch.failed_tickers)
            if batch.failed_tickers
            else "None"
        )
    )

    print()
    print("RESULTS")

    for result in batch.results:
        status = (
            "OPENED"
            if result.executed
            else (
                "APPROVED"
                if result.approved
                else "REJECTED"
            )
        )

        print(
            f"- {result.ticker:<14} "
            f"{status:<10} "
            f"{result.stage}"
        )

    print("=" * 100)

    if broker is not None:
        print_open_positions(
            broker.list_positions()
        )