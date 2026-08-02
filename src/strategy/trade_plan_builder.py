"""Build executable trade plans from radar pipeline candidates.

The builder converts a final DeepResult candidate into a TradePlan.

Paper Trading V1 rules:
- Long-only
- BUY candidates only
- No leverage
- Percentage-based initial stop and target
- Minimum score, confidence, and risk/reward validation
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.pipeline_models import DeepResult
from src.strategy.trade_plan import TradePlan


@dataclass(frozen=True, slots=True)
class TradePlanRules:
    """Configuration used when building a trade plan."""

    minimum_overall_score: int = 60
    minimum_technical_score: int = 50
    minimum_confidence: int = 50

    stock_stop_loss_percent: float = 5.0
    stock_take_profit_percent: float = 10.0

    crypto_stop_loss_percent: float = 7.0
    crypto_take_profit_percent: float = 14.0

    maximum_risk_percent: float = 1.0
    minimum_risk_reward: float = 1.5

    strategy_name: str = "Radar Paper V1"


@dataclass(frozen=True, slots=True)
class TradePlanBuildResult:
    """Result of attempting to build one trade plan."""

    ticker: str
    approved: bool

    trade_plan: TradePlan | None

    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _validate_percentage(
    value: float,
    field_name: str,
) -> None:
    """Validate a percentage configuration value."""

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    if value >= 100:
        raise ValueError(
            f"{field_name} must be lower than 100."
        )


def _validate_rules(
    rules: TradePlanRules,
) -> None:
    """Validate trade-plan builder rules."""

    if not 0 <= rules.minimum_overall_score <= 100:
        raise ValueError(
            "minimum_overall_score must be between 0 and 100."
        )

    if not 0 <= rules.minimum_technical_score <= 100:
        raise ValueError(
            "minimum_technical_score must be between 0 and 100."
        )

    if not 0 <= rules.minimum_confidence <= 100:
        raise ValueError(
            "minimum_confidence must be between 0 and 100."
        )

    _validate_percentage(
        rules.stock_stop_loss_percent,
        "stock_stop_loss_percent",
    )

    _validate_percentage(
        rules.stock_take_profit_percent,
        "stock_take_profit_percent",
    )

    _validate_percentage(
        rules.crypto_stop_loss_percent,
        "crypto_stop_loss_percent",
    )

    _validate_percentage(
        rules.crypto_take_profit_percent,
        "crypto_take_profit_percent",
    )

    _validate_percentage(
        rules.maximum_risk_percent,
        "maximum_risk_percent",
    )

    if rules.minimum_risk_reward <= 0:
        raise ValueError(
            "minimum_risk_reward must be greater than zero."
        )

    if not rules.strategy_name.strip():
        raise ValueError(
            "strategy_name cannot be empty."
        )


def _normalize_asset_type(
    asset_type: str,
) -> str:
    """Return a validated normalized asset type."""

    normalized = asset_type.strip().lower()

    if normalized not in {
        "stock",
        "crypto",
    }:
        raise ValueError(
            "asset_type must be either 'stock' or 'crypto'."
        )

    return normalized


def _get_stop_and_target_percentages(
    asset_type: str,
    rules: TradePlanRules,
) -> tuple[float, float]:
    """Return configured stop and target percentages."""

    if asset_type == "crypto":
        return (
            rules.crypto_stop_loss_percent,
            rules.crypto_take_profit_percent,
        )

    return (
        rules.stock_stop_loss_percent,
        rules.stock_take_profit_percent,
    )


def _calculate_price_levels(
    *,
    entry_price: float,
    stop_loss_percent: float,
    take_profit_percent: float,
) -> tuple[float, float]:
    """Calculate long-position stop-loss and take-profit prices."""

    if entry_price <= 0:
        raise ValueError(
            "entry_price must be greater than zero."
        )

    stop_loss = (
        entry_price
        * (
            1
            - stop_loss_percent
            / 100
        )
    )

    take_profit = (
        entry_price
        * (
            1
            + take_profit_percent
            / 100
        )
    )

    return (
        round(stop_loss, 6),
        round(take_profit, 6),
    )


def _calculate_risk_reward(
    *,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> float:
    """Calculate expected reward divided by expected risk."""

    risk = entry_price - stop_loss
    reward = take_profit - entry_price

    if risk <= 0:
        return 0.0

    return round(
        reward / risk,
        4,
    )


def _build_notes(
    candidate: DeepResult,
) -> str:
    """Build compact explainability notes for the trade plan."""

    parts = [
        (
            f"Pipeline candidate: "
            f"overall={candidate.overall_score}"
        ),
        (
            f"technical={candidate.technical_score}"
        ),
        (
            f"fundamental={candidate.fundamental_score}"
        ),
        (
            f"news={candidate.news_score}"
        ),
        (
            f"confidence={candidate.confidence}"
        ),
        (
            f"technical_signal="
            f"{candidate.technical_signal}"
        ),
        (
            f"news_sentiment="
            f"{candidate.news_sentiment}"
        ),
    ]

    if candidate.decision_reasons:
        decision_text = "; ".join(
            candidate.decision_reasons[:3]
        )

        parts.append(
            f"decision_reasons={decision_text}"
        )

    return " | ".join(parts)


def build_trade_plan(
    *,
    candidate: DeepResult,
    entry_price: float,
    rules: TradePlanRules | None = None,
) -> TradePlanBuildResult:
    """Build and validate one long-only trade plan."""

    if rules is None:
        rules = TradePlanRules()

    _validate_rules(rules)

    asset_type = _normalize_asset_type(
        candidate.asset_type
    )

    if entry_price <= 0:
        raise ValueError(
            "entry_price must be greater than zero."
        )

    rejection_reasons: list[str] = []
    warnings: list[str] = []

    recommendation = (
        candidate.signal
        .strip()
        .upper()
    )

    technical_signal = (
        candidate.technical_signal
        .strip()
        .upper()
    )

    # ---------------------------------------------------------
    # Recommendation validation
    # ---------------------------------------------------------
    if recommendation != "BUY":
        rejection_reasons.append(
            "Final pipeline recommendation is not BUY."
        )

    if technical_signal not in {
        "BUY",
        "HOLD",
    }:
        rejection_reasons.append(
            "Technical signal does not support a long position."
        )

    # ---------------------------------------------------------
    # Score validation
    # ---------------------------------------------------------
    if (
        candidate.overall_score
        < rules.minimum_overall_score
    ):
        rejection_reasons.append(
            "Overall score is below the configured minimum."
        )

    if (
        candidate.technical_score
        < rules.minimum_technical_score
    ):
        rejection_reasons.append(
            "Technical score is below the configured minimum."
        )

    if (
        candidate.confidence
        < rules.minimum_confidence
    ):
        rejection_reasons.append(
            "Confidence is below the configured minimum."
        )

    # ---------------------------------------------------------
    # Non-blocking warnings
    # ---------------------------------------------------------
    if technical_signal == "HOLD":
        warnings.append(
            "Final recommendation is BUY, but technical signal is HOLD."
        )

    if (
        candidate.news_sentiment
        .strip()
        .upper()
        == "NEGATIVE"
    ):
        warnings.append(
            "News sentiment is negative."
        )

    if (
        asset_type == "stock"
        and candidate.fundamental_score < 40
    ):
        warnings.append(
            "Fundamental score is weak."
        )

    # ---------------------------------------------------------
    # Price levels
    # ---------------------------------------------------------
    (
        stop_loss_percent,
        take_profit_percent,
    ) = _get_stop_and_target_percentages(
        asset_type,
        rules,
    )

    stop_loss, take_profit = (
        _calculate_price_levels(
            entry_price=entry_price,
            stop_loss_percent=(
                stop_loss_percent
            ),
            take_profit_percent=(
                take_profit_percent
            ),
        )
    )

    risk_reward = _calculate_risk_reward(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    if (
        risk_reward
        < rules.minimum_risk_reward
    ):
        rejection_reasons.append(
            "Calculated risk/reward ratio is below "
            "the configured minimum."
        )

    # ---------------------------------------------------------
    # Return rejection result
    # ---------------------------------------------------------
    if rejection_reasons:
        return TradePlanBuildResult(
            ticker=candidate.ticker,
            approved=False,
            trade_plan=None,
            rejection_reasons=tuple(
                dict.fromkeys(
                    rejection_reasons
                )
            ),
            warnings=tuple(
                dict.fromkeys(
                    warnings
                )
            ),
        )

    # ---------------------------------------------------------
    # Build TradePlan
    # ---------------------------------------------------------
    trade_plan = TradePlan.create(
        ticker=candidate.ticker,
        asset_type=asset_type,
        action="BUY",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=candidate.confidence,
        overall_score=candidate.overall_score,
        technical_score=candidate.technical_score,
        fundamental_score=candidate.fundamental_score,
        news_score=candidate.news_score,
        max_risk_percent=(
            rules.maximum_risk_percent
        ),
        strategy=rules.strategy_name,
        notes=_build_notes(candidate),
    )

    return TradePlanBuildResult(
        ticker=candidate.ticker,
        approved=True,
        trade_plan=trade_plan,
        rejection_reasons=(),
        warnings=tuple(
            dict.fromkeys(
                warnings
            )
        ),
    )


def build_trade_plans(
    *,
    candidates: list[DeepResult],
    prices: dict[str, float],
    rules: TradePlanRules | None = None,
) -> tuple[
    list[TradePlan],
    list[TradePlanBuildResult],
]:
    """Build plans for multiple pipeline candidates."""

    if rules is None:
        rules = TradePlanRules()

    approved_plans: list[TradePlan] = []
    build_results: list[TradePlanBuildResult] = []

    normalized_prices = {
        ticker.strip().upper(): float(price)
        for ticker, price in prices.items()
    }

    for candidate in candidates:
        ticker = candidate.ticker.upper()

        entry_price = normalized_prices.get(
            ticker
        )

        if entry_price is None:
            result = TradePlanBuildResult(
                ticker=ticker,
                approved=False,
                trade_plan=None,
                rejection_reasons=(
                    "No entry price was supplied.",
                ),
                warnings=(),
            )

            build_results.append(result)
            continue

        try:
            result = build_trade_plan(
                candidate=candidate,
                entry_price=entry_price,
                rules=rules,
            )

        except Exception as error:
            result = TradePlanBuildResult(
                ticker=ticker,
                approved=False,
                trade_plan=None,
                rejection_reasons=(
                    f"Trade-plan build failed: {error}",
                ),
                warnings=(),
            )

        build_results.append(result)

        if (
            result.approved
            and result.trade_plan is not None
        ):
            approved_plans.append(
                result.trade_plan
            )

    return (
        approved_plans,
        build_results,
    )


def print_trade_plan_build_result(
    result: TradePlanBuildResult,
) -> None:
    """Print one trade-plan builder result."""

    print()
    print("=" * 80)
    print("TRADE PLAN BUILDER")
    print("=" * 80)

    print(f"Ticker:     {result.ticker}")
    print(
        f"Approved:   "
        f"{'YES' if result.approved else 'NO'}"
    )

    if result.trade_plan is not None:
        plan = result.trade_plan

        print()
        print(f"Action:     {plan.action}")
        print(f"Asset:      {plan.asset_type}")
        print(f"Entry:      {plan.entry_price:.6f}")
        print(f"Stop:       {plan.stop_loss:.6f}")
        print(f"Target:     {plan.take_profit:.6f}")
        print(
            f"Risk/Reward: "
            f"{plan.risk_reward:.2f}"
        )
        print(
            f"Risk limit:  "
            f"{plan.max_risk_percent:.2f}%"
        )
        print(
            f"Confidence:  "
            f"{plan.confidence}%"
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

    print("=" * 80)


def print_trade_plan_batch(
    plans: list[TradePlan],
    results: list[TradePlanBuildResult],
) -> None:
    """Print a compact multi-candidate builder summary."""

    print()
    print("=" * 100)
    print("TRADE PLAN BUILDER — BATCH SUMMARY")
    print("=" * 100)

    print(f"Candidates processed: {len(results)}")
    print(f"Plans approved:       {len(plans)}")
    print(
        f"Plans rejected:       "
        f"{len(results) - len(plans)}"
    )

    print()

    for result in results:
        status = (
            "APPROVED"
            if result.approved
            else "REJECTED"
        )

        if result.trade_plan is None:
            details = ""
        else:
            details = (
                f"Entry={result.trade_plan.entry_price:.4f} "
                f"Stop={result.trade_plan.stop_loss:.4f} "
                f"Target={result.trade_plan.take_profit:.4f} "
                f"RR={result.trade_plan.risk_reward:.2f}"
            )

        print(
            f"{result.ticker:<14} "
            f"{status:<10} "
            f"{details}"
        )

    print("=" * 100)