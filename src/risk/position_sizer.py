"""Position sizing utilities for AI-Stock-Radar."""

from dataclasses import dataclass
from math import floor

from src.strategy.trade_plan import TradePlan


@dataclass(frozen=True, slots=True)
class PositionSizeResult:
    """Calculated position size and risk information."""

    ticker: str

    account_value: float
    available_cash: float

    risk_percent: float
    maximum_risk_amount: float

    entry_price: float
    stop_loss: float
    stop_distance: float

    quantity: float
    position_value: float
    actual_risk_amount: float
    actual_risk_percent: float

    cash_usage_percent: float

    approved: bool
    rejection_reasons: tuple[str, ...]


def _validate_positive(
    value: float,
    field_name: str,
) -> None:
    """Validate that a numeric value is positive."""

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )


def calculate_position_size(
    trade_plan: TradePlan,
    account_value: float,
    available_cash: float | None = None,
    risk_percent: float | None = None,
    maximum_position_percent: float = 25.0,
    allow_fractional: bool = False,
) -> PositionSizeResult:
    """Calculate a risk-based position size for one trade plan."""

    _validate_positive(
        account_value,
        "account_value",
    )

    if available_cash is None:
        available_cash = account_value

    _validate_positive(
        available_cash,
        "available_cash",
    )

    if risk_percent is None:
        risk_percent = trade_plan.max_risk_percent

    _validate_positive(
        risk_percent,
        "risk_percent",
    )

    _validate_positive(
        maximum_position_percent,
        "maximum_position_percent",
    )

    if risk_percent > 100:
        raise ValueError(
            "risk_percent cannot be greater than 100."
        )

    if maximum_position_percent > 100:
        raise ValueError(
            "maximum_position_percent cannot be greater than 100."
        )

    entry_price = float(trade_plan.entry_price)
    stop_loss = float(trade_plan.stop_loss)

    _validate_positive(
        entry_price,
        "entry_price",
    )

    _validate_positive(
        stop_loss,
        "stop_loss",
    )

    stop_distance = abs(
        entry_price - stop_loss
    )

    if stop_distance <= 0:
        raise ValueError(
            "Entry price and stop loss cannot be equal."
        )

    maximum_risk_amount = (
        account_value
        * risk_percent
        / 100
    )

    risk_based_quantity = (
        maximum_risk_amount
        / stop_distance
    )

    maximum_position_value = (
        account_value
        * maximum_position_percent
        / 100
    )

    maximum_affordable_value = min(
        available_cash,
        maximum_position_value,
    )

    cash_based_quantity = (
        maximum_affordable_value
        / entry_price
    )

    raw_quantity = min(
        risk_based_quantity,
        cash_based_quantity,
    )

    if allow_fractional:
        quantity = round(
            raw_quantity,
            6,
        )
    else:
        quantity = float(
            floor(raw_quantity)
        )

    position_value = round(
        quantity * entry_price,
        2,
    )

    actual_risk_amount = round(
        quantity * stop_distance,
        2,
    )

    actual_risk_percent = round(
        (
            actual_risk_amount
            / account_value
            * 100
        ),
        4,
    )

    cash_usage_percent = round(
        (
            position_value
            / available_cash
            * 100
        ),
        2,
    )

    rejection_reasons: list[str] = []

    if trade_plan.action.upper() != "BUY":
        rejection_reasons.append(
            "Only BUY trade plans are supported in Paper Trading V1."
        )

    if quantity <= 0:
        rejection_reasons.append(
            "Calculated quantity is zero."
        )

    if position_value > available_cash:
        rejection_reasons.append(
            "Position value exceeds available cash."
        )

    if actual_risk_amount > maximum_risk_amount:
        rejection_reasons.append(
            "Actual risk exceeds the allowed risk amount."
        )

    if (
        trade_plan.risk_reward
        < 1.5
    ):
        rejection_reasons.append(
            "Risk/reward ratio is below 1.5."
        )

    approved = not rejection_reasons

    return PositionSizeResult(
        ticker=trade_plan.ticker,
        account_value=round(
            account_value,
            2,
        ),
        available_cash=round(
            available_cash,
            2,
        ),
        risk_percent=round(
            risk_percent,
            4,
        ),
        maximum_risk_amount=round(
            maximum_risk_amount,
            2,
        ),
        entry_price=round(
            entry_price,
            4,
        ),
        stop_loss=round(
            stop_loss,
            4,
        ),
        stop_distance=round(
            stop_distance,
            4,
        ),
        quantity=quantity,
        position_value=position_value,
        actual_risk_amount=actual_risk_amount,
        actual_risk_percent=actual_risk_percent,
        cash_usage_percent=cash_usage_percent,
        approved=approved,
        rejection_reasons=tuple(
            rejection_reasons
        ),
    )


def print_position_size(
    result: PositionSizeResult,
) -> None:
    """Print a readable position-sizing report."""

    print()
    print("=" * 72)
    print("POSITION SIZE RESULT")
    print("=" * 72)

    print(f"Ticker:                 {result.ticker}")
    print(
        f"Account value:          "
        f"€{result.account_value:,.2f}"
    )
    print(
        f"Available cash:         "
        f"€{result.available_cash:,.2f}"
    )

    print()
    print(
        f"Risk limit:             "
        f"{result.risk_percent:.2f}%"
    )
    print(
        f"Maximum risk amount:    "
        f"€{result.maximum_risk_amount:,.2f}"
    )

    print()
    print(
        f"Entry price:            "
        f"€{result.entry_price:,.2f}"
    )
    print(
        f"Stop loss:              "
        f"€{result.stop_loss:,.2f}"
    )
    print(
        f"Stop distance:          "
        f"€{result.stop_distance:,.2f}"
    )

    print()
    print(
        f"Quantity:               "
        f"{result.quantity}"
    )
    print(
        f"Position value:         "
        f"€{result.position_value:,.2f}"
    )
    print(
        f"Actual risk amount:     "
        f"€{result.actual_risk_amount:,.2f}"
    )
    print(
        f"Actual account risk:    "
        f"{result.actual_risk_percent:.4f}%"
    )
    print(
        f"Available cash usage:   "
        f"{result.cash_usage_percent:.2f}%"
    )

    print()
    print(
        f"Approved:               "
        f"{'YES' if result.approved else 'NO'}"
    )

    if result.rejection_reasons:
        print()
        print("Rejection reasons:")

        for reason in result.rejection_reasons:
            print(f"- {reason}")

    print("=" * 72)