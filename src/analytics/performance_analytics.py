"""Performance analytics for the local paper-trading engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from src.paper.paper_account import PaperAccount
from src.paper.paper_broker import PaperBroker
from src.paper.paper_models import EquityPoint, PaperTrade


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Calculated paper-trading performance statistics."""

    initial_capital: float
    current_account_value: float

    total_return_amount: float
    total_return_percent: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    win_rate: float
    loss_rate: float

    gross_profit: float
    gross_loss: float
    net_profit: float

    average_trade: float
    average_win: float
    average_loss: float

    largest_win: float
    largest_loss: float

    profit_factor: float
    payoff_ratio: float
    expectancy: float

    maximum_drawdown_amount: float
    maximum_drawdown_percent: float

    open_position_count: int
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float


def _round_money(value: float) -> float:
    """Round a monetary value to two decimal places."""

    return round(float(value), 2)


def _round_percent(value: float) -> float:
    """Round a percentage value."""

    return round(float(value), 4)


def _calculate_maximum_drawdown(
    equity_points: list[EquityPoint],
    initial_capital: float,
) -> tuple[float, float]:
    """Calculate maximum peak-to-trough equity drawdown."""

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero."
        )

    equity_values = [
        initial_capital,
        *[
            float(point.account_value)
            for point in equity_points
        ],
    ]

    peak = equity_values[0]

    maximum_drawdown_amount = 0.0
    maximum_drawdown_percent = 0.0

    for equity in equity_values:
        if equity > peak:
            peak = equity

        drawdown_amount = peak - equity

        if peak <= 0:
            drawdown_percent = 0.0
        else:
            drawdown_percent = (
                drawdown_amount
                / peak
                * 100
            )

        if drawdown_amount > maximum_drawdown_amount:
            maximum_drawdown_amount = drawdown_amount

        if drawdown_percent > maximum_drawdown_percent:
            maximum_drawdown_percent = drawdown_percent

    return (
        _round_money(maximum_drawdown_amount),
        _round_percent(maximum_drawdown_percent),
    )


def _calculate_profit_factor(
    gross_profit: float,
    gross_loss: float,
) -> float:
    """Calculate gross profit divided by absolute gross loss."""

    if gross_loss < 0:
        denominator = abs(gross_loss)
    else:
        denominator = gross_loss

    if denominator == 0:
        if gross_profit > 0:
            return inf

        return 0.0

    return round(
        gross_profit / denominator,
        4,
    )


def _calculate_payoff_ratio(
    average_win: float,
    average_loss: float,
) -> float:
    """Calculate average win divided by absolute average loss."""

    if average_loss == 0:
        if average_win > 0:
            return inf

        return 0.0

    return round(
        average_win
        / abs(average_loss),
        4,
    )


def calculate_performance_metrics(
    *,
    account: PaperAccount,
    trades: list[PaperTrade],
    equity_points: list[EquityPoint],
    open_position_count: int,
    unrealized_pnl: float,
    current_account_value: float,
) -> PerformanceMetrics:
    """Calculate complete paper-trading performance metrics."""

    if account.initial_cash <= 0:
        raise ValueError(
            "Paper account initial capital must be positive."
        )

    winning_trades = [
        trade
        for trade in trades
        if trade.net_pnl > 0
    ]

    losing_trades = [
        trade
        for trade in trades
        if trade.net_pnl < 0
    ]

    breakeven_trades = [
        trade
        for trade in trades
        if trade.net_pnl == 0
    ]

    total_trades = len(trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    breakeven_count = len(breakeven_trades)

    gross_profit = _round_money(
        sum(
            trade.net_pnl
            for trade in winning_trades
        )
    )

    gross_loss = _round_money(
        sum(
            trade.net_pnl
            for trade in losing_trades
        )
    )

    net_profit = _round_money(
        sum(
            trade.net_pnl
            for trade in trades
        )
    )

    if total_trades == 0:
        win_rate = 0.0
        loss_rate = 0.0
        average_trade = 0.0
    else:
        win_rate = (
            winning_count
            / total_trades
            * 100
        )

        loss_rate = (
            losing_count
            / total_trades
            * 100
        )

        average_trade = (
            net_profit
            / total_trades
        )

    if winning_count == 0:
        average_win = 0.0
        largest_win = 0.0
    else:
        average_win = (
            gross_profit
            / winning_count
        )

        largest_win = max(
            trade.net_pnl
            for trade in winning_trades
        )

    if losing_count == 0:
        average_loss = 0.0
        largest_loss = 0.0
    else:
        average_loss = (
            gross_loss
            / losing_count
        )

        largest_loss = min(
            trade.net_pnl
            for trade in losing_trades
        )

    win_probability = (
        winning_count / total_trades
        if total_trades
        else 0.0
    )

    loss_probability = (
        losing_count / total_trades
        if total_trades
        else 0.0
    )

    expectancy = (
        win_probability * average_win
        + loss_probability * average_loss
    )

    profit_factor = _calculate_profit_factor(
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )

    payoff_ratio = _calculate_payoff_ratio(
        average_win=average_win,
        average_loss=average_loss,
    )

    (
        maximum_drawdown_amount,
        maximum_drawdown_percent,
    ) = _calculate_maximum_drawdown(
        equity_points=equity_points,
        initial_capital=account.initial_cash,
    )

    total_return_amount = _round_money(
        current_account_value
        - account.initial_cash
        - account.deposits
        + account.withdrawals
    )

    invested_capital = (
        account.initial_cash
        + account.deposits
        - account.withdrawals
    )

    if invested_capital <= 0:
        total_return_percent = 0.0
    else:
        total_return_percent = (
            total_return_amount
            / invested_capital
            * 100
        )

    return PerformanceMetrics(
        initial_capital=_round_money(
            account.initial_cash
        ),
        current_account_value=_round_money(
            current_account_value
        ),
        total_return_amount=total_return_amount,
        total_return_percent=_round_percent(
            total_return_percent
        ),
        total_trades=total_trades,
        winning_trades=winning_count,
        losing_trades=losing_count,
        breakeven_trades=breakeven_count,
        win_rate=_round_percent(win_rate),
        loss_rate=_round_percent(loss_rate),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        average_trade=_round_money(
            average_trade
        ),
        average_win=_round_money(
            average_win
        ),
        average_loss=_round_money(
            average_loss
        ),
        largest_win=_round_money(
            largest_win
        ),
        largest_loss=_round_money(
            largest_loss
        ),
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        expectancy=_round_money(
            expectancy
        ),
        maximum_drawdown_amount=(
            maximum_drawdown_amount
        ),
        maximum_drawdown_percent=(
            maximum_drawdown_percent
        ),
        open_position_count=int(
            open_position_count
        ),
        unrealized_pnl=_round_money(
            unrealized_pnl
        ),
        realized_pnl=_round_money(
            account.realized_pnl
        ),
        fees_paid=_round_money(
            account.fees_paid
        ),
    )


def calculate_broker_performance(
    broker: PaperBroker | None = None,
) -> PerformanceMetrics:
    """Calculate metrics directly from the current Paper Broker."""

    if broker is None:
        broker = PaperBroker()

    account = broker.account_store.load()
    trades = broker.list_trades()
    equity_points = broker.list_equity_points()
    positions = broker.list_positions()

    positions_market_value = sum(
        position.market_value
        for position in positions
    )

    unrealized_pnl = sum(
        position.unrealized_pnl
        for position in positions
    )

    current_account_value = (
        account.cash
        + positions_market_value
    )

    return calculate_performance_metrics(
        account=account,
        trades=trades,
        equity_points=equity_points,
        open_position_count=len(positions),
        unrealized_pnl=unrealized_pnl,
        current_account_value=(
            current_account_value
        ),
    )


def _format_ratio(value: float) -> str:
    """Format finite and infinite ratios."""

    if value == inf:
        return "∞"

    return f"{value:.4f}"


def print_performance_report(
    metrics: PerformanceMetrics,
    currency: str = "EUR",
) -> None:
    """Print a readable paper-trading performance report."""

    print()
    print("=" * 90)
    print("PAPER TRADING PERFORMANCE REPORT")
    print("=" * 90)

    print()
    print("ACCOUNT")

    print(
        f"Initial capital:          "
        f"{metrics.initial_capital:,.2f} {currency}"
    )

    print(
        f"Current account value:    "
        f"{metrics.current_account_value:,.2f} {currency}"
    )

    print(
        f"Total return:             "
        f"{metrics.total_return_amount:+,.2f} {currency}"
    )

    print(
        f"Total return %:           "
        f"{metrics.total_return_percent:+.4f}%"
    )

    print(
        f"Realized PnL:             "
        f"{metrics.realized_pnl:+,.2f} {currency}"
    )

    print(
        f"Unrealized PnL:           "
        f"{metrics.unrealized_pnl:+,.2f} {currency}"
    )

    print(
        f"Fees paid:                "
        f"{metrics.fees_paid:,.2f} {currency}"
    )

    print(
        f"Open positions:           "
        f"{metrics.open_position_count}"
    )

    print()
    print("TRADE STATISTICS")

    print(
        f"Total closed trades:      "
        f"{metrics.total_trades}"
    )

    print(
        f"Winning trades:           "
        f"{metrics.winning_trades}"
    )

    print(
        f"Losing trades:            "
        f"{metrics.losing_trades}"
    )

    print(
        f"Breakeven trades:         "
        f"{metrics.breakeven_trades}"
    )

    print(
        f"Win rate:                 "
        f"{metrics.win_rate:.4f}%"
    )

    print(
        f"Loss rate:                "
        f"{metrics.loss_rate:.4f}%"
    )

    print()
    print("PROFITABILITY")

    print(
        f"Gross profit:             "
        f"{metrics.gross_profit:+,.2f} {currency}"
    )

    print(
        f"Gross loss:               "
        f"{metrics.gross_loss:+,.2f} {currency}"
    )

    print(
        f"Net closed-trade profit:  "
        f"{metrics.net_profit:+,.2f} {currency}"
    )

    print(
        f"Average trade:            "
        f"{metrics.average_trade:+,.2f} {currency}"
    )

    print(
        f"Average win:              "
        f"{metrics.average_win:+,.2f} {currency}"
    )

    print(
        f"Average loss:             "
        f"{metrics.average_loss:+,.2f} {currency}"
    )

    print(
        f"Largest win:              "
        f"{metrics.largest_win:+,.2f} {currency}"
    )

    print(
        f"Largest loss:             "
        f"{metrics.largest_loss:+,.2f} {currency}"
    )

    print()
    print("EDGE AND RISK")

    print(
        f"Expectancy per trade:     "
        f"{metrics.expectancy:+,.2f} {currency}"
    )

    print(
        f"Profit factor:            "
        f"{_format_ratio(metrics.profit_factor)}"
    )

    print(
        f"Payoff ratio:             "
        f"{_format_ratio(metrics.payoff_ratio)}"
    )

    print(
        f"Maximum drawdown:         "
        f"{metrics.maximum_drawdown_amount:,.2f} {currency}"
    )

    print(
        f"Maximum drawdown %:       "
        f"{metrics.maximum_drawdown_percent:.4f}%"
    )

    print()
    print("INTERPRETATION")

    if metrics.total_trades < 30:
        print(
            "- Sample size is too small for reliable conclusions."
        )

    if metrics.expectancy > 0:
        print(
            "- Current closed trades show positive expectancy."
        )
    elif metrics.total_trades > 0:
        print(
            "- Current closed trades do not show positive expectancy."
        )
    else:
        print(
            "- No completed trades are available yet."
        )

    if (
        metrics.profit_factor != inf
        and metrics.profit_factor >= 1.3
    ):
        print(
            "- Profit factor currently exceeds the initial 1.30 target."
        )
    elif metrics.total_trades > 0:
        print(
            "- Profit factor is below the initial 1.30 target."
        )

    print("=" * 90)


def main() -> None:
    """Print current local Paper Broker performance."""

    broker = PaperBroker()
    account = broker.account_store.load()

    metrics = calculate_broker_performance(
        broker
    )

    print_performance_report(
        metrics=metrics,
        currency=account.currency,
    )


if __name__ == "__main__":
    main()