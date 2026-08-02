"""Run one complete AI-Stock-Radar paper-trading cycle.

Flow:
    Market universe
    -> Fast technical scan
    -> Deep analysis
    -> Candidate selection
    -> Current price retrieval
    -> Trade-plan generation
    -> Position sizing
    -> Risk validation
    -> Local paper-broker execution

Safety:
    Dry-run mode is the default.
    Paper positions are opened only when --execute is supplied.

This module cannot send real-money broker orders.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd

from src.data.download_stock import download_stock_data
from src.execution.paper_trade_runner import (
    PaperExecutionBatch,
    PaperRunnerRules,
    execute_candidates,
    print_execution_batch,
)
from src.paper.paper_account import print_paper_account
from src.paper.paper_broker import (
    PaperBroker,
    print_completed_trade,
    print_open_positions,
)
from src.pipeline.pipeline_models import (
    CandidateSelection,
    DeepResult,
)
from src.pipeline.run_pipeline import run_pipeline
from src.risk.risk_manager import RiskRules
from src.strategy.trade_plan_builder import TradePlanRules
from src.universe.universe_manager import (
    MarketUniverse,
    load_market_universe,
)


DEFAULT_PERIOD = "1y"
DEFAULT_PRICE_PERIOD = "5d"

DEFAULT_DEEP_STOCK_LIMIT = 10
DEFAULT_DEEP_CRYPTO_LIMIT = 5

DEFAULT_FINAL_STOCK_LIMIT = 3
DEFAULT_FINAL_CRYPTO_LIMIT = 3

DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_CURRENCY = "EUR"


@dataclass(frozen=True, slots=True)
class MarketPriceResult:
    """Latest market-price result for one ticker."""

    ticker: str
    price: float | None
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class PaperCycleResult:
    """Summary of one complete paper-trading cycle."""

    selection: CandidateSelection
    price_results: tuple[MarketPriceResult, ...]
    execution_batch: PaperExecutionBatch

    dry_run: bool
    duration_seconds: float

    @property
    def successful_price_count(self) -> int:
        """Return the number of successfully retrieved prices."""

        return sum(
            1
            for result in self.price_results
            if result.success
        )

    @property
    def failed_price_count(self) -> int:
        """Return the number of failed price requests."""

        return sum(
            1
            for result in self.price_results
            if not result.success
        )


def _create_test_universe() -> MarketUniverse:
    """Return a small universe for safe development testing."""

    return MarketUniverse(
        stocks=[
            "AAPL",
            "NVDA",
            "MSFT",
        ],
        crypto=[
            "BTC-USD",
            "ETH-USD",
        ],
    )


def _extract_latest_close(
    data: pd.DataFrame,
    ticker: str,
) -> float:
    """Extract the newest valid closing price."""

    if data.empty:
        raise ValueError(
            f"No market data returned for {ticker}."
        )

    if "Close" not in data.columns:
        raise ValueError(
            f"Close column is missing for {ticker}."
        )

    close_values = data["Close"].dropna()

    if close_values.empty:
        raise ValueError(
            f"No valid closing price found for {ticker}."
        )

    latest_value = close_values.iloc[-1]

    if isinstance(latest_value, pd.Series):
        if latest_value.empty:
            raise ValueError(
                f"Latest close value is empty for {ticker}."
            )

        latest_value = latest_value.iloc[0]

    price = float(latest_value)

    if price <= 0:
        raise ValueError(
            f"Latest close price is invalid for {ticker}."
        )

    return round(price, 6)


def get_latest_market_price(
    ticker: str,
    period: str = DEFAULT_PRICE_PERIOD,
) -> MarketPriceResult:
    """Download and return the newest valid close for one ticker."""

    normalized_ticker = ticker.strip().upper()

    if not normalized_ticker:
        return MarketPriceResult(
            ticker=ticker,
            price=None,
            success=False,
            message="Ticker cannot be empty.",
        )

    try:
        market_data = download_stock_data(
            normalized_ticker,
            period=period,
        )

        latest_price = _extract_latest_close(
            market_data,
            normalized_ticker,
        )

    except Exception as error:
        return MarketPriceResult(
            ticker=normalized_ticker,
            price=None,
            success=False,
            message=str(error),
        )

    return MarketPriceResult(
        ticker=normalized_ticker,
        price=latest_price,
        success=True,
        message="Latest market price retrieved.",
    )


def get_candidate_prices(
    candidates: list[DeepResult],
    period: str = DEFAULT_PRICE_PERIOD,
) -> tuple[
    dict[str, float],
    tuple[MarketPriceResult, ...],
]:
    """Retrieve current prices for final candidates."""

    prices: dict[str, float] = {}
    results: list[MarketPriceResult] = []

    print()
    print("=" * 90)
    print("CURRENT PRICE RETRIEVAL")
    print("=" * 90)

    total = len(candidates)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        ticker = candidate.ticker.upper()

        print(
            f"[{index}/{total}] "
            f"Retrieving price for {ticker}..."
        )

        result = get_latest_market_price(
            ticker=ticker,
            period=period,
        )

        results.append(result)

        if (
            result.success
            and result.price is not None
        ):
            prices[ticker] = result.price

            print(
                f"  Price: {result.price:.6f}"
            )

        else:
            print(
                f"  Failed: {result.message}"
            )

    print()
    print(
        f"Successful prices: {len(prices)}"
    )
    print(
        f"Failed prices: "
        f"{len(results) - len(prices)}"
    )
    print("=" * 90)

    return prices, tuple(results)


def _create_runner_rules() -> PaperRunnerRules:
    """Return conservative Paper Trading V1 rules."""

    trade_plan_rules = TradePlanRules(
        minimum_overall_score=60,
        minimum_technical_score=50,
        minimum_confidence=50,
        stock_stop_loss_percent=5.0,
        stock_take_profit_percent=10.0,
        crypto_stop_loss_percent=7.0,
        crypto_take_profit_percent=14.0,
        maximum_risk_percent=1.0,
        minimum_risk_reward=1.5,
        strategy_name="Radar Paper V1",
    )

    risk_rules = RiskRules(
        trading_enabled=True,
        maximum_open_positions=8,
        maximum_position_percent=25.0,
        maximum_total_open_risk_percent=5.0,
        maximum_crypto_allocation_percent=20.0,
        maximum_daily_loss_percent=2.0,
        minimum_risk_reward=1.5,
        minimum_confidence=50,
        minimum_overall_score=60,
        allow_duplicate_ticker=False,
    )

    return PaperRunnerRules(
        maximum_position_percent=25.0,
        allow_fractional_stocks=False,
        allow_fractional_crypto=True,
        stop_after_first_execution_failure=False,
        skip_existing_positions=True,
        trade_plan_rules=trade_plan_rules,
        risk_rules=risk_rules,
    )


def _format_duration(
    seconds: float,
) -> str:
    """Format elapsed seconds as minutes and seconds."""

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes == 0:
        return f"{remaining_seconds} seconds"

    return (
        f"{minutes} minutes "
        f"{remaining_seconds} seconds"
    )


def _print_cycle_header(
    *,
    dry_run: bool,
    test_mode: bool,
) -> None:
    """Print the paper-cycle startup header."""

    print()
    print("=" * 100)
    print("AI STOCK RADAR — PAPER TRADING CYCLE")
    print("=" * 100)

    print(
        f"Universe mode: "
        f"{'TEST' if test_mode else 'FULL'}"
    )

    print(
        f"Execution mode: "
        f"{'DRY RUN' if dry_run else 'PAPER EXECUTION'}"
    )

    print()
    print("Pipeline")
    print("1. Fast technical scan")
    print("2. Deep analysis")
    print("3. Final candidate selection")
    print("4. Latest price retrieval")
    print("5. Trade-plan generation")
    print("6. Position sizing")
    print("7. Portfolio risk validation")
    print("8. Local paper-broker execution")

    if dry_run:
        print()
        print(
            "Safety: no paper positions will be opened."
        )

    print("=" * 100)


def _print_price_failures(
    price_results: tuple[MarketPriceResult, ...],
) -> None:
    """Print failed price retrievals."""

    failed_results = [
        result
        for result in price_results
        if not result.success
    ]

    if not failed_results:
        return

    print()
    print("=" * 90)
    print("PRICE RETRIEVAL FAILURES")
    print("=" * 90)

    for result in failed_results:
        print(
            f"- {result.ticker}: "
            f"{result.message}"
        )

    print("=" * 90)


def _print_cycle_summary(
    result: PaperCycleResult,
) -> None:
    """Print final paper-cycle statistics."""

    print()
    print("=" * 100)
    print("PAPER CYCLE SUMMARY")
    print("=" * 100)

    print(
        f"Final candidates:           "
        f"{len(result.selection.ai_candidates)}"
    )

    print(
        f"Prices retrieved:           "
        f"{result.successful_price_count}"
    )

    print(
        f"Price failures:             "
        f"{result.failed_price_count}"
    )

    print(
        f"Candidates approved:        "
        f"{result.execution_batch.approved_count}"
    )

    print(
        f"Positions executed:         "
        f"{result.execution_batch.executed_count}"
    )

    print(
        f"Execution mode:             "
        f"{'DRY RUN' if result.dry_run else 'PAPER EXECUTION'}"
    )

    print(
        f"Duration:                   "
        f"{_format_duration(result.duration_seconds)}"
    )

    print("=" * 100)


def run_paper_cycle(
    *,
    universe: MarketUniverse,
    dry_run: bool = True,
    scan_period: str = DEFAULT_PERIOD,
    price_period: str = DEFAULT_PRICE_PERIOD,
    deep_stock_limit: int = DEFAULT_DEEP_STOCK_LIMIT,
    deep_crypto_limit: int = DEFAULT_DEEP_CRYPTO_LIMIT,
    final_stock_limit: int = DEFAULT_FINAL_STOCK_LIMIT,
    final_crypto_limit: int = DEFAULT_FINAL_CRYPTO_LIMIT,
    broker: PaperBroker | None = None,
) -> PaperCycleResult:
    """Run one complete radar-to-paper-trading cycle."""

    started_at = time.perf_counter()

    if broker is None:
        broker = PaperBroker()

    selection = run_pipeline(
        universe=universe,
        period=scan_period,
        deep_stock_limit=deep_stock_limit,
        deep_crypto_limit=deep_crypto_limit,
        final_stock_limit=final_stock_limit,
        final_crypto_limit=final_crypto_limit,
        save_raw_data=False,
    )

    candidates = selection.ai_candidates

    if not candidates:
        raise RuntimeError(
            "Pipeline did not produce any paper candidates."
        )

    prices, price_results = get_candidate_prices(
        candidates=candidates,
        period=price_period,
    )

    _print_price_failures(
        price_results
    )

    execution_batch = execute_candidates(
        candidates=candidates,
        prices=prices,
        broker=broker,
        rules=_create_runner_rules(),
        dry_run=dry_run,
    )

    print_execution_batch(
        execution_batch,
        broker=broker,
    )

    duration_seconds = (
        time.perf_counter()
        - started_at
    )

    result = PaperCycleResult(
        selection=selection,
        price_results=price_results,
        execution_batch=execution_batch,
        dry_run=dry_run,
        duration_seconds=duration_seconds,
    )

    _print_cycle_summary(result)

    return result


def monitor_open_positions(
    broker: PaperBroker | None = None,
    price_period: str = DEFAULT_PRICE_PERIOD,
) -> None:
    """Update open positions and process stop/target exits."""

    if broker is None:
        broker = PaperBroker()

    positions = broker.list_positions()

    print()
    print("=" * 90)
    print("PAPER POSITION MONITOR")
    print("=" * 90)

    if not positions:
        print("No open paper positions.")
        print("=" * 90)
        return

    prices: dict[str, float] = {}

    total = len(positions)

    for index, position in enumerate(
        positions,
        start=1,
    ):
        print(
            f"[{index}/{total}] "
            f"Updating {position.ticker}..."
        )

        result = get_latest_market_price(
            ticker=position.ticker,
            period=price_period,
        )

        if (
            result.success
            and result.price is not None
        ):
            prices[position.ticker] = (
                result.price
            )

            print(
                f"  Current price: "
                f"{result.price:.6f}"
            )

        else:
            print(
                f"  Price failed: "
                f"{result.message}"
            )

    if not prices:
        print()
        print("No valid prices were retrieved.")
        print("=" * 90)
        return

    closed_trades = broker.process_exit_rules(
        prices
    )

    print_open_positions(
        broker.list_positions()
    )

    if closed_trades:
        print()
        print("AUTOMATICALLY CLOSED TRADES")

        for trade in closed_trades:
            print_completed_trade(
                trade
            )

    else:
        print()
        print(
            "No stop-loss or take-profit "
            "levels were reached."
        )

    print("=" * 90)


def _parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one AI-Stock-Radar "
            "paper-trading cycle."
        )
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Use a small universe for "
            "development testing."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Open approved local paper positions. "
            "Without this flag, the command "
            "runs as dry-run."
        ),
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help=(
            "Only update open positions and process "
            "stop-loss or take-profit exits."
        ),
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Reset the local paper account "
            "before running."
        ),
    )

    parser.add_argument(
        "--initial-cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=(
            "Initial paper-account cash "
            "used with --reset."
        ),
    )

    parser.add_argument(
        "--currency",
        default=DEFAULT_CURRENCY,
        help=(
            "Paper-account currency "
            "used with --reset."
        ),
    )

    parser.add_argument(
        "--scan-period",
        default=DEFAULT_PERIOD,
        help=(
            "Historical period used by "
            "the radar scan."
        ),
    )

    parser.add_argument(
        "--price-period",
        default=DEFAULT_PRICE_PERIOD,
        help=(
            "Historical period used to retrieve "
            "latest prices."
        ),
    )

    parser.add_argument(
        "--deep-stocks",
        type=int,
        default=DEFAULT_DEEP_STOCK_LIMIT,
        help=(
            "Number of stock finalists sent "
            "to deep analysis."
        ),
    )

    parser.add_argument(
        "--deep-crypto",
        type=int,
        default=DEFAULT_DEEP_CRYPTO_LIMIT,
        help=(
            "Number of crypto finalists sent "
            "to deep analysis."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line paper-trading workflow."""

    arguments = _parse_arguments()

    broker = PaperBroker()

    if arguments.reset:
        account = broker.reset(
            initial_cash=arguments.initial_cash,
            currency=arguments.currency,
        )

        print_paper_account(
            account
        )

    if arguments.monitor:
        monitor_open_positions(
            broker=broker,
            price_period=arguments.price_period,
        )

        return

    dry_run = not arguments.execute

    _print_cycle_header(
        dry_run=dry_run,
        test_mode=arguments.test,
    )

    if arguments.test:
        universe = _create_test_universe()

        deep_stock_limit = min(
            arguments.deep_stocks,
            2,
        )

        deep_crypto_limit = min(
            arguments.deep_crypto,
            2,
        )

        final_stock_limit = 2
        final_crypto_limit = 2

    else:
        universe = load_market_universe()

        deep_stock_limit = (
            arguments.deep_stocks
        )

        deep_crypto_limit = (
            arguments.deep_crypto
        )

        final_stock_limit = (
            DEFAULT_FINAL_STOCK_LIMIT
        )

        final_crypto_limit = (
            DEFAULT_FINAL_CRYPTO_LIMIT
        )

    try:
        run_paper_cycle(
            universe=universe,
            dry_run=dry_run,
            scan_period=arguments.scan_period,
            price_period=arguments.price_period,
            deep_stock_limit=deep_stock_limit,
            deep_crypto_limit=deep_crypto_limit,
            final_stock_limit=final_stock_limit,
            final_crypto_limit=final_crypto_limit,
            broker=broker,
        )

    except KeyboardInterrupt:
        print()
        print("Paper cycle cancelled by the user.")

        raise SystemExit(130)

    except Exception as error:
        print()
        print("=" * 100)
        print("PAPER CYCLE FAILED")
        print("=" * 100)
        print(f"Error: {error}")
        print("=" * 100)

        raise SystemExit(1) from error

    print()
    print("Paper cycle completed successfully.")


if __name__ == "__main__":
    main()