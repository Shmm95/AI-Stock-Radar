"""Main runner for the efficient AI-Stock-Radar pipeline."""

import argparse
import time

from src.pipeline.candidate_selector import (
    print_candidate_selection,
    select_final_candidates,
)
from src.pipeline.deep_analysis import (
    print_deep_ranking,
    run_deep_analysis,
)
from src.pipeline.fast_scanner import (
    print_fast_scan_ranking,
    scan_market_universe,
    select_fast_scan_finalists,
)
from src.pipeline.pipeline_models import (
    CandidateSelection,
    PipelineSummary,
)
from src.universe.universe_manager import (
    MarketUniverse,
    load_market_universe,
    print_universe_summary,
)


DEFAULT_PERIOD = "1y"

DEFAULT_DEEP_STOCK_LIMIT = 30
DEFAULT_DEEP_CRYPTO_LIMIT = 5

DEFAULT_FINAL_STOCK_LIMIT = 3
DEFAULT_FINAL_CRYPTO_LIMIT = 3


def _create_test_universe() -> MarketUniverse:
    """Return a small universe for safe pipeline testing."""

    return MarketUniverse(
        stocks=[
            "AAPL",
            "NVDA",
        ],
        crypto=[
            "BTC-USD",
            "ETH-USD",
        ],
    )


def _format_duration(
    total_seconds: float,
) -> str:
    """Format a duration as minutes and seconds."""

    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    if minutes == 0:
        return f"{seconds} seconds"

    return f"{minutes} minutes {seconds} seconds"


def _print_pipeline_header(
    test_mode: bool,
) -> None:
    """Print the pipeline startup header."""

    print()
    print("=" * 90)
    print("AI STOCK RADAR — EFFICIENT PIPELINE")
    print("=" * 90)

    if test_mode:
        print("Mode: TEST")
    else:
        print("Mode: FULL UNIVERSE")

    print()
    print("Stage 1: Fast technical scan")
    print("Stage 2: Fundamental, news, and decision analysis")
    print("Stage 3: Final Python candidate selection")
    print("Stage 4: OpenAI report — not called automatically")
    print("=" * 90)


def _print_pipeline_summary(
    summary: PipelineSummary,
    selection: CandidateSelection,
    duration_seconds: float,
) -> None:
    """Print final pipeline statistics."""

    print()
    print("=" * 90)
    print("PIPELINE SUMMARY")
    print("=" * 90)

    print(f"Configured stocks: {summary.total_stocks}")
    print(f"Configured crypto: {summary.total_crypto}")
    print(f"Configured total: {summary.total_symbols}")

    print()
    print(
        "Successful fast scans: "
        f"{summary.successful_fast_total}"
    )
    print(
        "Successful stock scans: "
        f"{summary.successful_fast_stocks}"
    )
    print(
        "Successful crypto scans: "
        f"{summary.successful_fast_crypto}"
    )

    print()
    print(
        "Deep-analyzed stocks: "
        f"{summary.deep_stock_count}"
    )
    print(
        "Deep-analyzed crypto: "
        f"{summary.deep_crypto_count}"
    )

    print()
    print(
        "Final stock candidates: "
        f"{summary.final_stock_count}"
    )
    print(
        "Final crypto candidates: "
        f"{summary.final_crypto_count}"
    )

    print()
    print(
        "AI candidate count: "
        f"{len(selection.ai_candidates)}"
    )

    if selection.ai_candidates:
        ai_tickers = ", ".join(
            result.ticker
            for result in selection.ai_candidates
        )

        print(f"AI candidates: {ai_tickers}")

    print()
    print(
        f"Failed symbols: "
        f"{len(summary.failed_symbols)}"
    )

    if summary.failed_symbols:
        print(
            "Failed ticker list: "
            + ", ".join(summary.failed_symbols)
        )

    print()
    print(
        "Total pipeline duration: "
        f"{_format_duration(duration_seconds)}"
    )

    print("=" * 90)


def run_pipeline(
    universe: MarketUniverse,
    period: str = DEFAULT_PERIOD,
    deep_stock_limit: int = DEFAULT_DEEP_STOCK_LIMIT,
    deep_crypto_limit: int = DEFAULT_DEEP_CRYPTO_LIMIT,
    final_stock_limit: int = DEFAULT_FINAL_STOCK_LIMIT,
    final_crypto_limit: int = DEFAULT_FINAL_CRYPTO_LIMIT,
    save_raw_data: bool = False,
) -> CandidateSelection:
    """Run Stage 1, Stage 2, and Stage 3."""

    started_at = time.perf_counter()

    print_universe_summary(universe)

    # =========================================================
    # STAGE 1 — FAST TECHNICAL SCAN
    # =========================================================
    fast_stocks, fast_crypto, fast_failed = (
        scan_market_universe(
            stocks=universe.stocks,
            crypto=universe.crypto,
            period=period,
            save_raw_data=save_raw_data,
        )
    )

    print_fast_scan_ranking(
        fast_stocks,
        title="STAGE 1 — TOP TECHNICAL STOCKS",
        limit=10,
    )

    print_fast_scan_ranking(
        fast_crypto,
        title="STAGE 1 — TOP TECHNICAL CRYPTO",
        limit=10,
    )

    # =========================================================
    # FAST SCAN FINALISTS
    # =========================================================
    stock_finalists, crypto_finalists = (
        select_fast_scan_finalists(
            stock_results=fast_stocks,
            crypto_results=fast_crypto,
            stock_limit=deep_stock_limit,
            crypto_limit=deep_crypto_limit,
        )
    )

    print()
    print("=" * 90)
    print("STAGE 1 FINALISTS")
    print("=" * 90)
    print(
        f"Stocks selected for deep analysis: "
        f"{len(stock_finalists)}"
    )
    print(
        f"Crypto selected for deep analysis: "
        f"{len(crypto_finalists)}"
    )
    print("=" * 90)

    # =========================================================
    # STAGE 2 — DEEP ANALYSIS
    # =========================================================
    deep_stocks, deep_crypto, deep_failed = (
        run_deep_analysis(
            stock_candidates=stock_finalists,
            crypto_candidates=crypto_finalists,
        )
    )

    print_deep_ranking(
        deep_stocks,
        title="STAGE 2 — DEEP STOCK RANKING",
        limit=10,
    )

    print_deep_ranking(
        deep_crypto,
        title="STAGE 2 — DEEP CRYPTO RANKING",
        limit=10,
    )

    # =========================================================
    # STAGE 3 — FINAL CANDIDATE SELECTION
    # =========================================================
    selection = select_final_candidates(
        stock_results=deep_stocks,
        crypto_results=deep_crypto,
        stock_limit=final_stock_limit,
        crypto_limit=final_crypto_limit,
    )

    print_candidate_selection(selection)

    # =========================================================
    # SUMMARY
    # =========================================================
    all_failed = list(
        dict.fromkeys(
            fast_failed + deep_failed
        )
    )

    summary = PipelineSummary(
        total_stocks=universe.stock_count,
        total_crypto=universe.crypto_count,
        successful_fast_stocks=len(fast_stocks),
        successful_fast_crypto=len(fast_crypto),
        deep_stock_count=len(deep_stocks),
        deep_crypto_count=len(deep_crypto),
        final_stock_count=len(selection.top_stocks),
        final_crypto_count=len(selection.top_crypto),
        failed_symbols=all_failed,
    )

    duration_seconds = (
        time.perf_counter() - started_at
    )

    _print_pipeline_summary(
        summary=summary,
        selection=selection,
        duration_seconds=duration_seconds,
    )

    return selection


def _parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the efficient AI-Stock-Radar pipeline."
        )
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run a small test using two stocks "
            "and two crypto assets."
        ),
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help=(
            "Yahoo Finance history period. "
            "Default: 1y"
        ),
    )

    parser.add_argument(
        "--deep-stocks",
        type=int,
        default=DEFAULT_DEEP_STOCK_LIMIT,
        help=(
            "Maximum stock finalists sent to "
            "deep analysis. Default: 30"
        ),
    )

    parser.add_argument(
        "--deep-crypto",
        type=int,
        default=DEFAULT_DEEP_CRYPTO_LIMIT,
        help=(
            "Maximum crypto finalists sent to "
            "deep analysis. Default: 5"
        ),
    )

    parser.add_argument(
        "--save-raw-data",
        action="store_true",
        help=(
            "Save downloaded Stage 1 market data "
            "under data/raw."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the pipeline from the command line."""

    arguments = _parse_arguments()

    _print_pipeline_header(
        test_mode=arguments.test,
    )

    if arguments.test:
        universe = _create_test_universe()

        deep_stock_limit = 1
        deep_crypto_limit = 1

        final_stock_limit = 1
        final_crypto_limit = 1

    else:
        universe = load_market_universe()

        deep_stock_limit = arguments.deep_stocks
        deep_crypto_limit = arguments.deep_crypto

        final_stock_limit = DEFAULT_FINAL_STOCK_LIMIT
        final_crypto_limit = DEFAULT_FINAL_CRYPTO_LIMIT

    try:
        run_pipeline(
            universe=universe,
            period=arguments.period,
            deep_stock_limit=deep_stock_limit,
            deep_crypto_limit=deep_crypto_limit,
            final_stock_limit=final_stock_limit,
            final_crypto_limit=final_crypto_limit,
            save_raw_data=arguments.save_raw_data,
        )

    except KeyboardInterrupt:
        print()
        print("Pipeline cancelled by the user.")
        raise SystemExit(130)

    except Exception as error:
        print()
        print("=" * 90)
        print("PIPELINE FAILED")
        print("=" * 90)
        print(f"Error: {error}")
        print("=" * 90)

        raise SystemExit(1) from error

    print()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()