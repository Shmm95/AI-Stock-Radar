"""Main entry point for AI-Stock-Radar."""

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from src.analysis.technical_indicators import add_technical_indicators
from src.data.download_stock import (
    download_stock_data,
    save_stock_data,
)
from src.decision.decision_engine import calculate_decision
from src.fundamental.fundamental_score import (
    calculate_fundamental_score,
)
from src.news.news_sentiment import calculate_news_score
from src.reporting.csv_report import (
    append_history_report,
    save_ranking_report,
)
from src.scoring.radar_score import calculate_radar_score
from src.signals.signal_engine import generate_signal
from src.universe.universe_manager import (
    load_market_universe,
    print_universe_summary,
)


@dataclass
class RadarResult:
    """Final combined result for one market symbol."""

    ticker: str
    signal: str
    confidence: int
    score: int
    stars: int
    technical_score: int
    fundamental_score: int
    news_score: int
    technical_signal: str
    news_sentiment: str


def print_latest_summary(
    ticker: str,
    data: pd.DataFrame,
) -> None:
    """Print the latest valid price and technical indicators."""

    required_columns = [
        "Close",
        "EMA20",
        "EMA50",
        "RSI14",
        "MACD",
    ]

    valid = data.dropna(
        subset=required_columns,
    )

    if valid.empty:
        print(
            f"No valid technical data available for {ticker}"
        )
        return

    latest = valid.iloc[-1]

    print(f"Ticker: {ticker}")
    print(f"Close:  {float(latest['Close']):.2f}")
    print(f"EMA20:  {float(latest['EMA20']):.2f}")
    print(f"EMA50:  {float(latest['EMA50']):.2f}")
    print(f"RSI14:  {float(latest['RSI14']):.2f}")
    print(f"MACD:   {float(latest['MACD']):.4f}")


def print_fundamental_metrics(
    fundamental,
) -> None:
    """Print available fundamental metrics."""

    labels = {
        "forward_pe": "Forward P/E",
        "trailing_pe": "Trailing P/E",
        "peg_ratio": "PEG Ratio",
        "price_to_book": "Price / Book",
        "dividend_yield": "Dividend Yield",
        "return_on_equity": "Return on Equity",
        "profit_margin": "Profit Margin",
    }

    percentage_metrics = {
        "dividend_yield",
        "return_on_equity",
        "profit_margin",
    }

    print()
    print("Fundamental Metrics")

    for metric_name, label in labels.items():
        value = fundamental.metrics.get(metric_name)

        if value is None:
            print(f"- {label}: unavailable")
        elif metric_name in percentage_metrics:
            print(f"- {label}: {value * 100:.2f}%")
        else:
            print(f"- {label}: {value:.2f}")


def print_news_summary(
    news,
) -> None:
    """Print news sentiment and recent headlines."""

    print()
    print("News Analysis")
    print(f"News Score: {news.score}/100")
    print(f"Sentiment: {news.sentiment}")
    print(
        "Positive keyword matches: "
        f"{news.positive_matches}"
    )
    print(
        "Negative keyword matches: "
        f"{news.negative_matches}"
    )

    if news.headlines:
        print()
        print("Recent Headlines")

        for headline in news.headlines:
            print(f"- {headline}")
    else:
        print("- No recent headlines were available")


def print_ranking(
    results: list[RadarResult],
) -> None:
    """Print ranking ordered by final decision score."""

    ranked = sorted(
        results,
        key=lambda item: (
            item.score,
            item.confidence,
            item.technical_score,
        ),
        reverse=True,
    )

    print()
    print("=" * 110)
    print("AI STOCK RADAR — FINAL RANKING")
    print("=" * 110)

    if not ranked:
        print("No successful analysis results were produced.")
        return

    for position, result in enumerate(
        ranked,
        start=1,
    ):
        stars = "⭐" * result.stars

        if result.fundamental_score < 0:
            fundamental_display = "N/A"
        else:
            fundamental_display = (
                f"{result.fundamental_score}/100"
            )

        print(
            f"{position:>2}. "
            f"{result.ticker:<12} "
            f"Overall: {result.score:>3}/100  "
            f"Technical: {result.technical_score:>3}/100  "
            f"Fundamental: {fundamental_display:<7}  "
            f"News: {result.news_score:>3}/100  "
            f"{result.signal:<5}  "
            f"Confidence: {result.confidence:>3}%  "
            f"{stars}"
        )

    top = ranked[0]

    print()
    print("=" * 110)
    print("TOP RADAR CANDIDATE")
    print("=" * 110)
    print(f"Ticker: {top.ticker}")
    print(f"Final Recommendation: {top.signal}")
    print(f"Overall AI Score: {top.score}/100")
    print(
        f"Technical Score: "
        f"{top.technical_score}/100"
    )

    if top.fundamental_score >= 0:
        print(
            f"Fundamental Score: "
            f"{top.fundamental_score}/100"
        )
    else:
        print("Fundamental Score: Not applicable")

    print(f"News Score: {top.news_score}/100")
    print(f"News Sentiment: {top.news_sentiment}")
    print(f"Confidence: {top.confidence}%")
    print("⭐" * top.stars)


def create_fallback_news():
    """Return a neutral news result after a news error."""

    return SimpleNamespace(
        score=50,
        sentiment="NEUTRAL",
        stars=2,
        headlines=[],
        positive_matches=0,
        negative_matches=0,
    )


def main() -> None:
    """Download, analyze, score, rank, and report the universe."""

    period = "1y"

    universe = load_market_universe()
    watchlist = universe.all_symbols
    total = len(watchlist)

    results: list[RadarResult] = []

    print("=" * 80)
    print("AI STOCK RADAR")
    print("=" * 80)

    print_universe_summary(universe)

    print(
        f"Analyzing {universe.stock_count} stocks and "
        f"{universe.crypto_count} crypto assets "
        f"using {period} of daily data..."
    )
    print()

    for index, ticker in enumerate(
        watchlist,
        start=1,
    ):
        print()
        print("=" * 80)
        print(f"[{index}/{total}] PROCESSING {ticker}")
        print("=" * 80)

        try:
            # -----------------------------------------------------
            # 1. Download market data
            # -----------------------------------------------------
            data = download_stock_data(
                ticker,
                period=period,
            )

            raw_data_path = save_stock_data(
                data,
                ticker,
            )

            print(
                f"Market data saved to: {raw_data_path}"
            )

            # -----------------------------------------------------
            # 2. Technical analysis
            # -----------------------------------------------------
            data = add_technical_indicators(data)

            print()
            print("Technical Indicators")
            print_latest_summary(
                ticker,
                data,
            )

            technical_signal = generate_signal(data)
            technical_score = calculate_radar_score(data)

            # -----------------------------------------------------
            # 3. Asset type
            # -----------------------------------------------------
            is_crypto = ticker.endswith("-USD")

            # -----------------------------------------------------
            # 4. Fundamental analysis
            # -----------------------------------------------------
            fundamental = None
            fundamental_score_value = 0

            if is_crypto:
                print()
                print("Fundamental Analysis")
                print(
                    "Not applicable for cryptocurrency."
                )
            else:
                print()
                print("Downloading fundamental data...")

                try:
                    fundamental = (
                        calculate_fundamental_score(
                            ticker
                        )
                    )

                    fundamental_score_value = (
                        fundamental.score
                    )

                except Exception as error:
                    print(
                        "Fundamental analysis unavailable: "
                        f"{error}"
                    )

            # -----------------------------------------------------
            # 5. News analysis
            # -----------------------------------------------------
            print()
            print("Downloading recent news...")

            try:
                news = calculate_news_score(ticker)

            except Exception as error:
                print(
                    f"News analysis unavailable: {error}"
                )

                news = create_fallback_news()

            # -----------------------------------------------------
            # 6. Final decision
            # -----------------------------------------------------
            decision = calculate_decision(
                technical_score=technical_score.score,
                fundamental_score=(
                    fundamental_score_value
                ),
                news_score=news.score,
                is_crypto=is_crypto,
            )

            # -----------------------------------------------------
            # 7. Store result
            # -----------------------------------------------------
            result = RadarResult(
                ticker=ticker,
                signal=decision.recommendation,
                confidence=decision.confidence,
                score=decision.overall_score,
                stars=decision.stars,
                technical_score=technical_score.score,
                fundamental_score=(
                    -1
                    if is_crypto
                    else fundamental_score_value
                ),
                news_score=news.score,
                technical_signal=(
                    technical_signal.action
                ),
                news_sentiment=news.sentiment,
            )

            results.append(result)

            # -----------------------------------------------------
            # 8. Print technical result
            # -----------------------------------------------------
            print()
            print("Technical Analysis Result")
            print(
                f"Technical Signal: "
                f"{technical_signal.action}"
            )
            print(
                f"Technical Confidence: "
                f"{technical_signal.confidence}%"
            )
            print(
                f"Technical Score: "
                f"{technical_score.score}/100"
            )
            print("⭐" * technical_score.stars)

            print()
            print("Technical Score Reasons")

            for reason in technical_score.reasons:
                print(f"✓ {reason}")

            print()
            print("Technical Signal Reasons")

            for reason in technical_signal.reasons:
                print(f"✓ {reason}")

            # -----------------------------------------------------
            # 9. Print fundamental result
            # -----------------------------------------------------
            if fundamental is not None:
                print()
                print("Fundamental Analysis Result")
                print(
                    f"Fundamental Score: "
                    f"{fundamental.score}/100"
                )
                print("⭐" * fundamental.stars)

                print()
                print("Fundamental Reasons")

                for reason in fundamental.reasons:
                    print(f"✓ {reason}")

                print_fundamental_metrics(
                    fundamental
                )

            # -----------------------------------------------------
            # 10. Print news result
            # -----------------------------------------------------
            print_news_summary(news)

            # -----------------------------------------------------
            # 11. Print final decision
            # -----------------------------------------------------
            print()
            print("=" * 80)
            print("FINAL DECISION")
            print("=" * 80)
            print(f"Ticker: {ticker}")
            print(
                f"Recommendation: "
                f"{decision.recommendation}"
            )
            print(
                f"Overall AI Score: "
                f"{decision.overall_score}/100"
            )
            print(
                f"Decision Confidence: "
                f"{decision.confidence}%"
            )
            print("⭐" * decision.stars)

            print()
            print("Decision Reasons")

            for reason in decision.reasons:
                print(f"✓ {reason}")

        except Exception as error:
            print(
                f"Error processing {ticker}: {error}"
            )

        print()
        print("-" * 80)

    # ---------------------------------------------------------
    # 12. Final ranking
    # ---------------------------------------------------------
    print_ranking(results)

    # ---------------------------------------------------------
    # 13. Save reports
    # ---------------------------------------------------------
    if results:
        latest_path = save_ranking_report(
            results
        )

        history_path = append_history_report(
            results
        )

        print()
        print("=" * 80)
        print("REPORT FILES")
        print("=" * 80)
        print(
            f"Latest CSV saved to: {latest_path}"
        )
        print(
            f"History CSV updated at: "
            f"{history_path}"
        )
    else:
        print()
        print(
            "No successful results were available "
            "for reporting."
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()