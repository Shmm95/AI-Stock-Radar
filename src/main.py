"""Entry point for AI-Stock-Radar."""

from dataclasses import dataclass

from config.watchlist import WATCHLIST
from src.analysis.technical_indicators import add_technical_indicators
from src.data.download_stock import download_stock_data, save_stock_data
from src.reporting.csv_report import save_ranking_report
from src.scoring.radar_score import calculate_radar_score
from src.signals.signal_engine import generate_signal


@dataclass
class RadarResult:
    ticker: str
    signal: str
    confidence: int
    score: int
    stars: int


def print_latest_summary(ticker: str, data) -> None:
    valid = data.dropna(
        subset=["Close", "EMA20", "EMA50", "RSI14", "MACD"]
    )

    if valid.empty:
        print(f"No valid data for {ticker}")
        return

    latest = valid.iloc[-1]

    print(f"Ticker: {ticker}")
    print(f"Close: {float(latest['Close']):.2f}")
    print(f"EMA20: {float(latest['EMA20']):.2f}")
    print(f"EMA50: {float(latest['EMA50']):.2f}")
    print(f"RSI14: {float(latest['RSI14']):.2f}")
    print(f"MACD: {float(latest['MACD']):.4f}")


def print_ranking(results: list[RadarResult]) -> None:

    ranked = sorted(
        results,
        key=lambda x: (x.score, x.confidence),
        reverse=True,
    )

    print()
    print("=" * 60)
    print("AI STOCK RADAR RANKING")
    print("=" * 60)

    for i, item in enumerate(ranked, start=1):

        print(
            f"{i}. "
            f"{item.ticker:<10} "
            f"Score: {item.score}/100 "
            f"{item.signal:<4} "
            f"Confidence: {item.confidence}% "
            f"{'⭐' * item.stars}"
        )

    if ranked:

        best = ranked[0]

        print()
        print("=" * 60)
        print("TOP RADAR CANDIDATE")
        print("=" * 60)

        print(f"Ticker: {best.ticker}")
        print(f"Signal: {best.signal}")
        print(f"Score: {best.score}/100")
        print(f"Confidence: {best.confidence}%")
        print("⭐" * best.stars)


def main():

    period = "1y"

    total = len(WATCHLIST)

    results = []

    print(f"Downloading {total} symbols...\n")

    for index, ticker in enumerate(WATCHLIST, start=1):

        print(f"[{index}/{total}] Downloading {ticker}...")

        try:

            data = download_stock_data(
                ticker,
                period=period,
            )

            output_path = save_stock_data(
                data,
                ticker,
            )

            print(f"Saved to {output_path}")

            data = add_technical_indicators(data)

            print_latest_summary(
                ticker,
                data,
            )

            signal = generate_signal(data)

            radar = calculate_radar_score(data)

            results.append(
                RadarResult(
                    ticker=ticker,
                    signal=signal.action,
                    confidence=signal.confidence,
                    score=radar.score,
                    stars=radar.stars,
                )
            )

            print()
            print(f"Signal: {signal.action}")
            print(f"Confidence: {signal.confidence}%")

            print()
            print(f"Radar Score: {radar.score}/100")
            print("⭐" * radar.stars)

            print()
            print("Radar Reasons")

            for reason in radar.reasons:
                print(f"✓ {reason}")

            print()
            print("Signal Reasons")

            for reason in signal.reasons:
                print(f"✓ {reason}")

        except Exception as e:

            print(f"Error processing {ticker}: {e}")

        print("-" * 60)

    print_ranking(results)

    report_path = save_ranking_report(results)

    print()
    print(f"CSV report saved to: {report_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()