"""Stage 2 deep analysis for AI-Stock-Radar."""

from types import SimpleNamespace

from src.decision.decision_engine import calculate_decision
from src.fundamental.fundamental_score import (
    calculate_fundamental_score,
)
from src.news.news_sentiment import calculate_news_score
from src.pipeline.pipeline_models import (
    DeepResult,
    FastScanResult,
)


def _create_neutral_news():
    """Return a neutral fallback result after a news error."""

    return SimpleNamespace(
        score=50,
        sentiment="NEUTRAL",
        headlines=[],
        positive_matches=0,
        negative_matches=0,
        stars=2,
    )


def _create_missing_fundamental():
    """Return a fallback result after a fundamental error."""

    return SimpleNamespace(
        score=0,
        stars=0,
        reasons=["Fundamental data unavailable"],
        metrics={},
    )


def analyze_candidate(
    candidate: FastScanResult,
) -> DeepResult:
    """Run fundamental, news, and decision analysis for one asset."""

    ticker = candidate.ticker
    is_crypto = candidate.is_crypto

    print()
    print(f"Deep analyzing: {ticker}")

    # ---------------------------------------------------------
    # Fundamental analysis
    # ---------------------------------------------------------
    if is_crypto:
        fundamental_score = -1
        fundamental_reasons = [
            "Fundamental analysis is not applicable to cryptocurrency."
        ]

        print("  Fundamental: not applicable")

    else:
        try:
            fundamental = calculate_fundamental_score(ticker)

        except Exception as error:
            print(
                f"  Fundamental analysis failed: {error}"
            )

            fundamental = _create_missing_fundamental()

        fundamental_score = int(fundamental.score)
        fundamental_reasons = list(fundamental.reasons)

        print(
            f"  Fundamental score: "
            f"{fundamental_score}/100"
        )

    # ---------------------------------------------------------
    # News analysis
    # ---------------------------------------------------------
    try:
        news = calculate_news_score(ticker)

    except Exception as error:
        print(f"  News analysis failed: {error}")
        news = _create_neutral_news()

    news_score = int(news.score)
    news_sentiment = str(news.sentiment)
    news_headlines = list(news.headlines)

    print(f"  News score: {news_score}/100")
    print(f"  News sentiment: {news_sentiment}")

    # ---------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------
    decision = calculate_decision(
        technical_score=candidate.technical_score,
        fundamental_score=(
            0 if is_crypto else fundamental_score
        ),
        news_score=news_score,
        is_crypto=is_crypto,
    )

    result = DeepResult(
        ticker=ticker,
        asset_type=candidate.asset_type,
        overall_score=int(decision.overall_score),
        technical_score=int(candidate.technical_score),
        fundamental_score=int(fundamental_score),
        news_score=news_score,
        signal=str(decision.recommendation),
        confidence=int(decision.confidence),
        stars=int(decision.stars),
        technical_signal=str(candidate.technical_signal),
        news_sentiment=news_sentiment,
        technical_reasons=list(candidate.reasons),
        fundamental_reasons=fundamental_reasons,
        news_headlines=news_headlines,
        decision_reasons=list(decision.reasons),
    )

    print(
        f"  Overall: {result.overall_score}/100 | "
        f"Signal: {result.signal} | "
        f"Confidence: {result.confidence}%"
    )

    return result


def analyze_candidate_group(
    candidates: list[FastScanResult],
    group_name: str,
) -> tuple[list[DeepResult], list[str]]:
    """Deep-analyze one candidate group."""

    total = len(candidates)

    results: list[DeepResult] = []
    failed: list[str] = []

    print()
    print("=" * 80)
    print(f"STAGE 2 — DEEP ANALYSIS: {group_name}")
    print("=" * 80)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print()
        print(
            f"[{index}/{total}] "
            f"{candidate.ticker}"
        )

        try:
            result = analyze_candidate(candidate)

        except Exception as error:
            failed.append(candidate.ticker)

            print(
                f"  Deep analysis failed for "
                f"{candidate.ticker}: {error}"
            )

            continue

        results.append(result)

    results.sort(
        key=lambda item: (
            item.overall_score,
            item.confidence,
            item.technical_score,
        ),
        reverse=True,
    )

    return results, failed


def run_deep_analysis(
    stock_candidates: list[FastScanResult],
    crypto_candidates: list[FastScanResult],
) -> tuple[
    list[DeepResult],
    list[DeepResult],
    list[str],
]:
    """Run Stage 2 for stock and crypto finalists."""

    stock_results, failed_stocks = analyze_candidate_group(
        candidates=stock_candidates,
        group_name="STOCKS",
    )

    crypto_results, failed_crypto = analyze_candidate_group(
        candidates=crypto_candidates,
        group_name="CRYPTO",
    )

    failed_symbols = failed_stocks + failed_crypto

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print(f"Deep stock results: {len(stock_results)}")
    print(f"Deep crypto results: {len(crypto_results)}")
    print(f"Failed symbols: {len(failed_symbols)}")
    print("=" * 80)

    return (
        stock_results,
        crypto_results,
        failed_symbols,
    )


def print_deep_ranking(
    results: list[DeepResult],
    title: str,
    limit: int = 10,
) -> None:
    """Print a compact Stage 2 ranking."""

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if not results:
        print("No deep-analysis results.")
        return

    for rank, result in enumerate(
        results[:limit],
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
            f"{rank:>2}. "
            f"{result.ticker:<14} "
            f"Overall: {result.overall_score:>3}/100  "
            f"Technical: {result.technical_score:>3}/100  "
            f"Fundamental: {fundamental_display:<7}  "
            f"News: {result.news_score:>3}/100  "
            f"Signal: {result.signal:<5}  "
            f"Confidence: {result.confidence:>3}%  "
            f"{stars}"
        )
    