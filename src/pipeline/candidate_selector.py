"""Candidate selection for the AI-Stock-Radar pipeline."""

from src.pipeline.pipeline_models import (
    CandidateSelection,
    DeepResult,
)


def _sort_results(
    results: list[DeepResult],
) -> list[DeepResult]:
    """Sort candidates from strongest to weakest."""

    return sorted(
        results,
        key=lambda item: (
            item.overall_score,
            item.confidence,
            item.technical_score,
            item.news_score,
        ),
        reverse=True,
    )


def select_final_candidates(
    stock_results: list[DeepResult],
    crypto_results: list[DeepResult],
    stock_limit: int = 3,
    crypto_limit: int = 3,
) -> CandidateSelection:
    """Select final stock and crypto candidates before AI analysis."""

    if stock_limit < 1:
        raise ValueError(
            "stock_limit must be at least 1."
        )

    if crypto_limit < 1:
        raise ValueError(
            "crypto_limit must be at least 1."
        )

    ranked_stocks = _sort_results(
        stock_results
    )

    ranked_crypto = _sort_results(
        crypto_results
    )

    top_stocks = ranked_stocks[:stock_limit]
    top_crypto = ranked_crypto[:crypto_limit]

    selected_crypto = (
        top_crypto[0]
        if top_crypto
        else None
    )

    return CandidateSelection(
        top_stocks=top_stocks,
        top_crypto=top_crypto,
        selected_crypto=selected_crypto,
    )


def print_candidate_selection(
    selection: CandidateSelection,
) -> None:
    """Print final Python-selected candidates."""

    print()
    print("=" * 90)
    print("STAGE 3 — FINAL CANDIDATE SELECTION")
    print("=" * 90)

    print()
    print("TOP STOCKS")

    if not selection.top_stocks:
        print("No stock candidates.")
    else:
        for rank, result in enumerate(
            selection.top_stocks,
            start=1,
        ):
            stars = "⭐" * result.stars

            print(
                f"{rank:>2}. "
                f"{result.ticker:<14} "
                f"Overall: {result.overall_score:>3}/100  "
                f"Technical: {result.technical_score:>3}/100  "
                f"Fundamental: {result.fundamental_score:>3}/100  "
                f"News: {result.news_score:>3}/100  "
                f"Signal: {result.signal:<5}  "
                f"Confidence: {result.confidence:>3}%  "
                f"{stars}"
            )

    print()
    print("TOP CRYPTO")

    if not selection.top_crypto:
        print("No crypto candidates.")
    else:
        for rank, result in enumerate(
            selection.top_crypto,
            start=1,
        ):
            stars = "⭐" * result.stars

            print(
                f"{rank:>2}. "
                f"{result.ticker:<14} "
                f"Overall: {result.overall_score:>3}/100  "
                f"Technical: {result.technical_score:>3}/100  "
                f"News: {result.news_score:>3}/100  "
                f"Signal: {result.signal:<5}  "
                f"Confidence: {result.confidence:>3}%  "
                f"{stars}"
            )

    print()
    print("SELECTED CRYPTO FOR AI")

    if selection.selected_crypto is None:
        print("No crypto selected.")
    else:
        selected = selection.selected_crypto

        print(
            f"{selected.ticker} | "
            f"Overall: {selected.overall_score}/100 | "
            f"Technical: {selected.technical_score}/100 | "
            f"News: {selected.news_score}/100 | "
            f"Signal: {selected.signal}"
        )

    print()
    print("AI CANDIDATES")

    if not selection.ai_candidates:
        print("No AI candidates.")
    else:
        for rank, result in enumerate(
            selection.ai_candidates,
            start=1,
        ):
            print(
                f"{rank:>2}. "
                f"{result.ticker:<14} "
                f"{result.asset_type:<6} "
                f"Overall: {result.overall_score}/100"
            )

    print("=" * 90)