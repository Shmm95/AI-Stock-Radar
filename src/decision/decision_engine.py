"""Final decision engine for AI-Stock-Radar."""

from dataclasses import dataclass


@dataclass
class DecisionResult:
    """Combined investment decision result."""

    overall_score: int
    recommendation: str
    confidence: int
    stars: int
    reasons: list[str]


def calculate_decision(
    technical_score: int,
    fundamental_score: int,
    news_score: int,
    is_crypto: bool = False,
) -> DecisionResult:
    """Combine technical, fundamental, and news scores."""

    reasons: list[str] = []

    if is_crypto:
        overall_score = round(
            technical_score * 0.75
            + news_score * 0.25
        )

        reasons.append("Crypto weighting: 75% technical, 25% news")
    else:
        overall_score = round(
            technical_score * 0.50
            + fundamental_score * 0.30
            + news_score * 0.20
        )

        reasons.append(
            "Stock weighting: 50% technical, 30% fundamental, 20% news"
        )

    overall_score = max(0, min(overall_score, 100))

    if overall_score >= 75:
        recommendation = "BUY"
    elif overall_score >= 55:
        recommendation = "WATCH"
    elif overall_score >= 40:
        recommendation = "HOLD"
    else:
        recommendation = "SELL"

    if technical_score >= 75:
        reasons.append("Strong technical momentum")
    elif technical_score >= 50:
        reasons.append("Moderate technical setup")
    else:
        reasons.append("Weak technical setup")

    if not is_crypto:
        if fundamental_score >= 70:
            reasons.append("Strong fundamental quality")
        elif fundamental_score >= 45:
            reasons.append("Average fundamental quality")
        else:
            reasons.append("Weak fundamental quality")

    if news_score >= 70:
        reasons.append("Positive news sentiment")
    elif news_score >= 40:
        reasons.append("Neutral news sentiment")
    else:
        reasons.append("Negative news sentiment")

    confidence = abs(overall_score - 50) * 2
    confidence = max(20, min(confidence, 100))

    stars = min(overall_score // 20, 5)

    return DecisionResult(
        overall_score=overall_score,
        recommendation=recommendation,
        confidence=confidence,
        stars=stars,
        reasons=reasons,
    )