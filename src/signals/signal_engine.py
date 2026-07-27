"""Trading signal engine."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    """Result produced by the signal engine."""

    action: str
    confidence: int
    reasons: list[str]


def generate_signal(data: pd.DataFrame) -> Signal:
    """Generate a simple BUY, HOLD, or SELL signal."""

    valid = data.dropna(subset=["EMA20", "EMA50", "RSI14", "MACD"])

    if valid.empty:
        return Signal(
            action="HOLD",
            confidence=0,
            reasons=["Not enough complete indicator data"],
        )

    latest = valid.iloc[-1]

    score = 0
    reasons: list[str] = []

    if latest["EMA20"] > latest["EMA50"]:
        score += 1
        reasons.append("EMA20 is above EMA50")
    else:
        score -= 1
        reasons.append("EMA20 is below EMA50")

    if latest["RSI14"] < 30:
        score += 1
        reasons.append("RSI is oversold")
    elif latest["RSI14"] > 70:
        score -= 1
        reasons.append("RSI is overbought")
    else:
        reasons.append("RSI is neutral")

    if latest["MACD"] > 0:
        score += 1
        reasons.append("MACD is positive")
    else:
        score -= 1
        reasons.append("MACD is negative")

    if score >= 2:
        action = "BUY"
    elif score <= -2:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = min(abs(score) * 30 + 20, 100)

    return Signal(
        action=action,
        confidence=confidence,
        reasons=reasons,
    )
