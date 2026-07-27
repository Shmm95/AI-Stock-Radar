"""Radar score calculation."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class RadarScore:
    score: int
    stars: int
    reasons: list[str]


def calculate_radar_score(data: pd.DataFrame) -> RadarScore:
    valid = data.dropna(
        subset=["Close", "EMA20", "EMA50", "RSI14", "MACD"]
    )

    if valid.empty:
        return RadarScore(0, 0, ["No valid data"])

    latest = valid.iloc[-1]

    score = 0
    reasons = []

    # EMA Trend
    if latest["EMA20"] > latest["EMA50"]:
        score += 25
        reasons.append("EMA20 above EMA50")
    else:
        reasons.append("EMA20 below EMA50")

    # Price vs EMA20
    if latest["Close"] > latest["EMA20"]:
        score += 25
        reasons.append("Price above EMA20")
    else:
        reasons.append("Price below EMA20")

    # RSI
    if 45 <= latest["RSI14"] <= 70:
        score += 25
        reasons.append("Healthy RSI")
    elif latest["RSI14"] < 30:
        score += 20
        reasons.append("Oversold RSI")
    else:
        reasons.append("Weak RSI")

    # MACD
    if latest["MACD"] > 0:
        score += 25
        reasons.append("Positive MACD")
    else:
        reasons.append("Negative MACD")

    stars = min(score // 20, 5)

    return RadarScore(
        score=score,
        stars=stars,
        reasons=reasons,
    )