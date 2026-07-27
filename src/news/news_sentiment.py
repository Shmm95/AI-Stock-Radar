"""Simple news sentiment analysis using Yahoo Finance headlines."""

from dataclasses import dataclass
from typing import Any

import yfinance as yf


POSITIVE_KEYWORDS = {
    "approval",
    "beat",
    "bullish",
    "expansion",
    "growth",
    "partnership",
    "profit",
    "record",
    "strong",
    "surge",
    "upgrade",
}

NEGATIVE_KEYWORDS = {
    "bankruptcy",
    "bearish",
    "decline",
    "downgrade",
    "fraud",
    "investigation",
    "lawsuit",
    "layoff",
    "loss",
    "miss",
    "recall",
}


@dataclass
class NewsScore:
    """News sentiment result for one ticker."""

    score: int
    sentiment: str
    stars: int
    headlines: list[str]
    positive_matches: int
    negative_matches: int


def _extract_title(news_item: dict[str, Any]) -> str | None:
    """Extract a headline from different yfinance news response formats."""

    title = news_item.get("title")

    if isinstance(title, str) and title.strip():
        return title.strip()

    content = news_item.get("content")

    if isinstance(content, dict):
        content_title = content.get("title")

        if isinstance(content_title, str) and content_title.strip():
            return content_title.strip()

    return None


def _get_headlines(ticker: str, limit: int = 10) -> list[str]:
    """Download recent headlines for a ticker."""

    stock = yf.Ticker(ticker)
    news_items = stock.news or []

    headlines: list[str] = []

    for item in news_items:
        if not isinstance(item, dict):
            continue

        title = _extract_title(item)

        if title:
            headlines.append(title)

        if len(headlines) >= limit:
            break

    return headlines


def calculate_news_score(ticker: str) -> NewsScore:
    """Calculate a keyword-based news score between 0 and 100."""

    try:
        headlines = _get_headlines(ticker, limit=10)
    except Exception:
        headlines = []

    if not headlines:
        return NewsScore(
            score=50,
            sentiment="NEUTRAL",
            stars=2,
            headlines=[],
            positive_matches=0,
            negative_matches=0,
        )

    score = 50
    positive_matches = 0
    negative_matches = 0

    for headline in headlines:
        normalized = headline.lower()

        positive_found = {
            keyword
            for keyword in POSITIVE_KEYWORDS
            if keyword in normalized
        }

        negative_found = {
            keyword
            for keyword in NEGATIVE_KEYWORDS
            if keyword in normalized
        }

        positive_matches += len(positive_found)
        negative_matches += len(negative_found)

        score += len(positive_found) * 8
        score -= len(negative_found) * 8

    score = max(0, min(score, 100))

    if score >= 70:
        sentiment = "POSITIVE"
    elif score < 40:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    stars = min(score // 20, 5)

    return NewsScore(
        score=score,
        sentiment=sentiment,
        stars=stars,
        headlines=headlines[:5],
        positive_matches=positive_matches,
        negative_matches=negative_matches,
    )