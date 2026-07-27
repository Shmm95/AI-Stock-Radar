"""Fundamental scoring utilities."""

from dataclasses import dataclass

import yfinance as yf


@dataclass
class FundamentalScore:
    """Fundamental analysis result for one ticker."""

    score: int
    stars: int
    reasons: list[str]
    metrics: dict[str, float | None]


def _safe_number(value) -> float | None:
    """Convert a value to float when possible."""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_fundamental_score(ticker: str) -> FundamentalScore:
    """Calculate a simple 0-100 fundamental score."""

    stock = yf.Ticker(ticker)
    info = stock.get_info()

    forward_pe = _safe_number(info.get("forwardPE"))
    trailing_pe = _safe_number(info.get("trailingPE"))
    peg_ratio = _safe_number(info.get("pegRatio"))
    price_to_book = _safe_number(info.get("priceToBook"))
    dividend_yield = _safe_number(info.get("dividendYield"))
    return_on_equity = _safe_number(info.get("returnOnEquity"))
    profit_margin = _safe_number(info.get("profitMargins"))

    metrics = {
        "forward_pe": forward_pe,
        "trailing_pe": trailing_pe,
        "peg_ratio": peg_ratio,
        "price_to_book": price_to_book,
        "dividend_yield": dividend_yield,
        "return_on_equity": return_on_equity,
        "profit_margin": profit_margin,
    }

    score = 0
    reasons: list[str] = []

    # Forward P/E — maximum 20 points
    if forward_pe is None:
        reasons.append("Forward P/E unavailable")
    elif 0 < forward_pe <= 15:
        score += 20
        reasons.append("Attractive forward P/E")
    elif forward_pe <= 25:
        score += 12
        reasons.append("Reasonable forward P/E")
    else:
        score += 4
        reasons.append("High forward P/E")

    # PEG — maximum 15 points
    if peg_ratio is None:
        reasons.append("PEG unavailable")
    elif 0 < peg_ratio <= 1:
        score += 15
        reasons.append("Attractive PEG ratio")
    elif peg_ratio <= 2:
        score += 8
        reasons.append("Moderate PEG ratio")
    else:
        reasons.append("High PEG ratio")

    # Price / Book — maximum 10 points
    if price_to_book is None:
        reasons.append("Price to book unavailable")
    elif 0 < price_to_book <= 3:
        score += 10
        reasons.append("Reasonable price to book")
    elif price_to_book <= 6:
        score += 5
        reasons.append("Elevated price to book")
    else:
        reasons.append("High price to book")

    # Return on equity — maximum 20 points
    if return_on_equity is None:
        reasons.append("ROE unavailable")
    elif return_on_equity >= 0.20:
        score += 20
        reasons.append("Strong return on equity")
    elif return_on_equity >= 0.10:
        score += 12
        reasons.append("Healthy return on equity")
    elif return_on_equity > 0:
        score += 5
        reasons.append("Positive but weak return on equity")
    else:
        reasons.append("Negative return on equity")

    # Profit margin — maximum 20 points
    if profit_margin is None:
        reasons.append("Profit margin unavailable")
    elif profit_margin >= 0.20:
        score += 20
        reasons.append("Strong profit margin")
    elif profit_margin >= 0.10:
        score += 12
        reasons.append("Healthy profit margin")
    elif profit_margin > 0:
        score += 5
        reasons.append("Positive but low profit margin")
    else:
        reasons.append("Negative profit margin")

    # Dividend yield — maximum 10 points
    if dividend_yield is None:
        reasons.append("Dividend yield unavailable")
    elif dividend_yield >= 0.03:
        score += 10
        reasons.append("Strong dividend yield")
    elif dividend_yield > 0:
        score += 5
        reasons.append("Pays a dividend")
    else:
        reasons.append("No dividend")

    # Trailing P/E — maximum 5 bonus points
    if trailing_pe is not None and 0 < trailing_pe <= 20:
        score += 5
        reasons.append("Reasonable trailing P/E")

    score = min(score, 100)
    stars = min(score // 20, 5)

    return FundamentalScore(
        score=score,
        stars=stars,
        reasons=reasons,
        metrics=metrics,
    )