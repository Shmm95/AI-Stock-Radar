"""CSV reporting utilities."""

from pathlib import Path
from typing import Protocol

import pandas as pd


class RadarResultLike(Protocol):
    ticker: str
    signal: str
    confidence: int
    score: int
    stars: int


def save_ranking_report(results: list[RadarResultLike]) -> Path:
    """Save radar results ordered from strongest to weakest."""

    ranked = sorted(
        results,
        key=lambda item: (item.score, item.confidence),
        reverse=True,
    )

    rows = []

    for rank, result in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "ticker": result.ticker,
                "score": result.score,
                "signal": result.signal,
                "confidence": result.confidence,
                "stars": result.stars,
            }
        )

    project_root = Path(__file__).resolve().parent.parent.parent
    report_dir = project_root / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = report_dir / "radar_latest.csv"

    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)

    return output_path