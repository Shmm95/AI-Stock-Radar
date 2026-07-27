"""CSV reporting utilities for AI-Stock-Radar."""

from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd


class RadarResultLike(Protocol):
    """Minimum interface required from a radar result."""

    ticker: str
    signal: str
    confidence: int
    score: int
    stars: int


REPORT_COLUMNS = [
    "scan_date",
    "scan_time",
    "rank",
    "ticker",
    "overall_score",
    "technical_score",
    "fundamental_score",
    "news_score",
    "signal",
    "confidence",
    "stars",
    "technical_signal",
    "news_sentiment",
]


def _build_report_dataframe(
    results: list[RadarResultLike],
) -> pd.DataFrame:
    """Create a detailed ranked dataframe from radar results."""

    ranked = sorted(
        results,
        key=lambda item: (
            item.score,
            item.confidence,
            getattr(item, "technical_score", 0),
        ),
        reverse=True,
    )

    scan_time = datetime.now()
    rows: list[dict[str, object]] = []

    for rank, result in enumerate(ranked, start=1):
        rows.append(
            {
                "scan_date": scan_time.strftime("%Y-%m-%d"),
                "scan_time": scan_time.strftime("%H:%M:%S"),
                "rank": rank,
                "ticker": result.ticker,
                "overall_score": result.score,
                "technical_score": getattr(
                    result,
                    "technical_score",
                    0,
                ),
                "fundamental_score": getattr(
                    result,
                    "fundamental_score",
                    -1,
                ),
                "news_score": getattr(
                    result,
                    "news_score",
                    50,
                ),
                "signal": result.signal,
                "confidence": result.confidence,
                "stars": result.stars,
                "technical_signal": getattr(
                    result,
                    "technical_signal",
                    "UNKNOWN",
                ),
                "news_sentiment": getattr(
                    result,
                    "news_sentiment",
                    "NEUTRAL",
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=REPORT_COLUMNS,
    )


def _get_report_directory() -> Path:
    """Create and return the project report directory."""

    project_root = Path(__file__).resolve().parent.parent.parent
    report_dir = project_root / "data" / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return report_dir


def _archive_incompatible_history(
    history_path: Path,
) -> None:
    """Archive an old history file if its columns are incompatible."""

    if not history_path.exists():
        return

    try:
        existing_columns = list(
            pd.read_csv(
                history_path,
                nrows=0,
            ).columns
        )
    except (OSError, pd.errors.ParserError):
        existing_columns = []

    if existing_columns == REPORT_COLUMNS:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = (
        history_path.parent
        / f"radar_history_legacy_{timestamp}.csv"
    )

    history_path.rename(archive_path)

    print(
        "Previous history file archived because its columns changed: "
        f"{archive_path}"
    )


def save_ranking_report(
    results: list[RadarResultLike],
) -> Path:
    """Save the newest detailed ranking as radar_latest.csv."""

    report = _build_report_dataframe(results)
    report_dir = _get_report_directory()

    output_path = report_dir / "radar_latest.csv"
    report.to_csv(
        output_path,
        index=False,
    )

    return output_path


def append_history_report(
    results: list[RadarResultLike],
) -> Path:
    """Append detailed results to radar_history.csv."""

    report = _build_report_dataframe(results)
    report_dir = _get_report_directory()

    output_path = report_dir / "radar_history.csv"

    _archive_incompatible_history(output_path)

    if output_path.exists():
        report.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        report.to_csv(
            output_path,
            index=False,
        )

    return output_path