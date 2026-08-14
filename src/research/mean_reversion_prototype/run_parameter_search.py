"""Phase 2: run the full in-sample/out-of-sample parameter grid search
and save results. Read-only research; never touches data/live/, the
frozen engine, or CONTROLLED_TICKERS/LIVE_CONTROLLED_TICKERS. Run by
hand:

    .venv/bin/python -u -m src.research.mean_reversion_prototype.run_parameter_search
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.research.mean_reversion_prototype.data_loader import CANDIDATE_SYMBOLS, assess_quality, fetch_hourly_bars
from src.research.mean_reversion_prototype.parameter_search import full_grid, run_grid_search

OUTPUT_DIRECTORY = Path("data/research")


def main() -> None:
    raw_frames = {}
    for symbol in CANDIDATE_SYMBOLS:
        print(f"Fetching {symbol}...", flush=True)
        raw = fetch_hourly_bars(symbol)
        clean, report = assess_quality(symbol, raw)
        print(f"  {symbol}: rows={report.rows} missing={report.missing_percent}% invalid_ohlc={report.invalid_ohlc_rows}", flush=True)
        raw_frames[symbol] = clean

    grid = full_grid()
    print(f"\nGrid size: {len(grid)} combinations x {len(raw_frames)} tickers", flush=True)

    results, split_date = run_grid_search(raw_frames)
    print(f"Split date (in-sample < this <= out-of-sample): {split_date}", flush=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"mean_reversion_parameter_search_v1_{stamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "RESEARCH_CANDIDATE_NOT_OFFICIAL",
                "generated_at": datetime.now(UTC).isoformat(),
                "split_date": str(split_date),
                "grid_size": len(grid),
                "tickers": list(raw_frames.keys()),
                "results": results,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
