"""Rolling portfolio walk-forward analysis for position-count limits.

Candidate limits are ranked on training data only. The winner is then tested on
an immediately following unseen window and compared with a fixed baseline.
This module is historical research only and cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    DEFAULT_TICKERS,
    normalize_position_limits,
    run_position_ablation,
)

DEFAULT_PERIOD = "5y"
DEFAULT_REGIME_PERIOD = "10y"
DEFAULT_POSITION_LIMITS = (3, 4, 5)
DEFAULT_BASELINE_POSITION_LIMIT = 4
DEFAULT_TRAIN_MONTHS = 24
DEFAULT_TEST_MONTHS = 6
DEFAULT_STEP_MONTHS = 6
DEFAULT_OUTPUT_DIRECTORY = (
    Path("data") / "backtests" / "portfolio" / "walk_forward"
)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_id: str
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end_exclusive": self.train_end_exclusive.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end_exclusive": self.test_end_exclusive.isoformat(),
        }


def _normalize_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    frame = frame.sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def _positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def build_walk_forward_windows(
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> tuple[WalkForwardWindow, ...]:
    """Create complete rolling windows on the common universe history."""

    train_months = _positive_int("train_months", train_months)
    test_months = _positive_int("test_months", test_months)
    step_months = _positive_int("step_months", step_months)
    if not data_by_ticker:
        raise ValueError("data_by_ticker cannot be empty.")

    normalized = {
        ticker.strip().upper(): _normalize_frame(data)
        for ticker, data in data_by_ticker.items()
    }
    empty = [ticker for ticker, data in normalized.items() if data.empty]
    if empty:
        raise ValueError(f"Empty market data: {', '.join(sorted(empty))}")

    common_start = max(data.index.min() for data in normalized.values())
    common_last = min(data.index.max() for data in normalized.values())
    common_end_exclusive = pd.Timestamp(common_last) + pd.Timedelta(days=1)

    windows: list[WalkForwardWindow] = []
    train_start = pd.Timestamp(common_start)
    sequence = 1
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > common_end_exclusive:
            break

        windows.append(
            WalkForwardWindow(
                window_id=f"W{sequence:02d}",
                train_start=train_start,
                train_end_exclusive=train_end,
                test_start=test_start,
                test_end_exclusive=test_end,
            )
        )
        train_start += pd.DateOffset(months=step_months)
        sequence += 1

    if not windows:
        raise ValueError("Data range is too short for one complete window.")
    return tuple(windows)


def slice_prepared_data(
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Slice all tickers to the same half-open interval."""

    start = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end_exclusive)
    if end_exclusive <= start:
        raise ValueError("end_exclusive must be after start.")

    output: dict[str, pd.DataFrame] = {}
    for raw_ticker, raw_data in data_by_ticker.items():
        ticker = raw_ticker.strip().upper()
        data = _normalize_frame(raw_data)
        sliced = data.loc[
            (data.index >= start) & (data.index < end_exclusive)
        ].copy()
        if len(sliced) < 2:
            raise ValueError(
                f"{ticker} has fewer than two bars in "
                f"{start.date()} -> {end_exclusive.date()}."
            )
        output[ticker] = sliced
    return output


def rank_training_candidates(
    summary: pd.DataFrame,
    *,
    baseline_position_limit: int = DEFAULT_BASELINE_POSITION_LIMIT,
) -> pd.DataFrame:
    """Rank candidates by deterministic, scale-independent rank aggregation."""

    required = {
        "maximum_open_positions",
        "return_drawdown_ratio",
        "excess_return_vs_matched_percent",
        "profit_factor",
        "total_return_percent",
        "maximum_drawdown_percent",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Training summary is missing: {', '.join(missing)}")
    if summary.empty:
        raise ValueError("Training summary cannot be empty.")

    ranked = summary.copy().reset_index(drop=True)
    rules = (
        ("return_drawdown_ratio", False),
        ("excess_return_vs_matched_percent", False),
        ("profit_factor", False),
        ("total_return_percent", False),
        ("maximum_drawdown_percent", True),
    )
    rank_columns: list[str] = []
    for column, ascending in rules:
        values = pd.to_numeric(ranked[column], errors="coerce")
        values = values.fillna(float("inf") if ascending else float("-inf"))
        rank_column = f"rank_{column}"
        ranked[rank_column] = values.rank(method="min", ascending=ascending)
        rank_columns.append(rank_column)

    ranked["selection_rank_sum"] = ranked[rank_columns].sum(axis=1)
    ranked["distance_to_baseline"] = (
        ranked["maximum_open_positions"].astype(int)
        - int(baseline_position_limit)
    ).abs()
    ranked = ranked.sort_values(
        by=[
            "selection_rank_sum",
            "return_drawdown_ratio",
            "excess_return_vs_matched_percent",
            "profit_factor",
            "maximum_drawdown_percent",
            "total_return_percent",
            "distance_to_baseline",
            "maximum_open_positions",
        ],
        ascending=[True, False, False, False, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "training_rank", range(1, len(ranked) + 1))
    ranked["selected_for_test"] = ranked["training_rank"].eq(1)
    return ranked


def compound_returns(
    returns_percent: Iterable[float],
    *,
    initial_cash: float,
) -> float:
    capital = float(initial_cash)
    for value in returns_percent:
        capital *= 1 + float(value) / 100
    return round(capital, 6)


def maximum_drawdown_percent(equity_values: Sequence[float]) -> float:
    peak = 0.0
    maximum = 0.0
    for raw_value in equity_values:
        value = float(raw_value)
        peak = max(peak, value)
        if peak > 0:
            maximum = max(maximum, (peak - value) / peak * 100)
    return round(maximum, 4)


def _cagr(
    initial: float,
    ending: float,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> float:
    years = max((end_exclusive - start).days, 0) / 365.25
    if years <= 0 or initial <= 0 or ending <= 0:
        return 0.0
    return round(((ending / initial) ** (1 / years) - 1) * 100, 4)


def _return_drawdown_ratio(total_return: float, drawdown: float) -> float:
    if drawdown <= 0:
        return float("inf") if total_return > 0 else 0.0
    return round(total_return / drawdown, 4)


def _row_for_limit(frame: pd.DataFrame, limit: int) -> dict[str, Any]:
    matches = frame.loc[frame["maximum_open_positions"] == int(limit)]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for position limit {limit}.")
    return matches.iloc[0].to_dict()


def _filtered_records(
    frame: pd.DataFrame,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.loc[
        frame["maximum_open_positions"] == int(limit)
    ].to_dict(orient="records")


def _append_scaled_equity(
    output: list[dict[str, Any]],
    *,
    equity: pd.DataFrame,
    window_id: str,
    model: str,
    position_limit: int,
    opening_capital: float,
    initial_cash: float,
) -> float:
    if equity.empty:
        return opening_capital

    curve = equity.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])
    curve = curve.sort_values("timestamp")
    scale = opening_capital / initial_cash
    for row in curve.itertuples(index=False):
        output.append(
            {
                "window_id": window_id,
                "model": model,
                "position_limit": position_limit,
                "timestamp": pd.Timestamp(row.timestamp),
                "total_equity": round(float(row.total_equity) * scale, 6),
            }
        )
    return round(float(curve.iloc[-1]["total_equity"]) * scale, 6)


def _aggregate_model(
    test_runs: pd.DataFrame,
    stitched_equity: pd.DataFrame,
    *,
    model: str,
    initial_cash: float,
    first_test_start: pd.Timestamp,
    last_test_end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    rows = test_runs.loc[test_runs["model"] == model].sort_values("window_id")
    curve = stitched_equity.loc[
        stitched_equity["model"] == model
    ].sort_values("timestamp")
    ending_equity = compound_returns(
        rows["total_return_percent"],
        initial_cash=initial_cash,
    )
    total_return = (ending_equity / initial_cash - 1) * 100
    drawdown = maximum_drawdown_percent(curve["total_equity"])
    profit_factors = pd.to_numeric(rows["profit_factor"], errors="coerce")
    profit_factors = profit_factors[
        profit_factors.apply(lambda value: isfinite(float(value)))
    ]
    matched_ending = compound_returns(
        rows["matched_benchmark_return_percent"],
        initial_cash=initial_cash,
    )

    return {
        "model": model,
        "window_count": int(len(rows)),
        "initial_cash": round(initial_cash, 2),
        "ending_equity": round(ending_equity, 2),
        "compounded_return_percent": round(total_return, 4),
        "cagr_percent": _cagr(
            initial_cash,
            ending_equity,
            first_test_start,
            last_test_end_exclusive,
        ),
        "maximum_drawdown_percent": drawdown,
        "return_drawdown_ratio": _return_drawdown_ratio(
            total_return,
            drawdown,
        ),
        "matched_benchmark_ending_equity": round(matched_ending, 2),
        "matched_benchmark_compounded_return_percent": round(
            (matched_ending / initial_cash - 1) * 100,
            4,
        ),
        "average_window_return_percent": round(
            float(rows["total_return_percent"].mean()), 4
        ),
        "median_window_return_percent": round(
            float(rows["total_return_percent"].median()), 4
        ),
        "best_window_return_percent": round(
            float(rows["total_return_percent"].max()), 4
        ),
        "worst_window_return_percent": round(
            float(rows["total_return_percent"].min()), 4
        ),
        "positive_window_count": int((rows["total_return_percent"] > 0).sum()),
        "matched_benchmark_beat_count": int(
            (rows["excess_return_vs_matched_percent"] > 0).sum()
        ),
        "total_trades": int(rows["completed_trades"].sum()),
        "average_profit_factor": round(
            float(profit_factors.mean()) if len(profit_factors) else 0.0,
            4,
        ),
        "average_exposure_percent": round(
            float(rows["average_exposure_percent"].mean()), 4
        ),
    }


def run_portfolio_walk_forward(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    position_limits: Iterable[int] = DEFAULT_POSITION_LIMITS,
    baseline_position_limit: int = DEFAULT_BASELINE_POSITION_LIMIT,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> dict[str, Any]:
    """Run training-only selection and unseen portfolio tests."""

    limits = normalize_position_limits(position_limits)
    baseline_position_limit = int(baseline_position_limit)
    if baseline_position_limit not in limits:
        raise ValueError("baseline_position_limit must be in position_limits.")
    base_config.validate()

    windows = build_walk_forward_windows(
        data_by_ticker,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    training_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    stitched_rows: list[dict[str, Any]] = []
    capitals = {
        "DYNAMIC_SELECTED": float(base_config.initial_cash),
        "FIXED_BASELINE": float(base_config.initial_cash),
    }

    for sequence, window in enumerate(windows, start=1):
        print()
        print("=" * 108)
        print(
            f"PORTFOLIO WALK-FORWARD [{sequence}/{len(windows)}] — "
            f"{window.window_id}"
        )
        print(
            f"TRAIN {window.train_start.date()} -> "
            f"{window.train_end_exclusive.date()} | "
            f"TEST {window.test_start.date()} -> "
            f"{window.test_end_exclusive.date()}"
        )
        print("=" * 108)

        train_data = slice_prepared_data(
            data_by_ticker,
            start=window.train_start,
            end_exclusive=window.train_end_exclusive,
        )
        train_bundle = run_position_ablation(
            data_by_ticker=train_data,
            base_config=base_config,
            position_limits=limits,
        )
        ranked = rank_training_candidates(
            train_bundle["summary"],
            baseline_position_limit=baseline_position_limit,
        )
        selected_limit = int(ranked.iloc[0]["maximum_open_positions"])
        for record in ranked.to_dict(orient="records"):
            training_rows.append({**window.to_dict(), **record})

        test_data = slice_prepared_data(
            data_by_ticker,
            start=window.test_start,
            end_exclusive=window.test_end_exclusive,
        )
        test_limits = tuple(sorted({selected_limit, baseline_position_limit}))
        test_bundle = run_position_ablation(
            data_by_ticker=test_data,
            base_config=base_config,
            position_limits=test_limits,
        )
        dynamic = _row_for_limit(test_bundle["summary"], selected_limit)
        fixed = _row_for_limit(test_bundle["summary"], baseline_position_limit)
        advantage = round(
            dynamic["total_return_percent"] - fixed["total_return_percent"],
            4,
        )

        model_rows = (
            ("DYNAMIC_SELECTED", selected_limit, dynamic),
            ("FIXED_BASELINE", baseline_position_limit, fixed),
        )
        for model, limit, summary in model_rows:
            test_rows.append(
                {
                    "window_id": window.window_id,
                    "model": model,
                    "position_limit": limit,
                    "selected_position_limit": selected_limit,
                    "test_start": window.test_start.isoformat(),
                    "test_end_exclusive": window.test_end_exclusive.isoformat(),
                    **summary,
                    "return_advantage_vs_fixed_percent": (
                        advantage if model == "DYNAMIC_SELECTED" else 0.0
                    ),
                }
            )
            for record in _filtered_records(test_bundle["tickers"], limit=limit):
                ticker_rows.append(
                    {
                        "window_id": window.window_id,
                        "model": model,
                        "position_limit": limit,
                        **record,
                    }
                )
            for record in _filtered_records(
                test_bundle["rejections"],
                limit=limit,
            ):
                rejection_rows.append(
                    {
                        "window_id": window.window_id,
                        "model": model,
                        "position_limit": limit,
                        **record,
                    }
                )
            equity = test_bundle["equity"].loc[
                test_bundle["equity"]["maximum_open_positions"] == limit
            ].copy()
            capitals[model] = _append_scaled_equity(
                stitched_rows,
                equity=equity,
                window_id=window.window_id,
                model=model,
                position_limit=limit,
                opening_capital=capitals[model],
                initial_cash=base_config.initial_cash,
            )

        window_rows.append(
            {
                **window.to_dict(),
                "selected_position_limit": selected_limit,
                "fixed_baseline_position_limit": baseline_position_limit,
                "dynamic_return_percent": dynamic["total_return_percent"],
                "fixed_return_percent": fixed["total_return_percent"],
                "dynamic_return_advantage_vs_fixed_percent": advantage,
                "dynamic_max_drawdown_percent": dynamic[
                    "maximum_drawdown_percent"
                ],
                "fixed_max_drawdown_percent": fixed[
                    "maximum_drawdown_percent"
                ],
                "dynamic_profit_factor": dynamic["profit_factor"],
                "fixed_profit_factor": fixed["profit_factor"],
                "dynamic_excess_vs_matched_percent": dynamic[
                    "excess_return_vs_matched_percent"
                ],
                "fixed_excess_vs_matched_percent": fixed[
                    "excess_return_vs_matched_percent"
                ],
            }
        )
        print(
            f"SELECTED={selected_limit} | "
            f"dynamic={dynamic['total_return_percent']:+.4f}% | "
            f"fixed={fixed['total_return_percent']:+.4f}% | "
            f"advantage={advantage:+.4f}%"
        )

    windows_frame = pd.DataFrame(window_rows)
    training_frame = pd.DataFrame(training_rows)
    test_frame = pd.DataFrame(test_rows)
    tickers_frame = pd.DataFrame(ticker_rows)
    rejections_frame = pd.DataFrame(rejection_rows)
    stitched_frame = pd.DataFrame(stitched_rows)

    aggregate_frame = pd.DataFrame(
        [
            _aggregate_model(
                test_frame,
                stitched_frame,
                model=model,
                initial_cash=base_config.initial_cash,
                first_test_start=windows[0].test_start,
                last_test_end_exclusive=windows[-1].test_end_exclusive,
            )
            for model in ("DYNAMIC_SELECTED", "FIXED_BASELINE")
        ]
    )
    dynamic_aggregate = aggregate_frame.iloc[0]
    fixed_aggregate = aggregate_frame.iloc[1]
    comparison = {
        "dynamic_minus_fixed_compounded_return_percent": round(
            dynamic_aggregate["compounded_return_percent"]
            - fixed_aggregate["compounded_return_percent"],
            4,
        ),
        "dynamic_drawdown_advantage_percent": round(
            fixed_aggregate["maximum_drawdown_percent"]
            - dynamic_aggregate["maximum_drawdown_percent"],
            4,
        ),
        "dynamic_minus_fixed_return_drawdown_ratio": round(
            dynamic_aggregate["return_drawdown_ratio"]
            - fixed_aggregate["return_drawdown_ratio"],
            4,
        ),
        "dynamic_test_window_win_count": int(
            (windows_frame["dynamic_return_advantage_vs_fixed_percent"] > 0).sum()
        ),
        "dynamic_test_window_tie_count": int(
            (windows_frame["dynamic_return_advantage_vs_fixed_percent"] == 0).sum()
        ),
        "dynamic_test_window_loss_count": int(
            (windows_frame["dynamic_return_advantage_vs_fixed_percent"] < 0).sum()
        ),
    }
    selections = Counter(windows_frame["selected_position_limit"].astype(int))
    selection_summary = {
        "selection_counts": {
            str(limit): int(selections.get(limit, 0)) for limit in limits
        },
        "baseline_selected_count": int(
            selections.get(baseline_position_limit, 0)
        ),
        "selection_change_count": int(
            (
                windows_frame["selected_position_limit"]
                .astype(int)
                .diff()
                .fillna(0)
                != 0
            ).sum()
        ),
    }

    return {
        "windows": windows_frame,
        "training_candidates": training_frame,
        "test_runs": test_frame,
        "aggregate": aggregate_frame,
        "ticker_contributions": tickers_frame,
        "rejections": rejections_frame,
        "stitched_equity": stitched_frame,
        "comparison": comparison,
        "selection_summary": selection_summary,
        "window_definitions": [window.to_dict() for window in windows],
        "base_config": base_config.to_dict(),
        "position_limits": list(limits),
        "baseline_position_limit": baseline_position_limit,
        "train_months": train_months,
        "test_months": test_months,
        "step_months": step_months,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_portfolio_walk_forward(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths = {
        "json": output_directory / f"portfolio_walk_forward_{stamp}.json",
        "windows": output_directory / f"portfolio_walk_forward_windows_{stamp}.csv",
        "training": output_directory / f"portfolio_walk_forward_training_{stamp}.csv",
        "test_runs": output_directory / f"portfolio_walk_forward_test_runs_{stamp}.csv",
        "aggregate": output_directory / f"portfolio_walk_forward_aggregate_{stamp}.csv",
        "tickers": output_directory / f"portfolio_walk_forward_tickers_{stamp}.csv",
        "rejections": output_directory / f"portfolio_walk_forward_rejections_{stamp}.csv",
        "equity": output_directory / f"portfolio_walk_forward_equity_{stamp}.csv",
    }
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Rolling portfolio walk-forward. Candidate limits are ranked on "
            "training data only and evaluated on the next unseen window."
        ),
        "base_config": bundle["base_config"],
        "position_limits": bundle["position_limits"],
        "baseline_position_limit": bundle["baseline_position_limit"],
        "train_months": bundle["train_months"],
        "test_months": bundle["test_months"],
        "step_months": bundle["step_months"],
        "window_definitions": bundle["window_definitions"],
        "selection_summary": bundle["selection_summary"],
        "comparison": bundle["comparison"],
        "windows": bundle["windows"].to_dict(orient="records"),
        "training_candidates": bundle["training_candidates"].to_dict(
            orient="records"
        ),
        "test_runs": bundle["test_runs"].to_dict(orient="records"),
        "aggregate": bundle["aggregate"].to_dict(orient="records"),
        "ticker_contributions": bundle["ticker_contributions"].to_dict(
            orient="records"
        ),
        "rejections": bundle["rejections"].to_dict(orient="records"),
    }
    with paths["json"].open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")

    bundle["windows"].to_csv(paths["windows"], index=False)
    bundle["training_candidates"].to_csv(paths["training"], index=False)
    bundle["test_runs"].to_csv(paths["test_runs"], index=False)
    bundle["aggregate"].to_csv(paths["aggregate"], index=False)
    bundle["ticker_contributions"].to_csv(paths["tickers"], index=False)
    bundle["rejections"].to_csv(paths["rejections"], index=False)
    bundle["stitched_equity"].to_csv(paths["equity"], index=False)
    return paths


def print_portfolio_walk_forward(bundle: dict[str, Any]) -> None:
    windows = bundle["windows"]
    aggregate = bundle["aggregate"]
    print()
    print("=" * 132)
    print("AI STOCK RADAR — PORTFOLIO WALK-FORWARD")
    print("=" * 132)
    print(
        windows[
            [
                "window_id",
                "selected_position_limit",
                "dynamic_return_percent",
                "fixed_return_percent",
                "dynamic_return_advantage_vs_fixed_percent",
                "dynamic_max_drawdown_percent",
                "fixed_max_drawdown_percent",
            ]
        ].to_string(index=False)
    )
    print("-" * 132)
    print(
        aggregate[
            [
                "model",
                "compounded_return_percent",
                "cagr_percent",
                "maximum_drawdown_percent",
                "return_drawdown_ratio",
                "positive_window_count",
                "matched_benchmark_beat_count",
                "total_trades",
            ]
        ].to_string(index=False)
    )
    print("=" * 132)
    print(f"Selection summary: {bundle['selection_summary']}")
    print(f"Dynamic vs fixed: {bundle['comparison']}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portfolio position-limit walk-forward analysis."
    )
    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=list(DEFAULT_POSITION_LIMITS),
    )
    parser.add_argument(
        "--baseline-position",
        type=int,
        default=DEFAULT_BASELINE_POSITION_LIMIT,
    )
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--max-position", type=float, default=25.0)
    parser.add_argument("--max-total-risk", type=float, default=4.0)
    parser.add_argument("--max-crypto", type=float, default=25.0)
    parser.add_argument("--stock-stop", type=float, default=5.0)
    parser.add_argument("--crypto-stop", type=float, default=5.0)
    parser.add_argument("--stock-trailing", type=float, default=7.5)
    parser.add_argument("--crypto-trailing", type=float, default=7.5)
    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--minimum-fee", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fractional-stocks", action="store_true")
    parser.add_argument("--no-fractional-crypto", action="store_true")
    parser.add_argument("--no-crypto-regime", action="store_true")
    parser.add_argument("--no-force-close", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    tickers = arguments.tickers or list(DEFAULT_TICKERS)
    limits = normalize_position_limits(arguments.positions)
    config = PortfolioBacktestConfig(
        initial_cash=arguments.initial_cash,
        risk_per_trade_percent=arguments.risk,
        maximum_position_percent=arguments.max_position,
        maximum_total_open_risk_percent=arguments.max_total_risk,
        maximum_crypto_allocation_percent=arguments.max_crypto,
        maximum_open_positions=max(limits),
        stock_stop_loss_percent=arguments.stock_stop,
        crypto_stop_loss_percent=arguments.crypto_stop,
        stock_trailing_close_percent=arguments.stock_trailing,
        crypto_trailing_close_percent=arguments.crypto_trailing,
        commission_rate=arguments.commission_rate,
        minimum_fee=arguments.minimum_fee,
        slippage_bps=arguments.slippage_bps,
        allow_fractional_stocks=arguments.fractional_stocks,
        allow_fractional_crypto=(not arguments.no_fractional_crypto),
        force_close_at_end=(not arguments.no_force_close),
    )
    config.validate()

    from src.backtest.run_portfolio_backtest import prepare_portfolio_data

    prepared = prepare_portfolio_data(
        tickers=tickers,
        period=arguments.period,
        regime_period=arguments.regime_period,
        use_crypto_regime=(not arguments.no_crypto_regime),
    )
    bundle = run_portfolio_walk_forward(
        data_by_ticker=prepared,
        base_config=config,
        position_limits=limits,
        baseline_position_limit=arguments.baseline_position,
        train_months=arguments.train_months,
        test_months=arguments.test_months,
        step_months=arguments.step_months,
    )
    print_portfolio_walk_forward(bundle)

    if not arguments.no_save:
        paths = save_portfolio_walk_forward(
            bundle,
            output_directory=arguments.output_directory,
        )
        print()
        print("=" * 100)
        print("PORTFOLIO WALK-FORWARD FILES")
        print("=" * 100)
        for label, path in paths.items():
            print(f"{label.upper():<12} {path.resolve()}")
        print("=" * 100)

    print()
    print("Portfolio walk-forward completed successfully.")


if __name__ == "__main__":
    main()
