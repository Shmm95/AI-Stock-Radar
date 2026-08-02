"""Rolling portfolio walk-forward analysis for risk settings.

Training windows rank risk-per-trade and maximum-total-open-risk profiles using
only historical data available inside each training window. The selected
profile is evaluated on the immediately following unseen test window and is
compared with two fixed controls:

* the existing V1 baseline (1.00% risk per trade / 4.00% total open risk),
* the balanced full-period challenger (0.75% / 4.00%), and
* the aggressive full-period challenger (1.00% / 5.00%).

This module is historical research only. It cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    DEFAULT_REGIME_PERIOD,
    DEFAULT_TICKERS,
)
from src.backtest.run_portfolio_risk_ablation import (
    DEFAULT_MAXIMUM_TOTAL_RISK_VALUES,
    DEFAULT_RISK_PER_TRADE_VALUES,
    normalize_maximum_total_risk_values,
    normalize_risk_per_trade_values,
    risk_variant_id,
    run_portfolio_risk_ablation,
)
from src.backtest.run_portfolio_walk_forward import (
    WalkForwardWindow,
    build_walk_forward_windows,
    compound_returns,
    maximum_drawdown_percent,
    slice_prepared_data,
)

DEFAULT_PERIOD = "5y"
DEFAULT_MAXIMUM_OPEN_POSITIONS = 4
DEFAULT_BASELINE_RISK_PER_TRADE = 1.0
DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK = 4.0
DEFAULT_BALANCED_RISK_PER_TRADE = 0.75
DEFAULT_BALANCED_MAXIMUM_TOTAL_RISK = 4.0
DEFAULT_AGGRESSIVE_RISK_PER_TRADE = 1.0
DEFAULT_AGGRESSIVE_MAXIMUM_TOTAL_RISK = 5.0
DEFAULT_TRAIN_MONTHS = 24
DEFAULT_TEST_MONTHS = 6
DEFAULT_STEP_MONTHS = 6
DEFAULT_OUTPUT_DIRECTORY = (
    Path("data") / "backtests" / "portfolio" / "risk_walk_forward"
)

MODEL_DYNAMIC = "DYNAMIC_SELECTED"
MODEL_BASELINE = "FIXED_BASELINE"
MODEL_BALANCED = "FIXED_BALANCED"
MODEL_AGGRESSIVE = "FIXED_AGGRESSIVE"


@dataclass(frozen=True, slots=True, order=True)
class RiskProfile:
    risk_per_trade_percent: float
    maximum_total_open_risk_percent: float

    def __post_init__(self) -> None:
        risk = round(float(self.risk_per_trade_percent), 8)
        total = round(float(self.maximum_total_open_risk_percent), 8)
        if risk <= 0:
            raise ValueError("risk_per_trade_percent must be positive.")
        if total <= 0:
            raise ValueError(
                "maximum_total_open_risk_percent must be positive."
            )
        object.__setattr__(self, "risk_per_trade_percent", risk)
        object.__setattr__(
            self,
            "maximum_total_open_risk_percent",
            total,
        )

    @property
    def variant_id(self) -> str:
        return risk_variant_id(
            self.risk_per_trade_percent,
            self.maximum_total_open_risk_percent,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "variant_id": self.variant_id,
            "risk_per_trade_percent": self.risk_per_trade_percent,
            "maximum_total_open_risk_percent": (
                self.maximum_total_open_risk_percent
            ),
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model: str
    profile: RiskProfile


def _profile_from_row(row: pd.Series | dict[str, Any]) -> RiskProfile:
    return RiskProfile(
        risk_per_trade_percent=float(row["risk_per_trade_percent"]),
        maximum_total_open_risk_percent=float(
            row["maximum_total_open_risk_percent"]
        ),
    )


def _summary_row(
    frame: pd.DataFrame,
    profile: RiskProfile,
) -> dict[str, Any]:
    matches = frame.loc[frame["variant_id"] == profile.variant_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one summary row for {profile.variant_id}."
        )
    return matches.iloc[0].to_dict()


def _variant_records(
    frame: pd.DataFrame,
    profile: RiskProfile,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.loc[
        frame["variant_id"] == profile.variant_id
    ].to_dict(orient="records")


def _training_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    ranked = bundle["summary"].copy().reset_index(drop=True)
    ranked = ranked.rename(
        columns={
            "rank": "training_rank",
            "selected_candidate": "selected_for_test",
        }
    )
    return ranked


def _run_single_profile(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    profile: RiskProfile,
    maximum_open_positions: int,
    baseline_profile: RiskProfile,
) -> dict[str, Any]:
    return run_portfolio_risk_ablation(
        data_by_ticker=data_by_ticker,
        base_config=base_config,
        risk_per_trade_values=(profile.risk_per_trade_percent,),
        maximum_total_risk_values=(
            profile.maximum_total_open_risk_percent,
        ),
        maximum_open_positions=maximum_open_positions,
        baseline_risk_per_trade_percent=(
            baseline_profile.risk_per_trade_percent
        ),
        baseline_maximum_total_open_risk_percent=(
            baseline_profile.maximum_total_open_risk_percent
        ),
    )


def _append_scaled_equity(
    output: list[dict[str, Any]],
    *,
    equity: pd.DataFrame,
    window: WalkForwardWindow,
    model_profile: ModelProfile,
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
                "window_id": window.window_id,
                "model": model_profile.model,
                **model_profile.profile.to_dict(),
                "timestamp": pd.Timestamp(row.timestamp),
                "total_equity": round(float(row.total_equity) * scale, 6),
            }
        )

    return round(float(curve.iloc[-1]["total_equity"]) * scale, 6)


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


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if isfinite(float(value))]
    return round(sum(finite) / len(finite), 4) if finite else 0.0


def _aggregate_model(
    test_runs: pd.DataFrame,
    stitched_equity: pd.DataFrame,
    *,
    model: str,
    initial_cash: float,
    first_test_start: pd.Timestamp,
    last_test_end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    rows = test_runs.loc[test_runs["model"] == model].sort_values(
        "window_id"
    )
    curve = stitched_equity.loc[
        stitched_equity["model"] == model
    ].sort_values("timestamp")
    if rows.empty:
        raise ValueError(f"No test rows found for model {model}.")

    ending_equity = compound_returns(
        rows["total_return_percent"],
        initial_cash=initial_cash,
    )
    total_return = (ending_equity / initial_cash - 1) * 100
    drawdown = maximum_drawdown_percent(curve["total_equity"])
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
            float(rows["total_return_percent"].mean()),
            4,
        ),
        "median_window_return_percent": round(
            float(rows["total_return_percent"].median()),
            4,
        ),
        "best_window_return_percent": round(
            float(rows["total_return_percent"].max()),
            4,
        ),
        "worst_window_return_percent": round(
            float(rows["total_return_percent"].min()),
            4,
        ),
        "positive_window_count": int(
            (rows["total_return_percent"] > 0).sum()
        ),
        "matched_benchmark_beat_count": int(
            (rows["excess_return_vs_matched_percent"] > 0).sum()
        ),
        "total_trades": int(rows["completed_trades"].sum()),
        "average_profit_factor": _finite_mean(rows["profit_factor"]),
        "average_exposure_percent": round(
            float(rows["average_exposure_percent"].mean()),
            4,
        ),
    }


def _comparison_counts(
    windows: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    values = pd.to_numeric(windows[column], errors="coerce").fillna(0.0)
    return {
        "win_count": int((values > 1e-9).sum()),
        "tie_count": int((values.abs() <= 1e-9).sum()),
        "loss_count": int((values < -1e-9).sum()),
    }


def run_portfolio_risk_walk_forward(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    risk_per_trade_values: Iterable[float] = (
        DEFAULT_RISK_PER_TRADE_VALUES
    ),
    maximum_total_risk_values: Iterable[float] = (
        DEFAULT_MAXIMUM_TOTAL_RISK_VALUES
    ),
    maximum_open_positions: int = DEFAULT_MAXIMUM_OPEN_POSITIONS,
    baseline_profile: RiskProfile = RiskProfile(
        DEFAULT_BASELINE_RISK_PER_TRADE,
        DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK,
    ),
    balanced_profile: RiskProfile = RiskProfile(
        DEFAULT_BALANCED_RISK_PER_TRADE,
        DEFAULT_BALANCED_MAXIMUM_TOTAL_RISK,
    ),
    aggressive_profile: RiskProfile = RiskProfile(
        DEFAULT_AGGRESSIVE_RISK_PER_TRADE,
        DEFAULT_AGGRESSIVE_MAXIMUM_TOTAL_RISK,
    ),
    train_months: int = DEFAULT_TRAIN_MONTHS,
    test_months: int = DEFAULT_TEST_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> dict[str, Any]:
    """Run training-only risk selection and unseen portfolio tests."""

    risks = normalize_risk_per_trade_values(risk_per_trade_values)
    totals = normalize_maximum_total_risk_values(
        maximum_total_risk_values
    )
    maximum_open_positions = int(maximum_open_positions)
    if maximum_open_positions <= 0:
        raise ValueError("maximum_open_positions must be positive.")

    fixed_profiles = (
        ("Baseline", baseline_profile),
        ("Balanced", balanced_profile),
        ("Aggressive", aggressive_profile),
    )
    for label, profile in fixed_profiles:
        if profile.risk_per_trade_percent not in risks:
            raise ValueError(
                f"{label} risk must be included in risk candidates."
            )
        if profile.maximum_total_open_risk_percent not in totals:
            raise ValueError(
                f"{label} total risk must be included in "
                "total-risk candidates."
            )
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
        MODEL_DYNAMIC: float(base_config.initial_cash),
        MODEL_BASELINE: float(base_config.initial_cash),
        MODEL_BALANCED: float(base_config.initial_cash),
        MODEL_AGGRESSIVE: float(base_config.initial_cash),
    }

    for sequence, window in enumerate(windows, start=1):
        print()
        print("=" * 126)
        print(
            f"PORTFOLIO RISK WALK-FORWARD "
            f"[{sequence}/{len(windows)}] — {window.window_id}"
        )
        print(
            f"TRAIN {window.train_start.date()} -> "
            f"{window.train_end_exclusive.date()} | "
            f"TEST {window.test_start.date()} -> "
            f"{window.test_end_exclusive.date()}"
        )
        print("=" * 126)

        train_data = slice_prepared_data(
            data_by_ticker,
            start=window.train_start,
            end_exclusive=window.train_end_exclusive,
        )
        train_bundle = run_portfolio_risk_ablation(
            data_by_ticker=train_data,
            base_config=base_config,
            risk_per_trade_values=risks,
            maximum_total_risk_values=totals,
            maximum_open_positions=maximum_open_positions,
            baseline_risk_per_trade_percent=(
                baseline_profile.risk_per_trade_percent
            ),
            baseline_maximum_total_open_risk_percent=(
                baseline_profile.maximum_total_open_risk_percent
            ),
        )
        ranked = _training_frame(train_bundle)
        selected_profile = _profile_from_row(ranked.iloc[0])
        for record in ranked.to_dict(orient="records"):
            training_rows.append({**window.to_dict(), **record})

        test_data = slice_prepared_data(
            data_by_ticker,
            start=window.test_start,
            end_exclusive=window.test_end_exclusive,
        )
        profiles = {
            selected_profile.variant_id: selected_profile,
            baseline_profile.variant_id: baseline_profile,
            balanced_profile.variant_id: balanced_profile,
            aggressive_profile.variant_id: aggressive_profile,
        }
        profile_bundles = {
            profile_id: _run_single_profile(
                data_by_ticker=test_data,
                base_config=base_config,
                profile=profile,
                maximum_open_positions=maximum_open_positions,
                baseline_profile=baseline_profile,
            )
            for profile_id, profile in profiles.items()
        }

        model_profiles = (
            ModelProfile(MODEL_DYNAMIC, selected_profile),
            ModelProfile(MODEL_BASELINE, baseline_profile),
            ModelProfile(MODEL_BALANCED, balanced_profile),
            ModelProfile(MODEL_AGGRESSIVE, aggressive_profile),
        )
        model_summaries: dict[str, dict[str, Any]] = {}

        for model_profile in model_profiles:
            profile = model_profile.profile
            profile_bundle = profile_bundles[profile.variant_id]
            summary = _summary_row(profile_bundle["summary"], profile)
            model_summaries[model_profile.model] = summary
            test_rows.append(
                {
                    "window_id": window.window_id,
                    "model": model_profile.model,
                    "selected_variant_id": selected_profile.variant_id,
                    "selected_risk_per_trade_percent": (
                        selected_profile.risk_per_trade_percent
                    ),
                    "selected_maximum_total_open_risk_percent": (
                        selected_profile.maximum_total_open_risk_percent
                    ),
                    "test_start": window.test_start.isoformat(),
                    "test_end_exclusive": (
                        window.test_end_exclusive.isoformat()
                    ),
                    **profile.to_dict(),
                    **summary,
                }
            )

            for record in _variant_records(
                profile_bundle["tickers"],
                profile,
            ):
                ticker_rows.append(
                    {
                        "window_id": window.window_id,
                        "model": model_profile.model,
                        **record,
                    }
                )
            for record in _variant_records(
                profile_bundle["rejections"],
                profile,
            ):
                rejection_rows.append(
                    {
                        "window_id": window.window_id,
                        "model": model_profile.model,
                        **record,
                    }
                )

            equity = profile_bundle["equity"].loc[
                profile_bundle["equity"]["variant_id"]
                == profile.variant_id
            ].copy()
            capitals[model_profile.model] = _append_scaled_equity(
                stitched_rows,
                equity=equity,
                window=window,
                model_profile=model_profile,
                opening_capital=capitals[model_profile.model],
                initial_cash=base_config.initial_cash,
            )

        dynamic = model_summaries[MODEL_DYNAMIC]
        baseline = model_summaries[MODEL_BASELINE]
        balanced = model_summaries[MODEL_BALANCED]
        aggressive = model_summaries[MODEL_AGGRESSIVE]

        def advantage(row: dict[str, Any]) -> float:
            return round(
                row["total_return_percent"]
                - baseline["total_return_percent"],
                4,
            )

        window_rows.append(
            {
                **window.to_dict(),
                "selected_variant_id": selected_profile.variant_id,
                "selected_risk_per_trade_percent": (
                    selected_profile.risk_per_trade_percent
                ),
                "selected_maximum_total_open_risk_percent": (
                    selected_profile.maximum_total_open_risk_percent
                ),
                "baseline_variant_id": baseline_profile.variant_id,
                "balanced_variant_id": balanced_profile.variant_id,
                "aggressive_variant_id": aggressive_profile.variant_id,
                "dynamic_return_percent": dynamic["total_return_percent"],
                "baseline_return_percent": baseline[
                    "total_return_percent"
                ],
                "balanced_return_percent": balanced[
                    "total_return_percent"
                ],
                "aggressive_return_percent": aggressive[
                    "total_return_percent"
                ],
                "dynamic_return_advantage_vs_baseline_percent": (
                    advantage(dynamic)
                ),
                "balanced_return_advantage_vs_baseline_percent": (
                    advantage(balanced)
                ),
                "aggressive_return_advantage_vs_baseline_percent": (
                    advantage(aggressive)
                ),
                "dynamic_max_drawdown_percent": dynamic[
                    "maximum_drawdown_percent"
                ],
                "baseline_max_drawdown_percent": baseline[
                    "maximum_drawdown_percent"
                ],
                "balanced_max_drawdown_percent": balanced[
                    "maximum_drawdown_percent"
                ],
                "aggressive_max_drawdown_percent": aggressive[
                    "maximum_drawdown_percent"
                ],
                "dynamic_profit_factor": dynamic["profit_factor"],
                "baseline_profit_factor": baseline["profit_factor"],
                "balanced_profit_factor": balanced["profit_factor"],
                "aggressive_profit_factor": aggressive["profit_factor"],
                "dynamic_excess_vs_matched_percent": dynamic[
                    "excess_return_vs_matched_percent"
                ],
                "baseline_excess_vs_matched_percent": baseline[
                    "excess_return_vs_matched_percent"
                ],
                "balanced_excess_vs_matched_percent": balanced[
                    "excess_return_vs_matched_percent"
                ],
                "aggressive_excess_vs_matched_percent": aggressive[
                    "excess_return_vs_matched_percent"
                ],
            }
        )
        print(
            f"SELECTED={selected_profile.variant_id} | "
            f"dynamic={dynamic['total_return_percent']:+.4f}% | "
            f"baseline={baseline['total_return_percent']:+.4f}% | "
            f"balanced={balanced['total_return_percent']:+.4f}% | "
            f"aggressive={aggressive['total_return_percent']:+.4f}%"
        )

    windows_frame = pd.DataFrame(window_rows)
    training_frame = pd.DataFrame(training_rows)
    test_frame = pd.DataFrame(test_rows)
    tickers_frame = pd.DataFrame(ticker_rows)
    rejections_frame = pd.DataFrame(rejection_rows)
    stitched_frame = pd.DataFrame(stitched_rows)

    models = (
        MODEL_DYNAMIC,
        MODEL_BASELINE,
        MODEL_BALANCED,
        MODEL_AGGRESSIVE,
    )
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
            for model in models
        ]
    )
    aggregate_by_model = aggregate_frame.set_index("model")
    baseline_aggregate = aggregate_by_model.loc[MODEL_BASELINE]

    comparison: dict[str, Any] = {}
    comparison_specs = (
        ("dynamic", MODEL_DYNAMIC, "dynamic_return_advantage_vs_baseline_percent"),
        ("balanced", MODEL_BALANCED, "balanced_return_advantage_vs_baseline_percent"),
        ("aggressive", MODEL_AGGRESSIVE, "aggressive_return_advantage_vs_baseline_percent"),
    )
    for prefix, model, window_column in comparison_specs:
        candidate = aggregate_by_model.loc[model]
        counts = _comparison_counts(windows_frame, window_column)
        comparison.update(
            {
                f"{prefix}_minus_baseline_compounded_return_percent": round(
                    candidate["compounded_return_percent"]
                    - baseline_aggregate["compounded_return_percent"],
                    4,
                ),
                f"{prefix}_drawdown_advantage_vs_baseline_percent": round(
                    baseline_aggregate["maximum_drawdown_percent"]
                    - candidate["maximum_drawdown_percent"],
                    4,
                ),
                f"{prefix}_minus_baseline_return_drawdown_ratio": round(
                    candidate["return_drawdown_ratio"]
                    - baseline_aggregate["return_drawdown_ratio"],
                    4,
                ),
                f"{prefix}_test_window_win_count": counts["win_count"],
                f"{prefix}_test_window_tie_count": counts["tie_count"],
                f"{prefix}_test_window_loss_count": counts["loss_count"],
            }
        )

    selected_ids = windows_frame["selected_variant_id"].astype(str)
    selected_risks = windows_frame[
        "selected_risk_per_trade_percent"
    ].astype(float)
    selected_totals = windows_frame[
        "selected_maximum_total_open_risk_percent"
    ].astype(float)
    id_counts = Counter(selected_ids)
    risk_counts = Counter(selected_risks)
    total_counts = Counter(selected_totals)
    all_variant_ids = [
        risk_variant_id(risk, total)
        for risk in risks
        for total in totals
    ]
    selection_summary = {
        "variant_selection_counts": {
            variant_id: int(id_counts.get(variant_id, 0))
            for variant_id in all_variant_ids
        },
        "risk_per_trade_selection_counts": {
            f"{risk:.2f}": int(risk_counts.get(risk, 0))
            for risk in risks
        },
        "maximum_total_risk_selection_counts": {
            f"{total:.2f}": int(total_counts.get(total, 0))
            for total in totals
        },
        "baseline_selected_count": int(
            id_counts.get(baseline_profile.variant_id, 0)
        ),
        "balanced_selected_count": int(
            id_counts.get(balanced_profile.variant_id, 0)
        ),
        "aggressive_selected_count": int(
            id_counts.get(aggressive_profile.variant_id, 0)
        ),
        "selection_change_count": int(
            selected_ids.ne(selected_ids.shift(1)).iloc[1:].sum()
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
        "risk_per_trade_values": list(risks),
        "maximum_total_risk_values": list(totals),
        "maximum_open_positions": maximum_open_positions,
        "baseline_profile": baseline_profile.to_dict(),
        "balanced_profile": balanced_profile.to_dict(),
        "aggressive_profile": aggressive_profile.to_dict(),
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


def save_portfolio_risk_walk_forward(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths = {
        "json": output_directory
        / f"portfolio_risk_walk_forward_{stamp}.json",
        "windows": output_directory
        / f"portfolio_risk_walk_forward_windows_{stamp}.csv",
        "training": output_directory
        / f"portfolio_risk_walk_forward_training_{stamp}.csv",
        "test_runs": output_directory
        / f"portfolio_risk_walk_forward_test_runs_{stamp}.csv",
        "aggregate": output_directory
        / f"portfolio_risk_walk_forward_aggregate_{stamp}.csv",
        "tickers": output_directory
        / f"portfolio_risk_walk_forward_tickers_{stamp}.csv",
        "rejections": output_directory
        / f"portfolio_risk_walk_forward_rejections_{stamp}.csv",
        "equity": output_directory
        / f"portfolio_risk_walk_forward_equity_{stamp}.csv",
    }
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Rolling shared-cash portfolio risk walk-forward. Training data "
            "selects a risk profile which is evaluated on the next unseen "
            "window against fixed baseline, balanced, and aggressive "
            "profiles."
        ),
        "base_config": bundle["base_config"],
        "risk_per_trade_values": bundle["risk_per_trade_values"],
        "maximum_total_risk_values": bundle[
            "maximum_total_risk_values"
        ],
        "maximum_open_positions": bundle["maximum_open_positions"],
        "baseline_profile": bundle["baseline_profile"],
        "balanced_profile": bundle["balanced_profile"],
        "aggressive_profile": bundle["aggressive_profile"],
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


def print_portfolio_risk_walk_forward(bundle: dict[str, Any]) -> None:
    windows = bundle["windows"]
    aggregate = bundle["aggregate"]
    print()
    print("=" * 178)
    print("AI STOCK RADAR — PORTFOLIO RISK WALK-FORWARD")
    print("=" * 178)
    print(
        windows[
            [
                "window_id",
                "selected_variant_id",
                "dynamic_return_percent",
                "baseline_return_percent",
                "balanced_return_percent",
                "aggressive_return_percent",
                "dynamic_return_advantage_vs_baseline_percent",
                "balanced_return_advantage_vs_baseline_percent",
                "aggressive_return_advantage_vs_baseline_percent",
                "dynamic_max_drawdown_percent",
                "baseline_max_drawdown_percent",
                "balanced_max_drawdown_percent",
                "aggressive_max_drawdown_percent",
            ]
        ].to_string(index=False)
    )
    print("-" * 178)
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
                "average_profit_factor",
                "average_exposure_percent",
            ]
        ].to_string(index=False)
    )
    print("=" * 178)
    print(f"Selection summary: {bundle['selection_summary']}")
    print(f"Comparison: {bundle['comparison']}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portfolio risk-profile walk-forward analysis."
    )
    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--risks",
        nargs="+",
        type=float,
        default=list(DEFAULT_RISK_PER_TRADE_VALUES),
    )
    parser.add_argument(
        "--total-risks",
        nargs="+",
        type=float,
        default=list(DEFAULT_MAXIMUM_TOTAL_RISK_VALUES),
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAXIMUM_OPEN_POSITIONS,
    )
    parser.add_argument(
        "--baseline-risk",
        type=float,
        default=DEFAULT_BASELINE_RISK_PER_TRADE,
    )
    parser.add_argument(
        "--baseline-total-risk",
        type=float,
        default=DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK,
    )
    parser.add_argument(
        "--balanced-risk",
        type=float,
        default=DEFAULT_BALANCED_RISK_PER_TRADE,
    )
    parser.add_argument(
        "--balanced-total-risk",
        type=float,
        default=DEFAULT_BALANCED_MAXIMUM_TOTAL_RISK,
    )
    parser.add_argument(
        "--aggressive-risk",
        type=float,
        default=DEFAULT_AGGRESSIVE_RISK_PER_TRADE,
    )
    parser.add_argument(
        "--aggressive-total-risk",
        type=float,
        default=DEFAULT_AGGRESSIVE_MAXIMUM_TOTAL_RISK,
    )
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)

    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--max-position", type=float, default=25.0)
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
    risks = normalize_risk_per_trade_values(arguments.risks)
    totals = normalize_maximum_total_risk_values(arguments.total_risks)
    baseline_profile = RiskProfile(
        arguments.baseline_risk,
        arguments.baseline_total_risk,
    )
    balanced_profile = RiskProfile(
        arguments.balanced_risk,
        arguments.balanced_total_risk,
    )
    aggressive_profile = RiskProfile(
        arguments.aggressive_risk,
        arguments.aggressive_total_risk,
    )

    base_config = PortfolioBacktestConfig(
        initial_cash=arguments.initial_cash,
        risk_per_trade_percent=arguments.baseline_risk,
        maximum_position_percent=arguments.max_position,
        maximum_total_open_risk_percent=arguments.baseline_total_risk,
        maximum_crypto_allocation_percent=arguments.max_crypto,
        maximum_open_positions=arguments.max_positions,
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
    base_config.validate()

    from src.backtest.run_portfolio_backtest import prepare_portfolio_data

    prepared_data = prepare_portfolio_data(
        tickers=tickers,
        period=arguments.period,
        regime_period=arguments.regime_period,
        use_crypto_regime=(not arguments.no_crypto_regime),
    )
    bundle = run_portfolio_risk_walk_forward(
        data_by_ticker=prepared_data,
        base_config=base_config,
        risk_per_trade_values=risks,
        maximum_total_risk_values=totals,
        maximum_open_positions=arguments.max_positions,
        baseline_profile=baseline_profile,
        balanced_profile=balanced_profile,
        aggressive_profile=aggressive_profile,
        train_months=arguments.train_months,
        test_months=arguments.test_months,
        step_months=arguments.step_months,
    )
    print_portfolio_risk_walk_forward(bundle)

    if not arguments.no_save:
        paths = save_portfolio_risk_walk_forward(
            bundle,
            output_directory=arguments.output_directory,
        )
        print()
        print("=" * 110)
        print("PORTFOLIO RISK WALK-FORWARD FILES")
        print("=" * 110)
        for label, path in paths.items():
            print(f"{label.upper():<12} {path.resolve()}")
        print("=" * 110)

    print()
    print("Portfolio risk walk-forward completed successfully.")


if __name__ == "__main__":
    main()
