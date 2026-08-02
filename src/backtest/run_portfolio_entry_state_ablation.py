"""Causal entry-state ablation on a verified research snapshot.

The experiment changes only how an already-valid TREND_RSI setup becomes an
entry candidate. Portfolio risk, stop, trailing, execution, and ranking logic
remain in the validated portfolio engine. Historical research only; this
module cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import comb, isfinite
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from src.backtest import portfolio_backtest_engine as portfolio_engine
from src.backtest.portfolio_backtest_models import PortfolioSignal
from src.backtest.run_portfolio_position_ablation import (
    annual_returns_from_equity,
    build_variant_summary,
    rejection_counts,
    run_matched_benchmark,
    ticker_contributions,
)
from src.backtest.run_portfolio_risk_walk_forward import _aggregate_model
from src.backtest.run_portfolio_trade_timing_attribution import (
    _config_from_source,
    load_source,
)
from src.backtest.run_portfolio_walk_forward import slice_prepared_data
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/entry_state_ablation"
)
DEFAULT_STOP_MODELS = ("FIXED_BASELINE", "FIXED_MAX_RETURN")
BASELINE_POLICY = "EDGE_ONLY"


@dataclass(frozen=True, slots=True)
class EntryPolicy:
    """One pre-registered treatment in the 2x2 entry-state experiment."""

    name: str
    boundary_entry: bool
    persistence_bars: int

    def __post_init__(self) -> None:
        normalized = self.name.strip().upper()
        if not normalized:
            raise ValueError("Entry policy name cannot be empty.")
        if int(self.persistence_bars) < 1:
            raise ValueError("persistence_bars must be at least one.")
        object.__setattr__(self, "name", normalized)
        object.__setattr__(self, "persistence_bars", int(self.persistence_bars))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_policy": self.name,
            "boundary_entry": self.boundary_entry,
            "persistence_bars": self.persistence_bars,
        }


ENTRY_POLICIES = (
    EntryPolicy("EDGE_ONLY", False, 1),
    EntryPolicy("BOUNDARY_EDGE", True, 1),
    EntryPolicy("EDGE_PERSIST_5", False, 5),
    EntryPolicy("BOUNDARY_PERSIST_5", True, 5),
)
POLICY_BY_NAME = {policy.name: policy for policy in ENTRY_POLICIES}


def normalize_policy_names(values: Iterable[str]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        name = str(value).strip().upper()
        if name not in POLICY_BY_NAME:
            raise ValueError(
                f"Unknown entry policy {value!r}; expected one of "
                f"{', '.join(POLICY_BY_NAME)}."
            )
        if name not in names:
            names.append(name)
    if BASELINE_POLICY not in names:
        raise ValueError(f"{BASELINE_POLICY} must be included as the control.")
    return tuple(names)


def _setup_run_start(data: pd.DataFrame, local_bar_index: int) -> int:
    """Return the first bar of the current consecutive valid-setup run."""

    start = int(local_bar_index)
    while start > 0 and portfolio_engine._is_entry_setup(data.iloc[start - 1]):
        start -= 1
    return start


def build_policy_entry_signal(
    ticker: str,
    data: pd.DataFrame,
    local_bar_index: int,
    policy: EntryPolicy,
) -> PortfolioSignal | None:
    """Build a causal signal under one entry-state policy.

    A persistence value of five means the setup may be attempted on its edge
    and the following four closes while it remains valid. A boundary treatment
    treats the first test-window close as the observable edge when the setup is
    already valid at the beginning of the isolated test window.
    """

    if local_bar_index < 0:
        return None
    current = data.iloc[local_bar_index]
    if not portfolio_engine._is_entry_setup(current):
        return None

    if local_bar_index == 0:
        if not policy.boundary_entry:
            return None
        event_index = 0
        event_label = "boundary-valid"
    else:
        run_start = _setup_run_start(data, local_bar_index)
        if run_start == 0:
            if not policy.boundary_entry:
                return None
            event_index = 0
            event_label = "boundary-valid"
        else:
            event_index = run_start
            event_label = "newly-valid"

    age = local_bar_index - event_index
    if age < 0 or age >= policy.persistence_bars:
        return None

    score = portfolio_engine._entry_score(current)
    timestamp = portfolio_engine._timestamp_to_string(
        data.index[local_bar_index]
    )
    return PortfolioSignal(
        timestamp=timestamp,
        ticker=ticker,
        action="BUY",
        reference_price=float(current["Close"]),
        score=score,
        technical_score=score,
        confidence=score,
        reason=(
            f"ENTRY_STATE {policy.name} {event_label} attempt "
            f"{age + 1}/{policy.persistence_bars}: TREND_RSI valid."
        ),
    )


@contextmanager
def entry_policy_context(policy: EntryPolicy) -> Iterator[None]:
    """Temporarily route the unchanged engine through one entry policy."""

    original = portfolio_engine._build_entry_signal

    def builder(
        ticker: str,
        data: pd.DataFrame,
        local_bar_index: int,
    ) -> PortfolioSignal | None:
        return build_policy_entry_signal(
            ticker, data, local_bar_index, policy
        )

    portfolio_engine._build_entry_signal = builder
    try:
        yield
    finally:
        portfolio_engine._build_entry_signal = original


def _safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _model_key(stop_model: str, policy_name: str) -> str:
    return f"{stop_model}::{policy_name}"


def _append_scaled_equity(
    output: list[dict[str, Any]],
    *,
    result: Any,
    window_id: str,
    stop_model: str,
    policy: EntryPolicy,
    opening_capital: float,
) -> float:
    frame = pd.DataFrame([point.to_dict() for point in result.equity_curve])
    if frame.empty:
        return opening_capital
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp")
    scale = opening_capital / float(result.initial_cash)
    key = _model_key(stop_model, policy.name)
    for row in frame.itertuples(index=False):
        output.append(
            {
                "window_id": window_id,
                "model": key,
                "stop_model": stop_model,
                **policy.to_dict(),
                "timestamp": pd.Timestamp(row.timestamp),
                "total_equity": round(float(row.total_equity) * scale, 6),
            }
        )
    return round(float(frame.iloc[-1]["total_equity"]) * scale, 6)


def _exact_sign_test_two_sided(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    smaller = min(int(wins), int(losses))
    probability = 2 * sum(comb(n, k) for k in range(smaller + 1)) / (2**n)
    return round(min(probability, 1.0), 6)


def build_paired_effects(
    runs: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    baseline_policy: str = BASELINE_POLICY,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_model = aggregate.set_index("model")
    for stop_model in runs["stop_model"].drop_duplicates():
        baseline = runs.loc[
            (runs["stop_model"] == stop_model)
            & (runs["entry_policy"] == baseline_policy)
        ]
        if baseline.empty:
            raise ValueError(f"Missing {baseline_policy} for {stop_model}.")
        for policy_name in runs.loc[
            runs["stop_model"] == stop_model, "entry_policy"
        ].drop_duplicates():
            if policy_name == baseline_policy:
                continue
            candidate = runs.loc[
                (runs["stop_model"] == stop_model)
                & (runs["entry_policy"] == policy_name)
            ]
            paired = baseline.merge(
                candidate,
                on="window_id",
                suffixes=("_baseline", "_candidate"),
                validate="one_to_one",
            )
            delta = (
                paired["total_return_percent_candidate"]
                - paired["total_return_percent_baseline"]
            )
            wins = int((delta > 1e-9).sum())
            ties = int((delta.abs() <= 1e-9).sum())
            losses = int((delta < -1e-9).sum())
            base_aggregate = by_model.loc[
                _model_key(stop_model, baseline_policy)
            ]
            candidate_aggregate = by_model.loc[
                _model_key(stop_model, policy_name)
            ]
            records.append(
                {
                    "stop_model": stop_model,
                    "baseline_policy": baseline_policy,
                    "candidate_policy": policy_name,
                    "window_count": int(len(paired)),
                    "window_wins": wins,
                    "window_ties": ties,
                    "window_losses": losses,
                    "sign_test_p_value_two_sided": _exact_sign_test_two_sided(
                        wins, losses
                    ),
                    "average_window_return_delta_percent": round(
                        float(delta.mean()), 4
                    ),
                    "median_window_return_delta_percent": round(
                        float(delta.median()), 4
                    ),
                    "compounded_return_delta_percent": round(
                        float(
                            candidate_aggregate["compounded_return_percent"]
                            - base_aggregate["compounded_return_percent"]
                        ),
                        4,
                    ),
                    "drawdown_advantage_percent": round(
                        float(
                            base_aggregate["maximum_drawdown_percent"]
                            - candidate_aggregate["maximum_drawdown_percent"]
                        ),
                        4,
                    ),
                    "return_drawdown_ratio_delta": round(
                        float(
                            candidate_aggregate["return_drawdown_ratio"]
                            - base_aggregate["return_drawdown_ratio"]
                        ),
                        4,
                    ),
                    "average_exposure_delta_percent": round(
                        float(
                            candidate_aggregate["average_exposure_percent"]
                            - base_aggregate["average_exposure_percent"]
                        ),
                        4,
                    ),
                    "trade_count_delta": int(
                        candidate_aggregate["total_trades"]
                        - base_aggregate["total_trades"]
                    ),
                    "rejection_event_delta": int(
                        candidate["rejected_signals"].sum()
                        - baseline["rejected_signals"].sum()
                    ),
                }
            )
    return pd.DataFrame(records)


FACTOR_CONTRASTS = (
    ("BOUNDARY_WITHOUT_PERSISTENCE", "BOUNDARY_EDGE", "EDGE_ONLY"),
    ("PERSISTENCE_WITHOUT_BOUNDARY", "EDGE_PERSIST_5", "EDGE_ONLY"),
    (
        "BOUNDARY_WITH_PERSISTENCE",
        "BOUNDARY_PERSIST_5",
        "EDGE_PERSIST_5",
    ),
    (
        "PERSISTENCE_WITH_BOUNDARY",
        "BOUNDARY_PERSIST_5",
        "BOUNDARY_EDGE",
    ),
)


def build_factor_effects(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    available = set(runs["entry_policy"])
    for stop_model in runs["stop_model"].drop_duplicates():
        stop_rows = runs.loc[runs["stop_model"] == stop_model]
        for contrast, candidate_policy, control_policy in FACTOR_CONTRASTS:
            if candidate_policy not in available or control_policy not in available:
                continue
            control = stop_rows.loc[
                stop_rows["entry_policy"] == control_policy
            ]
            candidate = stop_rows.loc[
                stop_rows["entry_policy"] == candidate_policy
            ]
            paired = control.merge(
                candidate,
                on="window_id",
                suffixes=("_control", "_candidate"),
                validate="one_to_one",
            )
            delta = (
                paired["total_return_percent_candidate"]
                - paired["total_return_percent_control"]
            )
            wins = int((delta > 1e-9).sum())
            ties = int((delta.abs() <= 1e-9).sum())
            losses = int((delta < -1e-9).sum())
            records.append(
                {
                    "stop_model": stop_model,
                    "contrast": contrast,
                    "candidate_policy": candidate_policy,
                    "control_policy": control_policy,
                    "window_count": int(len(paired)),
                    "window_wins": wins,
                    "window_ties": ties,
                    "window_losses": losses,
                    "sign_test_p_value_two_sided": _exact_sign_test_two_sided(
                        wins, losses
                    ),
                    "average_window_return_delta_percent": round(
                        float(delta.mean()), 4
                    ),
                    "median_window_return_delta_percent": round(
                        float(delta.median()), 4
                    ),
                }
            )
    return pd.DataFrame(records)


def build_screen(paired_effects: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered robustness screen without selecting on rank."""

    rows: list[dict[str, Any]] = []
    for policy_name, group in paired_effects.groupby(
        "candidate_policy", sort=False
    ):
        return_positive = group["compounded_return_delta_percent"] > 0
        ratio_nonnegative = group["return_drawdown_ratio_delta"] >= 0
        majority_wins = group["window_wins"] > group["window_losses"]
        drawdown_bounded = group["drawdown_advantage_percent"] >= -2.5
        passed_by_stratum = (
            return_positive & ratio_nonnegative & majority_wins & drawdown_bounded
        )
        rows.append(
            {
                "candidate_policy": policy_name,
                "stop_strata_count": int(len(group)),
                "positive_compounded_return_strata": int(return_positive.sum()),
                "nonnegative_return_drawdown_strata": int(
                    ratio_nonnegative.sum()
                ),
                "majority_window_win_strata": int(majority_wins.sum()),
                "drawdown_within_2p5_point_strata": int(
                    drawdown_bounded.sum()
                ),
                "strata_passing_all_criteria": int(passed_by_stratum.sum()),
                "robust_screen_pass": bool(passed_by_stratum.all()),
            }
        )
    return pd.DataFrame(rows)


def _verify_reference_provenance(
    *,
    snapshot: dict[str, Any],
    source_directory: Path,
    stamp: str,
) -> dict[str, Any]:
    path = source_directory / (
        f"portfolio_stop_walk_forward_provenance_{stamp}.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Entry-state ablation requires a provenanced reference: {path}"
        )
    provenance = json.loads(path.read_text(encoding="utf-8"))
    expected = snapshot["manifest"]["fingerprint"]
    if provenance.get("snapshot_fingerprint") != expected:
        raise ValueError("Reference and experiment snapshot fingerprints differ.")
    return {"path": path, "payload": provenance}


def run_entry_state_ablation(
    *,
    snapshot_path: Path,
    source_directory: Path,
    stamp: str,
    stop_models: Sequence[str] = DEFAULT_STOP_MODELS,
    policy_names: Sequence[str] = tuple(POLICY_BY_NAME),
) -> dict[str, Any]:
    """Run the pre-registered 2x2 experiment on unseen test windows."""

    names = normalize_policy_names(policy_names)
    policies = tuple(POLICY_BY_NAME[name] for name in names)
    snapshot = load_snapshot(Path(snapshot_path), verify_code=True)
    source_directory = Path(source_directory)
    source = load_source(source_directory, stamp)
    reference_provenance = _verify_reference_provenance(
        snapshot=snapshot,
        source_directory=source_directory,
        stamp=stamp,
    )
    base_config = _config_from_source(source)
    profiles = source["payload"].get("profiles", {})
    normalized_stop_models = tuple(dict.fromkeys(str(x) for x in stop_models))
    missing = [name for name in normalized_stop_models if name not in profiles]
    if missing:
        raise ValueError(f"Source profiles are missing: {', '.join(missing)}")

    run_rows: list[dict[str, Any]] = []
    ticker_frames: list[pd.DataFrame] = []
    rejection_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    capitals = {
        _model_key(stop_model, policy.name): float(base_config.initial_cash)
        for stop_model in normalized_stop_models
        for policy in policies
    }
    source_runs = source["test_runs"].set_index(["window_id", "model"])

    windows = source["windows"].sort_values("window_id")
    for sequence, window in enumerate(windows.to_dict("records"), start=1):
        window_id = str(window["window_id"])
        print(f"\nENTRY STATE [{sequence}/{len(windows)}] {window_id}")
        test_data = slice_prepared_data(
            snapshot["data_by_ticker"],
            start=pd.Timestamp(window["test_start"]),
            end_exclusive=pd.Timestamp(window["test_end_exclusive"]),
        )
        for stop_model in normalized_stop_models:
            profile = profiles[stop_model]
            stock_stop = float(profile["stock_stop_loss_percent"])
            crypto_stop = float(profile["crypto_stop_loss_percent"])
            config = replace(
                base_config,
                stock_stop_loss_percent=stock_stop,
                crypto_stop_loss_percent=crypto_stop,
            )
            config.validate()
            source_return = float(
                source_runs.loc[(window_id, stop_model), "total_return_percent"]
            )
            for policy in policies:
                print(f"  {stop_model:<18} {policy.name}")
                with entry_policy_context(policy):
                    result = portfolio_engine.run_portfolio_backtest(
                        data_by_ticker=test_data,
                        config=config,
                        include_benchmark=False,
                    )
                matched, _ = run_matched_benchmark(
                    test_data,
                    initial_cash=config.initial_cash,
                    exposure_percent=max(
                        float(result.average_exposure_percent), 0.0001
                    ),
                    config=config,
                    label=f"MATCHED_{stop_model}_{policy.name}_{window_id}",
                )
                annual = annual_returns_from_equity(
                    result.equity_curve,
                    initial_cash=result.initial_cash,
                )
                tickers = ticker_contributions(result)
                summary = build_variant_summary(result, matched, annual, tickers)
                model = _model_key(stop_model, policy.name)
                run_rows.append(
                    {
                        "window_id": window_id,
                        "test_start": window["test_start"],
                        "test_end_exclusive": window["test_end_exclusive"],
                        "model": model,
                        "stop_model": stop_model,
                        "stock_stop_loss_percent": stock_stop,
                        "crypto_stop_loss_percent": crypto_stop,
                        **policy.to_dict(),
                        "source_edge_only_return_percent": source_return,
                        "source_replay_difference_percent": (
                            round(
                                result.total_return_percent - source_return, 6
                            )
                            if policy.name == BASELINE_POLICY
                            else None
                        ),
                        **summary,
                    }
                )
                if not tickers.empty:
                    frame = tickers.copy()
                    frame.insert(0, "window_id", window_id)
                    frame.insert(1, "model", model)
                    frame.insert(2, "stop_model", stop_model)
                    frame.insert(3, "entry_policy", policy.name)
                    ticker_frames.append(frame)
                for reason, count in sorted(rejection_counts(result).items()):
                    rejection_rows.append(
                        {
                            "window_id": window_id,
                            "model": model,
                            "stop_model": stop_model,
                            "entry_policy": policy.name,
                            "reason_code": reason,
                            "count": int(count),
                        }
                    )
                capitals[model] = _append_scaled_equity(
                    equity_rows,
                    result=result,
                    window_id=window_id,
                    stop_model=stop_model,
                    policy=policy,
                    opening_capital=capitals[model],
                )

    runs = pd.DataFrame(run_rows)
    equity = pd.DataFrame(equity_rows)
    first_start = pd.Timestamp(windows.iloc[0]["test_start"])
    last_end = pd.Timestamp(windows.iloc[-1]["test_end_exclusive"])
    aggregate_rows: list[dict[str, Any]] = []
    for stop_model in normalized_stop_models:
        for policy in policies:
            model = _model_key(stop_model, policy.name)
            row = _aggregate_model(
                runs,
                equity,
                model=model,
                initial_cash=base_config.initial_cash,
                first_test_start=first_start,
                last_test_end_exclusive=last_end,
            )
            aggregate_rows.append(
                {
                    "stop_model": stop_model,
                    **policy.to_dict(),
                    **row,
                    "total_rejection_events": int(
                        runs.loc[runs["model"] == model, "rejected_signals"].sum()
                    ),
                    "maximum_absolute_source_replay_difference_percent": (
                        round(
                            float(
                                pd.to_numeric(
                                    runs.loc[
                                        runs["model"] == model,
                                        "source_replay_difference_percent",
                                    ],
                                    errors="coerce",
                                )
                                .abs()
                                .max()
                            ),
                            6,
                        )
                        if policy.name == BASELINE_POLICY
                        else None
                    ),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    paired = build_paired_effects(runs, aggregate)
    factors = build_factor_effects(runs)
    screen = build_screen(paired)
    tickers = (
        pd.concat(ticker_frames, ignore_index=True)
        if ticker_frames
        else pd.DataFrame()
    )
    rejections = pd.DataFrame(rejection_rows)
    return {
        "runs": runs,
        "aggregate": aggregate,
        "paired_effects": paired,
        "factor_effects": factors,
        "screen": screen,
        "tickers": tickers,
        "rejections": rejections,
        "equity": equity,
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint": snapshot["manifest"]["fingerprint"],
        "snapshot_manifest_path": Path(snapshot_path) / "manifest.json",
        "source_stamp": stamp,
        "source_paths": source["paths"],
        "source_provenance_path": reference_provenance["path"],
        "stop_models": list(normalized_stop_models),
        "entry_policies": [policy.to_dict() for policy in policies],
        "base_config": base_config.to_dict(),
    }


def save_entry_state_ablation(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    frame_names = (
        "runs",
        "aggregate",
        "paired_effects",
        "factor_effects",
        "screen",
        "tickers",
        "rejections",
        "equity",
    )
    paths = {
        name: output_directory
        / f"portfolio_entry_state_ablation_{name}_{stamp}.csv"
        for name in frame_names
    }
    paths["json"] = (
        output_directory / f"portfolio_entry_state_ablation_{stamp}.json"
    )
    paths["provenance"] = output_directory / (
        f"portfolio_entry_state_ablation_provenance_{stamp}.json"
    )
    for name in frame_names:
        bundle[name].to_csv(paths[name], index=False)

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Frozen-snapshot 2x2 causal entry-state ablation on the same "
            "13 unseen stop walk-forward windows."
        ),
        "hypotheses": {
            "boundary": (
                "A valid setup at an isolated test-window boundary should be "
                "eligible at the next causal open."
            ),
            "persistence": (
                "A newly valid or boundary-valid setup may be retried for five "
                "bars after a capacity rejection while it remains valid."
            ),
        },
        "screen": {
            "criteria": [
                "positive compounded-return delta in every stop stratum",
                "nonnegative return/drawdown-ratio delta in every stratum",
                "more window wins than losses in every stratum",
                "maximum-drawdown deterioration no worse than 2.5 points",
            ],
            "results": bundle["screen"].to_dict("records"),
        },
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "source_stamp": bundle["source_stamp"],
        "stop_models": bundle["stop_models"],
        "entry_policies": bundle["entry_policies"],
        "base_config": bundle["base_config"],
        "aggregate": bundle["aggregate"].to_dict("records"),
        "paired_effects": bundle["paired_effects"].to_dict("records"),
        "factor_effects": bundle["factor_effects"].to_dict("records"),
        "limitations": [
            "This is an ablation, not a train-selected deployment rule.",
            "The same nine-asset universe and 13 non-overlapping test windows are reused.",
            "A passing policy requires a separate walk-forward selection test before adoption.",
        ],
    }
    paths["json"].write_text(
        json.dumps(_safe(payload), indent=2) + "\n", encoding="utf-8"
    )

    source_paths = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(Path(path)),
        }
        for name, path in bundle["source_paths"].items()
    }
    source_paths["provenance"] = {
        "path": str(Path(bundle["source_provenance_path"]).resolve()),
        "sha256": sha256_file(Path(bundle["source_provenance_path"])),
    }
    result_files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
        if name != "provenance"
    }
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_stamp": stamp,
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "snapshot_manifest_path": str(
            Path(bundle["snapshot_manifest_path"]).resolve()
        ),
        "snapshot_manifest_sha256": sha256_file(
            Path(bundle["snapshot_manifest_path"])
        ),
        "source_stamp": bundle["source_stamp"],
        "source_files": source_paths,
        "experiment_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "result_files": result_files,
    }
    paths["provenance"].write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-snapshot portfolio entry-state ablation"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path("data/backtests/portfolio/stop_walk_forward"),
    )
    parser.add_argument("--stamp", required=True)
    parser.add_argument(
        "--stop-models", nargs="+", default=list(DEFAULT_STOP_MODELS)
    )
    parser.add_argument(
        "--policies", nargs="+", default=list(POLICY_BY_NAME)
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bundle = run_entry_state_ablation(
        snapshot_path=args.snapshot,
        source_directory=args.source_directory,
        stamp=args.stamp,
        stop_models=args.stop_models,
        policy_names=args.policies,
    )
    print("\nAGGREGATE")
    print(bundle["aggregate"].to_string(index=False))
    print("\nPAIRED EFFECTS")
    print(bundle["paired_effects"].to_string(index=False))
    print("\nROBUSTNESS SCREEN")
    print(bundle["screen"].to_string(index=False))
    if not args.no_save:
        for name, path in save_entry_state_ablation(
            bundle, output_directory=args.output_directory
        ).items():
            print(name, path.resolve())


if __name__ == "__main__":
    main()
