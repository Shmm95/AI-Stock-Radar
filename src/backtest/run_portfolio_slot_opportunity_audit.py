"""Audit the opportunity cost of full-portfolio entry rejections.

This module does not replace positions or change trading rules. It instruments
the verified EDGE_ONLY portfolio replay and compares every candidate rejected
by MAX_OPEN_POSITIONS with the positions held at that causal timestamp.
Historical research only; this module cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from src.backtest import portfolio_backtest_engine as portfolio_engine
from src.backtest.run_portfolio_trade_timing_attribution import (
    _config_from_source,
    load_source,
    normalize_horizons,
)
from src.backtest.run_portfolio_walk_forward import slice_prepared_data
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/slot_opportunity_audit"
)
DEFAULT_STOP_MODELS = ("FIXED_BASELINE", "FIXED_MAX_RETURN")
DEFAULT_HORIZONS = (5, 10, 20)


def _asset_class(ticker: str) -> str:
    return "CRYPTO" if ticker.endswith(("-USD", "-EUR", "-GBP")) else "EQUITY"


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


def _position_at_or_after(data: pd.DataFrame, timestamp: Any) -> int | None:
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    position = int(index.searchsorted(pd.Timestamp(timestamp), side="left"))
    return position if position < len(index) else None


def forward_from_open(
    data: pd.DataFrame,
    *,
    timestamp: Any,
    reference_open: float | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Calculate causal open-to-future-close diagnostics."""

    position = _position_at_or_after(data, timestamp)
    output: dict[str, Any] = {}
    normalized = normalize_horizons(horizons)
    if position is None:
        for horizon in normalized:
            output[f"forward_return_{horizon}_bars_percent"] = None
            output[f"max_forward_return_{horizon}_bars_percent"] = None
        return output
    reference = (
        float(reference_open)
        if reference_open is not None
        else float(data.iloc[position]["Open"])
    )
    for horizon in normalized:
        end = position + horizon
        if end >= len(data) or reference <= 0:
            output[f"forward_return_{horizon}_bars_percent"] = None
            output[f"max_forward_return_{horizon}_bars_percent"] = None
            continue
        close = float(data.iloc[end]["Close"])
        future = data.iloc[position + 1 : end + 1]["Close"].astype(float)
        maximum = float(future.max()) if not future.empty else close
        output[f"forward_return_{horizon}_bars_percent"] = round(
            (close / reference - 1) * 100, 4
        )
        output[f"max_forward_return_{horizon}_bars_percent"] = round(
            (maximum / reference - 1) * 100, 4
        )
    return output


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2:
        return 0.0
    left_rank = frame["left"].rank(method="average")
    right_rank = frame["right"].rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return 0.0
    value = left_rank.corr(right_rank)
    return round(float(value), 6) if pd.notna(value) else 0.0


@dataclass(slots=True)
class SlotOpportunityCollector:
    data_by_ticker: dict[str, pd.DataFrame]
    window_id: str
    stop_model: str
    horizons: tuple[int, ...]
    events: list[dict[str, Any]]
    holdings: list[dict[str, Any]]
    sequence: int = 0

    def capture(
        self,
        *,
        state: Any,
        pending: Any,
        timestamp: str,
        portfolio_bar_index: int,
        raw_open_price: float,
        config: Any,
    ) -> None:
        signal = pending.signal
        if signal.ticker in state.positions:
            return
        if len(state.positions) < config.maximum_open_positions:
            return

        self.sequence += 1
        event_id = (
            f"{self.window_id}:{self.stop_model}:{self.sequence:04d}:"
            f"{signal.ticker}"
        )
        candidate_data = self.data_by_ticker[signal.ticker]
        candidate_metrics = forward_from_open(
            candidate_data,
            timestamp=timestamp,
            reference_open=raw_open_price,
            horizons=self.horizons,
        )
        holding_rows: list[dict[str, Any]] = []
        for held_ticker, position in sorted(state.positions.items()):
            held_data = self.data_by_ticker[held_ticker]
            current_position = _position_at_or_after(held_data, timestamp)
            if current_position is None:
                continue
            current_timestamp = pd.Timestamp(held_data.index[current_position])
            current_open = float(held_data.iloc[current_position]["Open"])
            entry_position = _position_at_or_after(
                held_data, position.entry_timestamp
            )
            holding_age = (
                current_position - entry_position
                if entry_position is not None
                else None
            )
            row = {
                "event_id": event_id,
                "window_id": self.window_id,
                "stop_model": self.stop_model,
                "candidate_ticker": signal.ticker,
                "candidate_asset_class": _asset_class(signal.ticker),
                "candidate_score": float(signal.score),
                "timestamp": timestamp,
                "held_ticker": held_ticker,
                "held_asset_class": _asset_class(held_ticker),
                "held_reference_timestamp": current_timestamp,
                "held_reference_open": round(current_open, 8),
                "held_entry_timestamp": position.entry_timestamp,
                "held_entry_price": float(position.entry_price),
                "held_score": float(position.signal_score),
                "candidate_minus_held_score": round(
                    float(signal.score) - float(position.signal_score), 4
                ),
                "holding_age_bars": holding_age,
                "held_unrealized_return_percent": round(
                    (current_open / float(position.entry_price) - 1) * 100,
                    4,
                ),
            }
            row.update(
                forward_from_open(
                    held_data,
                    timestamp=current_timestamp,
                    reference_open=current_open,
                    horizons=self.horizons,
                )
            )
            holding_rows.append(row)
            self.holdings.append(row)

        event = {
            "event_id": event_id,
            "window_id": self.window_id,
            "stop_model": self.stop_model,
            "timestamp": timestamp,
            "portfolio_bar_index": portfolio_bar_index,
            "candidate_ticker": signal.ticker,
            "candidate_asset_class": _asset_class(signal.ticker),
            "candidate_score": float(signal.score),
            "candidate_reference_open": round(float(raw_open_price), 8),
            "held_position_count": len(holding_rows),
            "mean_held_score": round(
                sum(row["held_score"] for row in holding_rows)
                / len(holding_rows),
                4,
            ),
        }
        event["candidate_minus_mean_held_score"] = round(
            event["candidate_score"] - event["mean_held_score"], 4
        )
        for horizon in self.horizons:
            candidate_value = candidate_metrics[
                f"forward_return_{horizon}_bars_percent"
            ]
            held_values = [
                row[f"forward_return_{horizon}_bars_percent"]
                for row in holding_rows
                if row[f"forward_return_{horizon}_bars_percent"] is not None
            ]
            prefix = f"{horizon}_bars"
            event[f"candidate_forward_return_{prefix}_percent"] = candidate_value
            event[f"mean_held_forward_return_{prefix}_percent"] = (
                round(sum(held_values) / len(held_values), 4)
                if held_values
                else None
            )
            event[f"worst_held_forward_return_{prefix}_percent"] = (
                round(min(held_values), 4) if held_values else None
            )
            event[f"best_held_forward_return_{prefix}_percent"] = (
                round(max(held_values), 4) if held_values else None
            )
            mean_held = event[f"mean_held_forward_return_{prefix}_percent"]
            worst_held = event[f"worst_held_forward_return_{prefix}_percent"]
            best_held = event[f"best_held_forward_return_{prefix}_percent"]
            event[f"candidate_minus_mean_held_{prefix}_percent"] = (
                round(candidate_value - mean_held, 4)
                if candidate_value is not None and mean_held is not None
                else None
            )
            event[f"candidate_beats_mean_held_{prefix}"] = (
                bool(candidate_value > mean_held)
                if candidate_value is not None and mean_held is not None
                else None
            )
            event[f"candidate_beats_worst_held_{prefix}"] = (
                bool(candidate_value > worst_held)
                if candidate_value is not None and worst_held is not None
                else None
            )
            event[f"candidate_beats_all_held_{prefix}"] = (
                bool(candidate_value > best_held)
                if candidate_value is not None and best_held is not None
                else None
            )
        self.events.append(event)


@contextmanager
def slot_opportunity_context(
    collector: SlotOpportunityCollector,
) -> Iterator[None]:
    """Instrument the engine without changing its decisions."""

    original = portfolio_engine._attempt_open_position

    def wrapper(
        state: Any,
        *,
        pending: Any,
        timestamp: str,
        portfolio_bar_index: int,
        raw_open_price: float,
        config: Any,
        prices: dict[str, float],
    ) -> None:
        collector.capture(
            state=state,
            pending=pending,
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_open_price=raw_open_price,
            config=config,
        )
        return original(
            state,
            pending=pending,
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_open_price=raw_open_price,
            config=config,
            prices=prices,
        )

    portfolio_engine._attempt_open_position = wrapper
    try:
        yield
    finally:
        portfolio_engine._attempt_open_position = original


def summarize_events(
    events: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    horizons: Sequence[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouping: Any = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in events.groupby(grouping, sort=False, dropna=False):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        record = dict(zip(group_columns, key_values))
        record.update(
            {
                "event_count": int(len(group)),
                "average_candidate_score": round(
                    float(group["candidate_score"].mean()), 4
                ),
                "average_candidate_minus_mean_held_score": round(
                    float(group["candidate_minus_mean_held_score"].mean()), 4
                ),
            }
        )
        for horizon in horizons:
            prefix = f"{horizon}_bars"
            candidate = f"candidate_forward_return_{prefix}_percent"
            held = f"mean_held_forward_return_{prefix}_percent"
            gap = f"candidate_minus_mean_held_{prefix}_percent"
            beat_mean = f"candidate_beats_mean_held_{prefix}"
            beat_worst = f"candidate_beats_worst_held_{prefix}"
            beat_all = f"candidate_beats_all_held_{prefix}"
            valid = group[[candidate, held, gap]].dropna()
            record[f"valid_event_count_{prefix}"] = int(len(valid))
            record[f"average_candidate_return_{prefix}_percent"] = round(
                float(valid[candidate].mean()), 4
            ) if len(valid) else 0.0
            record[f"average_mean_held_return_{prefix}_percent"] = round(
                float(valid[held].mean()), 4
            ) if len(valid) else 0.0
            record[f"average_opportunity_gap_{prefix}_percent"] = round(
                float(valid[gap].mean()), 4
            ) if len(valid) else 0.0
            record[f"median_opportunity_gap_{prefix}_percent"] = round(
                float(valid[gap].median()), 4
            ) if len(valid) else 0.0
            for source, label in (
                (beat_mean, "beat_mean_rate"),
                (beat_worst, "beat_worst_rate"),
                (beat_all, "beat_all_rate"),
            ):
                values = group[source].dropna().astype(bool)
                record[f"candidate_{label}_{prefix}_percent"] = round(
                    float(values.mean() * 100), 4
                ) if len(values) else 0.0
            record[f"score_premium_spearman_to_gap_{prefix}"] = (
                _rank_correlation(
                    group["candidate_minus_mean_held_score"], group[gap]
                )
            )
        records.append(record)
    return pd.DataFrame(records)


def build_score_quintiles(
    events: pd.DataFrame, horizons: Sequence[int]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stop_model, group in events.groupby("stop_model", sort=False):
        ranked = group.copy()
        ranked["score_premium_quintile"] = (
            pd.qcut(
                ranked["candidate_minus_mean_held_score"],
                5,
                labels=False,
                duplicates="drop",
            )
            + 1
        )
        frame = summarize_events(
            ranked,
            group_columns=("score_premium_quintile",),
            horizons=horizons,
        )
        frame.insert(0, "stop_model", stop_model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_screen(
    summary: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        window_rows = windows.loc[windows["stop_model"] == row.stop_model]
        positive_windows = int(
            (
                window_rows["average_opportunity_gap_20_bars_percent"] > 0
            ).sum()
        )
        criteria = {
            "positive_average_gap_10_bars": (
                row.average_opportunity_gap_10_bars_percent > 0
            ),
            "positive_average_gap_20_bars": (
                row.average_opportunity_gap_20_bars_percent > 0
            ),
            "candidate_beats_mean_at_least_55_percent_20_bars": (
                row.candidate_beat_mean_rate_20_bars_percent >= 55
            ),
            "positive_window_gap_at_least_8_of_13_20_bars": (
                positive_windows >= 8
            ),
            "score_premium_spearman_at_least_0p10_20_bars": (
                row.score_premium_spearman_to_gap_20_bars >= 0.10
            ),
        }
        rows.append(
            {
                "stop_model": row.stop_model,
                **criteria,
                "positive_window_count_20_bars": positive_windows,
                "criteria_passed": int(sum(criteria.values())),
                "stratum_pass": bool(all(criteria.values())),
            }
        )
    result = pd.DataFrame(rows)
    robust = bool(len(result) and result["stratum_pass"].all())
    result["robust_replacement_audit_pass"] = robust
    return result


def _verify_entry_predecessor(
    *,
    directory: Path,
    stamp: str,
    snapshot_fingerprint: str,
    source_stamp: str,
) -> dict[str, Any]:
    provenance_path = directory / (
        f"portfolio_entry_state_ablation_provenance_{stamp}.json"
    )
    json_path = directory / f"portfolio_entry_state_ablation_{stamp}.json"
    if not provenance_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            "Slot audit requires the official Entry State Ablation JSON and provenance."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if provenance.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise ValueError("Entry predecessor snapshot fingerprint differs.")
    if provenance.get("source_stamp") != source_stamp:
        raise ValueError("Entry predecessor source stamp differs.")
    for item in provenance.get("result_files", {}).values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Entry predecessor hash mismatch: {path}")
    screen = payload.get("screen", {}).get("results", [])
    if any(bool(row.get("robust_screen_pass")) for row in screen):
        raise ValueError(
            "Entry predecessor has a passing policy; slot audit is not the next gate."
        )
    return {
        "json_path": json_path,
        "provenance_path": provenance_path,
        "payload": payload,
    }


def _verify_stop_provenance(
    *, snapshot: dict[str, Any], directory: Path, stamp: str
) -> Path:
    path = directory / f"portfolio_stop_walk_forward_provenance_{stamp}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("snapshot_fingerprint") != snapshot["manifest"]["fingerprint"]:
        raise ValueError("Stop source snapshot fingerprint differs.")
    return path


def run_slot_opportunity_audit(
    *,
    snapshot_path: Path,
    stop_source_directory: Path,
    stop_stamp: str,
    entry_source_directory: Path,
    entry_stamp: str,
    stop_models: Sequence[str] = DEFAULT_STOP_MODELS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    normalized_horizons = normalize_horizons(horizons)
    snapshot = load_snapshot(Path(snapshot_path), verify_code=True)
    stop_source_directory = Path(stop_source_directory)
    stop_source = load_source(stop_source_directory, stop_stamp)
    stop_provenance = _verify_stop_provenance(
        snapshot=snapshot, directory=stop_source_directory, stamp=stop_stamp
    )
    entry_source = _verify_entry_predecessor(
        directory=Path(entry_source_directory),
        stamp=entry_stamp,
        snapshot_fingerprint=snapshot["manifest"]["fingerprint"],
        source_stamp=stop_stamp,
    )
    config = _config_from_source(stop_source)
    profiles = stop_source["payload"].get("profiles", {})
    models = tuple(dict.fromkeys(str(item) for item in stop_models))
    missing = [model for model in models if model not in profiles]
    if missing:
        raise ValueError(f"Stop profiles are missing: {', '.join(missing)}")
    source_runs = stop_source["test_runs"].set_index(["window_id", "model"])

    all_events: list[dict[str, Any]] = []
    all_holdings: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    windows_source = stop_source["windows"].sort_values("window_id")
    for sequence, window in enumerate(windows_source.to_dict("records"), start=1):
        window_id = str(window["window_id"])
        print(f"\nSLOT AUDIT [{sequence}/{len(windows_source)}] {window_id}")
        test_data = slice_prepared_data(
            snapshot["data_by_ticker"],
            start=pd.Timestamp(window["test_start"]),
            end_exclusive=pd.Timestamp(window["test_end_exclusive"]),
        )
        for model in models:
            profile = profiles[model]
            active_config = replace(
                config,
                stock_stop_loss_percent=float(
                    profile["stock_stop_loss_percent"]
                ),
                crypto_stop_loss_percent=float(
                    profile["crypto_stop_loss_percent"]
                ),
            )
            collector = SlotOpportunityCollector(
                data_by_ticker=test_data,
                window_id=window_id,
                stop_model=model,
                horizons=normalized_horizons,
                events=all_events,
                holdings=all_holdings,
            )
            with slot_opportunity_context(collector):
                result = portfolio_engine.run_portfolio_backtest(
                    data_by_ticker=test_data,
                    config=active_config,
                    include_benchmark=False,
                )
            source_return = float(
                source_runs.loc[(window_id, model), "total_return_percent"]
            )
            run_rows.append(
                {
                    "window_id": window_id,
                    "stop_model": model,
                    "test_start": window["test_start"],
                    "test_end_exclusive": window["test_end_exclusive"],
                    "source_return_percent": source_return,
                    "replay_return_percent": result.total_return_percent,
                    "replay_difference_percent": round(
                        result.total_return_percent - source_return, 6
                    ),
                    "maximum_drawdown_percent": result.maximum_drawdown_percent,
                    "completed_trades": result.completed_trades,
                    "rejected_signals": result.rejected_signals,
                    "captured_max_position_events": collector.sequence,
                }
            )

    events = pd.DataFrame(all_events)
    holdings = pd.DataFrame(all_holdings)
    runs = pd.DataFrame(run_rows)
    if events.empty:
        raise ValueError("No MAX_OPEN_POSITIONS opportunity events were captured.")
    summary = summarize_events(
        events, group_columns=("stop_model",), horizons=normalized_horizons
    )
    windows = summarize_events(
        events,
        group_columns=("stop_model", "window_id"),
        horizons=normalized_horizons,
    )
    tickers = summarize_events(
        events,
        group_columns=("stop_model", "candidate_ticker"),
        horizons=normalized_horizons,
    )
    score_quintiles = build_score_quintiles(events, normalized_horizons)
    screen = build_screen(summary, windows)
    if runs["replay_difference_percent"].abs().max() > 1e-9:
        raise ValueError("Instrumented EDGE_ONLY replay differs from stop source.")
    return {
        "events": events,
        "holdings": holdings,
        "runs": runs,
        "summary": summary,
        "windows": windows,
        "tickers": tickers,
        "score_quintiles": score_quintiles,
        "screen": screen,
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint": snapshot["manifest"]["fingerprint"],
        "snapshot_manifest_path": Path(snapshot_path) / "manifest.json",
        "stop_stamp": stop_stamp,
        "stop_paths": stop_source["paths"],
        "stop_provenance_path": stop_provenance,
        "entry_stamp": entry_stamp,
        "entry_json_path": entry_source["json_path"],
        "entry_provenance_path": entry_source["provenance_path"],
        "stop_models": list(models),
        "horizons": list(normalized_horizons),
    }


def save_slot_opportunity_audit(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    frame_names = (
        "events",
        "holdings",
        "runs",
        "summary",
        "windows",
        "tickers",
        "score_quintiles",
        "screen",
    )
    paths = {
        name: output_directory
        / f"portfolio_slot_opportunity_audit_{name}_{stamp}.csv"
        for name in frame_names
    }
    paths["json"] = output_directory / (
        f"portfolio_slot_opportunity_audit_{stamp}.json"
    )
    paths["provenance"] = output_directory / (
        f"portfolio_slot_opportunity_audit_provenance_{stamp}.json"
    )
    for name in frame_names:
        bundle[name].to_csv(paths[name], index=False)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Instrumented, decision-preserving EDGE_ONLY replay of every "
            "MAX_OPEN_POSITIONS candidate versus contemporaneous holdings."
        ),
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "stop_stamp": bundle["stop_stamp"],
        "entry_stamp": bundle["entry_stamp"],
        "stop_models": bundle["stop_models"],
        "horizons": bundle["horizons"],
        "summary": bundle["summary"].to_dict("records"),
        "screen": bundle["screen"].to_dict("records"),
        "limitations": [
            "Forward returns are non-additive diagnostics, not executable replacement PnL.",
            "Repeated events may overlap in time and share held positions.",
            "The audit changes no position; a passing result only authorizes a separate replacement-policy ablation.",
            "Score predictiveness is required before score-based replacement is tested.",
        ],
    }
    paths["json"].write_text(
        json.dumps(_safe(payload), indent=2) + "\n", encoding="utf-8"
    )
    source_paths = {
        **{
            f"stop_{name}": {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in bundle["stop_paths"].items()
        },
        "stop_provenance": {
            "path": str(Path(bundle["stop_provenance_path"]).resolve()),
            "sha256": sha256_file(Path(bundle["stop_provenance_path"])),
        },
        "entry_json": {
            "path": str(Path(bundle["entry_json_path"]).resolve()),
            "sha256": sha256_file(Path(bundle["entry_json_path"])),
        },
        "entry_provenance": {
            "path": str(Path(bundle["entry_provenance_path"]).resolve()),
            "sha256": sha256_file(Path(bundle["entry_provenance_path"])),
        },
    }
    result_files = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name != "provenance"
    }
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "audit_stamp": stamp,
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "snapshot_manifest_path": str(
            Path(bundle["snapshot_manifest_path"]).resolve()
        ),
        "snapshot_manifest_sha256": sha256_file(
            Path(bundle["snapshot_manifest_path"])
        ),
        "stop_stamp": bundle["stop_stamp"],
        "entry_stamp": bundle["entry_stamp"],
        "source_files": source_paths,
        "audit_code": {
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
        description="Frozen-snapshot portfolio slot opportunity audit"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--stop-source-directory",
        type=Path,
        default=Path("data/backtests/portfolio/stop_walk_forward"),
    )
    parser.add_argument("--stop-stamp", required=True)
    parser.add_argument(
        "--entry-source-directory",
        type=Path,
        default=Path("data/backtests/portfolio/entry_state_ablation"),
    )
    parser.add_argument("--entry-stamp", required=True)
    parser.add_argument(
        "--stop-models", nargs="+", default=list(DEFAULT_STOP_MODELS)
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS)
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bundle = run_slot_opportunity_audit(
        snapshot_path=args.snapshot,
        stop_source_directory=args.stop_source_directory,
        stop_stamp=args.stop_stamp,
        entry_source_directory=args.entry_source_directory,
        entry_stamp=args.entry_stamp,
        stop_models=args.stop_models,
        horizons=args.horizons,
    )
    print("\nSUMMARY")
    print(bundle["summary"].to_string(index=False))
    print("\nSCREEN")
    print(bundle["screen"].to_string(index=False))
    if not args.no_save:
        for name, path in save_slot_opportunity_audit(
            bundle, output_directory=args.output_directory
        ).items():
            print(name, path.resolve())


if __name__ == "__main__":
    main()
