"""Causal feature audit for full-portfolio entry opportunities.

The audit consumes the verified Slot Opportunity Audit and immutable research
snapshot.  It reconstructs only information available on the signal close and
tests whether pre-registered features rank the later candidate-versus-holdings
opportunity gap.  It does not alter entries, exits, sizing, or positions.

Historical research only; no broker integration is present.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/entry_score_feature_audit"
)
DEFAULT_SLOT_DIRECTORY = Path(
    "data/backtests/portfolio/slot_opportunity_audit"
)
DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_SCREEN_HORIZON = 20
EXPECTED_STOP_MODELS = ("FIXED_BASELINE", "FIXED_MAX_RETURN")

# Every value is oriented so that a larger number means "more attractive".
FEATURE_DEFINITIONS: dict[str, str] = {
    "CURRENT_SCORE": "Existing portfolio entry score recorded by the slot audit.",
    "SCORE_PREMIUM": "Candidate score minus the mean score of held positions.",
    "TREND_SPREAD_PERCENT": "100 * (EMA20 / EMA50 - 1) on the signal close.",
    "PRICE_EXTENSION_PERCENT": "100 * (Close / EMA20 - 1) on the signal close.",
    "RSI_QUALITY": "max(0, 15 - 1.2 * abs(RSI14 - 57.5)) on the signal close.",
    "MOMENTUM_20_PERCENT": "100 * (Close / Close[-20] - 1) on the signal close.",
    "MACD_PERCENT": "100 * MACD / Close on the signal close.",
    "LOW_ATR14_QUALITY": "Negative 14-bar simple true-range average as percent of Close.",
}


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


def normalize_horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in values}))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("Horizons must contain positive integers.")
    return horizons


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation without scipy or constant-input warnings."""

    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        return 0.0
    left_rank = frame["left"].rank(method="average")
    right_rank = frame["right"].rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return 0.0
    value = left_rank.corr(right_rank)
    return round(float(value), 6) if pd.notna(value) else 0.0


def _signal_position(data: pd.DataFrame, execution_timestamp: Any) -> int:
    """Return the last candidate bar strictly before the execution open."""

    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    timestamp = pd.Timestamp(execution_timestamp)
    if index.tz is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(index.tz)
    elif index.tz is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    position = int(index.searchsorted(timestamp, side="left")) - 1
    if position < 0:
        raise ValueError(f"No causal signal bar before {execution_timestamp}.")
    return position


def causal_atr14_percent(data: pd.DataFrame, position: int) -> float | None:
    """Fourteen-bar simple true-range average using no future observations."""

    if position < 13:
        return None
    history = data.iloc[: position + 1]
    previous_close = history["Close"].astype(float).shift(1)
    ranges = pd.concat(
        [
            history["High"].astype(float) - history["Low"].astype(float),
            (history["High"].astype(float) - previous_close).abs(),
            (history["Low"].astype(float) - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = ranges.rolling(14, min_periods=14).mean().iloc[-1]
    close = float(history["Close"].iloc[-1])
    if pd.isna(value) or close <= 0:
        return None
    return float(value / close * 100)


def feature_values(
    event: pd.Series | dict[str, Any], data: pd.DataFrame
) -> dict[str, Any]:
    """Reconstruct pre-registered features from the causal signal row."""

    source = dict(event)
    position = _signal_position(data, source["timestamp"])
    row = data.iloc[position]
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    rsi = float(row["RSI14"])
    momentum = (
        (close / float(data.iloc[position - 20]["Close"]) - 1) * 100
        if position >= 20 and float(data.iloc[position - 20]["Close"]) > 0
        else None
    )
    atr_percent = causal_atr14_percent(data, position)
    macd = float(row["MACD"])
    return {
        "signal_timestamp": pd.Timestamp(data.index[position]),
        "CURRENT_SCORE": float(source["candidate_score"]),
        "SCORE_PREMIUM": float(source["candidate_minus_mean_held_score"]),
        "TREND_SPREAD_PERCENT": (ema20 / ema50 - 1) * 100,
        "PRICE_EXTENSION_PERCENT": (close / ema20 - 1) * 100,
        "RSI_QUALITY": max(0.0, 15.0 - abs(rsi - 57.5) * 1.2),
        "MOMENTUM_20_PERCENT": momentum,
        "MACD_PERCENT": (macd / close * 100) if close > 0 else None,
        "LOW_ATR14_QUALITY": -atr_percent if atr_percent is not None else None,
    }


def extract_opportunities(
    events: pd.DataFrame,
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Attach causal features to every verified slot event."""

    normalized = normalize_horizons(horizons)
    required = {
        "event_id",
        "window_id",
        "stop_model",
        "timestamp",
        "candidate_ticker",
        "candidate_asset_class",
        "candidate_score",
        "candidate_minus_mean_held_score",
    }
    required.update(
        f"candidate_minus_mean_held_{horizon}_bars_percent"
        for horizon in normalized
    )
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Slot events missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for event in events.to_dict("records"):
        ticker = str(event["candidate_ticker"])
        if ticker not in data_by_ticker:
            raise ValueError(f"Snapshot missing candidate ticker: {ticker}")
        values = feature_values(event, data_by_ticker[ticker])
        record = {
            "event_id": event["event_id"],
            "window_id": event["window_id"],
            "stop_model": event["stop_model"],
            "timestamp": event["timestamp"],
            "signal_timestamp": values.pop("signal_timestamp"),
            "candidate_ticker": ticker,
            "candidate_asset_class": event["candidate_asset_class"],
            **values,
        }
        for horizon in normalized:
            record[f"opportunity_gap_{horizon}_bars_percent"] = event[
                f"candidate_minus_mean_held_{horizon}_bars_percent"
            ]
        records.append(record)
    return pd.DataFrame(records)


def build_feature_windows(
    opportunities: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active_features = tuple(features or FEATURE_DEFINITIONS)
    for (stop_model, window_id), group in opportunities.groupby(
        ["stop_model", "window_id"], sort=True
    ):
        for feature in active_features:
            for horizon in normalize_horizons(horizons):
                target = f"opportunity_gap_{horizon}_bars_percent"
                valid = group[[feature, target]].dropna()
                records.append(
                    {
                        "stop_model": stop_model,
                        "window_id": window_id,
                        "feature": feature,
                        "horizon_bars": horizon,
                        "event_count": int(len(valid)),
                        "spearman_rho": rank_correlation(
                            valid[feature], valid[target]
                        ),
                        "average_opportunity_gap_percent": round(
                            float(valid[target].mean()), 6
                        )
                        if len(valid)
                        else 0.0,
                    }
                )
    return pd.DataFrame(records)


def build_feature_quintiles(
    opportunities: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active_features = tuple(features or FEATURE_DEFINITIONS)
    for stop_model, stop_group in opportunities.groupby("stop_model", sort=True):
        for feature in active_features:
            for horizon in normalize_horizons(horizons):
                target = f"opportunity_gap_{horizon}_bars_percent"
                ranked = stop_group[[feature, target]].dropna().copy()
                if ranked.empty:
                    continue
                ranks = ranked[feature].rank(method="first")
                ranked["feature_quintile"] = (
                    pd.qcut(ranks, 5, labels=False, duplicates="drop") + 1
                )
                for quintile, group in ranked.groupby(
                    "feature_quintile", sort=True
                ):
                    records.append(
                        {
                            "stop_model": stop_model,
                            "feature": feature,
                            "horizon_bars": horizon,
                            "feature_quintile": int(quintile),
                            "event_count": int(len(group)),
                            "average_feature_value": round(
                                float(group[feature].mean()), 6
                            ),
                            "average_opportunity_gap_percent": round(
                                float(group[target].mean()), 6
                            ),
                            "median_opportunity_gap_percent": round(
                                float(group[target].median()), 6
                            ),
                            "positive_gap_rate_percent": round(
                                float((group[target] > 0).mean() * 100), 4
                            ),
                        }
                    )
    return pd.DataFrame(records)


def build_ticker_leave_one_out(
    opportunities: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active_features = tuple(features or FEATURE_DEFINITIONS)
    for stop_model, stop_group in opportunities.groupby("stop_model", sort=True):
        tickers = sorted(stop_group["candidate_ticker"].dropna().unique())
        for feature in active_features:
            for horizon in normalize_horizons(horizons):
                target = f"opportunity_gap_{horizon}_bars_percent"
                for ticker in tickers:
                    valid = stop_group.loc[
                        stop_group["candidate_ticker"] != ticker,
                        [feature, target],
                    ].dropna()
                    records.append(
                        {
                            "stop_model": stop_model,
                            "feature": feature,
                            "horizon_bars": horizon,
                            "omitted_ticker": ticker,
                            "event_count": int(len(valid)),
                            "spearman_rho": rank_correlation(
                                valid[feature], valid[target]
                            ),
                        }
                    )
    return pd.DataFrame(records)


def build_feature_summary(
    opportunities: pd.DataFrame,
    feature_windows: pd.DataFrame,
    feature_quintiles: pd.DataFrame,
    ticker_leave_one_out: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active_features = tuple(features or FEATURE_DEFINITIONS)
    for stop_model, stop_group in opportunities.groupby("stop_model", sort=True):
        for feature in active_features:
            for horizon in normalize_horizons(horizons):
                target = f"opportunity_gap_{horizon}_bars_percent"
                valid = stop_group[[feature, target]].dropna()
                window_rows = feature_windows.loc[
                    (feature_windows["stop_model"] == stop_model)
                    & (feature_windows["feature"] == feature)
                    & (feature_windows["horizon_bars"] == horizon)
                    & (feature_windows["event_count"] >= 3)
                ]
                quintile_rows = feature_quintiles.loc[
                    (feature_quintiles["stop_model"] == stop_model)
                    & (feature_quintiles["feature"] == feature)
                    & (feature_quintiles["horizon_bars"] == horizon)
                ].sort_values("feature_quintile")
                leave_rows = ticker_leave_one_out.loc[
                    (ticker_leave_one_out["stop_model"] == stop_model)
                    & (ticker_leave_one_out["feature"] == feature)
                    & (ticker_leave_one_out["horizon_bars"] == horizon)
                ]
                bottom = (
                    float(quintile_rows.iloc[0]["average_opportunity_gap_percent"])
                    if len(quintile_rows)
                    else 0.0
                )
                top = (
                    float(quintile_rows.iloc[-1]["average_opportunity_gap_percent"])
                    if len(quintile_rows)
                    else 0.0
                )
                records.append(
                    {
                        "stop_model": stop_model,
                        "feature": feature,
                        "horizon_bars": horizon,
                        "event_count": int(len(valid)),
                        "pooled_spearman_rho": rank_correlation(
                            valid[feature], valid[target]
                        ),
                        "valid_window_count": int(len(window_rows)),
                        "median_window_spearman_rho": round(
                            float(window_rows["spearman_rho"].median()), 6
                        )
                        if len(window_rows)
                        else 0.0,
                        "positive_window_count": int(
                            (window_rows["spearman_rho"] > 0).sum()
                        ),
                        "top_quintile_opportunity_gap_percent": round(top, 6),
                        "bottom_quintile_opportunity_gap_percent": round(
                            bottom, 6
                        ),
                        "top_minus_bottom_quintile_gap_percent": round(
                            top - bottom, 6
                        ),
                        "leave_one_ticker_out_count": int(len(leave_rows)),
                        "minimum_leave_one_ticker_out_rho": round(
                            float(leave_rows["spearman_rho"].min()), 6
                        )
                        if len(leave_rows)
                        else 0.0,
                        "positive_leave_one_ticker_out_count": int(
                            (leave_rows["spearman_rho"] > 0).sum()
                        ),
                    }
                )
    return pd.DataFrame(records)


def build_screen(
    summary: pd.DataFrame,
    *,
    horizon: int = DEFAULT_SCREEN_HORIZON,
    minimum_positive_windows: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = summary.loc[summary["horizon_bars"] == int(horizon)]
    for record in source.to_dict("records"):
        criteria = {
            "pooled_spearman_at_least_0p10": (
                record["pooled_spearman_rho"] >= 0.10
            ),
            "median_window_spearman_at_least_0p05": (
                record["median_window_spearman_rho"] >= 0.05
            ),
            "positive_windows_at_least_8_of_13": (
                record["positive_window_count"] >= minimum_positive_windows
            ),
            "top_minus_bottom_quintile_gap_at_least_1_percent": (
                record["top_minus_bottom_quintile_gap_percent"] >= 1.0
            ),
            "every_leave_one_ticker_out_rho_positive": (
                record["leave_one_ticker_out_count"] > 0
                and record["minimum_leave_one_ticker_out_rho"] > 0
            ),
        }
        rows.append(
            {
                "stop_model": record["stop_model"],
                "feature": record["feature"],
                "horizon_bars": int(horizon),
                **criteria,
                "positive_window_count": record["positive_window_count"],
                "valid_window_count": record["valid_window_count"],
                "pooled_spearman_rho": record["pooled_spearman_rho"],
                "median_window_spearman_rho": record[
                    "median_window_spearman_rho"
                ],
                "top_minus_bottom_quintile_gap_percent": record[
                    "top_minus_bottom_quintile_gap_percent"
                ],
                "minimum_leave_one_ticker_out_rho": record[
                    "minimum_leave_one_ticker_out_rho"
                ],
                "criteria_passed": int(sum(criteria.values())),
                "stratum_pass": bool(all(criteria.values())),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result["robust_feature_pass"] = pd.Series(dtype=bool)
        return result
    robust = result.groupby("feature")["stratum_pass"].transform(
        lambda values: len(values) == len(EXPECTED_STOP_MODELS) and bool(values.all())
    )
    result["robust_feature_pass"] = robust.astype(bool)
    return result.sort_values(["feature", "stop_model"]).reset_index(drop=True)


def _slot_paths(directory: Path, stamp: str) -> dict[str, Path]:
    prefix = f"portfolio_slot_opportunity_audit_"
    return {
        "events": directory / f"{prefix}events_{stamp}.csv",
        "holdings": directory / f"{prefix}holdings_{stamp}.csv",
        "runs": directory / f"{prefix}runs_{stamp}.csv",
        "score_quintiles": directory / f"{prefix}score_quintiles_{stamp}.csv",
        "screen": directory / f"{prefix}screen_{stamp}.csv",
        "summary": directory / f"{prefix}summary_{stamp}.csv",
        "tickers": directory / f"{prefix}tickers_{stamp}.csv",
        "windows": directory / f"{prefix}windows_{stamp}.csv",
        "json": directory / f"{prefix}{stamp}.json",
        "provenance": directory / f"{prefix}provenance_{stamp}.json",
    }


def verify_slot_source(
    *,
    directory: Path,
    stamp: str,
    snapshot_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify every official slot artifact and require its non-pass decision."""

    paths = _slot_paths(Path(directory), stamp)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing slot source files: " + ", ".join(missing))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if provenance.get("audit_stamp") != stamp:
        raise ValueError("Slot provenance stamp mismatch.")
    if provenance.get("snapshot_fingerprint") != snapshot_manifest.get("fingerprint"):
        raise ValueError("Slot source and snapshot fingerprints differ.")
    if payload.get("snapshot_fingerprint") != snapshot_manifest.get("fingerprint"):
        raise ValueError("Slot payload and snapshot fingerprints differ.")
    for name, metadata in provenance.get("result_files", {}).items():
        if name not in paths:
            raise ValueError(f"Unexpected slot result file: {name}")
        actual_path = paths[name]
        if sha256_file(actual_path) != metadata.get("sha256"):
            raise ValueError(f"Slot result hash mismatch: {name}")
    if set(provenance.get("result_files", {})) != set(paths).difference({"provenance"}):
        raise ValueError("Slot provenance does not cover every result artifact.")
    for name, metadata in provenance.get("source_files", {}).items():
        source_path = Path(metadata["path"])
        if not source_path.exists() or sha256_file(source_path) != metadata["sha256"]:
            raise ValueError(f"Slot predecessor hash mismatch: {name}")
    manifest_path = Path(provenance["snapshot_manifest_path"])
    if (
        not manifest_path.exists()
        or sha256_file(manifest_path) != provenance["snapshot_manifest_sha256"]
    ):
        raise ValueError("Slot snapshot manifest hash mismatch.")
    screen = pd.read_csv(paths["screen"])
    if screen.empty or "robust_replacement_audit_pass" not in screen:
        raise ValueError("Slot source has no replacement decision.")
    robust_values = screen["robust_replacement_audit_pass"].astype(str).str.lower()
    if robust_values.isin(("true", "1")).any():
        raise ValueError(
            "Slot replacement audit passed; run replacement ablation instead."
        )
    return {
        "paths": paths,
        "provenance": provenance,
        "payload": payload,
        "events": pd.read_csv(paths["events"]),
    }


def run_entry_score_feature_audit(
    *,
    snapshot_path: Path,
    slot_directory: Path,
    slot_stamp: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    normalized = normalize_horizons(horizons)
    if DEFAULT_SCREEN_HORIZON not in normalized:
        raise ValueError("The pre-registered screen requires the 20-bar horizon.")
    snapshot = load_snapshot(
        Path(snapshot_path), verify_code=True, project_root=Path(project_root)
    )
    source = verify_slot_source(
        directory=Path(slot_directory),
        stamp=slot_stamp,
        snapshot_manifest=snapshot["manifest"],
    )
    opportunities = extract_opportunities(
        source["events"], snapshot["data_by_ticker"], horizons=normalized
    )
    actual_models = tuple(sorted(opportunities["stop_model"].unique()))
    if actual_models != tuple(sorted(EXPECTED_STOP_MODELS)):
        raise ValueError(f"Unexpected stop strata: {actual_models}")
    feature_windows = build_feature_windows(
        opportunities, horizons=normalized
    )
    feature_quintiles = build_feature_quintiles(
        opportunities, horizons=normalized
    )
    leave_one_out = build_ticker_leave_one_out(
        opportunities, horizons=normalized
    )
    summary = build_feature_summary(
        opportunities,
        feature_windows,
        feature_quintiles,
        leave_one_out,
        horizons=normalized,
    )
    screen = build_screen(summary)
    return {
        "opportunities": opportunities,
        "feature_windows": feature_windows,
        "feature_quintiles": feature_quintiles,
        "ticker_leave_one_out": leave_one_out,
        "summary": summary,
        "screen": screen,
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint": snapshot["manifest"]["fingerprint"],
        "snapshot_manifest_path": Path(snapshot_path) / "manifest.json",
        "slot_stamp": slot_stamp,
        "slot_paths": source["paths"],
        "horizons": list(normalized),
        "features": FEATURE_DEFINITIONS,
    }


def save_entry_score_feature_audit(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    prefix = "portfolio_entry_score_feature_audit"
    paths: dict[str, Path] = {}
    for name in (
        "opportunities",
        "feature_windows",
        "feature_quintiles",
        "ticker_leave_one_out",
        "summary",
        "screen",
    ):
        path = output / f"{prefix}_{name}_{stamp}.csv"
        bundle[name].to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    robust_features = sorted(
        bundle["screen"].loc[
            bundle["screen"]["robust_feature_pass"], "feature"
        ].unique()
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Causal diagnostic ranking audit on verified MAX_OPEN_POSITIONS "
            "events; no portfolio decisions are changed."
        ),
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "slot_stamp": bundle["slot_stamp"],
        "horizons": bundle["horizons"],
        "feature_definitions": bundle["features"],
        "screen_horizon_bars": DEFAULT_SCREEN_HORIZON,
        "robust_features_authorized_for_separate_ablation": robust_features,
        "any_robust_feature_pass": bool(robust_features),
        "summary": _safe(bundle["summary"].to_dict("records")),
        "screen": _safe(bundle["screen"].to_dict("records")),
        "limitations": [
            "This is a diagnostic ranking audit, not executable replacement PnL.",
            "Overlapping forward horizons make events statistically dependent.",
            "No feature combination or weight is fitted in this stage.",
            "Only a robust passing feature may enter a separately specified ablation.",
        ],
    }
    json_path = output / f"{prefix}_{stamp}.json"
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    paths["json"] = json_path
    audit_code_path = Path(__file__).resolve()
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
        "slot_stamp": bundle["slot_stamp"],
        "audit_code": {
            "path": str(audit_code_path),
            "sha256": sha256_file(audit_code_path),
        },
        "source_files": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in bundle["slot_paths"].items()
        },
        "result_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    provenance_path = output / f"{prefix}_provenance_{stamp}.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["provenance"] = provenance_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--slot-source-directory", type=Path, default=DEFAULT_SLOT_DIRECTORY
    )
    parser.add_argument("--slot-stamp", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = run_entry_score_feature_audit(
        snapshot_path=args.snapshot,
        slot_directory=args.slot_source_directory,
        slot_stamp=args.slot_stamp,
        horizons=args.horizons,
        project_root=Path("."),
    )
    print(bundle["screen"].to_string(index=False))
    if args.no_save:
        print("NO_SAVE")
        return
    paths = save_entry_score_feature_audit(
        bundle, output_directory=args.output_directory
    )
    for name, path in paths.items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
