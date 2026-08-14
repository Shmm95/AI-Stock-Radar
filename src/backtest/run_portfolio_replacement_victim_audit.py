"""Audit causal victim-selection rules for portfolio replacement research.

The audit joins verified Slot Opportunity Audit holdings with the authorized
RSI_QUALITY candidate feature.  At every full-portfolio event it asks which
currently held position a pre-registered causal rule would select and compares
the candidate's later return with that holding's later return.

No position is replaced and no trading rule is changed.  Historical research
only; no broker integration is present.
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
    "data/backtests/portfolio/replacement_victim_audit"
)
DEFAULT_SLOT_DIRECTORY = Path(
    "data/backtests/portfolio/slot_opportunity_audit"
)
DEFAULT_FEATURE_DIRECTORY = Path(
    "data/backtests/portfolio/entry_score_feature_audit"
)
DEFAULT_HORIZONS = (5, 10, 20)
SCREEN_HORIZON = 20
EXPECTED_STOP_MODELS = ("FIXED_BASELINE", "FIXED_MAX_RETURN")
AUTHORIZED_CANDIDATE_FEATURE = "RSI_QUALITY"

# Direction is the side of the held-position variable selected as the victim.
RULE_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "LOWEST_CURRENT_RSI_QUALITY": (
        "held_current_rsi_quality",
        "min",
        "Held position with the lowest causal RSI quality.",
    ),
    "LOWEST_HELD_ENTRY_SCORE": (
        "held_entry_score",
        "min",
        "Held position with the lowest entry-time portfolio score.",
    ),
    "OLDEST_POSITION": (
        "holding_age_bars",
        "max",
        "Held position with the greatest causal holding age.",
    ),
    "WORST_UNREALIZED_RETURN": (
        "held_unrealized_return_percent",
        "min",
        "Held position with the lowest open-to-current-open unrealized return.",
    ),
    "LOWEST_CURRENT_TREND": (
        "held_current_trend_spread_percent",
        "min",
        "Held position with the lowest causal EMA20/EMA50 trend spread.",
    ),
    "LOWEST_CURRENT_EXTENSION": (
        "held_current_price_extension_percent",
        "min",
        "Held position with the lowest causal Close/EMA20 extension.",
    ),
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


def _prior_position(data: pd.DataFrame, execution_timestamp: Any) -> int:
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    timestamp = pd.Timestamp(execution_timestamp)
    if index.tz is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(index.tz)
    elif index.tz is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    position = int(index.searchsorted(timestamp, side="left")) - 1
    if position < 0:
        raise ValueError(f"No causal held-position bar before {execution_timestamp}.")
    return position


def held_current_features(
    data: pd.DataFrame, execution_timestamp: Any
) -> dict[str, Any]:
    """Calculate held-state features using the last close before execution."""

    position = _prior_position(data, execution_timestamp)
    row = data.iloc[position]
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    rsi = float(row["RSI14"])
    return {
        "held_state_timestamp": pd.Timestamp(data.index[position]),
        "held_current_rsi14": rsi,
        "held_current_rsi_quality": max(
            0.0, 15.0 - abs(rsi - 57.5) * 1.2
        ),
        "held_current_trend_spread_percent": (ema20 / ema50 - 1) * 100,
        "held_current_price_extension_percent": (close / ema20 - 1) * 100,
    }


def build_pairs(
    *,
    events: pd.DataFrame,
    holdings: pd.DataFrame,
    opportunities: pd.DataFrame,
    data_by_ticker: dict[str, pd.DataFrame],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Create one causal candidate-versus-held row per contemporaneous holding."""

    normalized = normalize_horizons(horizons)
    event_required = {
        "event_id",
        "window_id",
        "stop_model",
        "timestamp",
        "candidate_ticker",
        "candidate_asset_class",
        "candidate_score",
    }
    for horizon in normalized:
        event_required.update(
            {
                f"candidate_forward_return_{horizon}_bars_percent",
                f"candidate_minus_mean_held_{horizon}_bars_percent",
            }
        )
    holding_required = {
        "event_id",
        "held_ticker",
        "held_asset_class",
        "held_score",
        "holding_age_bars",
        "held_unrealized_return_percent",
    }
    for horizon in normalized:
        holding_required.add(f"forward_return_{horizon}_bars_percent")
    opportunity_required = {"event_id", AUTHORIZED_CANDIDATE_FEATURE}
    for label, frame, required in (
        ("events", events, event_required),
        ("holdings", holdings, holding_required),
        ("opportunities", opportunities, opportunity_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("Slot events contain duplicate event_id values.")
    if opportunities["event_id"].duplicated().any():
        raise ValueError("Feature opportunities contain duplicate event_id values.")
    event_map = events.set_index("event_id")
    feature_map = opportunities.set_index("event_id")[AUTHORIZED_CANDIDATE_FEATURE]
    if set(event_map.index) != set(feature_map.index):
        raise ValueError("Slot events and feature opportunities do not align.")
    records: list[dict[str, Any]] = []
    for holding in holdings.to_dict("records"):
        event_id = holding["event_id"]
        if event_id not in event_map.index:
            raise ValueError(f"Unknown holding event_id: {event_id}")
        event = event_map.loc[event_id]
        ticker = str(holding["held_ticker"])
        if ticker not in data_by_ticker:
            raise ValueError(f"Snapshot missing held ticker: {ticker}")
        current = held_current_features(
            data_by_ticker[ticker], event["timestamp"]
        )
        record = {
            "event_id": event_id,
            "window_id": event["window_id"],
            "stop_model": event["stop_model"],
            "timestamp": event["timestamp"],
            "candidate_ticker": event["candidate_ticker"],
            "candidate_asset_class": event["candidate_asset_class"],
            "candidate_score": float(event["candidate_score"]),
            "candidate_rsi_quality": float(feature_map.loc[event_id]),
            "held_ticker": ticker,
            "held_asset_class": holding["held_asset_class"],
            "held_entry_score": float(holding["held_score"]),
            "holding_age_bars": holding["holding_age_bars"],
            "held_unrealized_return_percent": float(
                holding["held_unrealized_return_percent"]
            ),
            **current,
        }
        record["candidate_minus_held_current_rsi_quality"] = (
            record["candidate_rsi_quality"]
            - record["held_current_rsi_quality"]
        )
        for horizon in normalized:
            candidate = event[
                f"candidate_forward_return_{horizon}_bars_percent"
            ]
            held = holding[f"forward_return_{horizon}_bars_percent"]
            record[f"candidate_forward_return_{horizon}_bars_percent"] = candidate
            record[f"held_forward_return_{horizon}_bars_percent"] = held
            record[f"candidate_minus_held_{horizon}_bars_percent"] = (
                float(candidate) - float(held)
                if pd.notna(candidate) and pd.notna(held)
                else None
            )
            record[f"candidate_minus_mean_held_{horizon}_bars_percent"] = event[
                f"candidate_minus_mean_held_{horizon}_bars_percent"
            ]
        records.append(record)
    pairs = pd.DataFrame(records)
    if pairs.empty:
        raise ValueError("No candidate/holding pairs were created.")
    return pairs


def select_victims(
    pairs: pd.DataFrame,
    *,
    rules: dict[str, tuple[str, str, str]] = RULE_DEFINITIONS,
) -> pd.DataFrame:
    """Apply deterministic victim rules; ticker breaks exact-value ties."""

    records: list[dict[str, Any]] = []
    for event_id, event_rows in pairs.groupby("event_id", sort=False):
        for rule, (column, direction, _) in rules.items():
            if column not in event_rows:
                raise ValueError(f"Victim column missing: {column}")
            ranked = event_rows.dropna(subset=[column]).sort_values(
                [column, "held_ticker"],
                ascending=[direction == "min", True],
                kind="mergesort",
            )
            if ranked.empty:
                continue
            selected = ranked.iloc[0].to_dict()
            selected["victim_rule"] = rule
            selected["victim_rule_value"] = selected[column]
            records.append(selected)
    return pd.DataFrame(records)


def top_tail_trimmed_mean(values: pd.Series, fraction: float = 0.05) -> float:
    """Mean after removing the largest positive fraction; protects from right tails."""

    valid = values.dropna().astype(float).sort_values().reset_index(drop=True)
    if valid.empty:
        return 0.0
    remove = max(1, int(len(valid) * fraction)) if len(valid) >= 20 else 0
    trimmed = valid.iloc[:-remove] if remove else valid
    return round(float(trimmed.mean()), 6) if len(trimmed) else 0.0


def build_rule_windows(
    selections: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stop, rule, window), group in selections.groupby(
        ["stop_model", "victim_rule", "window_id"], sort=True
    ):
        for horizon in normalize_horizons(horizons):
            target = f"candidate_minus_held_{horizon}_bars_percent"
            valid = group[target].dropna().astype(float)
            rows.append(
                {
                    "stop_model": stop,
                    "victim_rule": rule,
                    "window_id": window,
                    "horizon_bars": horizon,
                    "event_count": int(len(valid)),
                    "average_candidate_minus_victim_percent": round(
                        float(valid.mean()), 6
                    )
                    if len(valid)
                    else 0.0,
                    "median_candidate_minus_victim_percent": round(
                        float(valid.median()), 6
                    )
                    if len(valid)
                    else 0.0,
                    "candidate_beat_victim_rate_percent": round(
                        float((valid > 0).mean() * 100), 4
                    )
                    if len(valid)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_ticker_leave_one_out(
    selections: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stop, rule), group in selections.groupby(
        ["stop_model", "victim_rule"], sort=True
    ):
        for horizon in normalize_horizons(horizons):
            target = f"candidate_minus_held_{horizon}_bars_percent"
            for ticker in sorted(group["candidate_ticker"].dropna().unique()):
                valid = group.loc[
                    group["candidate_ticker"] != ticker, target
                ].dropna().astype(float)
                rows.append(
                    {
                        "stop_model": stop,
                        "victim_rule": rule,
                        "horizon_bars": horizon,
                        "omitted_candidate_ticker": ticker,
                        "event_count": int(len(valid)),
                        "average_candidate_minus_victim_percent": round(
                            float(valid.mean()), 6
                        )
                        if len(valid)
                        else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def build_rule_summary(
    selections: pd.DataFrame,
    windows: pd.DataFrame,
    ticker_leave_one_out: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stop, rule), group in selections.groupby(
        ["stop_model", "victim_rule"], sort=True
    ):
        for horizon in normalize_horizons(horizons):
            target = f"candidate_minus_held_{horizon}_bars_percent"
            mean_target = f"candidate_minus_mean_held_{horizon}_bars_percent"
            valid = group.dropna(subset=[target, mean_target]).copy()
            window_rows = windows.loc[
                (windows["stop_model"] == stop)
                & (windows["victim_rule"] == rule)
                & (windows["horizon_bars"] == horizon)
                & (windows["event_count"] > 0)
            ]
            leave_rows = ticker_leave_one_out.loc[
                (ticker_leave_one_out["stop_model"] == stop)
                & (ticker_leave_one_out["victim_rule"] == rule)
                & (ticker_leave_one_out["horizon_bars"] == horizon)
            ]
            values = valid[target].astype(float)
            rows.append(
                {
                    "stop_model": stop,
                    "victim_rule": rule,
                    "horizon_bars": horizon,
                    "event_count": int(len(valid)),
                    "average_candidate_minus_victim_percent": round(
                        float(values.mean()), 6
                    )
                    if len(values)
                    else 0.0,
                    "median_candidate_minus_victim_percent": round(
                        float(values.median()), 6
                    )
                    if len(values)
                    else 0.0,
                    "candidate_beat_victim_rate_percent": round(
                        float((values > 0).mean() * 100), 4
                    )
                    if len(values)
                    else 0.0,
                    "top_5_percent_trimmed_average_gap_percent": (
                        top_tail_trimmed_mean(values)
                    ),
                    "victim_selection_advantage_over_mean_percent": round(
                        float((valid[target] - valid[mean_target]).mean()), 6
                    )
                    if len(valid)
                    else 0.0,
                    "valid_window_count": int(len(window_rows)),
                    "positive_window_count": int(
                        (
                            window_rows[
                                "average_candidate_minus_victim_percent"
                            ]
                            > 0
                        ).sum()
                    ),
                    "leave_one_ticker_out_count": int(len(leave_rows)),
                    "minimum_leave_one_ticker_out_average_gap_percent": round(
                        float(
                            leave_rows[
                                "average_candidate_minus_victim_percent"
                            ].min()
                        ),
                        6,
                    )
                    if len(leave_rows)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_screen(
    summary: pd.DataFrame,
    *,
    horizon: int = SCREEN_HORIZON,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in summary.loc[
        summary["horizon_bars"] == int(horizon)
    ].to_dict("records"):
        criteria = {
            "positive_average_gap": (
                record["average_candidate_minus_victim_percent"] > 0
            ),
            "positive_median_gap": (
                record["median_candidate_minus_victim_percent"] > 0
            ),
            "candidate_beats_victim_at_least_55_percent": (
                record["candidate_beat_victim_rate_percent"] >= 55
            ),
            "positive_windows_at_least_8": (
                record["positive_window_count"] >= 8
            ),
            "positive_victim_selection_advantage_over_mean": (
                record["victim_selection_advantage_over_mean_percent"] > 0
            ),
            "positive_top_5_percent_trimmed_average": (
                record["top_5_percent_trimmed_average_gap_percent"] > 0
            ),
            "every_ticker_leave_one_out_average_positive": (
                record["leave_one_ticker_out_count"] > 0
                and record[
                    "minimum_leave_one_ticker_out_average_gap_percent"
                ]
                > 0
            ),
        }
        rows.append(
            {
                "stop_model": record["stop_model"],
                "victim_rule": record["victim_rule"],
                "horizon_bars": int(horizon),
                **criteria,
                "average_candidate_minus_victim_percent": record[
                    "average_candidate_minus_victim_percent"
                ],
                "median_candidate_minus_victim_percent": record[
                    "median_candidate_minus_victim_percent"
                ],
                "candidate_beat_victim_rate_percent": record[
                    "candidate_beat_victim_rate_percent"
                ],
                "positive_window_count": record["positive_window_count"],
                "valid_window_count": record["valid_window_count"],
                "top_5_percent_trimmed_average_gap_percent": record[
                    "top_5_percent_trimmed_average_gap_percent"
                ],
                "victim_selection_advantage_over_mean_percent": record[
                    "victim_selection_advantage_over_mean_percent"
                ],
                "minimum_leave_one_ticker_out_average_gap_percent": record[
                    "minimum_leave_one_ticker_out_average_gap_percent"
                ],
                "criteria_passed": int(sum(criteria.values())),
                "stratum_pass": bool(all(criteria.values())),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result["robust_victim_rule_pass"] = pd.Series(dtype=bool)
        return result
    robust = result.groupby("victim_rule")["stratum_pass"].transform(
        lambda values: len(values) == len(EXPECTED_STOP_MODELS) and bool(values.all())
    )
    result["robust_victim_rule_pass"] = robust.astype(bool)
    return result.sort_values(["victim_rule", "stop_model"]).reset_index(drop=True)


def _result_paths(directory: Path, prefix: str, stamp: str) -> dict[str, Path]:
    names = (
        "events",
        "holdings",
        "runs",
        "score_quintiles",
        "screen",
        "summary",
        "tickers",
        "windows",
    )
    paths = {
        name: directory / f"{prefix}_{name}_{stamp}.csv" for name in names
    }
    paths["json"] = directory / f"{prefix}_{stamp}.json"
    paths["provenance"] = directory / f"{prefix}_provenance_{stamp}.json"
    return paths


def _feature_paths(directory: Path, stamp: str) -> dict[str, Path]:
    prefix = "portfolio_entry_score_feature_audit"
    names = (
        "opportunities",
        "feature_quintiles",
        "feature_windows",
        "ticker_leave_one_out",
        "summary",
        "screen",
    )
    paths = {
        name: directory / f"{prefix}_{name}_{stamp}.csv" for name in names
    }
    paths["json"] = directory / f"{prefix}_{stamp}.json"
    paths["provenance"] = directory / f"{prefix}_provenance_{stamp}.json"
    return paths


def _verify_hashed_source(
    *,
    paths: dict[str, Path],
    expected_stamp: str,
    snapshot_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if provenance.get("audit_stamp") != expected_stamp:
        raise ValueError("Source provenance stamp mismatch.")
    if provenance.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise ValueError("Source provenance snapshot mismatch.")
    if payload.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise ValueError("Source payload snapshot mismatch.")
    audit_code = provenance.get("audit_code")
    if not audit_code:
        raise ValueError("Source provenance has no audit-code hash.")
    audit_path = Path(audit_code["path"])
    if (
        not audit_path.exists()
        or sha256_file(audit_path) != audit_code.get("sha256")
    ):
        raise ValueError("Source audit-code hash mismatch.")
    result_records = provenance.get("result_files", {})
    if set(result_records) != set(paths).difference({"provenance"}):
        raise ValueError("Source provenance does not cover every result file.")
    for name, metadata in result_records.items():
        if sha256_file(paths[name]) != metadata.get("sha256"):
            raise ValueError(f"Source result hash mismatch: {name}")
    for name, metadata in provenance.get("source_files", {}).items():
        source = Path(metadata["path"])
        if not source.exists() or sha256_file(source) != metadata.get("sha256"):
            raise ValueError(f"Source predecessor hash mismatch: {name}")
    manifest = Path(provenance["snapshot_manifest_path"])
    if (
        not manifest.exists()
        or sha256_file(manifest) != provenance["snapshot_manifest_sha256"]
    ):
        raise ValueError("Source snapshot manifest hash mismatch.")
    return provenance, payload


def verify_sources(
    *,
    slot_directory: Path,
    slot_stamp: str,
    feature_directory: Path,
    feature_stamp: str,
    snapshot_fingerprint: str,
) -> dict[str, Any]:
    slot_paths = _result_paths(
        Path(slot_directory), "portfolio_slot_opportunity_audit", slot_stamp
    )
    slot_provenance, slot_payload = _verify_hashed_source(
        paths=slot_paths,
        expected_stamp=slot_stamp,
        snapshot_fingerprint=snapshot_fingerprint,
    )
    feature_paths = _feature_paths(Path(feature_directory), feature_stamp)
    feature_provenance, feature_payload = _verify_hashed_source(
        paths=feature_paths,
        expected_stamp=feature_stamp,
        snapshot_fingerprint=snapshot_fingerprint,
    )
    if feature_provenance.get("slot_stamp") != slot_stamp:
        raise ValueError("Feature audit references a different slot stamp.")
    authorized = feature_payload.get(
        "robust_features_authorized_for_separate_ablation", []
    )
    if authorized != [AUTHORIZED_CANDIDATE_FEATURE]:
        raise ValueError(
            "Victim audit requires RSI_QUALITY as the sole robust feature."
        )
    feature_screen = pd.read_csv(feature_paths["screen"])
    passed = feature_screen.loc[
        feature_screen["robust_feature_pass"].astype(str).str.lower().isin(
            ("true", "1")
        ),
        "feature",
    ].unique()
    if list(passed) != [AUTHORIZED_CANDIDATE_FEATURE]:
        raise ValueError("Feature screen authorization mismatch.")
    return {
        "slot_paths": slot_paths,
        "feature_paths": feature_paths,
        "slot_payload": slot_payload,
        "feature_payload": feature_payload,
        "events": pd.read_csv(slot_paths["events"]),
        "holdings": pd.read_csv(slot_paths["holdings"]),
        "opportunities": pd.read_csv(feature_paths["opportunities"]),
    }


def run_replacement_victim_audit(
    *,
    snapshot_path: Path,
    slot_directory: Path,
    slot_stamp: str,
    feature_directory: Path,
    feature_stamp: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    normalized = normalize_horizons(horizons)
    if SCREEN_HORIZON not in normalized:
        raise ValueError("The pre-registered screen requires the 20-bar horizon.")
    snapshot = load_snapshot(
        Path(snapshot_path), verify_code=True, project_root=Path(project_root)
    )
    source = verify_sources(
        slot_directory=Path(slot_directory),
        slot_stamp=slot_stamp,
        feature_directory=Path(feature_directory),
        feature_stamp=feature_stamp,
        snapshot_fingerprint=snapshot["manifest"]["fingerprint"],
    )
    pairs = build_pairs(
        events=source["events"],
        holdings=source["holdings"],
        opportunities=source["opportunities"],
        data_by_ticker=snapshot["data_by_ticker"],
        horizons=normalized,
    )
    selections = select_victims(pairs)
    actual_models = tuple(sorted(selections["stop_model"].unique()))
    if actual_models != tuple(sorted(EXPECTED_STOP_MODELS)):
        raise ValueError(f"Unexpected stop strata: {actual_models}")
    windows = build_rule_windows(selections, horizons=normalized)
    ticker_leave_one_out = build_ticker_leave_one_out(
        selections, horizons=normalized
    )
    summary = build_rule_summary(
        selections,
        windows,
        ticker_leave_one_out,
        horizons=normalized,
    )
    screen = build_screen(summary)
    return {
        "pairs": pairs,
        "selections": selections,
        "rule_windows": windows,
        "ticker_leave_one_out": ticker_leave_one_out,
        "summary": summary,
        "screen": screen,
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint": snapshot["manifest"]["fingerprint"],
        "snapshot_manifest_path": Path(snapshot_path) / "manifest.json",
        "slot_stamp": slot_stamp,
        "feature_stamp": feature_stamp,
        "source_paths": {
            **{f"slot_{name}": path for name, path in source["slot_paths"].items()},
            **{
                f"feature_{name}": path
                for name, path in source["feature_paths"].items()
            },
        },
        "horizons": list(normalized),
    }


def save_replacement_victim_audit(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    prefix = "portfolio_replacement_victim_audit"
    paths: dict[str, Path] = {}
    for name in (
        "pairs",
        "selections",
        "rule_windows",
        "ticker_leave_one_out",
        "summary",
        "screen",
    ):
        path = output / f"{prefix}_{name}_{stamp}.csv"
        bundle[name].to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    robust_rules = sorted(
        bundle["screen"].loc[
            bundle["screen"]["robust_victim_rule_pass"], "victim_rule"
        ].unique()
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Causal, decision-preserving victim-selection audit on verified "
            "full-portfolio events; no replacement is executed."
        ),
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "slot_stamp": bundle["slot_stamp"],
        "feature_stamp": bundle["feature_stamp"],
        "authorized_candidate_feature": AUTHORIZED_CANDIDATE_FEATURE,
        "rule_definitions": {
            name: {"column": value[0], "direction": value[1], "description": value[2]}
            for name, value in RULE_DEFINITIONS.items()
        },
        "horizons": bundle["horizons"],
        "screen_horizon_bars": SCREEN_HORIZON,
        "robust_victim_rules_authorized_for_separate_ablation": robust_rules,
        "any_robust_victim_rule_pass": bool(robust_rules),
        "summary": _safe(bundle["summary"].to_dict("records")),
        "screen": _safe(bundle["screen"].to_dict("records")),
        "limitations": [
            "This is a diagnostic counterfactual audit, not executable swap PnL.",
            "Forward horizons overlap and repeated events are dependent.",
            "Six pre-registered victim rules are compared without fitting weights.",
            "A robust rule only authorizes a separately specified replacement ablation.",
            "The RSI feature was discovered on the same historical snapshot; future shadow validation remains required.",
        ],
    }
    json_path = output / f"{prefix}_{stamp}.json"
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    paths["json"] = json_path
    code_path = Path(__file__).resolve()
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
        "feature_stamp": bundle["feature_stamp"],
        "audit_code": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
        },
        "source_files": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in bundle["source_paths"].items()
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
    parser.add_argument(
        "--feature-source-directory",
        type=Path,
        default=DEFAULT_FEATURE_DIRECTORY,
    )
    parser.add_argument("--feature-stamp", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = run_replacement_victim_audit(
        snapshot_path=args.snapshot,
        slot_directory=args.slot_source_directory,
        slot_stamp=args.slot_stamp,
        feature_directory=args.feature_source_directory,
        feature_stamp=args.feature_stamp,
        horizons=args.horizons,
        project_root=Path("."),
    )
    print(bundle["screen"].to_string(index=False))
    if args.no_save:
        print("NO_SAVE")
        return
    paths = save_replacement_victim_audit(
        bundle, output_directory=args.output_directory
    )
    for name, path in paths.items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
