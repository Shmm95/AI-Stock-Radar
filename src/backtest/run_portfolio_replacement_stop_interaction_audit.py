"""Observational audit of RSI-replacement interaction with fixed stop strata.

The audit consumes the provenance-locked replacement walk-forward outputs.  It
does not rerun or modify the portfolio engine, tune thresholds, or authorize a
conditional strategy.  Its purpose is to explain why the global replacement
hypothesis passed the 5%/5% stratum but failed the 3.5%/3.5% stratum.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest.run_research_data_snapshot import sha256_file


DEFAULT_SOURCE_DIRECTORY = Path(
    "data/backtests/portfolio/rsi_replacement_walk_forward"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/replacement_stop_interaction_audit"
)
SOURCE_PREFIX = "portfolio_rsi_replacement_walk_forward"
OUTPUT_PREFIX = "portfolio_replacement_stop_interaction_audit"

CONTROL_POLICY = "CONTROL_NO_REPLACEMENT"
REPLACEMENT_POLICY = "RSI_Q12_LOSER_ONLY"
MODEL_DYNAMIC = "DYNAMIC_TRAIN_SELECTED"
MODEL_CONTROL = "FIXED_CONTROL"
MODEL_REPLACEMENT = "FIXED_RSI_Q12_LOSER_ONLY"
MODELS = (MODEL_DYNAMIC, MODEL_CONTROL, MODEL_REPLACEMENT)
STOP_BASELINE = "FIXED_BASELINE"
STOP_MAX_RETURN = "FIXED_MAX_RETURN"
STOP_MODELS = (STOP_BASELINE, STOP_MAX_RETURN)

SOURCE_FRAME_NAMES = (
    "windows",
    "training",
    "test_runs",
    "aggregate",
    "screen",
    "events",
    "tickers",
    "rejections",
    "trades",
    "equity",
)
OUTPUT_FRAME_NAMES = (
    "summary",
    "windows",
    "events",
    "event_pairs",
    "trade_pairs",
    "tickers",
    "exit_reasons",
    "rejections",
    "leave_one_window_out",
    "screen",
)


def _safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(("true", "1"))


def _source_paths(directory: Path, stamp: str) -> dict[str, Path]:
    directory = Path(directory)
    paths = {
        name: directory / f"{SOURCE_PREFIX}_{name}_{stamp}.csv"
        for name in SOURCE_FRAME_NAMES
    }
    paths["json"] = directory / f"{SOURCE_PREFIX}_{stamp}.json"
    paths["provenance"] = (
        directory / f"{SOURCE_PREFIX}_provenance_{stamp}.json"
    )
    return paths


def validate_source_decision(
    payload: Mapping[str, Any], screen: pd.DataFrame
) -> None:
    if payload.get("authorized_policy") != REPLACEMENT_POLICY:
        raise ValueError("Unexpected source replacement policy.")
    if not bool(payload.get("complete_window_set")):
        raise ValueError("Source walk-forward did not use the complete window set.")
    if bool(payload.get("robust_walk_forward_pass")):
        raise ValueError("Source global policy passed; failure audit is inapplicable.")
    if bool(payload.get("production_authorized")):
        raise ValueError("Source unexpectedly authorized production.")
    required = {
        "stop_model",
        "stratum_pass",
        "robust_walk_forward_pass",
        "production_authorized",
    }
    missing = sorted(required.difference(screen.columns))
    if missing:
        raise ValueError(f"Source screen columns are missing: {missing}")
    if sorted(screen["stop_model"].astype(str).unique()) != sorted(STOP_MODELS):
        raise ValueError("Source screen stop-stratum coverage mismatch.")
    if _truthy(screen["robust_walk_forward_pass"]).any():
        raise ValueError("Source screen robust decision mismatch.")
    if _truthy(screen["production_authorized"]).any():
        raise ValueError("Source screen production decision mismatch.")
    strata = screen.set_index("stop_model")["stratum_pass"]
    if not bool(_truthy(pd.Series([strata.loc[STOP_BASELINE]])).iloc[0]):
        raise ValueError("Expected the 5%/5% source stratum to pass.")
    if bool(_truthy(pd.Series([strata.loc[STOP_MAX_RETURN]])).iloc[0]):
        raise ValueError("Expected the 3.5%/3.5% source stratum to fail.")


def verify_source(
    *, directory: Path, stamp: str
) -> dict[str, Any]:
    paths = _source_paths(Path(directory), stamp)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing replacement walk-forward files: " + ", ".join(missing)
        )
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if provenance.get("audit_stamp") != stamp:
        raise ValueError("Source provenance stamp mismatch.")
    expected = set(paths).difference({"provenance"})
    recorded = provenance.get("result_files", {})
    if set(recorded) != expected:
        raise ValueError("Source provenance does not cover every result file.")
    for name in sorted(expected):
        if sha256_file(paths[name]) != recorded[name].get("sha256"):
            raise ValueError(f"Source result hash mismatch: {name}")
    code = provenance.get("audit_code")
    if not code:
        raise ValueError("Source provenance has no audit-code hash.")
    code_path = Path(code["path"])
    if not code_path.exists() or sha256_file(code_path) != code.get("sha256"):
        raise ValueError("Source audit-code hash mismatch.")
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
    screen = pd.read_csv(paths["screen"])
    validate_source_decision(payload, screen)
    return {
        "paths": paths,
        "provenance": provenance,
        "payload": payload,
    }


def _compound(returns: Sequence[float]) -> float:
    value = 1.0
    for item in returns:
        value *= 1 + float(item) / 100
    return round((value - 1) * 100, 6)


def build_leave_one_window_out(test_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stop_model, group in test_runs.groupby("stop_model", sort=True):
        pivot = group.pivot(
            index="window_id", columns="model", values="total_return_percent"
        ).sort_index()
        if set(pivot.columns) != set(MODELS):
            raise ValueError(f"Test model coverage mismatch for {stop_model}.")
        for model in (MODEL_DYNAMIC, MODEL_REPLACEMENT):
            full_candidate = _compound(pivot[model])
            full_control = _compound(pivot[MODEL_CONTROL])
            full_delta = full_candidate - full_control
            for excluded in pivot.index:
                reduced = pivot.drop(index=excluded)
                candidate = _compound(reduced[model])
                control = _compound(reduced[MODEL_CONTROL])
                delta = candidate - control
                rows.append(
                    {
                        "stop_model": stop_model,
                        "model": model,
                        "excluded_window_id": excluded,
                        "candidate_compounded_return_percent": round(candidate, 4),
                        "control_compounded_return_percent": round(control, 4),
                        "return_delta_vs_control_percent": round(delta, 4),
                        "full_return_delta_vs_control_percent": round(
                            full_delta, 4
                        ),
                        "delta_change_vs_full_percent": round(
                            delta - full_delta, 4
                        ),
                        "positive_after_exclusion": delta > 1e-9,
                    }
                )
    return pd.DataFrame(rows)


def build_ticker_deltas(tickers: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        tickers.groupby(["stop_model", "model", "ticker"], as_index=False)
        .agg(
            asset_class=("asset_class", "first"),
            completed_trades=("completed_trades", "sum"),
            scaled_net_pnl=("scaled_net_pnl", "sum"),
            scaled_gross_profit=("scaled_gross_profit", "sum"),
            scaled_gross_loss=("scaled_gross_loss", "sum"),
            scaled_total_fees=("scaled_total_fees", "sum"),
        )
    )
    rows: list[dict[str, Any]] = []
    for stop_model, stop_group in grouped.groupby("stop_model", sort=True):
        control = stop_group.loc[stop_group["model"] == MODEL_CONTROL].set_index(
            "ticker"
        )
        for model in (MODEL_DYNAMIC, MODEL_REPLACEMENT):
            candidate = stop_group.loc[stop_group["model"] == model].set_index(
                "ticker"
            )
            tickers_all = sorted(control.index.union(candidate.index))
            for ticker in tickers_all:
                c = control.loc[ticker] if ticker in control.index else None
                m = candidate.loc[ticker] if ticker in candidate.index else None
                c_pnl = float(c["scaled_net_pnl"]) if c is not None else 0.0
                m_pnl = float(m["scaled_net_pnl"]) if m is not None else 0.0
                rows.append(
                    {
                        "stop_model": stop_model,
                        "model": model,
                        "ticker": ticker,
                        "asset_class": (
                            str(m["asset_class"])
                            if m is not None
                            else str(c["asset_class"])
                        ),
                        "control_scaled_net_pnl": round(c_pnl, 6),
                        "model_scaled_net_pnl": round(m_pnl, 6),
                        "scaled_net_pnl_delta": round(m_pnl - c_pnl, 6),
                        "control_completed_trades": (
                            int(c["completed_trades"]) if c is not None else 0
                        ),
                        "model_completed_trades": (
                            int(m["completed_trades"]) if m is not None else 0
                        ),
                        "completed_trades_delta": (
                            (int(m["completed_trades"]) if m is not None else 0)
                            - (int(c["completed_trades"]) if c is not None else 0)
                        ),
                        "scaled_fee_delta": round(
                            (float(m["scaled_total_fees"]) if m is not None else 0.0)
                            - (float(c["scaled_total_fees"]) if c is not None else 0.0),
                            6,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _single_or_none(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    if len(frame) > 1:
        frame = frame.sort_values(list(frame.columns)).tail(1)
    return frame.iloc[0]


def build_event_outcomes(
    events: pd.DataFrame, trades: pd.DataFrame
) -> pd.DataFrame:
    executed = events.loc[events["status"] == "EXECUTED"].copy()
    rows: list[dict[str, Any]] = []
    for event in executed.to_dict("records"):
        common = (
            (trades["window_id"] == event["window_id"])
            & (trades["stop_model"] == event["stop_model"])
            & (trades["model"] == event["model"])
        )
        candidate = _single_or_none(
            trades.loc[
                common
                & (trades["ticker"] == event["candidate_ticker"])
                & (trades["entry_timestamp"] == event["timestamp"])
            ]
        )
        victim = _single_or_none(
            trades.loc[
                common
                & (trades["ticker"] == event["victim_ticker"])
                & (trades["exit_timestamp"] == event["timestamp"])
                & trades["exit_reason"].astype(str).str.contains(
                    "REPLACEMENT", na=False
                )
            ]
        )
        control_victim = None
        if victim is not None:
            control_victim = _single_or_none(
                trades.loc[
                    (trades["window_id"] == event["window_id"])
                    & (trades["stop_model"] == event["stop_model"])
                    & (trades["model"] == MODEL_CONTROL)
                    & (trades["ticker"] == event["victim_ticker"])
                    & (trades["entry_timestamp"] == victim["entry_timestamp"])
                ]
            )
        candidate_pnl = float(candidate["net_pnl"]) if candidate is not None else None
        victim_pnl = float(victim["net_pnl"]) if victim is not None else None
        control_pnl = (
            float(control_victim["net_pnl"])
            if control_victim is not None
            else None
        )
        local_delta = (
            candidate_pnl + victim_pnl - control_pnl
            if None not in (candidate_pnl, victim_pnl, control_pnl)
            else None
        )
        rows.append(
            {
                **event,
                "candidate_trade_matched": candidate is not None,
                "candidate_exit_timestamp": (
                    candidate["exit_timestamp"] if candidate is not None else None
                ),
                "candidate_exit_reason": (
                    candidate["exit_reason"] if candidate is not None else None
                ),
                "candidate_holding_period_bars": (
                    candidate["holding_period_bars"]
                    if candidate is not None
                    else None
                ),
                "candidate_return_percent": (
                    candidate["return_percent"] if candidate is not None else None
                ),
                "candidate_net_pnl": candidate_pnl,
                "candidate_scaled_net_pnl": (
                    candidate["scaled_net_pnl"] if candidate is not None else None
                ),
                "victim_trade_matched": victim is not None,
                "victim_entry_timestamp": (
                    victim["entry_timestamp"] if victim is not None else None
                ),
                "victim_forced_return_percent": (
                    victim["return_percent"] if victim is not None else None
                ),
                "victim_forced_net_pnl": victim_pnl,
                "victim_forced_scaled_net_pnl": (
                    victim["scaled_net_pnl"] if victim is not None else None
                ),
                "control_victim_continuation_matched": control_victim is not None,
                "control_victim_exit_timestamp": (
                    control_victim["exit_timestamp"]
                    if control_victim is not None
                    else None
                ),
                "control_victim_exit_reason": (
                    control_victim["exit_reason"]
                    if control_victim is not None
                    else None
                ),
                "control_victim_return_percent": (
                    control_victim["return_percent"]
                    if control_victim is not None
                    else None
                ),
                "control_victim_net_pnl": control_pnl,
                "local_pair_delta_vs_control_victim": local_delta,
                "local_pair_delta_is_observational_proxy": True,
            }
        )
    return pd.DataFrame(rows)


def build_event_pairs(event_outcomes: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "window_id", "timestamp", "candidate_ticker"]
    columns = keys + [
        "victim_ticker",
        "victim_unrealized_return_percent",
        "candidate_exit_reason",
        "candidate_return_percent",
        "candidate_net_pnl",
        "victim_forced_net_pnl",
        "control_victim_net_pnl",
        "local_pair_delta_vs_control_victim",
    ]
    baseline = event_outcomes.loc[
        event_outcomes["stop_model"] == STOP_BASELINE, columns
    ].copy()
    maximum = event_outcomes.loc[
        event_outcomes["stop_model"] == STOP_MAX_RETURN, columns
    ].copy()
    paired = baseline.merge(
        maximum,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_max_return"),
        indicator=True,
    )
    paired.insert(0, "event_pair_id", [f"EP{index:04d}" for index in range(1, len(paired) + 1)])
    paired["present_in_both_stops"] = paired["_merge"] == "both"
    paired["same_victim"] = (
        paired["victim_ticker_baseline"]
        == paired["victim_ticker_max_return"]
    ) & paired["present_in_both_stops"]
    paired["candidate_return_delta_max_minus_baseline"] = (
        paired["candidate_return_percent_max_return"]
        - paired["candidate_return_percent_baseline"]
    )
    paired["local_pair_delta_difference_max_minus_baseline"] = (
        paired["local_pair_delta_vs_control_victim_max_return"]
        - paired["local_pair_delta_vs_control_victim_baseline"]
    )
    return paired.drop(columns="_merge")


def build_trade_pairs(trades: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "window_id", "ticker", "entry_timestamp"]
    values = keys + [
        "asset_class",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "exit_reason",
        "holding_period_bars",
        "return_percent",
        "net_pnl",
    ]
    baseline = trades.loc[trades["stop_model"] == STOP_BASELINE, values].drop_duplicates(keys, keep="last")
    maximum = trades.loc[trades["stop_model"] == STOP_MAX_RETURN, values].drop_duplicates(keys, keep="last")
    paired = baseline.merge(
        maximum,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_max_return"),
        indicator=True,
    )
    paired.insert(0, "trade_pair_id", [f"TP{index:05d}" for index in range(1, len(paired) + 1)])
    paired["present_in_both_stops"] = paired["_merge"] == "both"
    paired["same_exit_timestamp"] = (
        paired["exit_timestamp_baseline"] == paired["exit_timestamp_max_return"]
    ) & paired["present_in_both_stops"]
    paired["same_exit_reason"] = (
        paired["exit_reason_baseline"] == paired["exit_reason_max_return"]
    ) & paired["present_in_both_stops"]
    paired["return_delta_max_minus_baseline"] = (
        paired["return_percent_max_return"]
        - paired["return_percent_baseline"]
    )
    paired["net_pnl_delta_max_minus_baseline"] = (
        paired["net_pnl_max_return"] - paired["net_pnl_baseline"]
    )
    return paired.drop(columns="_merge")


def build_exit_reason_effects(trades: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trades.groupby(["stop_model", "model", "exit_reason"], as_index=False)
        .agg(
            trade_count=("ticker", "size"),
            scaled_net_pnl=("scaled_net_pnl", "sum"),
            scaled_total_fees=("scaled_total_fees", "sum"),
        )
    )
    rows: list[dict[str, Any]] = []
    for stop_model, stop_group in grouped.groupby("stop_model", sort=True):
        control = stop_group.loc[stop_group["model"] == MODEL_CONTROL].set_index(
            "exit_reason"
        )
        for model in (MODEL_DYNAMIC, MODEL_REPLACEMENT):
            candidate = stop_group.loc[stop_group["model"] == model].set_index(
                "exit_reason"
            )
            reasons = sorted(control.index.union(candidate.index))
            for reason in reasons:
                c = control.loc[reason] if reason in control.index else None
                m = candidate.loc[reason] if reason in candidate.index else None
                c_pnl = float(c["scaled_net_pnl"]) if c is not None else 0.0
                m_pnl = float(m["scaled_net_pnl"]) if m is not None else 0.0
                rows.append(
                    {
                        "stop_model": stop_model,
                        "model": model,
                        "exit_reason": reason,
                        "control_trade_count": int(c["trade_count"]) if c is not None else 0,
                        "model_trade_count": int(m["trade_count"]) if m is not None else 0,
                        "trade_count_delta": (
                            (int(m["trade_count"]) if m is not None else 0)
                            - (int(c["trade_count"]) if c is not None else 0)
                        ),
                        "control_scaled_net_pnl": round(c_pnl, 6),
                        "model_scaled_net_pnl": round(m_pnl, 6),
                        "scaled_net_pnl_delta": round(m_pnl - c_pnl, 6),
                    }
                )
    return pd.DataFrame(rows)


def build_rejection_effects(rejections: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rejections.groupby(["stop_model", "model", "reason_code"], as_index=False)["count"]
        .sum()
    )
    rows: list[dict[str, Any]] = []
    for stop_model, stop_group in grouped.groupby("stop_model", sort=True):
        control = stop_group.loc[stop_group["model"] == MODEL_CONTROL].set_index(
            "reason_code"
        )
        for model in (MODEL_DYNAMIC, MODEL_REPLACEMENT):
            candidate = stop_group.loc[stop_group["model"] == model].set_index(
                "reason_code"
            )
            reasons = sorted(control.index.union(candidate.index))
            for reason in reasons:
                c_count = int(control.loc[reason, "count"]) if reason in control.index else 0
                m_count = int(candidate.loc[reason, "count"]) if reason in candidate.index else 0
                rows.append(
                    {
                        "stop_model": stop_model,
                        "model": model,
                        "reason_code": reason,
                        "control_count": c_count,
                        "model_count": m_count,
                        "count_delta": m_count - c_count,
                    }
                )
    return pd.DataFrame(rows)


def _comparison_counts(values: pd.Series) -> dict[str, int]:
    numbers = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return {
        "window_win_count": int((numbers > 1e-9).sum()),
        "window_tie_count": int((numbers.abs() <= 1e-9).sum()),
        "window_loss_count": int((numbers < -1e-9).sum()),
    }


def build_summary(
    *,
    aggregate: pd.DataFrame,
    test_runs: pd.DataFrame,
    source_screen: pd.DataFrame,
    event_outcomes: pd.DataFrame,
    ticker_deltas: pd.DataFrame,
    leave_one_window_out: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stop_model, group in aggregate.groupby("stop_model", sort=True):
        by_model = group.set_index("model")
        control = by_model.loc[MODEL_CONTROL]
        tests = test_runs.loc[test_runs["stop_model"] == stop_model]
        pivot = tests.pivot(
            index="window_id", columns="model", values="total_return_percent"
        )
        screen_row = source_screen.loc[
            source_screen["stop_model"] == stop_model
        ].iloc[0]
        for model in MODELS:
            record = by_model.loc[model]
            if model == MODEL_CONTROL:
                counts = {
                    "window_win_count": 0,
                    "window_tie_count": int(len(pivot)),
                    "window_loss_count": 0,
                }
                low_min = low_max = None
                model_tickers = pd.DataFrame()
            else:
                counts = _comparison_counts(pivot[model] - pivot[MODEL_CONTROL])
                low = leave_one_window_out.loc[
                    (leave_one_window_out["stop_model"] == stop_model)
                    & (leave_one_window_out["model"] == model),
                    "return_delta_vs_control_percent",
                ]
                low_min = float(low.min())
                low_max = float(low.max())
                model_tickers = ticker_deltas.loc[
                    (ticker_deltas["stop_model"] == stop_model)
                    & (ticker_deltas["model"] == model)
                ].sort_values("scaled_net_pnl_delta", ascending=False)
            positive = (
                model_tickers.loc[
                    model_tickers["scaled_net_pnl_delta"] > 0,
                    "scaled_net_pnl_delta",
                ]
                if not model_tickers.empty
                else pd.Series(dtype=float)
            )
            top = model_tickers.iloc[0] if not model_tickers.empty else None
            bottom = model_tickers.iloc[-1] if not model_tickers.empty else None
            model_tests = tests.loc[tests["model"] == model]
            model_events = event_outcomes.loc[
                (event_outcomes["stop_model"] == stop_model)
                & (event_outcomes["model"] == model)
            ]
            rows.append(
                {
                    **record.to_dict(),
                    "stop_model": stop_model,
                    "model": model,
                    "return_delta_vs_control_percent": round(
                        float(record["compounded_return_percent"])
                        - float(control["compounded_return_percent"]),
                        4,
                    ),
                    "drawdown_advantage_vs_control_percent": round(
                        float(control["maximum_drawdown_percent"])
                        - float(record["maximum_drawdown_percent"]),
                        4,
                    ),
                    "return_drawdown_ratio_delta_vs_control": round(
                        float(record["return_drawdown_ratio"])
                        - float(control["return_drawdown_ratio"]),
                        4,
                    ),
                    "pooled_profit_factor_delta_vs_control": round(
                        float(record["pooled_profit_factor"])
                        - float(control["pooled_profit_factor"]),
                        4,
                    ),
                    "excess_vs_matched_delta_vs_control_percent": round(
                        float(record["excess_return_vs_matched_percent"])
                        - float(control["excess_return_vs_matched_percent"]),
                        4,
                    ),
                    **counts,
                    "selected_replacement_window_count": int(
                        (model_tests["actual_policy"] == REPLACEMENT_POLICY).sum()
                    ),
                    "active_replacement_window_count": int(
                        (
                            (model_tests["actual_policy"] == REPLACEMENT_POLICY)
                            & (model_tests["executed_replacements"] > 0)
                        ).sum()
                    ),
                    "event_outcome_count": int(len(model_events)),
                    "candidate_trade_match_count": int(
                        model_events.get("candidate_trade_matched", pd.Series(dtype=bool)).sum()
                    ),
                    "control_victim_continuation_match_count": int(
                        model_events.get(
                            "control_victim_continuation_matched",
                            pd.Series(dtype=bool),
                        ).sum()
                    ),
                    "leave_one_window_out_min_return_delta_percent": low_min,
                    "leave_one_window_out_max_return_delta_percent": low_max,
                    "top_positive_delta_ticker": (
                        top["ticker"] if top is not None else None
                    ),
                    "top_positive_ticker_delta": (
                        round(float(top["scaled_net_pnl_delta"]), 4)
                        if top is not None
                        else None
                    ),
                    "worst_delta_ticker": (
                        bottom["ticker"] if bottom is not None else None
                    ),
                    "worst_ticker_delta": (
                        round(float(bottom["scaled_net_pnl_delta"]), 4)
                        if bottom is not None
                        else None
                    ),
                    "top_3_positive_delta_share_percent": (
                        round(float(positive.head(3).sum() / positive.sum() * 100), 4)
                        if float(positive.sum()) > 0
                        else 0.0
                    ),
                    "source_stratum_pass": (
                        bool(_truthy(pd.Series([screen_row["stratum_pass"]])).iloc[0])
                        if model == MODEL_DYNAMIC
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_interaction_screen(
    summary: pd.DataFrame,
    source_screen: pd.DataFrame,
    event_pairs: pd.DataFrame,
) -> pd.DataFrame:
    by_key = summary.set_index(["stop_model", "model"])
    baseline_dynamic = by_key.loc[(STOP_BASELINE, MODEL_DYNAMIC)]
    maximum_dynamic = by_key.loc[(STOP_MAX_RETURN, MODEL_DYNAMIC)]
    baseline_fixed = by_key.loc[(STOP_BASELINE, MODEL_REPLACEMENT)]
    maximum_fixed = by_key.loc[(STOP_MAX_RETURN, MODEL_REPLACEMENT)]
    paired_events = int(event_pairs["present_in_both_stops"].sum())
    source_by_stop = source_screen.set_index("stop_model")
    criteria = {
        "source_global_policy_rejected": not bool(
            _truthy(source_screen["robust_walk_forward_pass"]).any()
        ),
        "baseline_dynamic_stratum_passed": bool(
            _truthy(pd.Series([source_by_stop.loc[STOP_BASELINE, "stratum_pass"]])).iloc[0]
        ),
        "max_return_dynamic_stratum_failed": not bool(
            _truthy(pd.Series([source_by_stop.loc[STOP_MAX_RETURN, "stratum_pass"]])).iloc[0]
        ),
        "baseline_dynamic_leave_one_window_out_always_positive": (
            baseline_dynamic["leave_one_window_out_min_return_delta_percent"] > 0
        ),
        "max_return_dynamic_leave_one_window_out_can_be_negative": (
            maximum_dynamic["leave_one_window_out_min_return_delta_percent"] < 0
        ),
        "fixed_replacement_return_delta_changes_sign": (
            baseline_fixed["return_delta_vs_control_percent"] > 0
            and maximum_fixed["return_delta_vs_control_percent"] < 0
        ),
        "dynamic_return_delta_spread_at_least_10_percent": (
            baseline_dynamic["return_delta_vs_control_percent"]
            - maximum_dynamic["return_delta_vs_control_percent"]
            >= 10
        ),
        "at_least_5_cross_stop_event_pairs": paired_events >= 5,
    }
    interaction_supported = bool(all(criteria.values()))
    return pd.DataFrame(
        [
            {
                **criteria,
                "baseline_dynamic_return_delta_percent": baseline_dynamic[
                    "return_delta_vs_control_percent"
                ],
                "max_return_dynamic_return_delta_percent": maximum_dynamic[
                    "return_delta_vs_control_percent"
                ],
                "dynamic_return_delta_spread_percent": round(
                    baseline_dynamic["return_delta_vs_control_percent"]
                    - maximum_dynamic["return_delta_vs_control_percent"],
                    4,
                ),
                "baseline_fixed_return_delta_percent": baseline_fixed[
                    "return_delta_vs_control_percent"
                ],
                "max_return_fixed_return_delta_percent": maximum_fixed[
                    "return_delta_vs_control_percent"
                ],
                "cross_stop_event_pair_count": paired_events,
                "same_victim_cross_stop_event_pair_count": int(
                    event_pairs["same_victim"].sum()
                ),
                "evidence_criteria_passed": int(sum(criteria.values())),
                "stop_interaction_supported": interaction_supported,
                "conditional_policy_authorized": False,
                "production_authorized": False,
                "recommended_next_stage": (
                    "PREREGISTERED_FORWARD_ONLY_STOP_CONDITIONAL_HYPOTHESIS"
                    if interaction_supported
                    else "CLOSE_REPLACEMENT_RESEARCH_LINE"
                ),
            }
        ]
    )


def run_interaction_audit(
    *, source_directory: Path, source_stamp: str
) -> dict[str, Any]:
    verified = verify_source(directory=source_directory, stamp=source_stamp)
    frames = {
        name: pd.read_csv(verified["paths"][name])
        for name in SOURCE_FRAME_NAMES
    }
    leave_one = build_leave_one_window_out(frames["test_runs"])
    tickers = build_ticker_deltas(frames["tickers"])
    events = build_event_outcomes(frames["events"], frames["trades"])
    event_pairs = build_event_pairs(events)
    trade_pairs = build_trade_pairs(frames["trades"])
    exit_reasons = build_exit_reason_effects(frames["trades"])
    rejections = build_rejection_effects(frames["rejections"])
    summary = build_summary(
        aggregate=frames["aggregate"],
        test_runs=frames["test_runs"],
        source_screen=frames["screen"],
        event_outcomes=events,
        ticker_deltas=tickers,
        leave_one_window_out=leave_one,
    )
    screen = build_interaction_screen(summary, frames["screen"], event_pairs)
    windows = frames["windows"].copy()
    windows["dynamic_active_replacement_window"] = (
        windows["dynamic_executed_replacements"] > 0
    )
    return {
        "summary": summary,
        "windows": windows,
        "events": events,
        "event_pairs": event_pairs,
        "trade_pairs": trade_pairs,
        "tickers": tickers,
        "exit_reasons": exit_reasons,
        "rejections": rejections,
        "leave_one_window_out": leave_one,
        "screen": screen,
        "source_stamp": source_stamp,
        "source_paths": verified["paths"],
        "snapshot_id": verified["provenance"]["snapshot_id"],
        "snapshot_fingerprint": verified["provenance"]["snapshot_fingerprint"],
        "snapshot_manifest_path": Path(
            verified["provenance"]["snapshot_manifest_path"]
        ),
    }


def save_interaction_audit(
    bundle: Mapping[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}
    for name in OUTPUT_FRAME_NAMES:
        path = output / f"{OUTPUT_PREFIX}_{name}_{stamp}.csv"
        bundle[name].to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    screen_row = bundle["screen"].iloc[0]
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Observational provenance-locked attribution of replacement "
            "interaction with fixed 5%/5% and 3.5%/3.5% stop strata."
        ),
        "source_stamp": bundle["source_stamp"],
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "candidate_policy": REPLACEMENT_POLICY,
        "stop_interaction_supported": bool(
            screen_row["stop_interaction_supported"]
        ),
        "conditional_policy_authorized": False,
        "production_authorized": False,
        "summary": _safe(bundle["summary"].to_dict("records")),
        "screen": _safe(bundle["screen"].to_dict("records")),
        "limitations": [
            "The source global replacement policy failed its preregistered walk-forward gate.",
            "The 5%/5% conditional effect was observed after the global test and is a new hypothesis.",
            "Local event-pair deltas are observational proxies and do not remove portfolio path dependence.",
            "Leave-one-event-out cannot be reconstructed exactly without a newly preregistered engine replay.",
            "This audit cannot authorize baseline, paper, shadow, or production changes.",
        ],
        "recommended_next_stage": screen_row["recommended_next_stage"],
    }
    json_path = output / f"{OUTPUT_PREFIX}_{stamp}.json"
    json_path.write_text(
        json.dumps(_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["json"] = json_path
    code_path = Path(__file__).resolve()
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "audit_stamp": stamp,
        "source_stamp": bundle["source_stamp"],
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "snapshot_manifest_path": str(
            Path(bundle["snapshot_manifest_path"]).resolve()
        ),
        "snapshot_manifest_sha256": sha256_file(
            Path(bundle["snapshot_manifest_path"])
        ),
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
    provenance_path = output / f"{OUTPUT_PREFIX}_provenance_{stamp}.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["provenance"] = provenance_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--walk-forward-source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument("--walk-forward-stamp", required=True)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = run_interaction_audit(
        source_directory=args.walk_forward_source_directory,
        source_stamp=args.walk_forward_stamp,
    )
    print("\nSUMMARY")
    print(bundle["summary"].to_string(index=False))
    print("\nINTERACTION SCREEN")
    print(bundle["screen"].to_string(index=False))
    if args.no_save:
        print("NO_SAVE")
        return
    for name, path in save_interaction_audit(
        bundle, output_directory=args.output_directory
    ).items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
