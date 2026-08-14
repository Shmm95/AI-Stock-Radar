"""Post-discovery walk-forward stability test for RSI replacement.

This module does not authorize production trading.  It compares the unchanged
no-replacement control with the sole policy authorized by the predecessor
ablation, RSI_Q12_LOSER_ONLY, in rolling 24-month train / 6-month unseen-test
windows.  Training selection is deterministic and defaults to the control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from math import inf, isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    annual_returns_from_equity,
    build_variant_summary,
    rejection_counts,
    run_matched_benchmark,
    ticker_contributions,
)
from src.backtest.run_portfolio_rsi_replacement_ablation import (
    CONTROL_POLICY,
    REPLACEMENT_POLICIES,
    STOP_PROFILES,
    ReplacementCollector,
    ReplacementPolicy,
    _event_counts,
    replacement_policy_context,
)
from src.backtest.run_portfolio_walk_forward import (
    WalkForwardWindow,
    build_walk_forward_windows,
    compound_returns,
    maximum_drawdown_percent,
    slice_prepared_data,
)
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/rsi_replacement_walk_forward"
)
DEFAULT_ABLATION_DIRECTORY = Path(
    "data/backtests/portfolio/rsi_replacement_ablation"
)
REPLACEMENT_POLICY = "RSI_Q12_LOSER_ONLY"
MODEL_DYNAMIC = "DYNAMIC_TRAIN_SELECTED"
MODEL_CONTROL = "FIXED_CONTROL"
MODEL_REPLACEMENT = "FIXED_RSI_Q12_LOSER_ONLY"
MODELS = (MODEL_DYNAMIC, MODEL_CONTROL, MODEL_REPLACEMENT)

TRAIN_MONTHS = 24
TEST_MONTHS = 6
STEP_MONTHS = 6
MINIMUM_TRAIN_REPLACEMENTS = 2
MINIMUM_OOS_REPLACEMENTS = 5
MINIMUM_SELECTED_WINDOWS = 3
MAXIMUM_DRAWDOWN_WORSENING_PERCENT = 2.5

_ABLATION_RESULT_NAMES = (
    "summary",
    "screen",
    "events",
    "tickers",
    "annual",
    "rejections",
    "equity",
    "trades",
    "json",
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


def _replacement_policy() -> ReplacementPolicy:
    matches = [
        policy
        for policy in REPLACEMENT_POLICIES
        if policy.name == REPLACEMENT_POLICY
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {REPLACEMENT_POLICY} definition.")
    return matches[0]


def _control_policy() -> ReplacementPolicy:
    matches = [
        policy for policy in REPLACEMENT_POLICIES if policy.name == CONTROL_POLICY
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {CONTROL_POLICY} definition.")
    return matches[0]


POLICIES = (_control_policy(), _replacement_policy())


def _ablation_paths(directory: Path, stamp: str) -> dict[str, Path]:
    prefix = "portfolio_rsi_replacement_ablation"
    output = {
        name: Path(directory) / f"{prefix}_{name}_{stamp}.csv"
        for name in _ABLATION_RESULT_NAMES
        if name != "json"
    }
    output["json"] = Path(directory) / f"{prefix}_{stamp}.json"
    output["provenance"] = Path(directory) / f"{prefix}_provenance_{stamp}.json"
    return output


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1"))


def validate_ablation_authorization(
    payload: Mapping[str, Any], screen: pd.DataFrame
) -> None:
    authorized = payload.get("robust_policies_authorized_for_walk_forward")
    if authorized != [REPLACEMENT_POLICY]:
        raise ValueError(
            "Predecessor ablation did not solely authorize "
            f"{REPLACEMENT_POLICY}."
        )
    required = {
        "replacement_policy",
        "stop_model",
        "stratum_pass",
        "robust_policy_pass",
    }
    missing = sorted(required.difference(screen.columns))
    if missing:
        raise ValueError(f"Ablation screen columns are missing: {missing}")
    passed = screen.loc[
        _truthy(screen["robust_policy_pass"]),
        ["replacement_policy", "stop_model", "stratum_pass"],
    ]
    policies = sorted(passed["replacement_policy"].astype(str).unique())
    if policies != [REPLACEMENT_POLICY]:
        raise ValueError("Ablation screen robust-policy authorization mismatch.")
    policy_rows = passed.loc[passed["replacement_policy"] == REPLACEMENT_POLICY]
    strata = sorted(policy_rows.loc[_truthy(policy_rows["stratum_pass"]), "stop_model"])
    if strata != sorted(STOP_PROFILES):
        raise ValueError("Authorized policy did not pass every fixed stop stratum.")


def verify_ablation_source(
    *, directory: Path, stamp: str, snapshot_fingerprint: str
) -> dict[str, Any]:
    paths = _ablation_paths(Path(directory), stamp)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing RSI-ablation files: " + ", ".join(missing))

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if provenance.get("audit_stamp") != stamp:
        raise ValueError("RSI-ablation provenance stamp mismatch.")
    for source in (provenance, payload):
        if source.get("snapshot_fingerprint") != snapshot_fingerprint:
            raise ValueError("RSI-ablation snapshot fingerprint mismatch.")

    expected_results = set(paths).difference({"provenance"})
    result_files = provenance.get("result_files", {})
    if set(result_files) != expected_results:
        raise ValueError("RSI-ablation provenance does not cover every result.")
    for name in sorted(expected_results):
        if sha256_file(paths[name]) != result_files[name].get("sha256"):
            raise ValueError(f"RSI-ablation result hash mismatch: {name}")

    code = provenance.get("audit_code")
    if not code:
        raise ValueError("RSI-ablation provenance has no audit-code hash.")
    code_path = Path(code["path"])
    if not code_path.exists() or sha256_file(code_path) != code.get("sha256"):
        raise ValueError("RSI-ablation audit-code hash mismatch.")

    for name, metadata in provenance.get("source_files", {}).items():
        source = Path(metadata["path"])
        if not source.exists() or sha256_file(source) != metadata.get("sha256"):
            raise ValueError(f"RSI-ablation predecessor hash mismatch: {name}")
    manifest = Path(provenance["snapshot_manifest_path"])
    if (
        not manifest.exists()
        or sha256_file(manifest) != provenance["snapshot_manifest_sha256"]
    ):
        raise ValueError("RSI-ablation snapshot manifest hash mismatch.")

    screen = pd.read_csv(paths["screen"])
    validate_ablation_authorization(payload, screen)
    return {
        "paths": paths,
        "provenance": provenance,
        "payload": payload,
    }


def _run_policy(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    stop_model: str,
    policy: ReplacementPolicy,
) -> dict[str, Any]:
    if stop_model not in STOP_PROFILES:
        raise ValueError(f"Unknown fixed stop stratum: {stop_model}")
    stock_stop, crypto_stop = STOP_PROFILES[stop_model]
    config = replace(
        base_config,
        stock_stop_loss_percent=stock_stop,
        crypto_stop_loss_percent=crypto_stop,
    )
    config.validate()
    collector = ReplacementCollector(stop_model, policy, [])
    with replacement_policy_context(
        policy=policy,
        stop_model=stop_model,
        data_by_ticker=data_by_ticker,
        collector=collector,
    ):
        result = run_portfolio_backtest(
            data_by_ticker=data_by_ticker,
            config=config,
            include_benchmark=False,
        )
    events = pd.DataFrame(collector.events)
    annual = annual_returns_from_equity(
        result.equity_curve, initial_cash=result.initial_cash
    )
    tickers = ticker_contributions(result)
    matched, _ = run_matched_benchmark(
        data_by_ticker,
        initial_cash=config.initial_cash,
        exposure_percent=max(result.average_exposure_percent, 0.0001),
        config=config,
        label=f"MATCHED_WF_{stop_model}_{policy.name}",
    )
    summary = build_variant_summary(result, matched, annual, tickers)
    summary.update(
        {
            "stop_model": stop_model,
            "stock_stop_loss_percent": stock_stop,
            "crypto_stop_loss_percent": crypto_stop,
            **policy.to_dict(),
            **_event_counts(events),
            "gross_profit": result.gross_profit,
            "gross_loss": result.gross_loss,
        }
    )
    return {
        "summary": summary,
        "events": events,
        "tickers": tickers,
        "rejections": rejection_counts(result),
        "equity": pd.DataFrame(
            [point.to_dict() for point in result.equity_curve]
        ),
        "trades": pd.DataFrame([trade.to_dict() for trade in result.trades]),
    }


def training_decision(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_replacements: int = MINIMUM_TRAIN_REPLACEMENTS,
) -> dict[str, Any]:
    if int(minimum_replacements) <= 0:
        raise ValueError("minimum_replacements must be positive.")
    if control["replacement_policy"] != CONTROL_POLICY:
        raise ValueError("Training control policy mismatch.")
    if candidate["replacement_policy"] != REPLACEMENT_POLICY:
        raise ValueError("Training candidate policy mismatch.")

    return_delta = float(candidate["total_return_percent"]) - float(
        control["total_return_percent"]
    )
    ratio_delta = float(candidate["return_drawdown_ratio"]) - float(
        control["return_drawdown_ratio"]
    )
    profit_factor_delta = float(candidate["profit_factor"]) - float(
        control["profit_factor"]
    )
    drawdown_advantage = float(control["maximum_drawdown_percent"]) - float(
        candidate["maximum_drawdown_percent"]
    )
    excess_delta = float(candidate["excess_return_vs_matched_percent"]) - float(
        control["excess_return_vs_matched_percent"]
    )
    criteria = {
        "at_least_2_train_replacements": (
            int(candidate["executed_replacements"]) >= minimum_replacements
        ),
        "positive_train_return_delta": return_delta > 1e-9,
        "nonnegative_train_return_drawdown_ratio_delta": ratio_delta >= -1e-9,
        "nonnegative_train_profit_factor_delta": profit_factor_delta >= -1e-9,
        "train_drawdown_worsening_at_most_2p5_percent": (
            drawdown_advantage >= -MAXIMUM_DRAWDOWN_WORSENING_PERCENT
        ),
        "positive_train_excess_vs_matched_delta": excess_delta > 1e-9,
    }
    selected = REPLACEMENT_POLICY if all(criteria.values()) else CONTROL_POLICY
    return {
        **criteria,
        "train_total_return_delta_percent": round(return_delta, 4),
        "train_drawdown_advantage_percent": round(drawdown_advantage, 4),
        "train_return_drawdown_ratio_delta": round(ratio_delta, 4),
        "train_profit_factor_delta": round(profit_factor_delta, 4),
        "train_excess_vs_matched_delta_percent": round(excess_delta, 4),
        "train_criteria_passed": int(sum(criteria.values())),
        "selected_policy": selected,
        "replacement_selected": selected == REPLACEMENT_POLICY,
    }


def _append_scaled_equity(
    output: list[dict[str, Any]],
    *,
    equity: pd.DataFrame,
    window: WalkForwardWindow,
    stop_model: str,
    model: str,
    actual_policy: str,
    opening_capital: float,
    initial_cash: float,
) -> float:
    if equity.empty:
        return round(opening_capital, 6)
    curve = equity.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])
    curve = curve.sort_values("timestamp")
    scale = opening_capital / initial_cash
    for row in curve.itertuples(index=False):
        output.append(
            {
                "window_id": window.window_id,
                "stop_model": stop_model,
                "model": model,
                "actual_policy": actual_policy,
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


def _ratio(total_return: float, drawdown: float) -> float:
    if drawdown <= 0:
        return inf if total_return > 0 else 0.0
    return round(total_return / drawdown, 4)


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss <= 0:
        return inf if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 4)


def aggregate_model(
    test_runs: pd.DataFrame,
    stitched_equity: pd.DataFrame,
    *,
    stop_model: str,
    model: str,
    initial_cash: float,
    first_test_start: pd.Timestamp,
    last_test_end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    rows = test_runs.loc[
        (test_runs["stop_model"] == stop_model)
        & (test_runs["model"] == model)
    ].sort_values("window_id")
    if rows.empty:
        raise ValueError(f"No test rows for {stop_model}/{model}.")
    curve = stitched_equity.loc[
        (stitched_equity["stop_model"] == stop_model)
        & (stitched_equity["model"] == model)
    ].sort_values("timestamp")
    ending = compound_returns(rows["total_return_percent"], initial_cash=initial_cash)
    total_return = (ending / initial_cash - 1) * 100
    drawdown = maximum_drawdown_percent(curve["total_equity"])
    matched_ending = compound_returns(
        rows["matched_benchmark_return_percent"], initial_cash=initial_cash
    )
    matched_return = (matched_ending / initial_cash - 1) * 100
    gross_profit = float(rows["scaled_gross_profit"].sum())
    gross_loss = float(rows["scaled_gross_loss"].sum())
    actual = rows["actual_policy"].astype(str)
    return {
        "stop_model": stop_model,
        "model": model,
        "window_count": int(len(rows)),
        "initial_cash": round(initial_cash, 2),
        "ending_equity": round(ending, 2),
        "compounded_return_percent": round(total_return, 4),
        "cagr_percent": _cagr(
            initial_cash, ending, first_test_start, last_test_end_exclusive
        ),
        "maximum_drawdown_percent": round(drawdown, 4),
        "return_drawdown_ratio": _ratio(total_return, drawdown),
        "pooled_profit_factor": _profit_factor(gross_profit, gross_loss),
        "scaled_gross_profit": round(gross_profit, 2),
        "scaled_gross_loss": round(gross_loss, 2),
        "scaled_total_fees": round(float(rows["scaled_total_fees"].sum()), 2),
        "matched_benchmark_compounded_return_percent": round(matched_return, 4),
        "excess_return_vs_matched_percent": round(
            total_return - matched_return, 4
        ),
        "average_window_return_percent": round(
            float(rows["total_return_percent"].mean()), 4
        ),
        "median_window_return_percent": round(
            float(rows["total_return_percent"].median()), 4
        ),
        "positive_window_count": int((rows["total_return_percent"] > 0).sum()),
        "matched_benchmark_beat_count": int(
            (rows["excess_return_vs_matched_percent"] > 0).sum()
        ),
        "total_trades": int(rows["completed_trades"].sum()),
        "executed_replacements": int(rows["executed_replacements"].sum()),
        "replacement_policy_window_count": int(
            (actual == REPLACEMENT_POLICY).sum()
        ),
        "average_exposure_percent": round(
            float(rows["average_exposure_percent"].mean()), 4
        ),
    }


def _comparison_counts(values: pd.Series) -> dict[str, int]:
    numbers = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return {
        "window_win_count": int((numbers > 1e-9).sum()),
        "window_tie_count": int((numbers.abs() <= 1e-9).sum()),
        "window_loss_count": int((numbers < -1e-9).sum()),
    }


def build_research_screen(
    aggregate: pd.DataFrame,
    test_runs: pd.DataFrame,
    *,
    complete_window_set: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stop_model in sorted(STOP_PROFILES):
        group = aggregate.loc[aggregate["stop_model"] == stop_model].set_index(
            "model"
        )
        if set(group.index) != set(MODELS):
            raise ValueError(f"Aggregate model coverage mismatch for {stop_model}.")
        dynamic = group.loc[MODEL_DYNAMIC]
        control = group.loc[MODEL_CONTROL]
        window = test_runs.loc[
            test_runs["stop_model"] == stop_model,
            ["window_id", "model", "total_return_percent"],
        ].pivot(
            index="window_id", columns="model", values="total_return_percent"
        )
        counts = _comparison_counts(window[MODEL_DYNAMIC] - window[MODEL_CONTROL])
        return_delta = (
            dynamic["compounded_return_percent"]
            - control["compounded_return_percent"]
        )
        drawdown_advantage = (
            control["maximum_drawdown_percent"]
            - dynamic["maximum_drawdown_percent"]
        )
        ratio_delta = (
            dynamic["return_drawdown_ratio"] - control["return_drawdown_ratio"]
        )
        profit_factor_delta = (
            dynamic["pooled_profit_factor"] - control["pooled_profit_factor"]
        )
        excess_delta = (
            dynamic["excess_return_vs_matched_percent"]
            - control["excess_return_vs_matched_percent"]
        )
        criteria = {
            "complete_window_set": bool(complete_window_set),
            "positive_oos_compounded_return_delta": return_delta > 1e-9,
            "nonnegative_oos_return_drawdown_ratio_delta": ratio_delta >= -1e-9,
            "nonnegative_oos_pooled_profit_factor_delta": (
                profit_factor_delta >= -1e-9
            ),
            "oos_drawdown_worsening_at_most_2p5_percent": (
                drawdown_advantage >= -MAXIMUM_DRAWDOWN_WORSENING_PERCENT
            ),
            "positive_oos_excess_vs_matched_delta": excess_delta > 1e-9,
            "more_oos_window_wins_than_losses": (
                counts["window_win_count"] > counts["window_loss_count"]
            ),
            "at_least_5_oos_replacements": (
                int(dynamic["executed_replacements"]) >= MINIMUM_OOS_REPLACEMENTS
            ),
            "replacement_selected_in_at_least_3_windows": (
                int(dynamic["replacement_policy_window_count"])
                >= MINIMUM_SELECTED_WINDOWS
            ),
        }
        rows.append(
            {
                "stop_model": stop_model,
                **criteria,
                "oos_compounded_return_delta_percent": round(return_delta, 4),
                "oos_drawdown_advantage_percent": round(drawdown_advantage, 4),
                "oos_return_drawdown_ratio_delta": round(ratio_delta, 4),
                "oos_pooled_profit_factor_delta": round(profit_factor_delta, 4),
                "oos_excess_vs_matched_delta_percent": round(excess_delta, 4),
                **counts,
                "oos_executed_replacements": int(
                    dynamic["executed_replacements"]
                ),
                "replacement_selected_window_count": int(
                    dynamic["replacement_policy_window_count"]
                ),
                "criteria_passed": int(sum(criteria.values())),
                "stratum_pass": bool(all(criteria.values())),
            }
        )
    result = pd.DataFrame(rows)
    robust = len(result) == len(STOP_PROFILES) and bool(result["stratum_pass"].all())
    result["robust_walk_forward_pass"] = robust
    result["production_authorized"] = False
    return result


def _detail_records(
    frame: pd.DataFrame,
    *,
    window: WalkForwardWindow,
    stop_model: str,
    model: str,
    actual_policy: str,
    capital_scale: float,
    scaled_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    output: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        row = {
            "window_id": window.window_id,
            "stop_model": stop_model,
            "model": model,
            "actual_policy": actual_policy,
            "capital_scale": round(capital_scale, 8),
            **record,
        }
        for column in scaled_columns:
            if column in record:
                row[f"scaled_{column}"] = round(
                    float(record[column]) * capital_scale, 6
                )
        output.append(row)
    return output


def run_rsi_replacement_walk_forward(
    *,
    snapshot_path: Path,
    ablation_directory: Path,
    ablation_stamp: str,
    project_root: Path = Path("."),
    max_windows: int | None = None,
) -> dict[str, Any]:
    snapshot = load_snapshot(
        Path(snapshot_path), verify_code=True, project_root=Path(project_root)
    )
    source = verify_ablation_source(
        directory=Path(ablation_directory),
        stamp=ablation_stamp,
        snapshot_fingerprint=snapshot["manifest"]["fingerprint"],
    )
    base_config = snapshot["config"]
    base_config.validate()
    all_windows = build_walk_forward_windows(
        snapshot["data_by_ticker"],
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )
    if max_windows is not None:
        if int(max_windows) <= 0:
            raise ValueError("max_windows must be positive when provided.")
        windows = all_windows[: int(max_windows)]
    else:
        windows = all_windows
    complete_window_set = len(windows) == len(all_windows)

    training_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    capitals = {
        (stop_model, model): float(base_config.initial_cash)
        for stop_model in STOP_PROFILES
        for model in MODELS
    }

    for sequence, window in enumerate(windows, 1):
        print(f"\nRSI REPLACEMENT WALK-FORWARD [{sequence}/{len(windows)}] {window.window_id}")
        train_data = slice_prepared_data(
            snapshot["data_by_ticker"],
            start=window.train_start,
            end_exclusive=window.train_end_exclusive,
        )
        test_data = slice_prepared_data(
            snapshot["data_by_ticker"],
            start=window.test_start,
            end_exclusive=window.test_end_exclusive,
        )
        for stop_model in STOP_PROFILES:
            train_bundles = {
                policy.name: _run_policy(
                    data_by_ticker=train_data,
                    base_config=base_config,
                    stop_model=stop_model,
                    policy=policy,
                )
                for policy in POLICIES
            }
            decision = training_decision(
                train_bundles[CONTROL_POLICY]["summary"],
                train_bundles[REPLACEMENT_POLICY]["summary"],
            )
            selected = str(decision["selected_policy"])
            for policy in POLICIES:
                training_rows.append(
                    {
                        **window.to_dict(),
                        "stop_model": stop_model,
                        "candidate_policy": policy.name,
                        **train_bundles[policy.name]["summary"],
                        **decision,
                        "selected_for_test": policy.name == selected,
                    }
                )

            test_bundles = {
                policy.name: _run_policy(
                    data_by_ticker=test_data,
                    base_config=base_config,
                    stop_model=stop_model,
                    policy=policy,
                )
                for policy in POLICIES
            }
            model_policies = {
                MODEL_DYNAMIC: selected,
                MODEL_CONTROL: CONTROL_POLICY,
                MODEL_REPLACEMENT: REPLACEMENT_POLICY,
            }
            model_summaries: dict[str, Mapping[str, Any]] = {}
            for model, actual_policy in model_policies.items():
                bundle = test_bundles[actual_policy]
                summary = bundle["summary"]
                model_summaries[model] = summary
                capital_key = (stop_model, model)
                opening_capital = capitals[capital_key]
                capital_scale = opening_capital / float(base_config.initial_cash)
                ending_capital = _append_scaled_equity(
                    equity_rows,
                    equity=bundle["equity"],
                    window=window,
                    stop_model=stop_model,
                    model=model,
                    actual_policy=actual_policy,
                    opening_capital=opening_capital,
                    initial_cash=float(base_config.initial_cash),
                )
                capitals[capital_key] = ending_capital
                test_rows.append(
                    {
                        **window.to_dict(),
                        "stop_model": stop_model,
                        "model": model,
                        "actual_policy": actual_policy,
                        "train_selected_policy": selected,
                        "opening_capital": round(opening_capital, 6),
                        "ending_capital": round(ending_capital, 6),
                        "capital_scale": round(capital_scale, 8),
                        **summary,
                        "scaled_gross_profit": round(
                            float(summary["gross_profit"]) * capital_scale, 6
                        ),
                        "scaled_gross_loss": round(
                            float(summary["gross_loss"]) * capital_scale, 6
                        ),
                        "scaled_total_fees": round(
                            float(summary["total_fees"]) * capital_scale, 6
                        ),
                    }
                )
                event_rows.extend(
                    _detail_records(
                        bundle["events"],
                        window=window,
                        stop_model=stop_model,
                        model=model,
                        actual_policy=actual_policy,
                        capital_scale=capital_scale,
                    )
                )
                ticker_rows.extend(
                    _detail_records(
                        bundle["tickers"],
                        window=window,
                        stop_model=stop_model,
                        model=model,
                        actual_policy=actual_policy,
                        capital_scale=capital_scale,
                        scaled_columns=(
                            "gross_profit",
                            "gross_loss",
                            "net_pnl",
                            "total_fees",
                        ),
                    )
                )
                rejection_frame = pd.DataFrame(
                    [
                        {"reason_code": reason, "count": count}
                        for reason, count in sorted(bundle["rejections"].items())
                    ]
                )
                rejection_rows.extend(
                    _detail_records(
                        rejection_frame,
                        window=window,
                        stop_model=stop_model,
                        model=model,
                        actual_policy=actual_policy,
                        capital_scale=capital_scale,
                    )
                )
                trade_rows.extend(
                    _detail_records(
                        bundle["trades"],
                        window=window,
                        stop_model=stop_model,
                        model=model,
                        actual_policy=actual_policy,
                        capital_scale=capital_scale,
                        scaled_columns=("gross_pnl", "net_pnl", "total_fees"),
                    )
                )

            control_summary = model_summaries[MODEL_CONTROL]
            dynamic_summary = model_summaries[MODEL_DYNAMIC]
            fixed_summary = model_summaries[MODEL_REPLACEMENT]
            window_rows.append(
                {
                    **window.to_dict(),
                    "stop_model": stop_model,
                    "train_selected_policy": selected,
                    "replacement_selected": selected == REPLACEMENT_POLICY,
                    "train_criteria_passed": decision["train_criteria_passed"],
                    "dynamic_actual_policy": model_policies[MODEL_DYNAMIC],
                    "dynamic_return_percent": dynamic_summary[
                        "total_return_percent"
                    ],
                    "control_return_percent": control_summary[
                        "total_return_percent"
                    ],
                    "fixed_replacement_return_percent": fixed_summary[
                        "total_return_percent"
                    ],
                    "dynamic_return_delta_vs_control_percent": round(
                        float(dynamic_summary["total_return_percent"])
                        - float(control_summary["total_return_percent"]),
                        4,
                    ),
                    "fixed_replacement_return_delta_vs_control_percent": round(
                        float(fixed_summary["total_return_percent"])
                        - float(control_summary["total_return_percent"]),
                        4,
                    ),
                    "dynamic_maximum_drawdown_percent": dynamic_summary[
                        "maximum_drawdown_percent"
                    ],
                    "control_maximum_drawdown_percent": control_summary[
                        "maximum_drawdown_percent"
                    ],
                    "fixed_replacement_maximum_drawdown_percent": fixed_summary[
                        "maximum_drawdown_percent"
                    ],
                    "dynamic_profit_factor": dynamic_summary["profit_factor"],
                    "control_profit_factor": control_summary["profit_factor"],
                    "fixed_replacement_profit_factor": fixed_summary[
                        "profit_factor"
                    ],
                    "dynamic_executed_replacements": dynamic_summary[
                        "executed_replacements"
                    ],
                    "fixed_executed_replacements": fixed_summary[
                        "executed_replacements"
                    ],
                }
            )
            print(
                f"{stop_model} selected={selected} "
                f"dynamic={dynamic_summary['total_return_percent']:+.4f}% "
                f"control={control_summary['total_return_percent']:+.4f}%"
            )

    frames = {
        "windows": pd.DataFrame(window_rows),
        "training": pd.DataFrame(training_rows),
        "test_runs": pd.DataFrame(test_rows),
        "events": pd.DataFrame(event_rows),
        "tickers": pd.DataFrame(ticker_rows),
        "rejections": pd.DataFrame(rejection_rows),
        "trades": pd.DataFrame(trade_rows),
        "equity": pd.DataFrame(equity_rows),
    }
    aggregate = pd.DataFrame(
        [
            aggregate_model(
                frames["test_runs"],
                frames["equity"],
                stop_model=stop_model,
                model=model,
                initial_cash=float(base_config.initial_cash),
                first_test_start=windows[0].test_start,
                last_test_end_exclusive=windows[-1].test_end_exclusive,
            )
            for stop_model in STOP_PROFILES
            for model in MODELS
        ]
    )
    screen = build_research_screen(
        aggregate,
        frames["test_runs"],
        complete_window_set=complete_window_set,
    )
    selections = frames["windows"].groupby(
        ["stop_model", "train_selected_policy"], sort=True
    ).size()
    selection_summary = {
        stop_model: {
            policy.name: int(selections.get((stop_model, policy.name), 0))
            for policy in POLICIES
        }
        for stop_model in STOP_PROFILES
    }
    return {
        **frames,
        "aggregate": aggregate,
        "screen": screen,
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint": snapshot["manifest"]["fingerprint"],
        "snapshot_manifest_path": Path(snapshot_path) / "manifest.json",
        "ablation_stamp": ablation_stamp,
        "ablation_paths": source["paths"],
        "base_config": base_config.to_dict(),
        "window_definitions": [window.to_dict() for window in windows],
        "expected_window_count": len(all_windows),
        "actual_window_count": len(windows),
        "complete_window_set": complete_window_set,
        "selection_summary": selection_summary,
    }


def save_rsi_replacement_walk_forward(
    bundle: Mapping[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    prefix = "portfolio_rsi_replacement_walk_forward"
    frame_names = (
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
    paths: dict[str, Path] = {}
    for name in frame_names:
        path = output / f"{prefix}_{name}_{stamp}.csv"
        bundle[name].to_csv(path, index=False, lineterminator="\n")
        paths[name] = path

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Post-discovery 24-month train / 6-month unseen-test rolling "
            "stability validation of RSI_Q12_LOSER_ONLY versus no replacement."
        ),
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "ablation_stamp": bundle["ablation_stamp"],
        "authorized_policy": REPLACEMENT_POLICY,
        "models": list(MODELS),
        "stop_profiles": STOP_PROFILES,
        "train_months": TRAIN_MONTHS,
        "test_months": TEST_MONTHS,
        "step_months": STEP_MONTHS,
        "minimum_train_replacements": MINIMUM_TRAIN_REPLACEMENTS,
        "expected_window_count": bundle["expected_window_count"],
        "actual_window_count": bundle["actual_window_count"],
        "complete_window_set": bundle["complete_window_set"],
        "selection_summary": bundle["selection_summary"],
        "window_definitions": bundle["window_definitions"],
        "aggregate": _safe(bundle["aggregate"].to_dict("records")),
        "screen": _safe(bundle["screen"].to_dict("records")),
        "robust_walk_forward_pass": bool(
            bundle["screen"]["robust_walk_forward_pass"].all()
        ),
        "production_authorized": False,
        "limitations": [
            "RSI_Q12_LOSER_ONLY was discovered on the same frozen historical span.",
            "This is a post-discovery stability test, not pristine independent OOS discovery.",
            "Passing this screen cannot authorize live or paper execution.",
            "Independent forward shadow validation remains mandatory.",
        ],
    }
    json_path = output / f"{prefix}_{stamp}.json"
    json_path.write_text(
        json.dumps(_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
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
        "ablation_stamp": bundle["ablation_stamp"],
        "audit_code": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
        },
        "source_files": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in bundle["ablation_paths"].items()
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
        "--ablation-source-directory",
        type=Path,
        default=DEFAULT_ABLATION_DIRECTORY,
    )
    parser.add_argument("--ablation-stamp", required=True)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Diagnostic smoke run only; incomplete runs can never pass the screen.",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = run_rsi_replacement_walk_forward(
        snapshot_path=args.snapshot,
        ablation_directory=args.ablation_source_directory,
        ablation_stamp=args.ablation_stamp,
        project_root=Path("."),
        max_windows=args.max_windows,
    )
    print("\nAGGREGATE")
    print(bundle["aggregate"].to_string(index=False))
    print("\nSCREEN")
    print(bundle["screen"].to_string(index=False))
    print("\nSELECTIONS")
    print(json.dumps(bundle["selection_summary"], indent=2))
    if args.no_save:
        print("NO_SAVE")
        return
    paths = save_rsi_replacement_walk_forward(
        bundle, output_directory=args.output_directory
    )
    for name, path in paths.items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
