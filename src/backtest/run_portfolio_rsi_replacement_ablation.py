"""Exploratory RSI-quality portfolio replacement ablation.

The only authorized mechanism is:

* candidate feature: RSI_QUALITY;
* victim rule: WORST_UNREALIZED_RETURN;
* victim must have an executable Open at the candidate execution timestamp;
* positions opened on the same portfolio bar cannot be replaced;
* a shadow execution must prove the candidate can pass every post-exit risk,
  cash, and crypto-allocation control before the real victim is closed.

This full-period ablation compares pre-registered candidate-quality thresholds
and an optional losing-victim gate.  It is exploratory and cannot place orders.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from src.backtest import portfolio_backtest_engine as portfolio_engine
from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    annual_returns_from_equity,
    build_variant_summary,
    rejection_counts,
    run_matched_benchmark,
    ticker_contributions,
)
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/rsi_replacement_ablation"
)
DEFAULT_VICTIM_DIRECTORY = Path(
    "data/backtests/portfolio/replacement_victim_audit"
)
CONTROL_POLICY = "CONTROL_NO_REPLACEMENT"
AUTHORIZED_FEATURE = "RSI_QUALITY"
AUTHORIZED_VICTIM_RULE = "WORST_UNREALIZED_RETURN"


@dataclass(frozen=True, slots=True)
class ReplacementPolicy:
    name: str
    enabled: bool
    minimum_candidate_rsi_quality: float | None
    victim_must_be_losing: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Policy name cannot be empty.")
        if self.enabled:
            if self.minimum_candidate_rsi_quality is None:
                raise ValueError("Enabled policies require an RSI-quality threshold.")
            threshold = float(self.minimum_candidate_rsi_quality)
            if not 0 <= threshold <= 15:
                raise ValueError("RSI-quality threshold must be in [0, 15].")
            object.__setattr__(self, "minimum_candidate_rsi_quality", threshold)
        elif self.minimum_candidate_rsi_quality is not None:
            raise ValueError("Disabled control cannot have an RSI threshold.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "replacement_policy": self.name,
            "replacement_enabled": self.enabled,
            "minimum_candidate_rsi_quality": self.minimum_candidate_rsi_quality,
            "victim_must_be_losing": self.victim_must_be_losing,
        }


REPLACEMENT_POLICIES = (
    ReplacementPolicy(CONTROL_POLICY, False, None, False),
    ReplacementPolicy("RSI_Q9_ANY", True, 9.0, False),
    ReplacementPolicy("RSI_Q12_ANY", True, 12.0, False),
    ReplacementPolicy("RSI_Q9_LOSER_ONLY", True, 9.0, True),
    ReplacementPolicy("RSI_Q12_LOSER_ONLY", True, 12.0, True),
)

STOP_PROFILES: dict[str, tuple[float, float]] = {
    "FIXED_BASELINE": (5.0, 5.0),
    "FIXED_MAX_RETURN": (3.5, 3.5),
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


def normalize_policy_names(values: Iterable[str]) -> tuple[str, ...]:
    available = {policy.name: policy for policy in REPLACEMENT_POLICIES}
    names = tuple(dict.fromkeys(str(value).strip().upper() for value in values))
    unknown = sorted(set(names).difference(available))
    if unknown:
        raise ValueError(f"Unknown replacement policies: {unknown}")
    if CONTROL_POLICY not in names:
        raise ValueError(f"{CONTROL_POLICY} is required as the control.")
    return names


def _policy(name: str) -> ReplacementPolicy:
    normalized = str(name).strip().upper()
    for policy in REPLACEMENT_POLICIES:
        if policy.name == normalized:
            return policy
    raise ValueError(f"Unknown replacement policy: {name}")


def rsi_quality(value: float) -> float:
    return max(0.0, 15.0 - abs(float(value) - 57.5) * 1.2)


def candidate_rsi_quality(
    data: pd.DataFrame, signal_timestamp: Any
) -> tuple[pd.Timestamp, float]:
    """Read RSI quality from the signal close, never the execution bar."""

    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    timestamp = pd.Timestamp(signal_timestamp)
    if index.tz is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(index.tz)
    elif index.tz is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    position = int(index.searchsorted(timestamp, side="right")) - 1
    if position < 0:
        raise ValueError(f"No signal row at or before {signal_timestamp}.")
    row_timestamp = pd.Timestamp(index[position])
    if row_timestamp != timestamp:
        raise ValueError(f"Signal timestamp is absent from candidate data: {timestamp}")
    return row_timestamp, rsi_quality(float(data.iloc[position]["RSI14"]))


def select_executable_victim(
    *,
    state: Any,
    data_by_ticker: dict[str, pd.DataFrame],
    timestamp: Any,
    portfolio_bar_index: int,
) -> dict[str, Any] | None:
    """Select worst unrealized holding from positions tradable at this Open."""

    active_timestamp = pd.Timestamp(timestamp)
    candidates: list[dict[str, Any]] = []
    for ticker, position in state.positions.items():
        if position.entry_portfolio_bar_index >= portfolio_bar_index:
            continue
        data = data_by_ticker.get(ticker)
        if data is None:
            continue
        index = pd.DatetimeIndex(pd.to_datetime(data.index))
        if active_timestamp not in index:
            continue
        row = data.loc[active_timestamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        raw_open = float(row["Open"])
        unrealized = (raw_open / float(position.entry_price) - 1) * 100
        candidates.append(
            {
                "victim_ticker": ticker,
                "victim_raw_open": raw_open,
                "victim_entry_price": float(position.entry_price),
                "victim_unrealized_return_percent": unrealized,
                "victim_holding_age_bars": (
                    portfolio_bar_index - position.entry_portfolio_bar_index
                ),
            }
        )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            row["victim_unrealized_return_percent"],
            row["victim_ticker"],
        ),
    )[0]


def _shadow_state(state: Any) -> Any:
    """Small state clone for atomic post-exit entry feasibility checks."""

    return portfolio_engine._PortfolioState(
        cash=state.cash,
        positions=copy.deepcopy(state.positions),
        pending_buys=copy.deepcopy(state.pending_buys),
        pending_exits=copy.deepcopy(state.pending_exits),
        last_prices=dict(state.last_prices),
        trades=[],
        rejections=[],
        equity_curve=[],
        realized_pnl=state.realized_pnl,
        peak_equity=state.peak_equity,
        maximum_drawdown_amount=state.maximum_drawdown_amount,
        maximum_drawdown_percent=state.maximum_drawdown_percent,
        skipped_signals=state.skipped_signals,
        maximum_open_positions_observed=state.maximum_open_positions_observed,
    )


@dataclass(slots=True)
class ReplacementCollector:
    stop_model: str
    policy: ReplacementPolicy
    events: list[dict[str, Any]]
    sequence: int = 0

    def record(self, **values: Any) -> None:
        self.sequence += 1
        self.events.append(
            {
                "replacement_event_id": (
                    f"{self.stop_model}:{self.policy.name}:{self.sequence:05d}"
                ),
                "stop_model": self.stop_model,
                **self.policy.to_dict(),
                **values,
            }
        )


@contextmanager
def replacement_policy_context(
    *,
    policy: ReplacementPolicy,
    stop_model: str,
    data_by_ticker: dict[str, pd.DataFrame],
    collector: ReplacementCollector,
) -> Iterator[None]:
    """Instrument full-position attempts with atomic causal replacements."""

    original = portfolio_engine._attempt_open_position

    def wrapper(
        state: Any,
        *,
        pending: Any,
        timestamp: str,
        portfolio_bar_index: int,
        raw_open_price: float,
        config: PortfolioBacktestConfig,
        prices: dict[str, float],
    ) -> None:
        signal = pending.signal
        if (
            signal.ticker in state.positions
            or len(state.positions) < config.maximum_open_positions
        ):
            return original(
                state,
                pending=pending,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_open_price=raw_open_price,
                config=config,
                prices=prices,
            )

        signal_timestamp, quality = candidate_rsi_quality(
            data_by_ticker[signal.ticker], signal.timestamp
        )
        base = {
            "timestamp": timestamp,
            "portfolio_bar_index": portfolio_bar_index,
            "candidate_ticker": signal.ticker,
            "candidate_signal_timestamp": signal_timestamp,
            "candidate_rsi_quality": round(quality, 6),
            "candidate_score": float(signal.score),
            "open_position_count_before": len(state.positions),
        }
        if not policy.enabled:
            collector.record(status="CONTROL_MAX_POSITION_REJECTION", **base)
            return original(
                state,
                pending=pending,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_open_price=raw_open_price,
                config=config,
                prices=prices,
            )
        if quality < float(policy.minimum_candidate_rsi_quality):
            collector.record(status="CANDIDATE_QUALITY_BELOW_THRESHOLD", **base)
            return original(
                state,
                pending=pending,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_open_price=raw_open_price,
                config=config,
                prices=prices,
            )
        victim = select_executable_victim(
            state=state,
            data_by_ticker=data_by_ticker,
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
        )
        if victim is None:
            collector.record(status="NO_EXECUTABLE_VICTIM", **base)
            return original(
                state,
                pending=pending,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_open_price=raw_open_price,
                config=config,
                prices=prices,
            )
        event = {**base, **victim}
        if (
            policy.victim_must_be_losing
            and victim["victim_unrealized_return_percent"] > 0
        ):
            collector.record(status="VICTIM_NOT_LOSING", **event)
            return original(
                state,
                pending=pending,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_open_price=raw_open_price,
                config=config,
                prices=prices,
            )

        shadow = _shadow_state(state)
        portfolio_engine._close_position(
            shadow,
            ticker=victim["victim_ticker"],
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_exit_price=victim["victim_raw_open"],
            exit_reason=f"REPLACEMENT_{policy.name}",
            config=config,
        )
        original(
            shadow,
            pending=pending,
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_open_price=raw_open_price,
            config=config,
            prices=prices,
        )
        if signal.ticker not in shadow.positions:
            reason = shadow.rejections[-1].reason_code if shadow.rejections else None
            collector.record(
                status="POST_EXIT_ENTRY_NOT_FEASIBLE",
                post_exit_rejection_reason=reason,
                **event,
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

        portfolio_engine._close_position(
            state,
            ticker=victim["victim_ticker"],
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_exit_price=victim["victim_raw_open"],
            exit_reason=f"REPLACEMENT_{policy.name}",
            config=config,
        )
        original(
            state,
            pending=pending,
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_open_price=raw_open_price,
            config=config,
            prices=prices,
        )
        if signal.ticker not in state.positions:
            raise RuntimeError("Shadow feasibility and real replacement diverged.")
        collector.record(status="EXECUTED", post_exit_rejection_reason=None, **event)

    portfolio_engine._attempt_open_position = wrapper
    try:
        yield
    finally:
        portfolio_engine._attempt_open_position = original


def _victim_paths(directory: Path, stamp: str) -> dict[str, Path]:
    prefix = "portfolio_replacement_victim_audit"
    names = (
        "pairs",
        "selections",
        "rule_windows",
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


def validate_victim_authorization(
    payload: dict[str, Any], screen: pd.DataFrame
) -> None:
    if payload.get("authorized_candidate_feature") != AUTHORIZED_FEATURE:
        raise ValueError("Victim audit did not authorize RSI_QUALITY.")
    if payload.get("robust_victim_rules_authorized_for_separate_ablation") != [
        AUTHORIZED_VICTIM_RULE
    ]:
        raise ValueError("Victim audit did not solely authorize WORST_UNREALIZED_RETURN.")
    passed = sorted(
        screen.loc[
            screen["robust_victim_rule_pass"].astype(str).str.lower().isin(
                ("true", "1")
            ),
            "victim_rule",
        ].unique()
    )
    if passed != [AUTHORIZED_VICTIM_RULE]:
        raise ValueError("Victim audit screen authorization mismatch.")


def verify_victim_source(
    *,
    directory: Path,
    stamp: str,
    snapshot_fingerprint: str,
) -> dict[str, Any]:
    paths = _victim_paths(Path(directory), stamp)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing victim-audit files: " + ", ".join(missing))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if provenance.get("audit_stamp") != stamp:
        raise ValueError("Victim provenance stamp mismatch.")
    if provenance.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise ValueError("Victim provenance snapshot mismatch.")
    if payload.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise ValueError("Victim payload snapshot mismatch.")
    code = provenance.get("audit_code")
    if not code:
        raise ValueError("Victim provenance has no audit-code hash.")
    code_path = Path(code["path"])
    if not code_path.exists() or sha256_file(code_path) != code.get("sha256"):
        raise ValueError("Victim audit-code hash mismatch.")
    results = provenance.get("result_files", {})
    if set(results) != set(paths).difference({"provenance"}):
        raise ValueError("Victim provenance does not cover all result files.")
    for name, metadata in results.items():
        if sha256_file(paths[name]) != metadata.get("sha256"):
            raise ValueError(f"Victim result hash mismatch: {name}")
    for name, metadata in provenance.get("source_files", {}).items():
        source = Path(metadata["path"])
        if not source.exists() or sha256_file(source) != metadata.get("sha256"):
            raise ValueError(f"Victim predecessor hash mismatch: {name}")
    manifest = Path(provenance["snapshot_manifest_path"])
    if (
        not manifest.exists()
        or sha256_file(manifest) != provenance["snapshot_manifest_sha256"]
    ):
        raise ValueError("Victim snapshot manifest hash mismatch.")
    screen = pd.read_csv(paths["screen"])
    validate_victim_authorization(payload, screen)
    return {"paths": paths, "provenance": provenance, "payload": payload}


def _event_counts(events: pd.DataFrame) -> dict[str, int]:
    counts = Counter(events["status"].astype(str)) if not events.empty else Counter()
    return {
        "full_portfolio_attempts": int(sum(counts.values())),
        "executed_replacements": int(counts.get("EXECUTED", 0)),
        "quality_threshold_skips": int(
            counts.get("CANDIDATE_QUALITY_BELOW_THRESHOLD", 0)
        ),
        "victim_not_losing_skips": int(counts.get("VICTIM_NOT_LOSING", 0)),
        "no_executable_victim_skips": int(counts.get("NO_EXECUTABLE_VICTIM", 0)),
        "post_exit_entry_infeasible_skips": int(
            counts.get("POST_EXIT_ENTRY_NOT_FEASIBLE", 0)
        ),
    }


def build_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stop_model, group in summary.groupby("stop_model", sort=True):
        controls = group.loc[group["replacement_policy"] == CONTROL_POLICY]
        if len(controls) != 1:
            raise ValueError(f"Expected one control for {stop_model}.")
        control = controls.iloc[0]
        for record in group.to_dict("records"):
            rows.append(
                {
                    **record,
                    "total_return_delta_vs_control_percent": round(
                        record["total_return_percent"]
                        - control["total_return_percent"],
                        4,
                    ),
                    "drawdown_advantage_vs_control_percent": round(
                        control["maximum_drawdown_percent"]
                        - record["maximum_drawdown_percent"],
                        4,
                    ),
                    "return_drawdown_ratio_delta_vs_control": round(
                        record["return_drawdown_ratio"]
                        - control["return_drawdown_ratio"],
                        4,
                    ),
                    "profit_factor_delta_vs_control": round(
                        record["profit_factor"] - control["profit_factor"], 4
                    ),
                    "excess_vs_matched_delta_vs_control_percent": round(
                        record["excess_return_vs_matched_percent"]
                        - control["excess_return_vs_matched_percent"],
                        4,
                    ),
                    "completed_trades_delta_vs_control": int(
                        record["completed_trades"] - control["completed_trades"]
                    ),
                    "total_fees_delta_vs_control": round(
                        record["total_fees"] - control["total_fees"], 2
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_screen(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in summary.loc[
        summary["replacement_policy"] != CONTROL_POLICY
    ].to_dict("records"):
        criteria = {
            "positive_return_delta": (
                record["total_return_delta_vs_control_percent"] > 0
            ),
            "nonnegative_return_drawdown_ratio_delta": (
                record["return_drawdown_ratio_delta_vs_control"] >= 0
            ),
            "nonnegative_profit_factor_delta": (
                record["profit_factor_delta_vs_control"] >= 0
            ),
            "drawdown_worsening_at_most_2p5_percent": (
                record["drawdown_advantage_vs_control_percent"] >= -2.5
            ),
            "positive_excess_vs_matched_delta": (
                record["excess_vs_matched_delta_vs_control_percent"] > 0
            ),
            "at_least_10_executed_replacements": (
                record["executed_replacements"] >= 10
            ),
        }
        rows.append(
            {
                "stop_model": record["stop_model"],
                "replacement_policy": record["replacement_policy"],
                **criteria,
                "total_return_delta_vs_control_percent": record[
                    "total_return_delta_vs_control_percent"
                ],
                "drawdown_advantage_vs_control_percent": record[
                    "drawdown_advantage_vs_control_percent"
                ],
                "return_drawdown_ratio_delta_vs_control": record[
                    "return_drawdown_ratio_delta_vs_control"
                ],
                "profit_factor_delta_vs_control": record[
                    "profit_factor_delta_vs_control"
                ],
                "excess_vs_matched_delta_vs_control_percent": record[
                    "excess_vs_matched_delta_vs_control_percent"
                ],
                "executed_replacements": record["executed_replacements"],
                "criteria_passed": int(sum(criteria.values())),
                "stratum_pass": bool(all(criteria.values())),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result["robust_policy_pass"] = pd.Series(dtype=bool)
        return result
    robust = result.groupby("replacement_policy")["stratum_pass"].transform(
        lambda values: len(values) == len(STOP_PROFILES) and bool(values.all())
    )
    result["robust_policy_pass"] = robust.astype(bool)
    return result.sort_values(["replacement_policy", "stop_model"]).reset_index(
        drop=True
    )


def run_rsi_replacement_ablation(
    *,
    snapshot_path: Path,
    victim_directory: Path,
    victim_stamp: str,
    policy_names: Sequence[str] = tuple(
        policy.name for policy in REPLACEMENT_POLICIES
    ),
    project_root: Path = Path("."),
) -> dict[str, Any]:
    selected_names = normalize_policy_names(policy_names)
    selected_policies = tuple(_policy(name) for name in selected_names)
    snapshot = load_snapshot(
        Path(snapshot_path), verify_code=True, project_root=Path(project_root)
    )
    source = verify_victim_source(
        directory=Path(victim_directory),
        stamp=victim_stamp,
        snapshot_fingerprint=snapshot["manifest"]["fingerprint"],
    )
    base_config = snapshot["config"]
    summaries: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    ticker_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    rejection_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for stop_model, stops in STOP_PROFILES.items():
        config = replace(
            base_config,
            stock_stop_loss_percent=stops[0],
            crypto_stop_loss_percent=stops[1],
        )
        config.validate()
        for sequence, policy in enumerate(selected_policies, 1):
            print(
                f"RSI REPLACEMENT {stop_model} "
                f"[{sequence}/{len(selected_policies)}] {policy.name}"
            )
            collector = ReplacementCollector(stop_model, policy, [])
            with replacement_policy_context(
                policy=policy,
                stop_model=stop_model,
                data_by_ticker=snapshot["data_by_ticker"],
                collector=collector,
            ):
                result = run_portfolio_backtest(
                    data_by_ticker=snapshot["data_by_ticker"],
                    config=config,
                    include_benchmark=False,
                )
            events = pd.DataFrame(collector.events)
            if not events.empty:
                event_frames.append(events)
            annual = annual_returns_from_equity(
                result.equity_curve, initial_cash=result.initial_cash
            )
            tickers = ticker_contributions(result)
            matched, _ = run_matched_benchmark(
                snapshot["data_by_ticker"],
                initial_cash=config.initial_cash,
                exposure_percent=max(result.average_exposure_percent, 0.0001),
                config=config,
                label=f"MATCHED_{stop_model}_{policy.name}",
            )
            summary = build_variant_summary(result, matched, annual, tickers)
            summary.update(
                {
                    "stop_model": stop_model,
                    "stock_stop_loss_percent": stops[0],
                    "crypto_stop_loss_percent": stops[1],
                    **policy.to_dict(),
                    **_event_counts(events),
                }
            )
            summaries.append(summary)
            for frame, target in (
                (tickers, ticker_frames),
                (annual, annual_frames),
            ):
                if not frame.empty:
                    active = frame.copy()
                    active.insert(0, "stop_model", stop_model)
                    active.insert(1, "replacement_policy", policy.name)
                    target.append(active)
            for reason, count in sorted(rejection_counts(result).items()):
                rejection_rows.append(
                    {
                        "stop_model": stop_model,
                        "replacement_policy": policy.name,
                        "reason_code": reason,
                        "count": count,
                    }
                )
            equity = pd.DataFrame(
                [point.to_dict() for point in result.equity_curve]
            )
            if not equity.empty:
                equity.insert(0, "stop_model", stop_model)
                equity.insert(1, "replacement_policy", policy.name)
                equity_frames.append(equity)
            trades = pd.DataFrame([trade.to_dict() for trade in result.trades])
            if not trades.empty:
                trades.insert(0, "stop_model", stop_model)
                trades.insert(1, "replacement_policy", policy.name)
                trade_frames.append(trades)
    concat = lambda frames: (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    summary = build_comparisons(pd.DataFrame(summaries))
    screen = build_screen(summary)
    return {
        "summary": summary,
        "screen": screen,
        "events": concat(event_frames),
        "tickers": concat(ticker_frames),
        "annual": concat(annual_frames),
        "rejections": pd.DataFrame(rejection_rows),
        "equity": concat(equity_frames),
        "trades": concat(trade_frames),
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint": snapshot["manifest"]["fingerprint"],
        "snapshot_manifest_path": Path(snapshot_path) / "manifest.json",
        "victim_stamp": victim_stamp,
        "victim_paths": source["paths"],
        "policies": [policy.to_dict() for policy in selected_policies],
        "stop_profiles": STOP_PROFILES,
        "base_config": base_config.to_dict(),
    }


def save_rsi_replacement_ablation(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    prefix = "portfolio_rsi_replacement_ablation"
    paths: dict[str, Path] = {}
    for name in (
        "summary",
        "screen",
        "events",
        "tickers",
        "annual",
        "rejections",
        "equity",
        "trades",
    ):
        path = output / f"{prefix}_{name}_{stamp}.csv"
        bundle[name].to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    robust_policies = sorted(
        bundle["screen"].loc[
            bundle["screen"]["robust_policy_pass"], "replacement_policy"
        ].unique()
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Full-period exploratory ablation of causal RSI-quality candidate "
            "gates and executable worst-unrealized-return victim replacement."
        ),
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "victim_stamp": bundle["victim_stamp"],
        "authorized_candidate_feature": AUTHORIZED_FEATURE,
        "authorized_victim_rule": AUTHORIZED_VICTIM_RULE,
        "execution_correction": (
            "Victims require a market bar at the candidate Open; stale holdings "
            "are ineligible, eliminating next-session reference leakage."
        ),
        "atomicity_control": (
            "A shadow exit/entry proves every post-exit portfolio constraint "
            "before the real victim is closed."
        ),
        "policies": bundle["policies"],
        "stop_profiles": bundle["stop_profiles"],
        "robust_policies_authorized_for_walk_forward": robust_policies,
        "any_robust_policy_pass": bool(robust_policies),
        "summary": _safe(bundle["summary"].to_dict("records")),
        "screen": _safe(bundle["screen"].to_dict("records")),
        "limitations": [
            "This is a full-period exploratory ablation, not out-of-sample validation.",
            "The feature and victim rule were discovered on the same frozen history.",
            "Only robust policies may enter a separate train-selected walk-forward.",
            "Independent future shadow validation remains required before production use.",
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
        "victim_stamp": bundle["victim_stamp"],
        "audit_code": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
        },
        "source_files": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in bundle["victim_paths"].items()
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
        "--victim-source-directory", type=Path, default=DEFAULT_VICTIM_DIRECTORY
    )
    parser.add_argument("--victim-stamp", required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[policy.name for policy in REPLACEMENT_POLICIES],
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = run_rsi_replacement_ablation(
        snapshot_path=args.snapshot,
        victim_directory=args.victim_source_directory,
        victim_stamp=args.victim_stamp,
        policy_names=args.policies,
        project_root=Path("."),
    )
    print(bundle["summary"].to_string(index=False))
    print("\nSCREEN")
    print(bundle["screen"].to_string(index=False))
    if args.no_save:
        print("NO_SAVE")
        return
    paths = save_rsi_replacement_ablation(
        bundle, output_directory=args.output_directory
    )
    for name, path in paths.items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
