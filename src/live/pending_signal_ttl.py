"""Phase 2b: session-based, single-opportunity TTL for control-arm pending
BUY/EXIT signals.

WHY THIS EXISTS: `run_daily_decision()`'s own pending-order execution rule
is `submitted_portfolio_bar_index < portfolio_bar_index` -- ANY future run
with a higher bar index may execute a signal queued long ago, at whatever
price/context exists on the day the control arm finally gets around to it.
That is correct for the frozen backtest engine (which always advances one
bar at a time, never skips), but wrong for a live daily runner that can be
down for days (STOP, FREEZE, a crash, an outage): a BUY queued from a Close
a week ago executing today is not "the next available Open" the strategy's
own signal was scored against -- it is a stale decision on stale
information. This module closes that gap WITHOUT touching
`run_daily_decision.py` (forbidden -- see CLAUDE.md): it prunes stale
pending signals from `position_state.json` BEFORE `run_daily_decision()`
ever loads it, so the frozen engine simply never sees them.

THE ONE-SHOT RULE: a signal generated at Close session S is valid ONLY for
T, the first real equity session after S (an actual Alpaca-calendar
session, never calendar/weekend arithmetic). `target_execution_session_date`
(=T) is fixed EXACTLY ONCE, via one real `TradingClient.get_calendar()`
call, at the moment a signal is first recorded (`next_equity_session_date`,
called only from `stamp_new_pending_signals`). Every later run's staleness
check (`evaluate_pending_signals`) is then a PURE, local, no-API-call
comparison: `today's confirmed equity session > T` means the one-shot
window has passed.

THE CRYPTO-DATE BUG THIS WORKS AROUND (found in `run_daily_decision.py`,
never fixed there -- fixing it would mean editing the frozen file):
`run_daily_decision()` stamps EVERY `PortfolioSignal.timestamp` -- equity
and crypto alike -- with `crypto_date` (see that file's `timestamp_string
= crypto_date`, ~line 976, fed into `_queue_close_based_exits` and
`_queue_ranked_entry_signals`, ~lines 1013-1024). Crypto trades 24/7, so
`crypto_date` can differ from `equity_date` (any equity weekend/holiday).
This module NEVER reads a `PortfolioSignal`'s own `.timestamp` for an
equity ticker's `source_session_date` -- it always uses the run-level
`decision["as_of_bar_timestamp_equity"]` (already correct, unaffected by
the bug; see `run_daily_decision.py`'s own `decision` dict construction),
supplied by the caller.

STORAGE: `LiveRunnerState.pending_signal_metadata` (added to
`position_state.py` this task, backward-compatible, same pattern as
Phase 1's `last_processed_*` fields) -- one record per `f"{ticker}|{kind}"`
key (`kind` is `"BUY"` or `"EXIT"`, matching `pending_buys`/`pending_exits`
being ticker-keyed, at most one of each per ticker at a time). Fields:
`source_session_date`, `target_execution_session_date`, `created_at_utc`,
`signal_id`, `status` (`pending`/`expired`/`reconfirmed`/`canceled`, plus
the transient `reconfirming` used only mid-run between
`evaluate_pending_signals` and `finalize_after_run`), `expire_reason`.

Same "second, deliberate save" caveat as Phase 1's `last_processed_*`
fields: `run_daily_decision()`'s own internal `save_position_state()` call
builds a fresh `LiveRunnerState` that knows nothing about
`pending_signal_metadata` and wipes it to `{}` on every run. The caller
(`run_control_arm_decision.py`) is responsible for reloading, re-applying,
and re-saving this module's records AFTER `run_daily_decision()` returns
-- see `finalize_after_run`.

SIGNAL_ID / JOURNAL CORRELATION: `signal_id` is never a fresh random id --
it is computed with the SAME deterministic formula
`run_daily_decision._deterministic_client_order_id(ticker, action_kind,
source_session_date)` that `run_control_arm_decision._run_intent_protocol`
already uses for `order_intent.OrderIntent.client_order_id` on queued BUY
signals (`action_kind="QUEUED_ENTRY_SIGNAL"`). For a BUY, this means
`signal_id` here and the matching intent's `client_order_id` in
`order_intent.py` are IDENTICAL strings for the same ticker+session -- one
real correlation key, not a second ledger. This module never computes that
id itself (avoiding a `scripts.*` import, keeping this a plain `src/live`
module); the caller injects it via `compute_signal_id`. For EXIT signals,
`order_intent.py` does not currently create any journal entry at all
(`_run_intent_protocol` only walks `queued_for_next_run.buys` and
`executed_today.entries`) -- `signal_id` is still computed and stored for
forward-compatibility and audit-trail completeness, but
`has_unresolved_journal_trace` will correctly never find a match for an
EXIT today; that is an honest, disclosed Phase 1 gap, not something this
module papers over.

ROLLOUT COMPATIBILITY: a `pending_buys`/`pending_exits` entry with no
matching `pending_signal_metadata` record (or a record not in `pending`
status) is treated as `age_unknown` -- unconditionally routed through the
same stale-handling path as an expired record (never trusted as "still
within its window" just because no contrary evidence exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

KIND_BUY = "BUY"
KIND_EXIT = "EXIT"

STATUS_PENDING = "pending"
STATUS_EXPIRED = "expired"
STATUS_RECONFIRMED = "reconfirmed"
STATUS_CANCELED = "canceled"
_STATUS_RECONFIRMING = "reconfirming"  # transient, mid-run only -- never left on disk after finalize_after_run

REASON_STOP_MISSED_SESSION = "stop_missed_session"
REASON_FREEZE_MISSED_SESSION = "freeze_missed_session"
REASON_RUNNER_OUTAGE = "runner_outage"
# EXIT-only: the frozen engine re-evaluated the exit condition fresh
# against the latest completed session and it no longer holds. Never used
# for a BUY (a BUY's only fate is opportunity-window expiry, never
# reconfirmation).
REASON_RECONFIRMATION_INVALID = "reconfirmation_invalid"


class PendingSignalReconciliationRequiredError(RuntimeError):
    """Fail-closed: a stale pending EXIT could not be safely auto-resolved
    (no local position exists to reconfirm the exit condition against).
    Human review required -- see module docstring, Scenario 5e. Never
    raised for a BUY (a stale BUY's resolution, expire-or-quarantine, is
    always mechanical, never ambiguous in this way)."""


def pending_key(ticker: str, kind: str) -> str:
    return f"{ticker}|{kind}"


def next_equity_session_date(client: TradingClient, after_date_iso: str) -> str:
    """The ONE real Alpaca calendar call this module makes: the first real
    equity session strictly after `after_date_iso`. Called only once per
    signal, at creation time (`stamp_new_pending_signals`) -- every later
    staleness check is a pure local string comparison against the value
    this returns, never a fresh calendar call."""
    start = date.fromisoformat(after_date_iso) + timedelta(days=1)
    end = start + timedelta(days=14)  # comfortably covers any real holiday cluster
    calendar = client.get_calendar(GetCalendarRequest(start=start, end=end))
    if not calendar:
        raise RuntimeError(
            f"No equity session found in the 14 days after {after_date_iso} "
            f"({start} to {end}) -- unexpected for a real trading calendar; "
            f"investigate before proceeding."
        )
    return calendar[0].date.isoformat()


def _new_record(*, signal_id: str, source_session_date: str, target_execution_session_date: str) -> dict:
    return {
        "signal_id": signal_id,
        "source_session_date": source_session_date,
        "target_execution_session_date": target_execution_session_date,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": STATUS_PENDING,
        "expire_reason": None,
    }


@dataclass
class PendingSignalTTLOutcome:
    expired_buys: dict[str, dict] = field(default_factory=dict)
    quarantined_buys: dict[str, dict] = field(default_factory=dict)
    reconfirming_exits: dict[str, dict] = field(default_factory=dict)  # ticker -> the OLD (stale) record being reconfirmed
    quarantined_exits: dict[str, dict] = field(default_factory=dict)

    @property
    def touched_anything(self) -> bool:
        return bool(self.expired_buys or self.quarantined_buys or self.reconfirming_exits or self.quarantined_exits)


def evaluate_pending_signals(
    *,
    runner_state: Any,
    expected_session_date: str,
    freeze_active: bool,
    has_unresolved_journal_trace_for_signal: Callable[[str], bool],
    has_unresolved_journal_trace_for_ticker: Callable[[str, str], bool],
) -> PendingSignalTTLOutcome:
    """Called BEFORE `run_daily_decision()`. Pure local comparison, makes
    NO API call itself. MUTATES `runner_state.pending_buys`,
    `runner_state.pending_exits`, and `runner_state.pending_signal_metadata`
    in place.

    Two distinct journal-lookup callables, not one, because an
    `age_unknown` record (no metadata -- see module docstring's "ROLLOUT
    COMPATIBILITY") has no stored `signal_id` to look up exactly: we do
    not know what `source_session_date` the original signal (if any) was
    created with, so the deterministic-id formula cannot be reproduced
    for it. `has_unresolved_journal_trace_for_ticker` instead does a
    broader, ticker+kind-level journal sweep (any non-terminal intent for
    that ticker's BUY/EXIT family, any date) for that case specifically.
    A record WITH known metadata always uses the exact, narrow
    `has_unresolved_journal_trace_for_signal(record["signal_id"])` lookup.

    - A still-within-window signal (metadata exists, status=='pending',
      `expected_session_date <= target_execution_session_date`) is left
      completely untouched -- `run_daily_decision()` will process it
      normally this run, exactly as today.
    - A stale or age_unknown BUY with an unresolved journal trace is
      QUARANTINED: left in `pending_buys` untouched, reported separately,
      never auto-expired (see module docstring's "SIGNAL_ID / JOURNAL
      CORRELATION"). Without a trace, it is removed from `pending_buys`
      and its metadata record finalized to `expired` -- unconditionally,
      no human-approved resubmission path exists in this module.
    - A stale or age_unknown EXIT with an unresolved journal trace is
      QUARANTINED the same way. Without a trace: if the ticker has no
      local open position, raises `PendingSignalReconciliationRequiredError`
      (fail-closed -- ambiguous, needs a human). Otherwise the stale
      `pending_exits` entry is removed (so `run_daily_decision()`'s own
      unmodified `_queue_close_based_exits` step re-evaluates the exit
      condition fresh, against the CURRENT Close, later in this exact
      same run) and the metadata record is marked `reconfirming`
      (transient) pending `finalize_after_run`'s outcome check.

    Raises `PendingSignalReconciliationRequiredError` immediately on the
    first ambiguous EXIT found -- the caller's own exception handling
    (same as every other preflight check) then skips
    `run_daily_decision()` entirely for this run.
    """
    metadata = runner_state.pending_signal_metadata
    outcome = PendingSignalTTLOutcome()

    for ticker in list(runner_state.pending_buys.keys()):
        key = pending_key(ticker, KIND_BUY)
        record = metadata.get(key)
        age_unknown = record is None or record.get("status") != STATUS_PENDING
        if age_unknown:
            is_stale = True  # never trusted as still-fresh with no record
            has_trace = has_unresolved_journal_trace_for_ticker(ticker, KIND_BUY)
        else:
            is_stale = expected_session_date > record["target_execution_session_date"]
            has_trace = False if not is_stale else has_unresolved_journal_trace_for_signal(record["signal_id"])
        if not is_stale:
            continue
        if has_trace:
            outcome.quarantined_buys[ticker] = record or {"status": "age_unknown"}
            continue
        del runner_state.pending_buys[ticker]
        finalized = dict(record) if record else _new_record(
            signal_id=None, source_session_date="unknown", target_execution_session_date="unknown"
        )
        finalized["status"] = STATUS_EXPIRED
        finalized["expire_reason"] = REASON_FREEZE_MISSED_SESSION if freeze_active else REASON_RUNNER_OUTAGE
        metadata[key] = finalized
        outcome.expired_buys[ticker] = finalized

    for ticker in list(runner_state.pending_exits.keys()):
        key = pending_key(ticker, KIND_EXIT)
        record = metadata.get(key)
        age_unknown = record is None or record.get("status") != STATUS_PENDING
        if age_unknown:
            is_stale = True
            has_trace = has_unresolved_journal_trace_for_ticker(ticker, KIND_EXIT)
        else:
            is_stale = expected_session_date > record["target_execution_session_date"]
            has_trace = False if not is_stale else has_unresolved_journal_trace_for_signal(record["signal_id"])
        if not is_stale:
            continue
        if has_trace:
            outcome.quarantined_exits[ticker] = record or {"status": "age_unknown"}
            continue
        if ticker not in runner_state.positions:
            raise PendingSignalReconciliationRequiredError(
                f"Stale pending EXIT for {ticker} (target session "
                f"{record.get('target_execution_session_date') if record else 'unknown'}, "
                f"now {expected_session_date}) but no open local position "
                f"exists for {ticker} to reconfirm the exit condition "
                f"against -- ambiguous, fail-closed. A human must resolve "
                f"this; the stale order must never be directly resubmitted."
            )
        del runner_state.pending_exits[ticker]
        stale_record = dict(record) if record else _new_record(
            signal_id=None, source_session_date="unknown", target_execution_session_date="unknown"
        )
        stale_record["status"] = _STATUS_RECONFIRMING
        metadata[key] = stale_record
        outcome.reconfirming_exits[ticker] = stale_record

    return outcome


def stamp_new_pending_signals(
    *,
    client: TradingClient,
    runner_state: Any,
    decision: dict,
    equity_session_date: str,
    compute_signal_id: Callable[[str, str, str], str],
    reconfirming_exit_old_records: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Called AFTER `run_daily_decision()` returns (and after the caller's
    own audit checks pass). For every ticker in `decision['queued_for_next_run']`
    that does not already have a fresh `pending` metadata record for
    today's session, creates one -- computing `target_execution_session_date`
    via the ONE real calendar call this run makes for that purpose
    (`next_equity_session_date`). For a ticker present in
    `reconfirming_exit_old_records` (a stale EXIT that
    `evaluate_pending_signals` cleared this same run for reconfirmation),
    the OLD record is finalized to `reconfirmed` (moved to the audit
    trail, not deleted) exactly when a fresh queued exit for that same
    ticker is what triggered the new record below -- i.e. the frozen
    engine's own unmodified `_queue_close_based_exits` re-evaluation
    found the exit condition still valid.

    Returns `{ticker: outcome}` for tickers in `reconfirming_exit_old_records`
    that did NOT get a fresh queued exit this run -- `"expired_after_reconfirmation"`,
    finalized with `expire_reason=REASON_RECONFIRMATION_INVALID` -- so the
    caller can notify. Never touches `runner_state.positions` under any
    outcome."""
    metadata = runner_state.pending_signal_metadata
    reconfirming_exit_old_records = reconfirming_exit_old_records or {}
    reconfirmation_outcomes: dict[str, str] = {}

    queued_buy_tickers = {entry["ticker"] for entry in decision.get("queued_for_next_run", {}).get("buys", [])}
    queued_exit_tickers = {entry["ticker"] for entry in decision.get("queued_for_next_run", {}).get("exits", [])}

    for ticker in queued_buy_tickers:
        key = pending_key(ticker, KIND_BUY)
        existing = metadata.get(key)
        if existing is not None and existing.get("status") == STATUS_PENDING and existing.get("source_session_date") == equity_session_date:
            continue  # already stamped for today's session -- nothing to do
        signal_id = compute_signal_id(ticker, KIND_BUY, equity_session_date)
        target = next_equity_session_date(client, equity_session_date)
        metadata[key] = _new_record(
            signal_id=signal_id, source_session_date=equity_session_date, target_execution_session_date=target
        )

    for ticker in queued_exit_tickers:
        key = pending_key(ticker, KIND_EXIT)
        existing = metadata.get(key)
        was_reconfirming = ticker in reconfirming_exit_old_records
        if not was_reconfirming and existing is not None and existing.get("status") == STATUS_PENDING and existing.get("source_session_date") == equity_session_date:
            continue
        signal_id = compute_signal_id(ticker, KIND_EXIT, equity_session_date)
        target = next_equity_session_date(client, equity_session_date)
        metadata[key] = _new_record(
            signal_id=signal_id, source_session_date=equity_session_date, target_execution_session_date=target
        )
        if was_reconfirming:
            old_key = pending_key(ticker, KIND_EXIT)
            old_record = dict(reconfirming_exit_old_records[ticker])
            old_record["status"] = STATUS_RECONFIRMED
            old_record["expire_reason"] = None
            # The NEW record above already occupies `metadata[old_key]` --
            # the reconfirmed OLD record is what gets reported to the
            # caller for notification/audit; it is not re-stored under
            # the same live key (that would overwrite the fresh record
            # that replaced it), matching "old record finalized, new one
            # takes its place" from the module docstring.
            reconfirmation_outcomes[ticker] = STATUS_RECONFIRMED

    for ticker, old_record in reconfirming_exit_old_records.items():
        if ticker in reconfirmation_outcomes:
            continue  # already resolved as reconfirmed above
        finalized = dict(old_record)
        finalized["status"] = STATUS_EXPIRED
        finalized["expire_reason"] = REASON_RECONFIRMATION_INVALID
        metadata[pending_key(ticker, KIND_EXIT)] = finalized
        reconfirmation_outcomes[ticker] = "expired_after_reconfirmation"

    return reconfirmation_outcomes
