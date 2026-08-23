"""Missing equity-session detection + narrow auto-replay orchestrator --
"Missing-Equity-Session Remediation" design (workspace-c), sections A-E,
tied together. The real fix for the root cause documented in
`scripts/run_daily_decision.py`'s own module docstring and confirmed by
direct inspection this session: `run_daily_decision()`'s own final
`LiveRunnerState(...)` construction never passes
`last_processed_equity_date`/`last_processed_crypto_date`/
`last_processed_equity_session_date`/`last_processed_equity_bar_timestamp`,
so every real LIVE run (unlike control-arm's, which has its own
`_stamp_last_processed_dates` second save) silently resets those fields
to `None` -- there has never been a working session cursor for the live
path at all until this module's own second save (`_stamp_last_processed_dates`
below, deliberately named and structured the same as
`run_control_arm_decision.py`'s own function of that name, since it is
the exact same proven pattern, reused for a new caller).

PROTECTED-FILE DISCIPLINE: `src/backtest/portfolio_backtest_engine.py`
is never imported or called from this module (see
`equity_session_detection.py`'s own isolation note for detection's own
guarantee). `scripts/run_daily_decision.py` is never edited -- this
module only calls the public `run_daily_decision()` function, always
with `enable_equity_orders=False, enable_crypto_orders=False` (see the
module-level P0 note below for why real orders are never enabled
during a replay run), exactly the same "call the frozen function,
never reimplement its loop" discipline `run_control_arm_decision.py`
already established.

P0 -- REPLAY NEVER SUBMITS A REAL ORDER: every `run_daily_decision()`
call this module makes uses `enable_equity_orders=False,
enable_crypto_orders=False` unconditionally -- there is no parameter
anywhere in this module to override that. A missed historical Open is
completed for the market/DATA timeline only (positions/pending
orders/equity curve catch up in local state, exactly as the batch
engine's own logic would compute them); it is NEVER used as an excuse
to submit a real broker order against a bygone Open a real broker
never actually saw. Any real fill that genuinely happened during the
gap is picked up ONLY by `broker_reconciliation.py`'s own
Scenario C/`equity_stop_orders` invariant checks (already run,
unmodified, as part of this same orchestrator's preflight below) --
never inferred or fabricated here.

SCOPE (owner-approved): this module handles ONLY the narrow,
single-missing-session, execution-window-not-yet-passed auto-replay
case (section B). Multiple missing sessions, or a missing session whose
own execution window has already passed without being replayable, are
explicitly OUT OF SCOPE for automatic handling -- section C's
`handle_missed_execution_window` (stale-BUY-expire/stale-EXIT-clear)
covers the narrower "the window passed, don't fabricate a fill" case;
anything beyond that (multiple gaps) is deferred to a separate,
not-yet-built, human-approved, orders-disabled recovery tool (section
C's own explicit ask), never auto-replayed by this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca.trading.client import TradingClient

import scripts.run_daily_decision as rdd
import src.live.broker_reconciliation as broker_reconciliation
import src.live.equity_session_detection as detection
import src.live.missed_session_window_handling as missed_window
import src.live.session_replay_journal as session_journal
from src.live import order_submission
from src.live.live_universe import LIVE_CONTROLLED_TICKERS
from src.live.single_instance_lock import single_instance_lock

DEFAULT_PROVENANCE_LOG_DIRECTORY = Path("data/live/session_replay_provenance")

# Section E, item 10: "Önerilen otomatik sınır: yalnız execution
# penceresi henüz geçmemiş tek gecikmiş seans." -- a hard, non-tunable
# structural limit, deliberately not a parameter anywhere in this
# module. Anything beyond exactly one missing, still-replayable session
# routes to `handle_missed_execution_window` (never fabricates a fill)
# or raises directing to the separate, not-yet-built manual recovery
# tool -- never a bigger automatic batch replay.
_MAX_AUTO_REPLAY_SESSIONS = 1


class SessionReplayFailClosedError(RuntimeError):
    """Base class for every fail-closed condition this orchestrator
    enforces (design section E's 10-item table) -- see each raise
    site's own message for which specific condition fired."""


class SessionCursorAheadOfCalendarError(SessionReplayFailClosedError):
    """Section E, item 5: `last_processed_equity_session_date` is AFTER
    today's real calendar-confirmed date -- a rollback-like anomaly,
    never silently trusted."""


class StaleSessionReplayJournalError(SessionReplayFailClosedError):
    """Section E, item 7: a PREPARED/VERIFIED session-replay journal
    entry from a prior, interrupted run still exists on disk -- refuses
    to start a NEW replay attempt until that is resolved (human
    review), never silently ignored or overwritten."""


class TooManyMissingSessionsError(SessionReplayFailClosedError):
    """Section E, item 10: more than `_MAX_AUTO_REPLAY_SESSIONS` session(s)
    are missing -- automatic replay is refused entirely; this is a
    multi-gap recovery case, out of this module's scope (see module
    docstring)."""


class SessionOutcomeMismatchError(SessionReplayFailClosedError):
    """The actually-processed session
    (`decision["as_of_bar_timestamp_equity"]`) does not match the
    session this replay attempt targeted -- mirrors
    `run_control_arm_decision._verify_actual_equity_session`'s own
    real-vs-expected check, applied to the replay path."""


@dataclass(frozen=True)
class OrchestratorResult:
    outcome: str  # "CLEAN_NO_OP" | "REPLAYED_ONE_SESSION" | "MISSED_WINDOW_HANDLED"
    detection: detection.DetectionResult | None = None
    replayed_session_date: str | None = None
    missed_window_outcome: missed_window.MissedWindowOutcome | None = None
    provenance_log_path: Path | None = None


def _hash_state_file(state_path: Path) -> str | None:
    path = Path(state_path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bar_set_sha256(raw_bars_by_ticker: dict[str, Any], session_date: str) -> str:
    """Deterministic hash of exactly what this session's bar set looked
    like at detection time -- included in the session_id itself (see
    `session_replay_journal.build_session_id`'s own docstring for why),
    so a bar set that later turns out to have been incomplete produces
    a genuinely different session_id, never silently conflated with an
    earlier attempt for the same calendar date."""
    parts = []
    for ticker in sorted(raw_bars_by_ticker):
        frame = raw_bars_by_ticker[ticker]
        if session_date in {ts.date().isoformat() for ts in frame.index}:
            row = frame.loc[[ts for ts in frame.index if ts.date().isoformat() == session_date][0]]
            parts.append(f"{ticker}:{row['Open']}:{row['High']}:{row['Low']}:{row['Close']}:{row['Volume']}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _stamp_last_processed_dates(state_path: Path, guard_path: Path, *, actual_session_date: str) -> None:
    """The real fix for the root cause this whole module exists to
    close -- see module docstring. Identical pattern to
    `run_control_arm_decision._stamp_last_processed_dates` (that
    function's own docstring explains the full "why a second save"
    reasoning; not repeated here, just reused): `run_daily_decision()`'s
    own internal save never populates these fields, so a second,
    separate read-modify-write after it returns is the only way they
    ever become durable for the LIVE path -- which, unlike control-arm,
    has never had ANY caller doing this until now."""
    today = datetime.now(timezone.utc).date().isoformat()
    runner_state = rdd.ps.load_position_state(state_path, guard_path=guard_path)
    runner_state.last_processed_equity_date = today
    runner_state.last_processed_crypto_date = today
    runner_state.last_processed_equity_session_date = actual_session_date
    runner_state.last_processed_equity_bar_timestamp = actual_session_date
    rdd.ps.save_position_state(runner_state, state_path, guard_path=guard_path)


def _write_provenance_log(
    *,
    directory: Path,
    processing_mode: str,
    equity_session_date: str | None,
    expected_latest_completed_session: str | None,
    last_processed_session_before: str | None,
    last_processed_session_after: str | None,
    replay_sequence_index: int | None,
    replay_sequence_total: int | None,
    bar_set_sha256: str | None,
    missing_tickers: tuple[str, ...],
    state_hash_before: str | None,
    state_hash_after: str | None,
    real_decision_log_path: str | None,
    account_number_masked: str | None,
) -> Path:
    """Section F: the full provenance schema, written as a SEPARATE
    JSON file cross-referencing the real decision log
    `run_daily_decision()` itself already writes unchanged -- never
    injected into that function's own decision dict (which would
    require editing the protected file). One record per orchestrator
    run, whether or not anything was actually replayed."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"session_replay_provenance_{stamp}.json"
    payload = {
        "processing_mode": processing_mode,
        "equity_session_date": equity_session_date,
        "expected_latest_completed_session": expected_latest_completed_session,
        "last_processed_session_before": last_processed_session_before,
        "last_processed_session_after": last_processed_session_after,
        "replay_sequence_index": replay_sequence_index,
        "replay_sequence_total": replay_sequence_total,
        "replay_detected_at_utc": datetime.now(timezone.utc).isoformat(),
        "calendar_source": "alpaca.trading.client.TradingClient.get_calendar",
        "bar_set_sha256": bar_set_sha256,
        "missing_tickers": list(missing_tickers),
        "market_data_provider": "alpaca",
        "market_data_feed": "iex",
        "market_data_adjustment": "all",
        "market_data_timeframe": "1Day",
        "state_hash_before": state_hash_before,
        "state_hash_after": state_hash_after,
        "real_decision_log_path": real_decision_log_path,
        "broker_account_masked": account_number_masked,
        "order_submission_allowed": False,
        "order_submission_block_reason": "replay never submits real orders -- P0 principle, see module docstring",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run_missing_session_replay(
    *,
    state_path: Path = rdd.DEFAULT_STATE_PATH,
    decision_log_directory: Path = rdd.DEFAULT_DECISION_LOG_DIRECTORY,
    guard_path: Path = rdd.ps.HIGH_WATER_MARK_PATH,
    provenance_log_directory: Path = DEFAULT_PROVENANCE_LOG_DIRECTORY,
    trading_client: TradingClient | None = None,
    tickers: tuple[str, ...] = LIVE_CONTROLLED_TICKERS,
) -> OrchestratorResult:
    """The real orchestration entry point -- sections A-E, tied
    together. `state_path`/`decision_log_directory`/`guard_path` default
    to the SAME paths `run_daily_decision.py`'s own `main()` uses,
    DELIBERATELY -- this module fixes the LIVE path's own session
    cursor, not a separate, isolated universe. Isolation from any OTHER
    deployment sharing this codebase (e.g. the control arm) comes from
    the SAME convention already established throughout this codebase:
    process `cwd` plus the `AI_STOCK_RADAR_GUARD_DIR` environment
    variable (see `scripts/run_control_arm_decision.py`'s own module
    docstring, "Isolation from the live deployment is NOT implemented
    in this file") -- this script must be launched from the LIVE
    deployment's own directory, never a shared/other one. See this
    module's own test file for a real, explicit assertion that this
    module's default `state_path` and `run_control_arm_decision.py`'s
    own default `--state-path` argparse setting resolve to the
    identical path constant, so this documented isolation model is
    checked, not merely asserted in prose (a real, disclosed gap this
    session's own earlier fact-finding found: nothing previously tested
    this).
    """
    with single_instance_lock(state_path):
        if rdd.STOP_FLAG_PATH.exists():
            return OrchestratorResult(outcome="CLEAN_NO_OP")

        if trading_client is None:
            trading_client = order_submission.get_trading_client()

        stray_journal_entries = [
            intent for intent in session_journal.list_session_intents()
            if intent.status in (session_journal.PREPARED, session_journal.VERIFIED, session_journal.COMMITTED)
        ]
        if stray_journal_entries:
            raise StaleSessionReplayJournalError(
                f"{len(stray_journal_entries)} session-replay journal entr(y/ies) from a prior, "
                f"interrupted run are still non-TERMINAL (PREPARED/VERIFIED/COMMITTED) -- refusing "
                f"to start a new replay attempt until these are resolved (human review): "
                f"{[(i.intent_id, i.session_id, i.status) for i in stray_journal_entries]}"
            )

        live_account_suffix = rdd._require_live_account_suffix()
        runner_state = rdd.ps.load_position_state(state_path, guard_path=guard_path)

        # Section E, item 8: broker reconciliation must pass before
        # anything else -- reused unmodified, dry-run-only (this
        # orchestrator never enables real orders, see module docstring).
        broker_reconciliation.reconcile(
            trading_client, runner_state,
            expected_account_suffix=live_account_suffix,
            equity_orders_enabled=False, crypto_orders_enabled=False,
        )
        account_number_masked = broker_reconciliation._mask_account_number(str(trading_client.get_account().account_number))

        last_processed_session_before = runner_state.last_processed_equity_session_date
        today_iso = datetime.now(timezone.utc).date().isoformat()
        if last_processed_session_before is not None and last_processed_session_before > today_iso:
            raise SessionCursorAheadOfCalendarError(
                f"last_processed_equity_session_date ({last_processed_session_before}) is AFTER "
                f"today's real calendar date ({today_iso}) -- a rollback-like anomaly. Fail-closed."
            )

        open_equity_positions = frozenset(
            ticker for ticker, position in runner_state.positions.items() if position.asset_class == "EQUITY"
        )

        if last_processed_session_before is None:
            # No session cursor has EVER been recorded for this state
            # file (exactly the root-cause bug this module exists to
            # fix, on its very first run) -- nothing to detect a GAP
            # against yet. A clean no-op; the next real
            # run_daily_decision() call will process today normally,
            # and this module's own second save will start the cursor
            # from there.
            return OrchestratorResult(outcome="CLEAN_NO_OP")

        detection_result = detection.detect_missing_equity_sessions(
            trading_client,
            tickers=tuple(t for t in tickers if not detection._is_crypto_ticker(t)),
            open_position_tickers=open_equity_positions,
            since_date=last_processed_session_before,
        )
        # detect_missing_equity_sessions never raises merely because
        # something is missing (see that module's own docstring,
        # "FAIL-CLOSED IS THE ORCHESTRATOR'S DEFAULT") -- everything
        # from here on is THIS function deciding which of section B
        # (narrow auto-replay), section C (missed-window handling), or
        # fail-closed applies to the real facts just detected.

        if not detection_result.expected_sessions:
            return OrchestratorResult(outcome="CLEAN_NO_OP", detection=detection_result)

        if detection_result.unexpected_future_bars:
            raise SessionReplayFailClosedError(
                f"Unexpected bar(s) dated after the latest calendar-expected session: "
                f"{detection_result.unexpected_future_bars}. Fail-closed; investigate before proceeding."
            )

        latest_expected = detection_result.expected_sessions[-1]

        if not detection_result.missing_by_session:
            # Every expected session's bar set is fully complete RIGHT
            # NOW. If the cursor is exactly one session behind, this is
            # section B's narrow eligible case: `latest_expected` is (by
            # `fetch_expected_equity_sessions`'s own settle-buffer
            # filtering) the single, most-recently-settled session, so
            # no newer session has settled yet -- its own next execution
            # window has not passed. More than one session behind with
            # fully-complete data is a genuine multi-gap catch-up case,
            # out of this module's scope.
            if len(detection_result.expected_sessions) > _MAX_AUTO_REPLAY_SESSIONS:
                raise TooManyMissingSessionsError(
                    f"{len(detection_result.expected_sessions)} expected session(s) "
                    f"{detection_result.expected_sessions} are all bar-complete, but the session "
                    f"cursor ({last_processed_session_before!r}) is more than "
                    f"{_MAX_AUTO_REPLAY_SESSIONS} session(s) behind -- a multi-session gap. Out of "
                    f"scope for automatic replay; use the (separate, not-yet-built) manual "
                    f"recovery tool."
                )
            return _replay_one_session(
                latest_expected,
                detection_result,
                state_path=Path(state_path),
                decision_log_directory=Path(decision_log_directory),
                guard_path=Path(guard_path),
                trading_client=trading_client,
                provenance_log_directory=provenance_log_directory,
                last_processed_session_before=last_processed_session_before,
                account_number_masked=account_number_masked,
            )

        # At least one expected session is still missing bar data right now.
        if len(detection_result.missing_by_session) > _MAX_AUTO_REPLAY_SESSIONS:
            raise TooManyMissingSessionsError(
                f"{len(detection_result.missing_by_session)} expected session(s) are missing bar "
                f"data: {detection_result.missing_by_session}. Out of scope for automatic "
                f"handling (more than {_MAX_AUTO_REPLAY_SESSIONS}); use the (separate, "
                f"not-yet-built) manual recovery tool."
            )

        missing_session = next(iter(detection_result.missing_by_session))
        if missing_session == latest_expected and len(detection_result.expected_sessions) == 1:
            # The ONLY expected session simply has not arrived yet -- no
            # newer session has settled on top of it, so its own
            # execution window has not passed either. Not an error and
            # not yet replayable; a clean no-op until the data arrives.
            return OrchestratorResult(outcome="CLEAN_NO_OP", detection=detection_result)

        # The missing session is not simply "hasn't arrived yet in a
        # single-session context": either a newer session has already
        # settled on top of it (its own execution window has passed) or
        # multiple sessions are affected. Either way, section C applies
        # -- never fabricate a fill; just clear the stale queued signals
        # so the next real run re-evaluates fresh from today's own Close.
        missed_dates = tuple(detection_result.missing_by_session)
        window_outcome = missed_window.handle_missed_execution_window(
            runner_state, missed_session_dates=missed_dates,
        )
        rdd.ps.save_position_state(runner_state, state_path, guard_path=guard_path)
        provenance_path = _write_provenance_log(
            directory=provenance_log_directory,
            processing_mode="MISSED_WINDOW_HANDLED",
            equity_session_date=missing_session,
            expected_latest_completed_session=latest_expected,
            last_processed_session_before=last_processed_session_before,
            last_processed_session_after=runner_state.last_processed_equity_session_date,
            replay_sequence_index=None,
            replay_sequence_total=None,
            bar_set_sha256=None,
            missing_tickers=detection_result.missing_by_session[missing_session],
            state_hash_before=None,
            state_hash_after=None,
            real_decision_log_path=None,
            account_number_masked=account_number_masked,
        )
        return OrchestratorResult(
            outcome="MISSED_WINDOW_HANDLED",
            detection=detection_result,
            missed_window_outcome=window_outcome,
            provenance_log_path=provenance_path,
        )


def _replay_one_session(
    session_date: str,
    detection_result: detection.DetectionResult,
    *,
    state_path: Path,
    decision_log_directory: Path,
    guard_path: Path,
    trading_client: TradingClient,
    provenance_log_directory: Path,
    last_processed_session_before: str | None,
    account_number_masked: str | None,
) -> OrchestratorResult:
    """Section D's sequential replay transaction for exactly one
    session -- the narrow, section-B-eligible auto-replay case. The
    caller (`run_missing_session_replay`) has already confirmed
    eligibility (this session's own bar set is fully complete, and the
    cursor is exactly one session behind it) before calling this;
    nothing here re-derives that determination."""
    bar_hash = _bar_set_sha256(detection_result.raw_bars_by_ticker, session_date)
    session_id = session_journal.build_session_id(session_date, bar_hash)

    existing = session_journal.find_session_intent_by_id(session_id)
    if existing is not None and existing.status == session_journal.TERMINAL:
        # Already fully replayed in a prior run of this same orchestrator
        # (e.g. re-invoked after a clean exit before the cursor's own
        # next detection pass moved past it) -- never replay the same
        # session twice.
        return OrchestratorResult(outcome="CLEAN_NO_OP", detection=detection_result)

    state_hash_before = _hash_state_file(state_path)
    intent = session_journal.create_session_intent(
        session_id=session_id,
        session_date=session_date,
        bar_set_sha256=bar_hash,
        replay_sequence_index=1,
        replay_sequence_total=1,
        state_hash_before=state_hash_before,
    )

    try:
        result = rdd.run_daily_decision(
            state_path=state_path,
            decision_log_directory=decision_log_directory,
            enable_equity_orders=False,
            enable_crypto_orders=False,
            guard_path=guard_path,
            trading_client=trading_client,
            skip_broker_reconciliation=True,
        )
    except Exception as error:
        session_journal.transition_session_intent(
            intent, session_journal.TERMINAL, last_error=f"{type(error).__name__}: {error}",
        )
        raise

    decision = result["decision"]
    actual_session = decision.get("as_of_bar_timestamp_equity")
    if actual_session != session_date:
        session_journal.transition_session_intent(
            intent, session_journal.TERMINAL,
            last_error=(
                f"Replay targeted session {session_date!r} but run_daily_decision() actually "
                f"processed {actual_session!r}."
            ),
        )
        raise SessionOutcomeMismatchError(
            f"Replay for {session_date} produced decision['as_of_bar_timestamp_equity'] = "
            f"{actual_session!r} -- does not match the targeted session. Fail-closed; "
            f"session-replay journal entry {intent.intent_id} closed TERMINAL with this error."
        )

    session_journal.transition_session_intent(intent, session_journal.VERIFIED)

    _stamp_last_processed_dates(state_path, guard_path, actual_session_date=actual_session)

    state_hash_after = _hash_state_file(state_path)
    session_journal.transition_session_intent(intent, session_journal.COMMITTED, state_hash_after=state_hash_after)
    session_journal.transition_session_intent(intent, session_journal.TERMINAL)

    provenance_path = _write_provenance_log(
        directory=provenance_log_directory,
        processing_mode="REPLAYED_ONE_SESSION",
        equity_session_date=session_date,
        expected_latest_completed_session=detection_result.expected_sessions[-1],
        last_processed_session_before=last_processed_session_before,
        last_processed_session_after=actual_session,
        replay_sequence_index=1,
        replay_sequence_total=1,
        bar_set_sha256=bar_hash,
        missing_tickers=(),
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        real_decision_log_path=str(result["log_path"]),
        account_number_masked=account_number_masked,
    )

    return OrchestratorResult(
        outcome="REPLAYED_ONE_SESSION",
        detection=detection_result,
        replayed_session_date=session_date,
        provenance_log_path=provenance_path,
    )
