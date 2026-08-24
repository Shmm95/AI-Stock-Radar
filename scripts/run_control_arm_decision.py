"""Control-arm operational entry point for the frozen TREND_RSI strategy.

THIS FILE, NOT `run_daily_decision.py`, is what the control arm's cron
job must invoke. `run_daily_decision.py` itself is never edited (see
CLAUDE.md's protected-areas policy and this task's own instruction);
this is a separate, standalone wrapper that reuses its
`run_daily_decision()` function (never reimplements the frozen
entry/exit/stop logic) while replacing:
  - the ticker universe it operates on (44 MidCap tickers from
    `src.live.control_universe`, + BTC-USD as regime-reference data only)
  - preventing BTC-USD from ever being a tradable candidate itself
  - restoring the STOP/FREEZE/guard kill-switch checks that a bare call
    to `run_daily_decision()` bypassing `main()` would skip entirely
    (found in an external audit of this session's earlier scratchpad
    dry-run script, which called `run_daily_decision()` directly)
  - tagging every Telegram notification `[CONTROL]` so it can never be
    confused with the live system's own messages
  - failing loudly, never silently, if a credential-file override this
    script is asked to apply cannot actually be resolved/loaded

Why not a CLI flag on `run_daily_decision.py` itself: modifying that
file is explicitly out of scope. Every technique here is data/monkey-
patch-only, applied to the SAME two names `run_daily_decision()`
already resolves as plain module globals at call time
(`LIVE_CONTROLLED_TICKERS`, `prepare_live_market_data`) -- confirmed by
reading that function's body directly: both are bare-name references
inside it, so CPython's LOAD_GLOBAL resolves them from
`scripts.run_daily_decision`'s own `__dict__` at CALL time, not at
import/def time, which is exactly what makes patching those two module
attributes before calling `run_daily_decision()` equivalent to editing
the file. Verified empirically, not just reasoned about: an earlier ad
hoc version of the ticker-only half of this technique (this session's
throwaway scratchpad `dry_run.py`) produced a real PINS entry signal --
PINS is one of the 44 control tickers, not a live-universe-only ticker
-- proving the patched name really was what the function used.

BTC-USD "regime reference only, never tradable" mechanism: BTC-USD MUST
be present in the tickers passed to `prepare_live_market_data` (that
function enforces this itself, raising `ValueError` otherwise -- see
its `REGIME_SYMBOL` check) because the crypto bull-regime filter needs
BTC's own EMA200. But `prepare_live_market_data` has no parameter to
compute that and then exclude BTC-USD from its returned dict, and
`_bars_today_by_asset_class` (called inside `run_daily_decision()`)
separately REQUIRES at least one crypto ticker to be present in that
same dict, or it raises `ValueError` too -- so BTC-USD cannot simply be
deleted from the dict returned to `run_daily_decision()`. Instead:
after the real `prepare_live_market_data()` call computes BTC-USD's
data (and its regime EMA200, which nothing else in this 44-equity
control universe actually needs -- equities always get
`RegimeAllowed=True` unconditionally, independent of BTC), this wrapper
force-sets BTC-USD's own `RegimeAllowed` column to `False` for every
row. Confirmed by reading `portfolio_backtest_engine.py` directly:
`RegimeAllowed` is referenced in exactly one place in the whole engine,
`_is_entry_setup()`, as a hard `and` condition -- `False` there makes
`_is_entry_setup` return `False` unconditionally for every bar, which
makes `_build_entry_signal` return `None`, which means BTC-USD can
never generate an entry signal, never queue a pending buy, never
consume one of the 6 shared position slots, and never appear in
`open_positions_after_run`. `RegimeAllowed` is never referenced
anywhere in the engine's exit/stop logic (same grep, zero other hits),
so this cannot affect an exit even in the structurally-impossible case
BTC-USD somehow already held a position. This is the exact same
data-only, engine-unmodified `RegimeAllowed`-gating technique already
used by `src/research/performance_report_v1/run_equity_regime_gate_spy.py`
in this same codebase -- not a new trick invented for this file.

MARKET HOLIDAY / SESSION-DATE VERIFICATION (added 2026-08-16, on top of
Phase 1's intent journal + single-instance lock): `_preflight_double_run_check`'s
own docstring already flags an accepted gap -- it compares WALL-CLOCK
date, so on a US market holiday that isn't also a weekend, the fetched
equity bar would be the SAME stale session as last time, but wall-clock
date would still differ from the stored one, silently letting a
same-session re-process through. This section closes that specific gap
with a SEPARATE, real Alpaca-calendar-based check -- deliberately kept
independent of the wall-clock guard (see `position_state.py`'s own
docstring: `last_processed_equity_date` stays untouched, unrepurposed).

BEFORE any market-data fetch: `_expected_equity_session_date()` makes a
real `TradingClient.get_calendar()` call for today. Three outcomes:
- No calendar entry for today (weekend/holiday) -> clean no-op, zero
  progress, this run returns before `rdd.run_daily_decision()` is ever
  called (same early-return discipline as STOP/FREEZE/the two existing
  preflight checks).
- A calendar entry exists but today's session has not yet closed (real
  `open`/`close` times, naive Eastern-time values from Alpaca, resolved
  DST-aware via stdlib `zoneinfo` -- confirmed via a real call this
  session that August correctly resolves to EDT/UTC-4) plus a 15-minute
  settle buffer -> also a clean no-op ("not yet settled").
- Otherwise -> `expected_equity_session_date` is the calendar's own
  confirmed session date for today.

AFTER `rdd.run_daily_decision()` returns: `_verify_actual_equity_session()`
compares the decision's own `as_of_bar_timestamp_equity` (the ACTUAL
session the fetch produced) against both `expected_equity_session_date`
(computed above, before the fetch) and the state's own
`last_processed_equity_session_date` (the value from BEFORE this run,
captured before `run_daily_decision()`'s internal save could touch
anything). Every case:
- actual == expected AND actual > last_processed -> normal, proceed.
- actual == last_processed -> idempotent skip, not an error (this run
  fetched the exact same session already fully processed before --
  logged, not raised).
- actual < last_processed, OR actual < expected, OR actual > expected
  -> fail-closed, `RuntimeError`, zero further progress in this run.

ONE HONEST LIMITATION, not glossed over: this check runs AFTER
`rdd.run_daily_decision()` has ALREADY made its own internal
`save_position_state()` call (that function is one atomic black-box
call this wrapper cannot pause mid-way through without modifying it,
explicitly out of scope). So an anomaly caught here is detected and
raised LOUDLY -- notified, never silently absorbed -- but it is NOT
prevented from having already been written to `position_state.json`
by `run_daily_decision()`'s own save. True prevention would need either
a real transactional rollback (explicitly deferred, same "atomic
order+state" work named as out of scope for the whole Phase 1/1.5
effort) or pausing `run_daily_decision()` itself mid-call (would
require editing that file, forbidden). Loud failure over silent
corruption is the actual guarantee this section provides -- not full
prevention. The one case this section DOES fully prevent before any
fetch happens at all is the weekend/holiday/not-yet-closed no-op path.

`_bars_today_by_asset_class` inside `run_daily_decision.py` (untouched)
already independently enforces that all EQUITY tickers agree on one
calendar date (raises `ValueError` otherwise) -- the "ticker'lar arası
tutarsız seans tarihi" case from this task's own instruction is
therefore already covered by existing, unmodified engine code; nothing
new was added here to re-check what that function already guarantees.

PHASE 2a: BROKER RECONCILIATION (added 2026-08-16, on top of the market-
holiday/session-date verification above): `src/live/broker_reconciliation.py`'s
`reconcile()` runs at the `[PHASE 2a integration point]` inside `main()`,
BEFORE `_expected_equity_session_date()` and therefore before any
market-data fetch or decision computation. It verifies the connected
Alpaca account's identity (masked, never logs the raw account_number),
takes a consistency-verified snapshot of the broker's open orders and
positions, and fail-closed-raises on any of four documented mismatch
scenarios against `submitted_actions`/`equity_stop_orders`/`positions`
(never `pending_buys`/`pending_exits` -- those are local-only, not yet at
the broker). See that module's own docstring for the full case table.
A reconciliation failure here is caught by the same try/except this
section sits inside, which sends a `[CONTROL]` failure notification and
re-raises -- `rdd.run_daily_decision()` is never called, so the decision
log, position state, and any real broker order are all left untouched.
The one case that is a SUCCESS, not a failure -- Scenario C's narrow
"order fully matches and is confirmed filled" resolution -- updates only
that order's status field on a freshly-loaded state and persists that one
correction to disk before `rdd.run_daily_decision()` runs, the same as
any other successful pre-decision state fix.

HEALTHCHECKS.IO DEAD-MAN'S-SWITCH (added 2026-08-16, on top of Phase 2):
two independent checks -- see `src/live/healthchecks_ping.py`'s own
module docstring for the mechanical ping contract. LIVENESS is pinged
`start` as the very first action in `main()`, then `success`/`fail`
purely on whether `_execute()` raised -- independent of what the run
actually decided to do (a clean STOP/FREEZE/lock-conflict return is
still a liveness SUCCESS: the process ran and did not crash). OPERATIONAL
STATE is pinged `success` only for a genuinely normal run or an expected
market-closed no-op; every other reachable outcome (STOP active, FREEZE
active, a lock conflict, or the final Telegram delivery failing) pings
`fail` with a short, non-sensitive detail code -- see `_execute()`'s own
return-value contract for the complete outcome table. `main()` wraps the
call to `_execute()` in a try/except/finally that covers the ENTIRE
outcome (not just the earlier preflight+run_daily_decision() section --
see the widened inner try/except inside `_execute()` itself, fixed after
an independent audit found audit-check/intent-transition/stamp failures
were previously not caught or notified at all).

Configuration is entirely optional and silently no-ops if absent: no
`--healthchecks-env-file` given, the file it points to missing, or the
`HEALTHCHECKS_LIVENESS_URL`/`HEALTHCHECKS_OPERATIONAL_URL` variables
being blank all result in `ping_healthcheck()` simply returning `False`
without ever blocking `run_daily_decision()` -- Healthchecks integration
must never gate trading. A one-time (marker-file-guarded, not per-run)
`[CONTROL]` Telegram notice is sent the first time this control arm runs
with Healthchecks not configured, so the gap doesn't go unnoticed
forever, without spamming the owner on every subsequent cron invocation.

INTENT JOURNAL ORDERING GAP (found by an independent audit; the
write-ahead mechanism below is now BUILT and sandbox-tested, but real
order activation stays manually gated -- see "STILL GUARDED" below):
`_run_intent_protocol` originally only wrote PREPARED intents from
`decision["queued_for_next_run"]`/`["executed_today"]` -- i.e. AFTER
`rdd.run_daily_decision()` had already returned, and (for a real order,
a case that has never yet happened in production -- this control arm
has never passed `--enable-equity-orders`/`--enable-crypto-orders`)
after any real broker call `_execute_equity_orders`/`_execute_crypto_orders`
inside that function would already have made. That defeated the entire
point of a WRITE-AHEAD journal for exactly the case it exists to
protect: a crash between the real broker call and this wrapper
regaining control would leave zero on-disk evidence of what was
attempted, one atomic black-box call (`run_daily_decision()`) providing
no hook to intervene earlier without editing that frozen file.

THE FIX, NOW BUILT: `_write_blind_prepared_intents` runs BEFORE
`rdd.run_daily_decision()` is called, reading the SAME `pending_buys`/
`pending_exits` (already TTL-adjusted, freshly loaded from this exact
state file) that `_execute_pending_buys_at_open`/
`_execute_pending_exits_at_open` are about to act on inside that call --
NOT re-deriving the frozen engine's own signal-selection logic, only
reading what it already queued. It writes a "blind" PREPARED intent for
each candidate (the SAME deterministic `client_order_id` formula
`_run_intent_protocol` uses post-hoc, so the SAME intent record is what
gets found/updated afterward, never a second, divergent one; idempotent
for a leftover PREPARED intent from an interrupted prior run -- see that
function's own docstring for the crash-recovery case and
`StrayPreparedIntentConflictError` for the fail-closed case beyond it).
`_verify_write_ahead_evidence_before_broker_call` then confirms, by
actually reading the on-disk journal (not by trusting a flag), that
every candidate really has durable evidence, right before
`rdd.run_daily_decision()` is called. AFTER that call returns,
`_run_intent_protocol` correlates: an intent whose ticker appears in the
real result's `executed_today.entries`/`.exits` transitions
SUBMITTING -> BROKER_ACKNOWLEDGED -> COMMITTED as today (its
quantity/notional/stop_price updated with the now-known real numbers);
a "blind" intent that did NOT materialize (the engine decided
differently this bar than the pre-run snapshot implied -- e.g. the
position cap was full, FREEZE blocked new entries, or a stop/condition
changed the outcome before that ticker's turn) transitions to TERMINAL
as an abandoned guess, never COMMITTED. This still cannot inject a
checkpoint DURING `run_daily_decision()`'s own internal broker call --
only editing that frozen file could -- but it closes the window this
section used to describe: the intent is durably PREPARED before the
call that might submit a real order starts, not after it returns.

STILL GUARDED: `_guard_against_premature_order_activation` (called as
the very first thing `_execute()` does, before any other check) still
makes it impossible to pass either order-enabling flag unless the
manual, source-level `_WRITE_AHEAD_JOURNAL_OWNER_APPROVED` flag (see its
own comment, just above that guard) is True -- "sandbox-tested" is
deliberately NOT treated as equivalent to "proven safe with a real
broker in real production use." That flag stays False until the owner
explicitly flips it after real production validation; this task did not
flip it. Even if it were True, `_verify_write_ahead_evidence_before_broker_call`
is the real, runtime-checked enforcement (see its own docstring) -- the
early guard passing is necessary, not sufficient.

Isolation from the live deployment is NOT implemented in this file --
it comes entirely from process cwd (`data/live/STOP`, `data/live/FREEZE`,
and the default `--state-path`/`--decision-log-directory` are all
relative paths, resolved against whatever directory this script is
launched from) plus the deploy-time `AI_STOCK_RADAR_GUARD_DIR`
environment variable (read by `src/live/position_state.py` at import
time). This script must always be launched with cwd set to the control
arm's own deploy directory (see `control_arm_crontab.draft`'s
`cd /root/AI-Stock-Radar-Control && ...`), never the live system's --
exactly the same convention the live cron already uses for its own
isolation from anything else on the box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from dotenv import load_dotenv

import scripts.run_daily_decision as rdd
import src.live.broker_reconciliation as broker_reconciliation
import src.live.healthchecks_ping as healthchecks_ping
import src.live.issuer_identity_preflight as issuer_identity_preflight
import src.live.order_intent as order_intent
import src.live.order_intent_reconciliation as order_intent_reconciliation
import src.live.pending_signal_ttl as pending_signal_ttl
from src.live.authorized_execution_context import authorize_order_execution
from src.live.control_universe import CONTROL_UNIVERSE_TICKERS
from src.live.single_instance_lock import SingleInstanceLockError, single_instance_lock

_EASTERN = ZoneInfo("America/New_York")  # real US equity market timezone, DST-aware (stdlib zoneinfo, no new dependency)
_SESSION_CLOSE_SETTLE_BUFFER = timedelta(minutes=15)

REGIME_ONLY_TICKER = "BTC-USD"
CONTROL_TICKERS_WITH_REGIME: tuple[str, ...] = CONTROL_UNIVERSE_TICKERS + (REGIME_ONLY_TICKER,)
_NOTIFICATION_PREFIX = "[CONTROL] "
ACCOUNT_IDENTITY = "control_arm_v1"

# Captured BEFORE patching rdd.prepare_live_market_data below -- the
# patched wrapper calls this real reference, never `rdd.prepare_live_market_data`
# itself (which by the time the wrapper runs points back at the wrapper,
# and would recurse infinitely).
_real_prepare_live_market_data = rdd.prepare_live_market_data


def _prepare_control_arm_market_data(tickers, **kwargs):
    """Replaces `rdd.prepare_live_market_data` for this process's
    lifetime. Calls the real, unmodified function (needed for its BTC
    regime-EMA computation and its own REGIME_SYMBOL/validation checks),
    then force-blocks BTC-USD from ever being an entry candidate. See
    module docstring for the full "why" and the engine-read confirmation
    that RegimeAllowed=False is a hard, exit-logic-safe entry block."""
    prepared = _real_prepare_live_market_data(tickers, **kwargs)
    if REGIME_ONLY_TICKER in prepared:
        btc_frame = prepared[REGIME_ONLY_TICKER].copy()
        btc_frame["RegimeAllowed"] = False
        prepared[REGIME_ONLY_TICKER] = btc_frame
    return prepared


rdd.LIVE_CONTROLLED_TICKERS = CONTROL_TICKERS_WITH_REGIME
rdd.prepare_live_market_data = _prepare_control_arm_market_data


def _load_env_file_fail_closed(env_file: str) -> None:
    """Fail loudly, never silently, if the given .env path does not
    resolve to a real file on disk -- the exact failure mode an
    external audit flagged (a symlink/resolve problem could otherwise
    leave the LIVE .env's already-loaded credentials silently in
    effect instead of erroring). Must be called AFTER
    `import scripts.run_daily_decision as rdd` (already done, at
    module level above) -- that import transitively triggers two other
    modules' own `load_dotenv(MAIN_ENV_PATH, override=True)` calls at
    import time (`src/data/alpaca_market_data.py`,
    `src/notify/telegram_notifier.py`); this override must run after
    both to actually win, exactly the ordering discipline established
    earlier this session.
    """
    resolved = Path(env_file).resolve(strict=True)  # raises FileNotFoundError -- never a silent no-op
    loaded = load_dotenv(resolved, override=True)
    if not loaded:
        raise RuntimeError(
            f"load_dotenv() reported nothing loaded from {resolved} -- "
            "refusing to silently continue with whatever credentials were "
            "already in os.environ (would be the main .env's, not this "
            "file's). Fail-closed by design; see this module's docstring."
        )
    print(f"[CONTROL] Credential override applied from: {resolved}")


def _notify_control(text: str) -> bool:
    return rdd._notify_safe(_NOTIFICATION_PREFIX + text)


def _preflight_universe_check(state_path: Path, control_tickers: frozenset[str]) -> None:
    """Refuses to proceed if the ALREADY-PERSISTED state file contains any
    position/pending-buy/pending-exit for a ticker outside this run's own
    44-ticker control universe -- BTC-USD included, since BTC-USD must
    never hold a real tradable position under this universe's design
    (see the module docstring's RegimeAllowed=False mechanism). Runs
    BEFORE any network/API call -- before `rdd.run_daily_decision()` is
    ever invoked, i.e. before `get_live_cash_balance()`,
    `prepare_live_market_data()`, or any order-submission code path can
    run -- so a stale out-of-universe entry left behind by a previous
    run cannot silently be acted on by this one.

    This is a genuinely different, STRONGER guarantee than the post-hoc
    audit check further down in `main()` (kept, not replaced -- see that
    check's own comment): the post-hoc check only inspects what THIS
    run's OWN decision produced, so it could never have caught a
    pre-existing stale entry that was already sitting in state before
    this run started (a real gap an external audit correctly flagged --
    read/write ordering matters here, not just whether a check exists at
    all).
    """
    path = Path(state_path)
    if not path.is_file():
        return  # nothing persisted yet on a fresh deployment -- nothing to check
    payload = json.loads(path.read_text(encoding="utf-8"))
    offending: dict[str, list[str]] = {}
    for collection_name in ("positions", "pending_buys", "pending_exits"):
        bad = sorted(set(payload.get(collection_name, {})) - control_tickers)
        if bad:
            offending[collection_name] = bad
    if offending:
        raise RuntimeError(
            f"Pre-flight check failed: {path} already contains ticker(s) "
            f"outside this run's 44-ticker control universe: {offending}. "
            f"Refusing to call run_daily_decision() at all -- zero Alpaca "
            f"API calls were made. This normally means stale state from a "
            f"different universe/run leaked into this file; investigate "
            f"and repair the state file by hand before retrying."
        )


def _preflight_double_run_check(state_path: Path) -> None:
    """Refuses to proceed if TODAY's real calendar date already equals
    the state's own `last_processed_equity_date` -- the fix for a real,
    demonstrated bug an external audit found via an actual double-run
    test: the SAME equity bar's signal (concretely, AA) got queued
    TWICE when this script ran twice on the same day. Runs BEFORE any
    Alpaca API call, same zero-Alpaca-API-calls discipline as
    `_preflight_universe_check` -- reads the state file directly, never
    calls `rdd.run_daily_decision()` if this fires.

    Uses today's real UTC calendar date as the comparison point, NOT a
    market-calendar-aware "last actual trading session" concept -- that
    would require either an Alpaca API call (defeating the zero-API-call
    guarantee this check exists to provide) or a market-calendar
    library this codebase does not have. This is a deliberate, NARROWER
    fix than the full problem: it reliably catches an accidental
    same-day cron re-trigger or manual re-run (the exact scenario this
    task required proving with a real test), but does NOT fully solve
    the broader US-market-holiday case also raised in the same audit --
    if the market is closed on a later real calendar day, today's date
    would differ from the stored one, this check would PASS, and
    `run_daily_decision()` would still internally reprocess the same
    underlying (most-recent-available) equity session. Closing that gap
    needs either a real market calendar or a change inside
    `run_daily_decision.py` itself -- both explicitly out of scope for
    this task (broker-reconciliation, atomic order+state, and a
    single-instance lock are separate, later, pre-real-order hardening).
    Flagged honestly here rather than silently claimed as solved.
    """
    path = Path(state_path)
    if not path.is_file():
        return  # nothing persisted yet -- nothing to compare against
    payload = json.loads(path.read_text(encoding="utf-8"))
    last_equity_date = payload.get("last_processed_equity_date")
    today = datetime.now(timezone.utc).date().isoformat()
    if last_equity_date is not None and last_equity_date == today:
        raise RuntimeError(
            f"Pre-flight check failed: {path}'s last_processed_equity_date "
            f"({last_equity_date}) already equals today's real UTC calendar "
            f"date ({today}). Refusing to call run_daily_decision() at all "
            f"-- zero Alpaca API calls were made. This normally means this "
            f"script already completed a successful run today (a cron "
            f"re-trigger, a manual re-run, or similar) -- re-processing the "
            f"same equity bar a second time is the exact double-queuing bug "
            f"an external audit demonstrated (a real signal, e.g. AA, "
            f"queued twice). If a genuine intentional re-run is needed "
            f"today, investigate why first rather than bypassing this check."
        )


def _expected_equity_session_date(client: TradingClient) -> str | None:
    """Real Alpaca `TradingClient.get_calendar()` call, made BEFORE any
    market-data fetch. See module docstring's "MARKET HOLIDAY /
    SESSION-DATE VERIFICATION" section for the full design. Returns the
    calendar-confirmed session date (ISO string) for today if today's
    session exists AND has already closed (+ a 15-minute settle
    buffer), or `None` if this run should clean no-op (no session
    today, or today's session has not yet closed).

    `client` is REQUIRED, never constructed internally -- see
    `_execute()`'s own `trading_client` injection-seam parameter for why:
    every real client construction in this file happens at exactly one
    point (`_execute()`'s own resolution of `trading_client`), so a test
    harness supplying a fake client there transparently reaches every
    Alpaca call this whole run makes, this one included."""
    today = datetime.now(timezone.utc).date()
    calendar = client.get_calendar(GetCalendarRequest(start=today, end=today))
    if not calendar:
        return None  # no equity session today -- weekend or market holiday
    entry = calendar[0]
    close_utc = entry.close.replace(tzinfo=_EASTERN).astimezone(timezone.utc)
    if datetime.now(timezone.utc) < close_utc + _SESSION_CLOSE_SETTLE_BUFFER:
        return None  # today's session exists but has not settled yet
    return entry.date.isoformat()


def _read_last_processed_equity_session_date(state_path: Path) -> str | None:
    """Reads `last_processed_equity_session_date` directly from the
    ON-DISK state file, BEFORE `rdd.run_daily_decision()` is ever
    called. Must be read at this point, not after: `run_daily_decision()`'s
    own internal `save_position_state()` call builds a fresh
    `LiveRunnerState` that never sets this field, so it would be wiped
    to `None` by that save if read afterward -- the exact same reason
    `_stamp_last_processed_dates` below has to be a second, separate
    save."""
    path = Path(state_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("last_processed_equity_session_date")


def _verify_actual_equity_session(
    decision: dict, expected_session_date: str | None, last_processed_session_date: str | None
) -> None:
    """AFTER `rdd.run_daily_decision()` returns. See module docstring's
    "MARKET HOLIDAY / SESSION-DATE VERIFICATION" section for the full
    case table and the one honest limitation (detection, not full
    prevention -- the internal save has already happened by the time
    this runs). Raises `RuntimeError` (fail-closed) for any anomalous
    case; the normal and idempotent-skip cases return cleanly."""
    actual = decision.get("as_of_bar_timestamp_equity")
    if actual is None:
        raise RuntimeError(
            "decision['as_of_bar_timestamp_equity'] is missing/None -- cannot "
            "verify the actual equity session date. Fail-closed."
        )
    if last_processed_session_date is not None and actual == last_processed_session_date:
        print(
            f"[CONTROL] Idempotent skip signal: actual equity session "
            f"({actual}) equals the already-processed session -- this run's "
            f"fetch produced no new session, not an error."
        )
        return
    if last_processed_session_date is not None and actual < last_processed_session_date:
        raise RuntimeError(
            f"Actual equity session ({actual}) is BEFORE the last processed "
            f"session ({last_processed_session_date}) -- a rollback-like "
            f"anomaly. Fail-closed; investigate before retrying."
        )
    if expected_session_date is not None and actual < expected_session_date:
        raise RuntimeError(
            f"Actual equity session ({actual}) is BEFORE the calendar-expected "
            f"session ({expected_session_date}) -- the fetched data appears "
            f"stale relative to Alpaca's own trading calendar. Fail-closed; "
            f"investigate before retrying."
        )
    if expected_session_date is not None and actual > expected_session_date:
        raise RuntimeError(
            f"Actual equity session ({actual}) is AFTER the calendar-expected "
            f"session ({expected_session_date}) -- the fetched data is ahead "
            f"of what Alpaca's own trading calendar confirmed for today, a "
            f"genuine anomaly. Fail-closed; investigate before retrying."
        )
    # actual == expected (or expected is None, e.g. a defensive skip of
    # the pre-fetch check somehow) AND actual > last_processed (or no
    # prior value) -- the normal case, nothing further to do here.


def _stamp_last_processed_dates(state_path: Path, actual_session_date: str | None = None) -> None:
    """After a successful `run_daily_decision()` call, record that this
    process completed successfully TODAY onto `position_state.json`'s
    `last_processed_equity_date`/`last_processed_crypto_date` fields
    (added to `src/live/position_state.py` this task, backward-compatible
    -- old files without these keys still load fine, see that module's
    own docstring). Also stamps `last_processed_equity_session_date`/
    `last_processed_equity_bar_timestamp` with `actual_session_date`
    when provided (the real, calendar-verified session
    `_verify_actual_equity_session` just confirmed) -- see
    `position_state.py`'s own docstring for why these are separate,
    deliberately un-merged with the wall-clock pair below.

    STAMPS TODAY'S REAL WALL-CLOCK UTC DATE, NOT the decision's own
    `as_of_bar_timestamp_equity`/`_crypto` bar dates -- a deliberate
    correction made DURING this task's own real testing, not the
    original design. The bar-date version was tried first and FAILED
    the required real double-run test: run on a real Sunday
    (2026-08-16), the fetched equity bar was genuinely Friday's
    (2026-08-14, the last real trading session -- confirmed correct,
    not a data bug), so a bar-date stamp compared against wall-clock
    "today" in `_preflight_double_run_check` never matched, and a
    second same-day run would have been silently ALLOWED through on any
    non-trading day -- precisely the class of gap already flagged as
    unsolved in that function's own docstring, just discovered to bite
    on the very first real test rather than only in the deferred
    holiday case. Stamping wall-clock date instead directly answers the
    only question this check actually needs answered -- "did this
    script already complete successfully TODAY (real calendar day)" --
    independent of which trading session happened to be fetched, and
    passes the literal required test (same real day, two consecutive
    runs) regardless of whether that day is a trading day.

    A SECOND, deliberate save -- `run_daily_decision()`'s own internal
    `save_position_state()` call (inside `run_daily_decision.py`, never
    touched by this wrapper) has no knowledge of these two new fields
    and would leave them at their `None` default every time, which would
    make `_preflight_double_run_check` above unable to ever detect
    anything. Read-modify-write, NOT atomic against the first (real)
    save -- a crash between the two would leave the stamp missing for
    this run, silently re-permitting a same-day re-run. An accepted,
    documented gap for this task's own narrow scope; atomic order+state
    is explicitly a LATER, separate hardening round.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    runner_state = rdd.ps.load_position_state(state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH)
    runner_state.last_processed_equity_date = today
    runner_state.last_processed_crypto_date = today
    if actual_session_date is not None:
        runner_state.last_processed_equity_session_date = actual_session_date
        runner_state.last_processed_equity_bar_timestamp = actual_session_date
    rdd.ps.save_position_state(runner_state, state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH)


def _hash_state_file(state_path: Path) -> str | None:
    """SHA-256 of the state file's exact bytes at the moment this is
    called -- `pre_state_hash` on every intent created this run, taken
    BEFORE `rdd.run_daily_decision()` is called (i.e. before this run's
    own mutation of that file). Returns `None` if no file exists yet
    (a fresh deployment's first run) rather than hashing an absence."""
    path = Path(state_path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrayPreparedIntentConflictError(RuntimeError):
    """Raised by `_write_blind_prepared_intents` when a candidate's
    deterministic `client_order_id` already has an on-disk intent in
    status SUBMITTING/BROKER_ACKNOWLEDGED/COMMITTED/UNCERTAIN --
    meaning a PRIOR run got further than this pre-write step before
    crashing/exiting, or before this control arm's normal end-of-run
    cleanup ran, and Phase 1.5's own stray-recovery (which runs before
    this) did not (or could not) resolve it to something this step
    knows how to handle.

    CORRECTED DOCSTRING (reboot-drill finding #1, 2026-08-24): this
    previously (incorrectly) claimed "non-terminal status" here while
    the actual code ALSO raised for a plain TERMINAL record -- a real
    documentation bug, not just an implementation one; the code itself
    is what changed (TERMINAL is no longer a conflict at all, see
    `_write_blind_prepared_intents`'s own updated docstring), and this
    docstring is now accurate to what the code actually checks.

    This pre-write step only knows how to safely resume from a leftover
    PREPARED intent (the idempotent crash-before-`rdd.run_daily_decision()`
    case -- see the "crash simulation" test in
    tests/test_run_control_arm_decision_guards.py) or to ignore a
    TERMINAL one (writing a fresh new intent instead, see
    `_write_blind_prepared_intents`); anything else needs real
    broker-side reconciliation, which Phase 1.5 (immediately before this
    step) is responsible for -- reaching this error means Phase 1.5
    itself did not run, or found something it could not resolve.
    Fail-closed rather than silently creating a second, divergent intent
    for the same `client_order_id`."""


def _stray_intent_matches_live_candidate(intent: order_intent.OrderIntent, runner_state) -> bool:
    """Reboot-drill finding #1 (2026-08-24): does THIS run still have a
    live, EXACTLY matching candidate for a stray submission intent --
    ticker, action_kind, client_order_id, account_identity, and side ALL
    agree with a currently-queued `pending_buys`/`pending_exits` entry.
    Used by Phase 1.5 to decide whether a confirmed-404-from-PREPARED
    stray should be left RESUMABLE (reused by `_write_blind_prepared_intents`
    below, exactly as if this were a plain crash-before-`rdd.run_daily_decision()`
    case) rather than closed `ABANDONED_NO_SUBMISSION` -- see
    `order_intent_reconciliation.resolve_stray_order_submission_intent`'s
    own docstring for the full "why" this parameter exists.

    Only ENTRY_MARKET_BUY/SIGNAL_EXIT_MARKET_SELL intents have a
    "pending candidate" concept at all -- PROTECTIVE_STOP/
    CANCEL_PROTECTIVE_STOP are decided reactively mid-run, never
    pre-queued (see `_order_intent_hook_for_run`'s own docstring) --
    `False` immediately for anything else.

    `client_order_id` is re-derived from `intent.ticker`/`intent.action_kind`/
    `intent.source_signal_timestamp` and compared against the intent's
    own STORED `client_order_id`, rather than trusted blindly -- this is
    what makes the match "exact" with respect to session date too: if
    the intent was originally prepared for a DIFFERENT session than
    today's real `expected_session_date`, `_write_blind_prepared_intents`
    would compute a different id entirely for today's own candidate and
    would never even look this intent up by that id in the first place,
    so an intentionally-stale intent can never spuriously match here."""
    if intent.action_kind == "ENTRY_MARKET_BUY":
        pending = runner_state.pending_buys.get(intent.ticker)
        expected_side = "BUY"
    elif intent.action_kind == "SIGNAL_EXIT_MARKET_SELL":
        pending = runner_state.pending_exits.get(intent.ticker)
        expected_side = "SELL"
    else:
        return False
    if pending is None:
        return False
    if intent.account_identity != ACCOUNT_IDENTITY:
        return False
    if intent.side != expected_side:
        return False
    recomputed_client_order_id = rdd.client_order_id_for_action(
        intent.ticker, intent.action_kind, str(intent.source_signal_timestamp)
    )
    return recomputed_client_order_id == intent.client_order_id


def _write_blind_prepared_intents(
    runner_state,
    expected_session_date: str,
    pre_state_hash: str | None,
) -> list[order_intent.OrderIntent]:
    """THE FIX for this module's own "INTENT JOURNAL ORDERING GAP"
    docstring section ("blind" pre-run PREPARED intents): writes one
    durable PREPARED intent for every candidate already queued in
    `runner_state.pending_buys`/`pending_exits` -- the SAME,
    TTL-adjusted dicts `rdd.run_daily_decision()` is about to load fresh
    from this exact `position_state.json` file, i.e. exactly what
    `_execute_pending_buys_at_open`/`_execute_pending_exits_at_open`
    (inside that frozen, never-reimplemented call) are about to attempt
    at today's Open. Called BEFORE that call, and therefore before any
    real broker call it might make.

    Real quantity/notional/stop_price are not yet known at this point
    (the frozen engine decides the real fill/rejection when it actually
    processes each ticker) -- `quantity`/`stop_price` are left `None`;
    `notional` is filled with the pending signal's own
    `reference_price` (the best "intended price" this pre-run snapshot
    can offer) and gets overwritten with the real fill/exit price by
    `_run_intent_protocol` once that exists.

    `client_order_id` uses the EXACT SAME deterministic formula
    (`rdd._deterministic_client_order_id(ticker, action_kind,
    expected_session_date)`) that `_run_intent_protocol` computes
    AFTER the run for a materialized entry/exit -- `expected_session_date`
    is the real, Alpaca-calendar-confirmed session date for today,
    verified equal to `decision["as_of_bar_timestamp_equity"]` by
    `_verify_actual_equity_session` right after the fetch (module
    docstring's "MARKET HOLIDAY / SESSION-DATE VERIFICATION" section) --
    so the id computed here always resolves back to the SAME intent
    `_run_intent_protocol` looks for afterward, never a second,
    divergent one.

    IDEMPOTENT for a leftover PREPARED intent from an interrupted prior
    run (a crash between this function returning and
    `rdd.run_daily_decision()` being called): reuses it rather than
    creating a duplicate -- the candidate is still genuinely pending
    (never consumed), so the SAME client_order_id, SAME PREPARED intent
    is still exactly correct.

    A leftover TERMINAL intent for the SAME `client_order_id` is IGNORED
    (reboot-drill finding #1, 2026-08-24, real bug fixed) -- never
    raised on, never reused/mutated. `intent_id`, not `client_order_id`,
    is this journal's own storage key, so multiple records legitimately
    sharing one `client_order_id` is not a structural problem (e.g. an
    earlier attempt genuinely completed or was abandoned
    `ABANDONED_NO_SUBMISSION`, and THIS run's own candidate deserves its
    own fresh PREPARED record, own `intent_id`, leaving the old TERMINAL
    one on disk untouched as historical evidence). Before this fix, ANY
    non-PREPARED status -- TERMINAL included, despite this function's
    own now-corrected docstring previously claiming otherwise -- raised
    `StrayPreparedIntentConflictError` here, which deadlocked a run
    whenever Phase 1.5's own stray-recovery had (correctly, at the time)
    closed a confirmed-404 PREPARED intent TERMINAL earlier in the SAME
    run, for a candidate that was in fact still live. Confirmed via a
    real, end-to-end reboot-drill reproduction. Phase 1.5's own
    RESUMABLE_PREPARED fix (`_stray_intent_matches_live_candidate`) now
    prevents that specific TERMINAL-with-a-live-candidate case from ever
    happening in the first place -- this TERMINAL-ignoring fix is the
    second, independent layer: this step must be robust to finding a
    TERMINAL record regardless of why one exists.

    Raises `StrayPreparedIntentConflictError` (fail-closed) if a
    leftover intent for that client_order_id exists in status
    SUBMITTING/BROKER_ACKNOWLEDGED/COMMITTED/UNCERTAIN -- see that
    exception's own docstring.
    """
    candidates: list[tuple[str, str, object]] = []
    for ticker, pending in sorted(runner_state.pending_buys.items()):
        candidates.append((ticker, "ENTRY_MARKET_BUY", pending))
    for ticker, pending in sorted(runner_state.pending_exits.items()):
        candidates.append((ticker, "SIGNAL_EXIT_MARKET_SELL", pending))

    intents: list[order_intent.OrderIntent] = []
    for ticker, action_kind, pending in candidates:
        # REAL BUG FOUND AND FIXED (2026-08-22, independent audit finding
        # #2): this used to call rdd._deterministic_client_order_id
        # directly with the underscore-spelled action_kind literal
        # ("ENTRY_MARKET_BUY"), while run_daily_decision.py's own real
        # broker-call sites built their client_order_id with a
        # dash-spelled literal ("ENTRY-MARKET-BUY") -- two different
        # strings for the same real ticker+action+date, so this
        # journal's own id never matched what actually got submitted to
        # Alpaca. rdd.client_order_id_for_action is the one shared,
        # canonical mapping both files now go through.
        client_order_id = rdd.client_order_id_for_action(ticker, action_kind, str(expected_session_date))
        existing = order_intent.find_intent_by_client_order_id(client_order_id)
        if existing is not None and existing.status == order_intent.PREPARED:
            intent = existing
            print(
                f"[CONTROL] Blind intent {intent.intent_id} ({ticker} {action_kind}): "
                f"reused existing PREPARED intent left by an interrupted prior run "
                f"(idempotent, same client_order_id)."
            )
        elif existing is not None and existing.status != order_intent.TERMINAL:
            raise StrayPreparedIntentConflictError(
                f"{ticker} {action_kind}: an intent for client_order_id={client_order_id!r} "
                f"already exists in status={existing.status!r} (intent_id={existing.intent_id}), "
                f"not PREPARED -- this pre-write step cannot safely resume past PREPARED. "
                f"Needs real reconciliation before this run can proceed."
            )
        else:
            # Either no prior intent exists at all, or one exists but is
            # TERMINAL (reboot-drill finding #1, 2026-08-24: ignored,
            # never raised on, never reused/mutated -- see this
            # function's own docstring) -- both cases get a genuinely
            # fresh PREPARED intent, own new `intent_id`, for this run's
            # own candidate.
            if existing is not None:
                print(
                    f"[CONTROL] Blind intent for {ticker} {action_kind} (client_order_id={client_order_id!r}): "
                    f"a TERMINAL record already exists (intent_id={existing.intent_id}) -- ignored, writing a "
                    f"fresh PREPARED intent for this run's own candidate."
                )
            intent = order_intent.create_intent(
                client_order_id=client_order_id,
                account_identity=ACCOUNT_IDENTITY,
                ticker=ticker,
                side="BUY" if action_kind == "ENTRY_MARKET_BUY" else "SELL",
                order_type="market",
                action_kind=action_kind,
                source_signal_timestamp=str(expected_session_date),
                quantity=None,
                notional=pending.signal.reference_price,
                stop_price=None,
                pre_state_hash=pre_state_hash,
            )
            print(
                f"[CONTROL] Blind intent {intent.intent_id} ({ticker} {action_kind}): "
                f"PREPARED (write-ahead, before rdd.run_daily_decision())."
            )
        intents.append(intent)
    return intents


def _order_intent_hook_for_run(blind_intents: list[order_intent.OrderIntent]) -> "rdd.OrderIntentHook":
    """Builds the real callback passed as `rdd.run_daily_decision(...,
    order_intent_hook=...)` (2026-08-22, independent audit finding #3,
    approach (B) -- see rdd.OrderIntentHook's own module-level comment
    for the full contract and why a callback, not a direct import,
    closes this gap).

    THE REAL GAP THIS CLOSES: before this hook existed, SUBMITTING was
    only ever recorded by `_run_intent_protocol`, AFTER
    `rdd.run_daily_decision()` returned -- i.e. after every real broker
    call inside `_execute_equity_orders`/`_execute_crypto_orders` had
    already happened. A crash DURING one of those real calls left every
    candidate's on-disk intent stuck at PREPARED, with no way to tell
    which one (if any) actually reached the broker before the crash.
    This hook transitions the matching blind-written PREPARED intent to
    SUBMITTING immediately BEFORE, and to BROKER_ACKNOWLEDGED
    immediately AFTER, EACH real broker call -- for the exact
    ticker/action about to be attempted, not a blanket pre-run guess.

    `blind_intents` is the SAME list `_write_blind_prepared_intents`
    already wrote before this run's `rdd.run_daily_decision()` call,
    keyed here by `client_order_id` for lookup -- covers
    `ENTRY_MARKET_BUY`/`SIGNAL_EXIT_MARKET_SELL`, the two action kinds
    that have a pre-run "blind" candidate (`pending_buys`/`pending_exits`).

    `PROTECTIVE_STOP`/`CANCEL_PROTECTIVE_STOP` (Task 3, 2026-08-22, closing
    the real gap the ORIGINAL version of this docstring's own "silent
    no-op" wording described): unlike a BUY/EXIT, these are decided
    REACTIVELY mid-run (a stop is only placed once its entry fill
    confirms; a cancel is only decided once a signal-exit needs to sell
    through it) -- there is no pre-run candidate to have blind-journaled
    ahead of time. For these two, this hook creates a REAL PREPARED
    intent itself, JUST-IN-TIME, using the real quantity/stop_price/
    parent_order_id/operation_id the caller now passes (see
    `OrderIntentHook`'s own module-level comment in run_daily_decision.py)
    -- still durably on disk BEFORE the broker call, the same guarantee
    the blind-pre-journaled path provides, just created one step later in
    time because that is the earliest point this information exists.
    `PROTECTIVE_STOP` is keyed by `client_order_id` (it submits a real
    order, same as BUY/EXIT); `CANCEL_PROTECTIVE_STOP` has none and is
    keyed by `operation_id` instead (`OrderIntent.operation_id`/
    `target_broker_order_id` -- see that dataclass's own field comments).

    An unrecognized `action_kind` raises `RuntimeError` -- fail-closed,
    never a silent pass-through for a future action kind this hook does
    not yet know how to journal.

    Inert today: `_guard_against_premature_order_activation` blocks
    `--enable-equity-orders`/`--enable-crypto-orders` entirely (see that
    function's own docstring), so `_execute_equity_orders`/
    `_execute_crypto_orders` -- and therefore this hook -- never run in
    production. Built now so the real fix is ready and tested the
    moment the owner ever flips `_WRITE_AHEAD_JOURNAL_OWNER_APPROVED`,
    rather than needing a second round of protected-file changes then.
    """
    blind_by_client_order_id: dict[str, order_intent.OrderIntent] = {
        intent.client_order_id: intent for intent in blind_intents
    }
    # Just-in-time intents THIS hook creates itself, for the two action
    # kinds with no pre-run blind candidate -- keyed by client_order_id
    # (PROTECTIVE_STOP) or operation_id (CANCEL_PROTECTIVE_STOP), never
    # mixed with `blind_by_client_order_id` above (a different dict, so
    # a real client_order_id collision between the two action families
    # is structurally impossible even in principle).
    just_in_time_by_key: dict[str, order_intent.OrderIntent] = {}

    _KNOWN_ACTION_KINDS = frozenset(
        {"ENTRY_MARKET_BUY", "SIGNAL_EXIT_MARKET_SELL", "PROTECTIVE_STOP", "CANCEL_PROTECTIVE_STOP"}
    )

    def _find_or_create_just_in_time_intent(
        *, key: str, ticker: str, action_kind: str, client_order_id: str, operation_id: str | None,
        side: str | None, order_type: str | None, quantity: float | None, notional: float | None,
        stop_price: float | None, parent_order_id: str | None,
    ) -> order_intent.OrderIntent:
        existing = just_in_time_by_key.get(key)
        if existing is not None:
            return existing
        # Also check disk directly -- a crash-restart within the same
        # calendar run could have already durably written this exact
        # intent (same deterministic key) on a prior, interrupted attempt;
        # reuse it rather than creating a second, divergent one. Accepts
        # PREPARED *or* SUBMITTING (unlike `_write_blind_prepared_intents`'s
        # own blind-candidate resume, which only accepts PREPARED and
        # fails closed on anything further along): that function runs
        # BEFORE rdd.run_daily_decision() even starts, so finding
        # anything past PREPARED there means an earlier, more serious
        # crash needing human review. This hook instead fires DURING the
        # real broker call itself -- finding a leftover SUBMITTING
        # record here means a prior attempt (this run or an interrupted
        # one) got as far as querying/calling the broker but never
        # recorded the result; it is still safe to resume, because the
        # REAL protection against a duplicate submission is
        # `_resolve_or_submit_order`'s own broker-authoritative query by
        # client_order_id (always run before any real submit call,
        # tested separately) -- this journal-level reuse only avoids a
        # second, divergent LOCAL record, it is not itself the safety
        # mechanism against a real duplicate order.
        found = (
            order_intent.find_intent_by_client_order_id(client_order_id)
            if action_kind == "PROTECTIVE_STOP"
            else order_intent.find_intent_by_operation_id(operation_id) if operation_id else None
        )
        if found is not None and found.status in (order_intent.PREPARED, order_intent.SUBMITTING):
            just_in_time_by_key[key] = found
            return found
        created = order_intent.create_intent(
            client_order_id=client_order_id,
            account_identity=ACCOUNT_IDENTITY,
            ticker=ticker,
            side=side or "",
            order_type=order_type or "",
            action_kind=action_kind,
            source_signal_timestamp=datetime.now(timezone.utc).isoformat(),
            quantity=quantity,
            notional=notional,
            stop_price=stop_price,
            parent_intent_id=None,
            operation_id=operation_id,
            target_broker_order_id=parent_order_id if action_kind == "CANCEL_PROTECTIVE_STOP" else None,
        )
        just_in_time_by_key[key] = created
        print(
            f"[CONTROL] Just-in-time intent {created.intent_id} ({ticker} {action_kind}): "
            f"PREPARED (created immediately before this broker call, key={key!r})."
        )
        return created

    def hook(
        phase: str, *, ticker: str, action_kind: str, client_order_id: str = "", order=None,
        operation_id: str | None = None, side: str | None = None, order_type: str | None = None,
        quantity: float | None = None, notional: float | None = None, stop_price: float | None = None,
        parent_order_id: str | None = None, broker_status: str | None = None,
    ) -> None:
        if action_kind not in _KNOWN_ACTION_KINDS:
            raise RuntimeError(
                f"order_intent_hook received an unrecognized action_kind {action_kind!r} "
                f"(ticker={ticker!r}) -- refusing to silently skip journaling it. Fail-closed; "
                f"add it to _KNOWN_ACTION_KINDS only after deciding how it should be journaled."
            )

        if action_kind in ("ENTRY_MARKET_BUY", "SIGNAL_EXIT_MARKET_SELL"):
            intent = blind_by_client_order_id.get(client_order_id)
            if intent is None:
                return  # e.g. a candidate that only appeared this run with no prior pending state
            key = client_order_id
            store = blind_by_client_order_id
        else:
            key = client_order_id if action_kind == "PROTECTIVE_STOP" else (operation_id or "")
            if phase == "SUBMITTING":
                intent = _find_or_create_just_in_time_intent(
                    key=key, ticker=ticker, action_kind=action_kind, client_order_id=client_order_id,
                    operation_id=operation_id, side=side, order_type=order_type, quantity=quantity,
                    notional=notional, stop_price=stop_price, parent_order_id=parent_order_id,
                )
            else:
                intent = just_in_time_by_key.get(key)
                if intent is None:
                    return  # SUBMITTING was never recorded for this key -- nothing to acknowledge
            store = just_in_time_by_key

        if phase == "SUBMITTING":
            if intent.status != order_intent.PREPARED:
                return  # already advanced (e.g. a retry within the same run) -- do not re-transition
            intent = order_intent.transition_intent(intent, order_intent.SUBMITTING, increment_attempt=True)
            store[key] = intent
            print(
                f"[CONTROL] Intent {intent.intent_id} ({ticker} {action_kind}): PREPARED -> "
                f"SUBMITTING (real broker call about to start, key={key!r})."
            )
        elif phase == "BROKER_ACKNOWLEDGED":
            if intent.status != order_intent.SUBMITTING:
                return  # nothing to acknowledge if SUBMITTING was never durably recorded
            # independent-audit-round-2 finding #3 (2026-08-23): a
            # cancel has no `order` object (Alpaca's cancel API returns
            # none), only the plain `broker_status` string
            # `run_daily_decision.py`'s own CANCEL_STOP call site now
            # threads through. Prefer that explicit value when given --
            # it is the more specific, deliberately-supplied one for
            # exactly the callers that have no `order` at all; a real
            # `order` (the ENTRY_MARKET_BUY/SIGNAL_EXIT_MARKET_SELL/
            # PROTECTIVE_STOP submission callers) still derives its own
            # status via `_order_status_str` as before.
            resolved_broker_status = (
                broker_status if broker_status is not None
                else (rdd._order_status_str(order) if order is not None else None)
            )
            intent = order_intent.transition_intent(
                intent, order_intent.BROKER_ACKNOWLEDGED,
                broker_order_id=str(order.id) if order is not None else None,
                broker_status=resolved_broker_status,
            )
            store[key] = intent
            print(
                f"[CONTROL] Intent {intent.intent_id} ({ticker} {action_kind}): SUBMITTING -> "
                f"BROKER_ACKNOWLEDGED (real broker order_id={intent.broker_order_id!r}, "
                f"status={intent.broker_status!r})."
            )

    # Exposed as a plain function attribute (not a second return value)
    # so this hook's own call signature/assignment at the call site stays
    # a single `order_intent_hook=_order_intent_hook_for_run(blind_intents)`
    # -- the caller reads `hook.just_in_time_intents.values()` AFTER
    # rdd.run_daily_decision() returns, to fold PROTECTIVE_STOP/
    # CANCEL_PROTECTIVE_STOP intents into the same COMMITTED/TERMINAL
    # finalization loop `_run_intent_protocol`'s own return value already
    # goes through -- these are never included in that function's own
    # return value (it only ever knows about ENTRY_MARKET_BUY/
    # SIGNAL_EXIT_MARKET_SELL).
    hook.just_in_time_intents = just_in_time_by_key
    return hook


def _run_intent_protocol(
    decision: dict,
    pre_state_hash: str | None,
    blind_intents: list[order_intent.OrderIntent] | None = None,
) -> list[order_intent.OrderIntent]:
    """See `order_intent.py`'s own module docstring for the full state
    machine. Correlates this run's REAL decision output against the
    "blind" PREPARED intents `_write_blind_prepared_intents` already
    wrote before `rdd.run_daily_decision()` was called (`blind_intents`
    -- `None` only when this function is called directly, e.g. from a
    unit test that does not exercise the pre-write step; production
    always passes the real list).

    For every entry/exit this run's decision actually materialized
    (`executed_today.entries`/`executed_today.exits`), the matching
    blind intent (found by the SAME deterministic `client_order_id`
    formula, see `_write_blind_prepared_intents`'s own docstring) is
    reused and updated with the now-known real quantity/price, then
    walked PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED. If no blind
    intent exists for that id (the `blind_intents=None` direct-call
    case, or a candidate that only appeared THIS run with no prior
    pending state -- structurally rare but not assumed impossible), a
    fresh intent is created instead, exactly like this function's
    original (pre-write-ahead) behavior.

    Every blind intent that did NOT materialize this run (the frozen
    engine decided differently than the pre-run snapshot implied --
    e.g. the position cap was full, FREEZE blocked new entries, or a
    stop/condition changed the outcome before that ticker's turn) is
    closed PREPARED -> TERMINAL as an abandoned guess, never COMMITTED
    -- see module docstring's "INTENT JOURNAL ORDERING GAP" section,
    "a 'blind' intent that did NOT materialize ... transitions to
    TERMINAL as an abandoned guess." THIS is the ordinary case
    (PREPARED, since real order submission has never been enabled here
    in production). If the blind intent's status is instead SUBMITTING
    or BROKER_ACKNOWLEDGED (independent-audit-round-2 finding #4,
    2026-08-23 -- the real order_intent_hook DID fire for it during
    `rdd.run_daily_decision()`, yet it still does not appear in
    `executed_today`), it is routed to UNCERTAIN instead -- PREPARED is
    the only status `_VALID_TRANSITIONS` allows a direct move to
    TERMINAL from; a genuinely ambiguous "hook advanced it, but it's not
    in the output" case is never silently forced through the same path.

    NO REAL BROKER CALL IS MADE ANYWHERE IN THIS FUNCTION.
    BROKER_ACKNOWLEDGED here means "the local decision is confirmed as
    this run's final output" -- a scaffold for where a real Alpaca
    acknowledgment will plug in once real order submission AND Phase 2
    broker-reconciliation both exist for this control arm (neither
    exists today; this control arm has never enabled
    --enable-equity-orders/--enable-crypto-orders in production).

    Returns the created/updated/closed intents so the caller can
    transition the materialized ones to COMMITTED only after
    `position_state.json` has actually been durably re-saved with
    today's stamp (see main()'s own ordering).
    """
    equity_date = decision.get("as_of_bar_timestamp_equity") or decision.get("as_of_bar_timestamp")
    materialized: list[tuple[str, dict, str, str]] = []
    for entry in decision.get("executed_today", {}).get("entries", []):
        materialized.append(("ENTRY_MARKET_BUY", entry, "fill_price", "BUY"))
    for entry in decision.get("executed_today", {}).get("exits", []):
        materialized.append(("SIGNAL_EXIT_MARKET_SELL", entry, "exit_price", "SELL"))

    blind_by_client_order_id: dict[str, order_intent.OrderIntent] = {
        blind.client_order_id: blind for blind in (blind_intents or [])
    }

    intents: list[order_intent.OrderIntent] = []
    materialized_client_order_ids: set[str] = set()
    for action_kind, entry, price_field, side in materialized:
        ticker = entry["ticker"]
        # Same fix as _write_blind_prepared_intents above -- see its own
        # comment for the real id-mismatch bug this closes.
        client_order_id = rdd.client_order_id_for_action(ticker, action_kind, str(equity_date))
        materialized_client_order_ids.add(client_order_id)
        intent = blind_by_client_order_id.get(client_order_id)
        if intent is None:
            intent = order_intent.create_intent(
                client_order_id=client_order_id,
                account_identity=ACCOUNT_IDENTITY,
                ticker=ticker,
                side=side,
                order_type="market",
                action_kind=action_kind,
                source_signal_timestamp=str(equity_date),
                quantity=entry.get("quantity"),
                notional=entry.get(price_field),
                stop_price=entry.get("stop_loss_price"),
                pre_state_hash=pre_state_hash,
            )

        # REAL ORDER-INTENT HOOK ALREADY RAN (2026-08-22, independent
        # audit finding #3, approach (B)): if rdd.run_daily_decision()
        # was given an order_intent_hook (see _order_intent_hook_for_run
        # below) and this candidate's real broker call actually
        # happened, the hook has ALREADY transitioned this exact intent
        # PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED, with the REAL
        # broker order id/status, DURING run_daily_decision() -- not a
        # post-hoc guess. Re-running the two-step transition below on an
        # intent that is no longer PREPARED would raise
        # InvalidTransitionError (SUBMITTING/BROKER_ACKNOWLEDGED have no
        # self-transition). This intent already carries the real,
        # hook-recorded data; nothing further to do here.
        #
        # STILL PREPARED (today's only reachable case in production --
        # real order submission has never been enabled here, so
        # _execute_equity_orders/_execute_crypto_orders, and therefore
        # the hook, never run): falls through to the original post-hoc
        # PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED transition,
        # unchanged from before this task -- this is a "local decision
        # confirmed" bookkeeping transition, not a real broker
        # acknowledgment (see the broker_status string below).
        if intent.status in (order_intent.SUBMITTING, order_intent.BROKER_ACKNOWLEDGED):
            print(
                f"[CONTROL] Intent {intent.intent_id} ({ticker} {action_kind}): already "
                f"{intent.status} (real order_intent_hook already ran during "
                f"run_daily_decision()) -- not re-transitioning."
            )
            intents.append(intent)
            continue

        intent = order_intent.transition_intent(
            intent, order_intent.SUBMITTING, increment_attempt=True,
            quantity=entry.get("quantity"), notional=entry.get(price_field), stop_price=entry.get("stop_loss_price"),
        )
        intent = order_intent.transition_intent(
            intent, order_intent.BROKER_ACKNOWLEDGED,
            broker_status="local_decision_confirmed -- no real broker call made (Phase 1 scaffold)",
        )
        intents.append(intent)
        print(f"[CONTROL] Intent {intent.intent_id} ({ticker} {action_kind}): PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED")

    for client_order_id, blind in blind_by_client_order_id.items():
        if client_order_id in materialized_client_order_ids:
            continue
        # independent-audit-round-2 finding #4 (2026-08-23): this used
        # to transition straight to TERMINAL without checking the
        # blind intent's CURRENT status first. The ordinary case is
        # PREPARED (never materialized this run at all) -- but if the
        # real order_intent_hook DID fire for this exact candidate
        # during run_daily_decision() (e.g. it got as far as
        # SUBMITTING, or even BROKER_ACKNOWLEDGED, before a crash or a
        # rejection kept it out of this run's own executed_today),
        # `_VALID_TRANSITIONS` has no path from SUBMITTING/
        # BROKER_ACKNOWLEDGED straight to TERMINAL -- the old code would
        # raise InvalidTransitionError uncaught in exactly that case.
        if blind.status == order_intent.PREPARED:
            closed = order_intent.transition_intent(
                blind, order_intent.TERMINAL,
                last_error=(
                    "attempted_but_not_executed -- this pre-run write-ahead candidate did not "
                    "appear in this run's executed_today (cap full / rejected / FREEZE / a stop "
                    "or another condition changed the outcome before this ticker was reached)."
                ),
            )
            print(f"[CONTROL] Intent {closed.intent_id} ({closed.ticker} {closed.action_kind}): PREPARED -> TERMINAL (attempted_but_not_executed)")
            intents.append(closed)
        elif blind.status in (order_intent.SUBMITTING, order_intent.BROKER_ACKNOWLEDGED):
            # The hook DID advance this candidate during
            # run_daily_decision() -- a real broker call was at least
            # attempted, possibly acknowledged -- yet it does not appear
            # in this run's own executed_today. Never assumed to be the
            # ordinary "never attempted" case (that would be a false
            # claim); routes to UNCERTAIN instead (a valid transition
            # from both statuses) -- fail-closed, human review required.
            # The NEXT run's own Phase 1.5 stray-intent recovery (see
            # order_intent_reconciliation.py) is what actually resolves
            # this against the broker; this function makes no broker
            # call itself (see its own docstring).
            closed = order_intent.transition_intent(
                blind, order_intent.UNCERTAIN,
                last_error=(
                    f"blind candidate reached {blind.status} via the real order_intent_hook during "
                    f"run_daily_decision(), but does not appear in this run's own executed_today -- "
                    f"ambiguous outcome, never assumed to be 'never attempted'. Fail-closed, human "
                    f"review (or the next run's own stray-intent recovery) required."
                ),
            )
            print(
                f"[CONTROL] Intent {closed.intent_id} ({closed.ticker} {closed.action_kind}): "
                f"{blind.status} -> UNCERTAIN (hook advanced it but it is not in executed_today)"
            )
            intents.append(closed)
        else:
            # UNCERTAIN/COMMITTED/TERMINAL: already resolved, or already
            # flagged ambiguous, by something else -- never re-transitioned
            # here.
            print(f"[CONTROL] Intent {blind.intent_id} ({blind.ticker} {blind.action_kind}): already {blind.status} -- not re-transitioning.")
            intents.append(blind)

    return intents


def _finalize_run_intents(all_run_intents: list[order_intent.OrderIntent]) -> list[order_intent.OrderIntent]:
    """Closes out every intent this SAME run touched -- extracted from
    `_execute()`'s own inline loop (2026-08-23, independent-audit-round-2
    test-quality note) so this is directly, behaviorally testable rather
    than only checkable via source-text matching against `_execute`.

    Intents committed only AFTER state has actually been durably
    re-saved -- COMMITTED is meant to mean "local state reflects this,"
    not merely "we decided to." `_execute()`'s own caller only invokes
    this function after `_stamp_last_processed_dates` has already run,
    preserving that ordering guarantee; this function itself does not
    re-verify it.

    Folds in the PROTECTIVE_STOP/CANCEL_PROTECTIVE_STOP just-in-time
    intents the hook created during this same run (never returned by
    `_run_intent_protocol` itself -- that function only knows about
    ENTRY_MARKET_BUY/SIGNAL_EXIT_MARKET_SELL) so they go through the
    exact same COMMITTED/TERMINAL finalization, not a second, divergent
    code path (Task 3, 2026-08-22).

    Abandoned "blind" guesses (already TERMINAL by this point, or
    UNCERTAIN -- see `_run_intent_protocol`'s own docstring) are left
    untouched: TERMINAL has no further valid transition, and COMMITTED
    would be a false claim for a candidate that was never actually
    acted on this run, or one whose outcome is still genuinely
    ambiguous. Only materialized (BROKER_ACKNOWLEDGED) intents commit
    -- closing the real, pre-existing gap where COMMITTED was
    previously a dead end in practice, even though `order_intent.py`'s
    own `_VALID_TRANSITIONS` has always allowed COMMITTED -> TERMINAL.

    Returns the same list, each element updated in place to its final
    status -- purely for test convenience; `_execute()`'s own caller
    does not use the return value today."""
    for intent in all_run_intents:
        if intent.status != order_intent.BROKER_ACKNOWLEDGED:
            continue
        committed = order_intent.transition_intent(intent, order_intent.COMMITTED)
        print(f"[CONTROL] Intent {committed.intent_id} ({committed.ticker}): BROKER_ACKNOWLEDGED -> COMMITTED")
        order_intent.transition_intent(committed, order_intent.TERMINAL)
        print(f"[CONTROL] Intent {committed.intent_id} ({committed.ticker}): COMMITTED -> TERMINAL (run finished successfully)")
    return all_run_intents


_PENDING_SIGNAL_ACTION_KIND_FAMILIES: dict[str, tuple[str, ...]] = {
    pending_signal_ttl.KIND_BUY: ("QUEUED_ENTRY_SIGNAL", "ENTRY_MARKET_BUY"),
    pending_signal_ttl.KIND_EXIT: ("QUEUED_EXIT_SIGNAL", "SIGNAL_EXIT_MARKET_SELL"),
}
_INTENT_NON_TERMINAL_STATUSES = frozenset(
    {order_intent.PREPARED, order_intent.SUBMITTING, order_intent.BROKER_ACKNOWLEDGED, order_intent.UNCERTAIN}
)


def _compute_pending_signal_id(ticker: str, kind: str, source_session_date: str) -> str:
    """The SAME deterministic-id formula `_run_intent_protocol` already
    uses for a queued BUY's `client_order_id` (`action_kind=
    "QUEUED_ENTRY_SIGNAL"`) -- see `pending_signal_ttl.py`'s own module
    docstring ("SIGNAL_ID / JOURNAL CORRELATION") for why this must be
    the exact same formula, not a fresh id."""
    action_kind = "QUEUED_ENTRY_SIGNAL" if kind == pending_signal_ttl.KIND_BUY else "QUEUED_EXIT_SIGNAL"
    return rdd._deterministic_client_order_id(ticker, action_kind, source_session_date)


def _has_unresolved_journal_trace_for_signal(signal_id: str) -> bool:
    """Exact match: used when a pending-signal metadata record's own
    `signal_id` is known (a real, non-`age_unknown` record)."""
    for intent in order_intent.list_intents():
        if intent.client_order_id == signal_id and intent.status in _INTENT_NON_TERMINAL_STATUSES:
            return True
    return False


def _has_unresolved_journal_trace_for_ticker(ticker: str, kind: str) -> bool:
    """Broader, ticker+kind-family match (any date): used ONLY for the
    `age_unknown` case (no metadata record exists, so no exact
    `signal_id` can be reproduced -- see `pending_signal_ttl.evaluate_pending_signals`'s
    own docstring)."""
    families = _PENDING_SIGNAL_ACTION_KIND_FAMILIES[kind]
    for intent in order_intent.list_intents():
        if intent.ticker == ticker and intent.action_kind in families and intent.status in _INTENT_NON_TERMINAL_STATUSES:
            return True
    return False


def _finalize_pending_signals_after_run(
    state_path: Path,
    *,
    client,
    decision: dict,
    ttl_outcome: "pending_signal_ttl.PendingSignalTTLOutcome",
) -> dict[str, str]:
    """Second, deliberate save -- same discipline as
    `_stamp_last_processed_dates` (see that function's own docstring for
    the full "why a second save" rationale): `rdd.run_daily_decision()`'s
    internal save has already wiped `pending_signal_metadata` to `{}` by
    this point (it builds a fresh `LiveRunnerState` that never passes
    this kwarg). Reloads, restores every record `evaluate_pending_signals`
    set earlier THIS run (pre-run TTL outcomes: expired/quarantined
    records, and the transient `reconfirming` markers for stale EXITs),
    then calls `pending_signal_ttl.stamp_new_pending_signals` to create
    fresh records for whatever this run's real decision newly queued, and
    to finalize each `reconfirming` EXIT to either `reconfirmed` (a fresh
    queued exit for that ticker exists -- the frozen engine's own
    unmodified Close-phase re-evaluation found the exit condition still
    valid) or `expired_after_reconfirmation` (it does not). Saves once
    more. Returns the reconfirmation outcomes for the caller to notify."""
    runner_state = rdd.ps.load_position_state(state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH)
    restored: dict[str, dict] = {}
    for ticker, record in ttl_outcome.expired_buys.items():
        restored[pending_signal_ttl.pending_key(ticker, pending_signal_ttl.KIND_BUY)] = record
    for ticker, record in ttl_outcome.quarantined_buys.items():
        if record:
            restored[pending_signal_ttl.pending_key(ticker, pending_signal_ttl.KIND_BUY)] = record
    for ticker, record in ttl_outcome.reconfirming_exits.items():
        restored[pending_signal_ttl.pending_key(ticker, pending_signal_ttl.KIND_EXIT)] = record
    for ticker, record in ttl_outcome.quarantined_exits.items():
        if record:
            restored[pending_signal_ttl.pending_key(ticker, pending_signal_ttl.KIND_EXIT)] = record
    runner_state.pending_signal_metadata.update(restored)

    equity_session_date = decision.get("as_of_bar_timestamp_equity")
    reconfirmation_outcomes = pending_signal_ttl.stamp_new_pending_signals(
        client=client,
        runner_state=runner_state,
        decision=decision,
        equity_session_date=equity_session_date,
        compute_signal_id=_compute_pending_signal_id,
        reconfirming_exit_old_records=ttl_outcome.reconfirming_exits,
    )
    rdd.ps.save_position_state(runner_state, state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH)
    return reconfirmation_outcomes


# Outcome codes _execute() can return -- see module docstring's
# "HEALTHCHECKS.IO DEAD-MAN'S-SWITCH" section and main()'s own mapping
# from these strings to liveness/operational pings. Plain strings, not an
# Enum, kept deliberately simple since these are also the literal
# Healthchecks POST-body detail values (see healthchecks_ping.py's own
# "POST BODY" docstring section -- non-sensitive status codes only).
OUTCOME_RUN_OK = "RUN_OK"
OUTCOME_STOP_ACTIVE = "STOP_ACTIVE"
OUTCOME_FREEZE_ACTIVE = "FREEZE_ACTIVE"
OUTCOME_LOCK_CONFLICT = "LOCK_CONFLICT"
OUTCOME_MARKET_CLOSED_EXPECTED = "MARKET_CLOSED_EXPECTED"
OUTCOME_TELEGRAM_DELIVERY_FAILED = "TELEGRAM_DELIVERY_FAILED"
# Only these two outcomes ping the operational check `success` -- every
# other reachable outcome pings `fail` (still liveness `success`, since
# none of them are a crash). See module docstring.
_OPERATIONAL_SUCCESS_OUTCOMES = frozenset({OUTCOME_RUN_OK, OUTCOME_MARKET_CLOSED_EXPECTED})

_HEALTHCHECKS_NOT_CONFIGURED_NOTICE_MARKER = (
    rdd.ps.HIGH_WATER_MARK_PATH.parent / "healthchecks_not_configured_notice_sent"
)


def _ping_healthcheck(base_url: str | None, event: str, detail: str) -> bool:
    """Thin wrapper around `healthchecks_ping.ping_healthcheck` -- kept as
    a separate module-level name (same pattern as `_notify_control`
    wrapping `rdd._notify_safe`) so call sites in this file read as
    control-arm-specific operations, and so tests can monkeypatch this
    one name without reaching into `healthchecks_ping` internals."""
    return healthchecks_ping.ping_healthcheck(base_url, event, detail)


def _notify_and_warn(text: str) -> bool:
    """`_notify_control(text)`, but captures and reports a delivery
    failure at every call site uniformly -- the fix for an independent
    audit finding: several `_notify_control(...)` call sites (STOP,
    FREEZE, the lock-conflict path, the final daily success message)
    previously ignored the return value entirely, so a silently-failed
    Telegram delivery for any of them was invisible. Never raises, never
    affects trading/state -- purely a reporting fix. Returns the same
    bool `_notify_control` returned, for callers that need to fold it
    into an outcome decision (see `_execute()`'s final RUN_OK vs
    TELEGRAM_DELIVERY_FAILED branch)."""
    delivered = _notify_control(text)
    if not delivered:
        print(
            "WARNING: [CONTROL] Telegram notification could not be confirmed "
            "sent (check TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID and network "
            f"reachability). Message was: {text[:200]!r}",
            file=sys.stderr,
        )
    return delivered


def _load_healthchecks_config(healthchecks_env_file: str | None) -> tuple[str | None, str | None]:
    """Returns `(liveness_url, operational_url)`, or `(None, None)` if
    Healthchecks integration is not available this run -- NEVER raises.

    If `healthchecks_env_file` is given, attempts to load it with
    `healthchecks_ping.load_healthchecks_env_file_fail_closed` -- the
    EXACT same fail-closed mechanics as `--env-file`'s own loader
    (`resolve(strict=True)`, `load_dotenv()` result verified). But unlike
    `--env-file`, any failure at that stage (missing file, unreadable,
    nothing loaded) is caught HERE and downgraded to "not available this
    run" rather than propagated to block the actual trading run --
    Healthchecks is optional best-effort infrastructure, and per this
    task's own explicit instruction, its own absence/misconfiguration
    must never block the system. A missing/empty `HEALTHCHECKS_LIVENESS_URL`/
    `HEALTHCHECKS_OPERATIONAL_URL` (e.g. the file loaded fine but the real
    UUIDs haven't been filled in yet) is treated the same way -- silently
    disabled, not an error."""
    if healthchecks_env_file:
        try:
            healthchecks_ping.load_healthchecks_env_file_fail_closed(healthchecks_env_file)
        except (FileNotFoundError, RuntimeError) as error:
            print(
                f"[CONTROL] Healthchecks config could not be loaded from "
                f"{healthchecks_env_file!r} ({type(error).__name__}: {error}) "
                f"-- Healthchecks integration disabled for this run, trading "
                f"proceeds normally."
            )
            return None, None
    liveness_url = os.environ.get(healthchecks_ping.LIVENESS_URL_ENV_VAR) or None
    operational_url = os.environ.get(healthchecks_ping.OPERATIONAL_URL_ENV_VAR) or None
    return liveness_url, operational_url


def _notify_healthchecks_not_configured_once() -> None:
    """Sends a ONE-TIME (marker-file-guarded, not per-cron-run)
    `[CONTROL]` Telegram notice that Healthchecks integration is not
    configured. A cron job runs as a fresh process every day, so a plain
    "notify every time this is true" would resend this notice daily
    forever -- the marker file (in the same shared guard directory as
    the single-instance lock and the high-water mark, so it survives
    across cron invocations) makes this genuinely one-time until a human
    deletes it."""
    if _HEALTHCHECKS_NOT_CONFIGURED_NOTICE_MARKER.exists():
        return
    _notify_and_warn(
        "Healthchecks.io integration is not configured (no "
        "--healthchecks-env-file given, the file it points to is "
        "missing, or HEALTHCHECKS_LIVENESS_URL/HEALTHCHECKS_OPERATIONAL_URL "
        "are blank). The control arm will keep running normally without "
        "dead-man's-switch monitoring until this is configured. This "
        "notice is sent once; delete "
        f"{_HEALTHCHECKS_NOT_CONFIGURED_NOTICE_MARKER} to see it again."
    )
    _HEALTHCHECKS_NOT_CONFIGURED_NOTICE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _HEALTHCHECKS_NOT_CONFIGURED_NOTICE_MARKER.write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


# MANUAL, SOURCE-LEVEL GATE -- not flipped by the task that built the
# write-ahead mechanism below. See module docstring's "INTENT JOURNAL
# ORDERING GAP" section: `_write_blind_prepared_intents` +
# `_verify_write_ahead_evidence_before_broker_call` now exist and are
# covered by real, deterministic tests (see
# tests/test_run_control_arm_decision_guards.py) -- but "sandbox-tested"
# is not "proven safe with a real broker in real production use," and
# this control arm has still never enabled --enable-equity-orders/
# --enable-crypto-orders in production. Only the owner, after real
# production validation, should ever flip this to True. Until then it
# stays False and `_guard_against_premature_order_activation` blocks
# exactly as before -- zero behavior change in production from the task
# that added this mechanism.
_WRITE_AHEAD_JOURNAL_OWNER_APPROVED = False


def _guard_against_premature_order_activation(arguments: argparse.Namespace) -> None:
    """Hard, zero-API-call guard -- see module docstring's "INTENT
    JOURNAL ORDERING GAP" section for the full "why". Real order
    submission (`--enable-equity-orders`/`--enable-crypto-orders`) is
    BLOCKED unless `_WRITE_AHEAD_JOURNAL_OWNER_APPROVED` is True (see
    that constant's own comment -- a manual, source-level flag this
    guard does not and cannot flip itself). Called as the FIRST thing
    `_execute()` does, before any other check or API call.

    This early guard is deliberately NOT the only enforcement point: it
    cannot yet know whether write-ahead evidence genuinely exists for
    THIS invocation's own candidates (`runner_state.pending_buys`/
    `pending_exits` are not even loaded yet at this point in `_execute()`
    -- see its own docstring's "trading_client -- INJECTION SEAM"
    section for the overall call order). Even when
    `_WRITE_AHEAD_JOURNAL_OWNER_APPROVED` is True, the REAL, runtime
    check is `_verify_write_ahead_evidence_before_broker_call`, called
    later, once state is loaded and `_write_blind_prepared_intents` has
    actually run for this invocation -- it reads `order_intent.py`'s own
    on-disk journal and fails closed if evidence is missing for even one
    candidate. This function alone passing a True flag is a necessary,
    not sufficient, condition for `rdd.run_daily_decision()` ever being
    called with a real order-enabling flag."""
    if not (arguments.enable_equity_orders or arguments.enable_crypto_orders):
        return
    if not _WRITE_AHEAD_JOURNAL_OWNER_APPROVED:
        raise RuntimeError(
            "Real order activation (--enable-equity-orders/--enable-crypto-orders) "
            "is BLOCKED: _WRITE_AHEAD_JOURNAL_OWNER_APPROVED is False. The "
            "write-ahead intent mechanism (_write_blind_prepared_intents / "
            "_verify_write_ahead_evidence_before_broker_call) now exists and is "
            "sandbox-tested, but this manual, source-level flag has not been "
            "set by the owner after real production validation -- see this "
            "constant's own comment just above this function. Do not pass "
            "either flag until the owner explicitly approves and flips it."
        )


def _verify_write_ahead_evidence_before_broker_call(
    runner_state,
    expected_session_date: str,
    arguments: argparse.Namespace,
) -> None:
    """The REAL runtime gate `_guard_against_premature_order_activation`'s
    own docstring refers to. A no-op unless either order-enabling flag is
    set (today: never true in production). Confirms, by actually reading
    `order_intent.py`'s on-disk journal via
    `order_intent.find_intent_by_client_order_id` (not by trusting the
    static `_WRITE_AHEAD_JOURNAL_OWNER_APPROVED` flag alone), that a
    durable, non-terminal intent genuinely exists for EVERY candidate in
    `runner_state.pending_buys`/`pending_exits` -- i.e. that
    `_write_blind_prepared_intents` really ran, for real, for THIS exact
    invocation, not merely that the function exists in this file. Called
    right after that pre-write step, right before `rdd.run_daily_decision()`.
    Fail-closed: raises `RuntimeError` if evidence is missing for even one
    candidate while either order-enabling flag is set."""
    if not (arguments.enable_equity_orders or arguments.enable_crypto_orders):
        return
    # Same fix as _write_blind_prepared_intents -- see its own comment
    # for the real id-mismatch bug this closes. This function must use
    # the exact same id formula that function used, or it would always
    # report every candidate as "missing" even when the journal is
    # genuinely complete.
    missing: list[str] = []
    for ticker in runner_state.pending_buys:
        client_order_id = rdd.client_order_id_for_action(ticker, "ENTRY_MARKET_BUY", str(expected_session_date))
        intent = order_intent.find_intent_by_client_order_id(client_order_id)
        if intent is None or intent.status not in _INTENT_NON_TERMINAL_STATUSES:
            missing.append(f"{ticker} (BUY, client_order_id={client_order_id!r})")
    for ticker in runner_state.pending_exits:
        client_order_id = rdd.client_order_id_for_action(ticker, "SIGNAL_EXIT_MARKET_SELL", str(expected_session_date))
        intent = order_intent.find_intent_by_client_order_id(client_order_id)
        if intent is None or intent.status not in _INTENT_NON_TERMINAL_STATUSES:
            missing.append(f"{ticker} (EXIT, client_order_id={client_order_id!r})")
    if missing:
        raise RuntimeError(
            f"Write-ahead evidence missing for {len(missing)} candidate(s) about to be "
            f"acted on by rdd.run_daily_decision() with a real order-enabling flag set: "
            f"{missing}. Refusing to proceed -- _write_blind_prepared_intents did not "
            f"durably record a non-terminal intent for every candidate this run would act on."
        )


def _execute(arguments: argparse.Namespace, *, trading_client: TradingClient | None = None) -> str:
    """Everything `main()` used to do directly, now returning an outcome
    code (see the `OUTCOME_*` constants) instead of bare `return`
    statements, so the caller can classify liveness/operational
    Healthchecks pings without needing its own copy of this control flow.
    Any genuinely unexpected exception still propagates uncaught (after
    attempting a `[CONTROL]` failure notification) -- `main()`'s own
    try/except/finally around the call to this function is what pings
    Healthchecks `fail` for that case and re-raises so the process still
    exits non-zero.

    The inner try/except below is DELIBERATELY widened to cover
    EVERYTHING from the first pre-flight check through the final daily
    success notification -- fixed after an independent audit found the
    ORIGINAL version only wrapped preflight+`run_daily_decision()`, so a
    failure in the audit checks, `_finalize_pending_signals_after_run`,
    `_run_intent_protocol`, `_stamp_last_processed_dates`, or the intent
    COMMITTED loop previously propagated with NO `[CONTROL]` failure
    notification at all -- a real, silent gap in owner-facing visibility
    for exactly the phase where a bug would be most consequential (after
    the frozen engine has already computed and partially acted on a real
    decision).

    `trading_client` -- INJECTION SEAM, test-harness-only, deliberately
    NOT exposed as a CLI flag anywhere (see `main()`'s own argparse
    setup, which has no such option): when `None` (every real production
    invocation -- `main()` always calls `_execute(arguments)` with no
    override), behavior is completely unchanged from before this
    parameter existed -- the real `order_submission.get_trading_client()`
    factory is used, exactly once, at the same point in the flow this
    call already happened at. When a caller supplies a client directly
    (only reachable by importing this module and calling `_execute` or
    `main` as a Python function, e.g. from a test -- there is no
    argparse path to it), that ONE object is reused for every real
    Alpaca call this run makes (`get_all_assets` for issuer-identity,
    `get_account`/`get_orders`/`get_all_positions`/`get_order_by_id` for
    broker reconciliation, `get_calendar` for the session-date check) --
    a single resolution point, not a client constructed independently at
    each call site, so a fake client transparently reaches the whole
    real flow.
    """
    _guard_against_premature_order_activation(arguments)

    print(
        f"[CONTROL] Ticker universe in effect ({len(CONTROL_TICKERS_WITH_REGIME)} tickers: "
        f"{len(CONTROL_UNIVERSE_TICKERS)} strategy + 1 regime-reference-only): "
        f"{sorted(CONTROL_TICKERS_WITH_REGIME)}"
    )
    assert rdd.LIVE_CONTROLLED_TICKERS == CONTROL_TICKERS_WITH_REGIME
    assert rdd.prepare_live_market_data is _prepare_control_arm_market_data

    # STOP checked FIRST, before even attempting the lock -- matches
    # run_daily_decision.main()'s own precedence (STOP wins,
    # unconditionally, before anything else) and means a STOP file alone
    # is enough to skip a run with zero contention on the lock at all.
    if rdd.STOP_FLAG_PATH.exists():
        text = (
            f"AI-Stock-Radar CONTROL ARM daily decision runner -- STOP flag "
            f"detected ({rdd.STOP_FLAG_PATH.resolve()}); this run was "
            f"skipped entirely, no Alpaca API calls were made. Remove the file to resume."
        )
        print(text)
        _notify_and_warn(text)
        return OUTCOME_STOP_ACTIVE

    # Single-instance lock -- acquired BEFORE any of the checks below,
    # held for the rest of this run via the `with` block, released the
    # instant it exits (success, `return`, or exception) -- see
    # single_instance_lock.py's own module docstring for why this is a
    # deliberately DIFFERENT mechanism from position_state.py's
    # _guard_write_lock (kernel-released via fd closure on process exit,
    # no age-based staleness heuristic at all, fail-closed with no
    # override if another real process holds it).
    try:
        with single_instance_lock(arguments.state_path) as lock_metadata:
            print(f"[CONTROL] Single-instance lock acquired: {lock_metadata}")

            # STOP re-checked NOW that the lock is held -- closes the
            # (tiny, but real) window between the first check above and
            # lock acquisition during which a STOP file could have been
            # created.
            if rdd.STOP_FLAG_PATH.exists():
                text = (
                    f"AI-Stock-Radar CONTROL ARM daily decision runner -- STOP flag "
                    f"detected after acquiring the lock ({rdd.STOP_FLAG_PATH.resolve()}); "
                    f"this run was skipped entirely, no Alpaca API calls were made. "
                    f"Remove the file to resume."
                )
                print(text)
                _notify_and_warn(text)
                return OUTCOME_STOP_ACTIVE

            freeze = rdd.FREEZE_FLAG_PATH.exists()
            if freeze:
                text = (
                    f"AI-Stock-Radar CONTROL ARM daily decision runner -- FREEZE flag "
                    f"detected ({rdd.FREEZE_FLAG_PATH.resolve()}); this run will still "
                    f"fetch data and monitor/exit existing positions as usual, but will "
                    f"NOT open any new positions. Remove the file to resume normal entries."
                )
                print(text)
                _notify_and_warn(text)

            try:
                # Tracks whether EVERY [CONTROL] Telegram notification
                # this run attempts actually delivered -- folded into the
                # final RUN_OK vs TELEGRAM_DELIVERY_FAILED outcome code
                # below (see _execute()'s own docstring / module
                # docstring's Healthchecks section). A failure here never
                # affects trading/state, only which outcome code gets
                # reported.
                telegram_all_ok = True

                # Pre-flight, BEFORE any API call: refuse to even call
                # run_daily_decision() if the persisted state already
                # contains a stale out-of-universe (incl. BTC-USD)
                # position/pending entry. See _preflight_universe_check's
                # own docstring for why this is a stronger guarantee
                # than the post-hoc audit check below.
                _preflight_universe_check(arguments.state_path, frozenset(CONTROL_UNIVERSE_TICKERS))

                # Second pre-flight, same zero-Alpaca-API-calls
                # discipline: refuse to double-process today's equity
                # bar. See its own docstring for exactly what this does
                # and does not solve.
                _preflight_double_run_check(arguments.state_path)

                # PHASE 3: issuer-identity preflight -- BEFORE broker
                # reconciliation and before any market-data fetch. Real,
                # unfiltered Alpaca get_all_assets() (daily) + a
                # weekly-cadence-refreshed SEC EDGAR CIK cross-check for
                # all 44 control tickers against the pre-registered
                # anchor artifact (config/control_universe_identity_anchors_v1.json)
                # -- see src/live/issuer_identity_preflight.py's own
                # module docstring for the full nine-hard-condition/
                # two-soft-condition table. A hard mismatch raises
                # IssuerIdentityMismatchError, caught by this same
                # try/except below (notifies [CONTROL], re-raises,
                # run_daily_decision() never called -- zero progress, no
                # single-ticker exclusion exists anywhere in that
                # module). A soft (review-required) drift does NOT raise
                # -- this run continues normally, but gets a [CONTROL]
                # notice below.
                # Single resolution point for the whole run -- see this
                # function's own docstring, "trading_client -- INJECTION
                # SEAM". Real factory unless a test supplied one.
                reconciliation_client = trading_client if trading_client is not None else rdd.order_submission.get_trading_client()
                issuer_identity_result = issuer_identity_preflight.run_issuer_identity_preflight(reconciliation_client)
                if issuer_identity_result.soft_findings:
                    for finding in issuer_identity_result.soft_findings:
                        telegram_all_ok &= _notify_and_warn(
                            f"Issuer-identity SOFT drift (review required, NOT blocking): "
                            f"{finding.ticker} {finding.field} changed from "
                            f"{finding.old_value!r} to {finding.new_value!r} -- {finding.detail}"
                        )
                print(
                    f"[CONTROL] Issuer-identity preflight passed: "
                    f"{issuer_identity_result.checked_ticker_count} ticker(s) verified, "
                    f"status={issuer_identity_result.status}, "
                    f"SEC cache age={issuer_identity_result.sec_data_age_days}."
                )

                # Loaded HERE, before Phase 1.5, rather than at Phase 2a
                # as before (reboot-drill finding #1, 2026-08-24): Phase
                # 1.5's own stray-recovery loop now needs
                # `pending_buys`/`pending_exits` to decide whether a
                # confirmed-404 PREPARED intent still has a live,
                # matching candidate this run (see
                # `_stray_intent_matches_live_candidate` below) --
                # RESUMABLE_PREPARED vs ABANDONED_NO_SUBMISSION. The SAME
                # loaded object is reused, unmodified, at Phase 2a below
                # (no second, redundant load).
                reconciliation_state = rdd.ps.load_position_state(
                    arguments.state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH
                )

                # PHASE 1.5 (Task 3, 2026-08-22, item 6; finalization
                # closed independent-audit-round-2, 2026-08-23, findings
                # #1/#2; two-phase recovery model fixed reboot-drill
                # findings #1/#2, 2026-08-24): post-crash order-intent
                # recovery -- BEFORE broker reconciliation's own snapshot
                # below, using the SAME already-resolved
                # reconciliation_client (one real TradingClient for this
                # whole run, same discipline as every other call here).
                # Resolves any stray PREPARED/SUBMITTING/BROKER_ACKNOWLEDGED/
                # COMMITTED/UNCERTAIN intent a PRIOR, interrupted run left
                # behind against the real broker -- never auto-resubmits,
                # never auto-retries a cancel (see
                # order_intent_reconciliation.py's own module docstring,
                # "TWO-PHASE RECOVERY", for the full resolution table).
                #
                # `resolve_stray_intent` now returns one of FOUR
                # legitimate resting statuses, not just TERMINAL/UNCERTAIN
                # as before: TERMINAL (fully resolved), PREPARED
                # (RESUMABLE_PREPARED -- a confirmed 404 whose candidate
                # is still live THIS run; `_write_blind_prepared_intents`
                # below will naturally reuse it), BROKER_ACKNOWLEDGED
                # (broker truth is known, but local-state reconciliation
                # has not run/completed yet -- Phase 2a below is what
                # determines whether that catches up cleanly or halts via
                # `JournalAcknowledgedButLocalStateMissingError`), or
                # UNCERTAIN (genuinely ambiguous, always blocks). Only
                # UNCERTAIN (or anything else entirely unexpected) blocks
                # HERE, fail-closed -- human review required before
                # proceeding; PREPARED/BROKER_ACKNOWLEDGED are legitimate
                # resting states this phase deliberately does NOT try to
                # resolve further itself (see module docstring for why:
                # moving broker_reconciliation's own reconcile() call earlier does
                # not help, it would still hit the exact same halt for
                # the BROKER_ACKNOWLEDGED case, just reordered).
                stray_intents = order_intent_reconciliation.find_stray_session_intents_from_prior_run()
                if stray_intents:
                    print(
                        f"[CONTROL] Found {len(stray_intents)} stray order-intent(s) from a "
                        f"prior run -- resolving against the broker before proceeding."
                    )
                    unresolved_stray_intents: list[order_intent.OrderIntent] = []
                    for stray in stray_intents:
                        matches_live_candidate = _stray_intent_matches_live_candidate(stray, reconciliation_state)
                        resolved = order_intent_reconciliation.resolve_stray_intent(
                            reconciliation_client, stray, matches_live_candidate=matches_live_candidate,
                        )
                        print(
                            f"[CONTROL] Stray intent {resolved.intent_id} ({resolved.ticker} "
                            f"{resolved.action_kind}): {stray.status} -> {resolved.status}"
                            + (" (RESUMABLE_PREPARED -- live candidate still matches this run)"
                               if resolved.status == order_intent.PREPARED else "")
                            + (" (broker truth known; awaiting Phase 2a local-state reconciliation)"
                               if resolved.status == order_intent.BROKER_ACKNOWLEDGED else "")
                        )
                        if resolved.status not in (
                            order_intent.TERMINAL, order_intent.PREPARED, order_intent.BROKER_ACKNOWLEDGED,
                        ):
                            unresolved_stray_intents.append(resolved)
                    if unresolved_stray_intents:
                        raise RuntimeError(
                            f"{len(unresolved_stray_intents)} stray order-intent(s) from a prior run "
                            f"remain unresolved (UNCERTAIN) after broker-authoritative recovery -- "
                            f"refusing to proceed. Human review required: "
                            f"{[(i.intent_id, i.ticker, i.action_kind, i.status) for i in unresolved_stray_intents]}"
                        )

                # PHASE 2a: broker-vs-local reconciliation, BEFORE any
                # market-data fetch or decision computation. Account
                # identity check, then a consistency-verified broker
                # snapshot (open orders + positions), then the four
                # documented mismatch scenarios -- see
                # src/live/broker_reconciliation.py's own module docstring
                # for the full case table. Raises (fail-closed) on any
                # anomaly, caught by this same try/except below, which
                # notifies [CONTROL] and re-raises WITHOUT ever calling
                # rdd.run_daily_decision() -- so a reconciliation failure
                # leaves the decision log, position state, and any real
                # broker order completely untouched. The one narrow
                # exception (Scenario C's fully-matched "filled" case) is
                # itself a successful reconciliation outcome, not a
                # failure -- it updates only an order-status field on the
                # freshly-loaded state below and persists that correction
                # to disk before proceeding, exactly like any other
                # successful pre-decision state fix. `broker_reconciliation.py`'s
                # own Scenario D now DOES inspect `order_intent.list_intents()`
                # (reboot-drill finding #2, 2026-08-24) -- narrowly, only
                # to distinguish a journal-correlated crash-recovery gap
                # from a genuinely mystery broker order; still fail-closed
                # either way, only the diagnostic differs. Reuses
                # `reconciliation_client`, already constructed above for
                # the issuer-identity preflight -- one real TradingClient
                # instance for this whole run, not a fresh one per check.
                # `reconciliation_state` reuses the SAME object loaded
                # before Phase 1.5 above -- not reloaded here.
                reconciliation_result = broker_reconciliation.reconcile(
                    reconciliation_client, reconciliation_state,
                    # REAL REGRESSION FOUND AND FIXED (2026-08-22,
                    # independent audit): reconcile()'s signature moved
                    # from a single blended `orders_enabled` to two
                    # independent flags (see that function's own
                    # docstring, "ORDERS-DISABLED / DRY-RUN AWARENESS")
                    # when the equity/crypto split was fixed for
                    # run_daily_decision.py's own caller -- this call
                    # site was never updated to match, so every real
                    # control-arm run has been raising a real TypeError
                    # (unexpected keyword argument 'orders_enabled')
                    # before ever reaching run_daily_decision(). Each
                    # ticker's own asset-class flag now passed through
                    # directly, exactly as run_daily_decision.py's own
                    # call already does.
                    equity_orders_enabled=arguments.enable_equity_orders,
                    crypto_orders_enabled=arguments.enable_crypto_orders,
                )
                if reconciliation_result.order_status_updates:
                    print(
                        f"[CONTROL] Broker reconciliation applied narrow "
                        f"order-status correction(s) (Scenario C, "
                        f"filled-and-fully-matched only): "
                        f"{reconciliation_result.order_status_updates}"
                    )
                    rdd.ps.save_position_state(
                        reconciliation_state, arguments.state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH
                    )
                print(
                    f"[CONTROL] Broker reconciliation passed: account "
                    f"{reconciliation_result.account_number_masked}, "
                    f"{reconciliation_result.local_position_count} local "
                    f"position(s), {reconciliation_result.broker_position_count} "
                    f"broker position(s), "
                    f"{reconciliation_result.broker_open_order_count} broker "
                    f"open order(s), checked at {reconciliation_result.checked_at}."
                )

                # Real Alpaca calendar check, BEFORE any market-data
                # fetch. See module docstring's "MARKET HOLIDAY /
                # SESSION-DATE VERIFICATION" section.
                expected_session_date = _expected_equity_session_date(reconciliation_client)
                last_processed_session_date = _read_last_processed_equity_session_date(arguments.state_path)
                if expected_session_date is None:
                    text = (
                        "AI-Stock-Radar CONTROL ARM daily decision runner -- no "
                        "settled equity session today (no calendar entry, or "
                        "today's session has not yet closed + settled). Clean "
                        "no-op, zero progress, zero Alpaca market-data calls."
                    )
                    print(text)
                    _notify_and_warn(text)
                    return OUTCOME_MARKET_CLOSED_EXPECTED

                # PHASE 2b: pending-signal TTL, BEFORE run_daily_decision()
                # is ever called -- see src/live/pending_signal_ttl.py's
                # own module docstring for the full one-shot-session
                # design and the crypto-date bug it works around WITHOUT
                # editing run_daily_decision.py. Pure local comparison,
                # no broker/API call (the one real calendar call this
                # feature needs -- computing a NEW signal's
                # target_execution_session_date -- happens only in the
                # POST-run stamping step below, never here). Mutates
                # reconciliation_state's pending_buys/pending_exits/
                # pending_signal_metadata in place and persists that
                # BEFORE run_daily_decision() loads state, so a stale
                # signal is pruned before the frozen engine can ever see
                # it. A raised PendingSignalReconciliationRequiredError
                # (ambiguous stale EXIT, no local position to reconfirm
                # against) is caught by the same except block below,
                # exactly like every other preflight failure -- nothing
                # is saved on that path (see the function's own docstring
                # for why), run_daily_decision() is never called.
                ttl_outcome = pending_signal_ttl.evaluate_pending_signals(
                    runner_state=reconciliation_state,
                    expected_session_date=expected_session_date,
                    freeze_active=freeze,
                    has_unresolved_journal_trace_for_signal=_has_unresolved_journal_trace_for_signal,
                    has_unresolved_journal_trace_for_ticker=_has_unresolved_journal_trace_for_ticker,
                )
                if ttl_outcome.touched_anything:
                    rdd.ps.save_position_state(
                        reconciliation_state, arguments.state_path, guard_path=rdd.ps.HIGH_WATER_MARK_PATH
                    )
                    for ticker, record in ttl_outcome.expired_buys.items():
                        telegram_all_ok &= _notify_and_warn(
                            f"Pending BUY EXPIRED (TTL, one-shot window missed): {ticker} -- "
                            f"source session {record.get('source_session_date')}, target "
                            f"execution session {record.get('target_execution_session_date')}, "
                            f"now {expected_session_date}, expire_reason="
                            f"{record.get('expire_reason')}. No broker order was ever created "
                            f"for this expired signal; it was never resubmitted."
                        )
                    for ticker, record in ttl_outcome.quarantined_buys.items():
                        telegram_all_ok &= _notify_and_warn(
                            f"Pending BUY for {ticker} looks stale but has an unresolved "
                            f"order-journal trace -- QUARANTINED, left in place, NOT expired. "
                            f"Needs manual/broker reconciliation before this can resolve."
                        )
                    for ticker in ttl_outcome.reconfirming_exits:
                        telegram_all_ok &= _notify_and_warn(
                            f"Pending EXIT for {ticker} is stale (one-shot window missed) -- "
                            f"cleared for deterministic reconfirmation against the current "
                            f"Close, using the frozen engine's own unmodified exit rule, later "
                            f"in this same run."
                        )
                    for ticker, record in ttl_outcome.quarantined_exits.items():
                        telegram_all_ok &= _notify_and_warn(
                            f"Pending EXIT for {ticker} looks stale but has an unresolved "
                            f"order-journal trace -- QUARANTINED, left in place, NOT touched. "
                            f"Needs manual/broker reconciliation before this can resolve."
                        )
                    print(
                        f"[CONTROL] Pending-signal TTL: {len(ttl_outcome.expired_buys)} BUY(s) "
                        f"expired, {len(ttl_outcome.quarantined_buys)} BUY(s) quarantined, "
                        f"{len(ttl_outcome.reconfirming_exits)} EXIT(s) sent for reconfirmation, "
                        f"{len(ttl_outcome.quarantined_exits)} EXIT(s) quarantined."
                    )

                pre_state_hash = _hash_state_file(arguments.state_path)

                # WRITE-AHEAD INTENT JOURNAL -- closes this module docstring's
                # own "INTENT JOURNAL ORDERING GAP": write one durable
                # "blind" PREPARED intent per candidate already queued in
                # reconciliation_state.pending_buys/pending_exits (TTL-
                # adjusted above, the SAME dicts rdd.run_daily_decision()
                # is about to load fresh from this exact state file) --
                # BEFORE that call, and therefore before any real broker
                # call it might make. See _write_blind_prepared_intents's
                # own docstring for the full design and the crash-recovery
                # (idempotent reuse) case.
                blind_intents = _write_blind_prepared_intents(
                    reconciliation_state, expected_session_date, pre_state_hash
                )

                # The REAL runtime gate for real order activation -- see
                # _guard_against_premature_order_activation's own
                # docstring. No-op today (never true in production, see
                # that guard), but genuinely verifies write-ahead evidence
                # exists on disk for this exact invocation whenever either
                # order-enabling flag IS set.
                _verify_write_ahead_evidence_before_broker_call(
                    reconciliation_state, expected_session_date, arguments
                )

                # Independent audit finding "Madde E" (2026-08-22):
                # self-authorizes ONLY here, AFTER every preflight guard
                # above has already passed (identity, reconciliation,
                # TTL, session, write-ahead evidence) -- see
                # src.live.authorized_execution_context's own module
                # docstring for the full design. Constructed only when
                # actually needed (an order-enabling flag is set); the
                # dry-run path never touches this.
                authorization = (
                    authorize_order_execution()
                    if (arguments.enable_equity_orders or arguments.enable_crypto_orders)
                    else None
                )

                # Task 3 (2026-08-22): "hook required in control-arm's own
                # real-order mode" fail-closed check -- deliberately placed
                # HERE, in this file's own preflight, never inside
                # run_daily_decision.py's shared functions. Those functions
                # are caller-agnostic by design and are also called by the
                # LIVE system's own main() (see authorized_execution_context's
                # module docstring: main() self-authorizes but never
                # supplies an order_intent_hook -- order_intent.py is
                # control-arm-only). A check keyed only on
                # `authorization is not None` inside a shared function would
                # therefore have raised on the live cron's very first real
                # order submission. This assertion is currently a pure
                # defense-in-depth measure -- the hook below is already
                # always built and passed regardless of the flags -- guarding
                # against a future refactor of this call site silently
                # dropping the `order_intent_hook=` kwarg while order-enabling
                # flags stay set.
                real_order_intent_hook = _order_intent_hook_for_run(blind_intents)
                if authorization is not None and real_order_intent_hook is None:
                    raise RuntimeError(
                        "Real order submission is authorized (an order-enabling flag is "
                        "set) but no order_intent_hook was built -- refusing to proceed "
                        "without durable write-ahead journal coverage. Fail-closed."
                    )

                result = rdd.run_daily_decision(
                    state_path=arguments.state_path,
                    decision_log_directory=arguments.decision_log_directory,
                    enable_equity_orders=arguments.enable_equity_orders,
                    enable_crypto_orders=arguments.enable_crypto_orders,
                    freeze=freeze,
                    authorization=authorization,
                    # REAL BUG FOUND AND FIXED (2026-08-22, independent
                    # audit): without these two, run_daily_decision()
                    # would build its OWN fresh trading client and run a
                    # SECOND, redundant broker_reconciliation.reconcile()
                    # internally -- using the wrong (LIVE, not control-arm)
                    # expected account suffix, and 3-4 wasted real API
                    # calls -- on top of THIS run's own reconciliation
                    # already done above (reconciliation_client, already
                    # identity-verified against THIS account). Reuse that
                    # same client, skip the internal duplicate.
                    trading_client=reconciliation_client,
                    skip_broker_reconciliation=True,
                    # Independent audit finding #3, approach (B): the
                    # real fix for the SUBMITTING-timing gap -- see
                    # _order_intent_hook_for_run's own docstring. Inert
                    # today (real order submission is still blocked by
                    # _guard_against_premature_order_activation), wired
                    # now so it is already correct and tested once real
                    # orders are ever enabled.
                    order_intent_hook=real_order_intent_hook,
                )

                decision = result["decision"]
                print(json.dumps(decision, indent=2, sort_keys=True, default=str))
                print()
                print(f"Decision log written to: {result['log_path'].resolve()}")
                print(f"Position state updated at: {arguments.state_path.resolve()}")

                # Real-session verification, AFTER the fetch. See module
                # docstring's own honest limitation note: this detects and
                # raises loudly, it does not undo run_daily_decision()'s own
                # internal save that already happened by this point.
                _verify_actual_equity_session(decision, expected_session_date, last_processed_session_date)

                # Audit assertions -- fail loudly if BTC-USD or any
                # non-control ticker ever slipped through despite the
                # patches above. Defense in depth: the module docstring's
                # reasoning says this should be structurally impossible, but
                # "should be" is exactly the class of claim this session's
                # own earlier investigation (a silently wrong dict key
                # produced a false "0 signals" report) taught not to trust
                # without a runtime check.
                open_positions = set(decision.get("open_positions_after_run", {}))
                queued_buys = {b["ticker"] for b in decision.get("queued_for_next_run", {}).get("buys", [])}
                executed_entries = {e["ticker"] for e in decision.get("executed_today", {}).get("entries", [])}
                all_tickers_seen = open_positions | queued_buys | executed_entries

                leaked_btc = all_tickers_seen & {REGIME_ONLY_TICKER}
                if leaked_btc:
                    raise RuntimeError(
                        f"BTC-USD leaked into the tradable control-arm universe despite "
                        f"the RegimeAllowed=False gate: {leaked_btc}. This should be "
                        f"structurally impossible -- stop and investigate before this "
                        f"script is ever used operationally again."
                    )
                non_control = all_tickers_seen - set(CONTROL_UNIVERSE_TICKERS)
                if non_control:
                    raise RuntimeError(
                        f"Ticker(s) outside the 44-ticker control universe appeared in "
                        f"this decision: {non_control}. The universe patch may not have "
                        f"taken effect -- stop and investigate."
                    )
                print("[CONTROL] Audit check passed: zero BTC-USD leakage, zero non-control-universe tickers.")

                # Post-run pending-signal TTL finalize -- see
                # _finalize_pending_signals_after_run's own docstring for why
                # this is a second, deliberate save (run_daily_decision()'s
                # own internal save already wiped pending_signal_metadata to
                # {}). Creates fresh metadata for anything newly queued this
                # run and resolves every `reconfirming` stale EXIT to either
                # `reconfirmed` or `expired_after_reconfirmation`.
                reconfirmation_outcomes = _finalize_pending_signals_after_run(
                    arguments.state_path,
                    client=reconciliation_client,
                    decision=decision,
                    ttl_outcome=ttl_outcome,
                )
                for ticker, outcome_label in reconfirmation_outcomes.items():
                    if outcome_label == pending_signal_ttl.STATUS_RECONFIRMED:
                        telegram_all_ok &= _notify_and_warn(
                            f"Pending EXIT reconfirmation for {ticker}: the frozen engine's own "
                            f"exit rule, re-evaluated against the current Close, STILL finds the "
                            f"exit condition valid -- re-queued with a fresh source/target session."
                        )
                    else:
                        telegram_all_ok &= _notify_and_warn(
                            f"Pending EXIT reconfirmation for {ticker}: the frozen engine's own "
                            f"exit rule, re-evaluated against the current Close, no longer finds "
                            f"the exit condition valid -- closed as {outcome_label}, not re-queued."
                        )

                # PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED (materialized)
                # or PREPARED -> TERMINAL (abandoned guess) -- correlates
                # against the blind intents written above, before
                # rdd.run_daily_decision() was ever called. See
                # _run_intent_protocol's own docstring for exactly what
                # this does and does not represent (no real broker call yet).
                intents = _run_intent_protocol(decision, pre_state_hash, blind_intents=blind_intents)

                # Stamp last_processed_equity_date/crypto_date AFTER the
                # audit checks above pass, deliberately -- if this run's own
                # output were untrustworthy (BTC leak / non-control ticker,
                # both raise before reaching here), we do not want to record
                # it as "successfully processed today" and block a corrected
                # re-run via _preflight_double_run_check.
                _stamp_last_processed_dates(arguments.state_path, actual_session_date=decision.get("as_of_bar_timestamp_equity"))

                # Intents committed only AFTER state has actually been
                # durably re-saved above -- COMMITTED is meant to mean
                # "local state reflects this," not merely "we decided to."
                # See `_finalize_run_intents`'s own docstring for the
                # full reasoning (extracted into its own function
                # 2026-08-23 so it is directly, behaviorally testable --
                # independent-audit-round-2's own test-quality note --
                # rather than only checkable via source-text matching).
                all_run_intents = list(intents) + list(getattr(real_order_intent_hook, "just_in_time_intents", {}).values())
                _finalize_run_intents(all_run_intents)

                telegram_all_ok &= _notify_and_warn(rdd._build_daily_notification_text(decision))
                # `with single_instance_lock(...)` releases the lock here,
                # on normal exit from this block.
                if freeze:
                    # Being frozen is itself the operationally-notable
                    # fact for the healthchecks operational check,
                    # regardless of whether every notification above
                    # delivered -- takes priority over
                    # TELEGRAM_DELIVERY_FAILED in the outcome code below
                    # (a delivery hiccup is still captured and logged to
                    # stderr by _notify_and_warn either way, just not
                    # surfaced as the PRIMARY outcome here).
                    return OUTCOME_FREEZE_ACTIVE
                if not telegram_all_ok:
                    return OUTCOME_TELEGRAM_DELIVERY_FAILED
                return OUTCOME_RUN_OK
            except Exception as error:
                # WIDENED to cover the entire block above (preflight
                # through the final daily notification) -- see this
                # function's own docstring for the audit finding this
                # fixes: previously this except only wrapped up through
                # `rdd.run_daily_decision()`, so an audit-assertion/
                # intent-transition/stamp failure below that point
                # propagated with NO [CONTROL] failure notification at
                # all. Still re-raises -- main()'s own outer
                # try/except/finally is what pings Healthchecks `fail`
                # for this case and preserves the non-zero exit code.
                _notify_and_warn(rdd._build_failure_notification_text(error))
                raise
    except SingleInstanceLockError as error:
        text = (
            f"AI-Stock-Radar CONTROL ARM daily decision runner -- could not "
            f"acquire the single-instance lock; another run is already in "
            f"progress. Refusing to proceed -- zero Alpaca API calls were "
            f"made. {error}"
        )
        print(text)
        _notify_and_warn(text)
        return OUTCOME_LOCK_CONFLICT
    # Any OTHER exception raised inside the `with` block (e.g. from
    # run_daily_decision() failing, or an audit-assertion RuntimeError)
    # is deliberately NOT caught here -- it propagates normally, the
    # `with` block's __exit__ still releases the lock on the way out
    # (standard context-manager semantics), and the inner try/except
    # above has already sent the [CONTROL] failure notification before
    # re-raising. main()'s own outer try/except/finally pings
    # Healthchecks liveness `fail` for this case.


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one day of the frozen TREND_RSI strategy against the "
            "isolated 44-ticker MidCap control-arm universe (see "
            "src/live/control_universe.py). Reuses run_daily_decision()'s "
            "own engine-call logic unmodified; re-implements main()'s "
            "STOP/FREEZE checks identically below rather than inheriting "
            "them, since this script never calls that main()."
        )
    )
    parser.add_argument("--state-path", type=Path, default=rdd.DEFAULT_STATE_PATH)
    parser.add_argument("--decision-log-directory", type=Path, default=rdd.DEFAULT_DECISION_LOG_DIRECTORY)
    parser.add_argument(
        "--enable-equity-orders", action="store_true",
        help="Submit real Alpaca PAPER orders for the 44 equity tickers. Omit to stay dry-run/log-only.",
    )
    parser.add_argument(
        "--enable-crypto-orders", action="store_true",
        help=(
            "BTC-USD will never generate a NEW entry signal in this universe "
            "(RegimeAllowed is force-set False -- see module docstring), so "
            "this flag cannot by itself cause a new BTC-USD trade. It is NOT "
            "an unconditional no-op, though: if a pre-existing BTC-USD "
            "pending/open position were ever sitting in the state file (e.g. "
            "from a bug, a hand-edit, or state copied from elsewhere), this "
            "flag could still act on it via crypto order-submission logic "
            "reused unmodified from run_daily_decision.py. In practice this "
            "script's own pre-flight check (_preflight_universe_check) "
            "refuses to run AT ALL whenever such a stale entry exists, so "
            "that scenario is blocked before this flag is ever consulted -- "
            "but the flag's own semantics are 'acts on crypto state if any "
            "exists,' not 'always inert.' Kept for CLI symmetry with the "
            "live script."
        ),
    )
    parser.add_argument(
        "--env-file", type=str, default=None,
        help=(
            "Optional .env path to layer on top of whatever .env this "
            "process already loaded at import time, override=True. Fails "
            "loudly (FileNotFoundError / RuntimeError) rather than "
            "silently falling back if the path does not resolve or "
            "load_dotenv reports nothing loaded. REQUIRED, always passed, "
            "in the production crontab (control_arm_crontab_v3.draft's "
            "command line passes --env-file /root/AI-Stock-Radar-Control/.env "
            "on every run) -- this is what makes the fail-closed check "
            "above actually run in production, not just in local/sandbox "
            "testing. Technically still optional at the argparse level "
            "(omitting it falls back to whatever .env this process already "
            "loaded at import time via the transitive alpaca_market_data.py/"
            "telegram_notifier.py load_dotenv calls, unverified) -- but no "
            "real deployment should omit it."
        ),
    )
    parser.add_argument(
        "--healthchecks-env-file", type=str, default=None,
        help=(
            "Optional .env path providing HEALTHCHECKS_LIVENESS_URL/"
            "HEALTHCHECKS_OPERATIONAL_URL, loaded with the SAME fail-closed "
            "loading mechanics as --env-file (resolve(strict=True), "
            "load_dotenv() result verified) -- but unlike --env-file, a "
            "failure at that stage never blocks this run; it only disables "
            "Healthchecks integration for this invocation (see "
            "_load_healthchecks_config's own docstring). Genuinely optional "
            "-- omit entirely until a real Healthchecks.io account and its "
            "two check UUIDs exist; the production crontab (see "
            "control_arm_crontab_v3.draft) passes "
            "/root/.config/ai-stock-radar-control/healthchecks.env once "
            "they do."
        ),
    )
    arguments = parser.parse_args()

    if arguments.env_file:
        _load_env_file_fail_closed(arguments.env_file)

    liveness_url, operational_url = _load_healthchecks_config(arguments.healthchecks_env_file)
    if liveness_url is None and operational_url is None:
        _notify_healthchecks_not_configured_once()
    _ping_healthcheck(liveness_url, "start", "RUN_STARTED")

    crashed = False
    outcome: str | None = None
    try:
        outcome = _execute(arguments)
    except Exception:
        crashed = True
        raise
    finally:
        # TRUE try/finally coverage of the entire run -- fires whether
        # _execute() returned normally, returned early (STOP/FREEZE/
        # lock-conflict/market-closed), or raised. See module docstring's
        # "HEALTHCHECKS.IO DEAD-MAN'S-SWITCH" section for the full
        # liveness-vs-operational mapping this implements.
        if crashed:
            _ping_healthcheck(liveness_url, "fail", "UNCAUGHT_EXCEPTION")
            # Operational is deliberately NOT pinged on a genuine crash --
            # the literal outcome table this task specified only requires
            # a liveness fail here; the operational check's own
            # missing-ping timeout is what surfaces this case on that
            # side, rather than this code guessing at a detail string for
            # a failure it may not have enough context to describe.
        else:
            _ping_healthcheck(liveness_url, "success", "RUN_COMPLETED")
            operational_event = "success" if outcome in _OPERATIONAL_SUCCESS_OUTCOMES else "fail"
            _ping_healthcheck(operational_url, operational_event, outcome or "UNKNOWN_OUTCOME")


if __name__ == "__main__":
    main()
