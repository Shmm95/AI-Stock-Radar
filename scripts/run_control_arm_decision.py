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
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.requests import GetCalendarRequest
from dotenv import load_dotenv

import scripts.run_daily_decision as rdd
import src.live.order_intent as order_intent
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


def _expected_equity_session_date() -> str | None:
    """Real Alpaca `TradingClient.get_calendar()` call, made BEFORE any
    market-data fetch. See module docstring's "MARKET HOLIDAY /
    SESSION-DATE VERIFICATION" section for the full design. Returns the
    calendar-confirmed session date (ISO string) for today if today's
    session exists AND has already closed (+ a 15-minute settle
    buffer), or `None` if this run should clean no-op (no session
    today, or today's session has not yet closed)."""
    client = rdd.order_submission.get_trading_client()
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


def _run_intent_protocol(decision: dict, pre_state_hash: str | None) -> list[order_intent.OrderIntent]:
    """PHASE 1 SCAFFOLD -- see order_intent.py's own module docstring
    for the full "why." Creates one durable intent per real signal this
    run's decision produced (today's queued-for-next-run buys and
    today's executed entries), and walks each through PREPARED ->
    SUBMITTING -> BROKER_ACKNOWLEDGED.

    NO REAL BROKER CALL IS MADE ANYWHERE IN THIS FUNCTION.
    BROKER_ACKNOWLEDGED here means "the local decision is confirmed as
    this run's final output" -- a scaffold for where a real Alpaca
    acknowledgment will plug in once real order submission AND Phase 2
    broker-reconciliation both exist for this control arm (neither
    exists today; this control arm has never enabled
    --enable-equity-orders/--enable-crypto-orders in production).

    Returns the created intents so the caller can transition them to
    COMMITTED only after `position_state.json` has actually been
    durably re-saved with today's stamp (see main()'s own ordering).
    """
    equity_date = decision.get("as_of_bar_timestamp_equity") or decision.get("as_of_bar_timestamp")
    signals: list[tuple[str, dict, str]] = []
    for entry in decision.get("queued_for_next_run", {}).get("buys", []):
        signals.append(("QUEUED_ENTRY_SIGNAL", entry, "reference_close"))
    for entry in decision.get("executed_today", {}).get("entries", []):
        signals.append(("ENTRY_MARKET_BUY", entry, "fill_price"))

    intents: list[order_intent.OrderIntent] = []
    for action_kind, entry, price_field in signals:
        ticker = entry["ticker"]
        client_order_id = rdd._deterministic_client_order_id(ticker, action_kind, str(equity_date))
        intent = order_intent.create_intent(
            client_order_id=client_order_id,
            account_identity=ACCOUNT_IDENTITY,
            ticker=ticker,
            side="BUY",
            order_type="market",
            action_kind=action_kind,
            source_signal_timestamp=str(equity_date),
            quantity=entry.get("quantity"),
            notional=entry.get(price_field),
            stop_price=entry.get("stop_loss_price"),
            pre_state_hash=pre_state_hash,
        )
        order_intent.transition_intent(intent, order_intent.SUBMITTING, increment_attempt=True)
        order_intent.transition_intent(
            intent, order_intent.BROKER_ACKNOWLEDGED,
            broker_status="local_decision_confirmed -- no real broker call made (Phase 1 scaffold)",
        )
        intents.append(intent)
        print(f"[CONTROL] Intent {intent.intent_id} ({ticker} {action_kind}): PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED")
    return intents


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
    arguments = parser.parse_args()

    if arguments.env_file:
        _load_env_file_fail_closed(arguments.env_file)

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
        _notify_control(text)
        return

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
                _notify_control(text)
                return

            freeze = rdd.FREEZE_FLAG_PATH.exists()
            if freeze:
                text = (
                    f"AI-Stock-Radar CONTROL ARM daily decision runner -- FREEZE flag "
                    f"detected ({rdd.FREEZE_FLAG_PATH.resolve()}); this run will still "
                    f"fetch data and monitor/exit existing positions as usual, but will "
                    f"NOT open any new positions. Remove the file to resume normal entries."
                )
                print(text)
                _notify_control(text)

            try:
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

                # [PHASE 2 INTEGRATION POINT: broker-reconciliation logic
                # goes here -- inspecting order_intent.list_intents() for
                # any PREPARED/SUBMITTING/UNCERTAIN intent left over from
                # an interrupted prior run, and resolving it against the
                # broker's own order history BEFORE this run computes a
                # new decision. Not implemented this task -- explicitly
                # out of scope.]

                # Real Alpaca calendar check, BEFORE any market-data
                # fetch. See module docstring's "MARKET HOLIDAY /
                # SESSION-DATE VERIFICATION" section.
                expected_session_date = _expected_equity_session_date()
                last_processed_session_date = _read_last_processed_equity_session_date(arguments.state_path)
                if expected_session_date is None:
                    text = (
                        "AI-Stock-Radar CONTROL ARM daily decision runner -- no "
                        "settled equity session today (no calendar entry, or "
                        "today's session has not yet closed + settled). Clean "
                        "no-op, zero progress, zero Alpaca market-data calls."
                    )
                    print(text)
                    _notify_control(text)
                    return

                pre_state_hash = _hash_state_file(arguments.state_path)

                result = rdd.run_daily_decision(
                    state_path=arguments.state_path,
                    decision_log_directory=arguments.decision_log_directory,
                    enable_equity_orders=arguments.enable_equity_orders,
                    enable_crypto_orders=arguments.enable_crypto_orders,
                    freeze=freeze,
                )
            except Exception as error:
                notified = _notify_control(rdd._build_failure_notification_text(error))
                if not notified:
                    print(
                        "WARNING: [CONTROL] Telegram failure notification could not be "
                        "confirmed sent (check TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID and "
                        "network reachability). Original error follows: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
                raise

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

            # PREPARED -> SUBMITTING -> BROKER_ACKNOWLEDGED protocol --
            # see _run_intent_protocol's own docstring for exactly what
            # this does and does not represent (no real broker call yet).
            intents = _run_intent_protocol(decision, pre_state_hash)

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
            for intent in intents:
                order_intent.transition_intent(intent, order_intent.COMMITTED)
                print(f"[CONTROL] Intent {intent.intent_id} ({intent.ticker}): BROKER_ACKNOWLEDGED -> COMMITTED")

            _notify_control(rdd._build_daily_notification_text(decision))
            # `with single_instance_lock(...)` releases the lock here,
            # on normal exit from this block.
    except SingleInstanceLockError as error:
        text = (
            f"AI-Stock-Radar CONTROL ARM daily decision runner -- could not "
            f"acquire the single-instance lock; another run is already in "
            f"progress. Refusing to proceed -- zero Alpaca API calls were "
            f"made. {error}"
        )
        print(text)
        _notify_control(text)
        return
    # Any OTHER exception raised inside the `with` block (e.g. from
    # run_daily_decision() failing, or an audit-assertion RuntimeError)
    # is deliberately NOT caught here -- it propagates normally, the
    # `with` block's __exit__ still releases the lock on the way out
    # (standard context-manager semantics), and the inner try/except
    # above has already sent the [CONTROL] failure notification before
    # re-raising.


if __name__ == "__main__":
    main()
