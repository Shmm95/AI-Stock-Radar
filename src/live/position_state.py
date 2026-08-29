"""JSON persistence for the live daily runner's portfolio state.

Dry-run research tooling only — no broker order is placed here.

Cash is intentionally NOT persisted. Every run reads the real Alpaca
paper account's cash balance live (see `src/live/account_state.py`),
so our own bookkeeping can never silently drift from the broker's
authoritative balance.

Why this persists more than "just positions": the batch engine's
`_execute_pending_buys_at_open` / `_execute_pending_exits_at_open`
only execute a queued signal once `submitted_portfolio_bar_index <
portfolio_bar_index` — i.e. a signal queued on day N is only allowed
to fill on day N+1 or later, never the same day it was generated.
A daily runner that persisted only filled positions would silently
drop every signal generated "today" (it would be queued in memory and
then discarded when the process exits), which is not a faithful
reuse of the engine's own next-available-Open execution rule. This
module therefore also persists `pending_buys`, `pending_exits`, and
the running `portfolio_bar_index` counter across runs.

Private-API coupling (documented per CLAUDE.md's "protected areas"
policy — never modified, only imported and reconstructed as-is):
- `src.backtest.portfolio_backtest_engine._MutablePosition`
- `src.backtest.portfolio_backtest_engine._PendingOrder`
Both are reconstructed directly from the persisted JSON instead of via
an independently-defined, drift-prone equivalent, so whatever the six
per-bar step functions actually read/write is exactly what gets saved
and loaded.

Real-order tracking (schema v2, equity-only): `equity_stop_orders`
maps a ticker to the Alpaca order id of its currently resting
protective stop, so it can be canceled the moment a signal-based exit
closes that position. `submitted_actions` is the idempotency ledger,
keyed per checkpoint by the real calendar date the action belongs to
(not `portfolio_bar_index` — see the high-water-mark note below for
why), checked before any real order submission so re-running the
script for the same day never re-submits an order that was already
sent. Crypto tickers never appear in either dict.

High-water-mark rollback guard (added after a real incident: an rsync
deploy overwrote the server's `data/live/` with a stale local copy,
making `portfolio_bar_index` jump backward). `load_position_state`
refuses to proceed if the loaded `portfolio_bar_index` is behind the
persisted high-water mark, raising `RollbackDetectedError` rather than
silently continuing with stale state. This is a real, but deliberately
partial, defense -- not the sole guarantee against a stale-ledger
collision. See `scripts/run_daily_decision.py`'s module docstring for
the complementary, arguably more important fix: every real order
submission now queries the broker by a deterministic `client_order_id`
BEFORE submitting, so even a lost/corrupted/never-written local ledger
entry (this guard's own blind spot, e.g. a crash between a real fill
and the next save) cannot cause a silent duplicate -- the broker, not
this file or `position_state.json`, is the actual authority on "did
this already happen." An independent review (2026-08-14) correctly
flagged that framing the date-keyed ledger alone as "structurally
impossible to collide" overclaimed what one key format change can
guarantee; this module's guard and that broker-query check are two
independent, partial layers, not a single complete proof.

Deliberately stored OUTSIDE `data/live/` (and outside the repo
entirely), at `HIGH_WATER_MARK_PATH`: the whole point is to survive
the exact failure mode that caused the incident. If the mark lived
inside `data/live/` -- even in a sibling file next to
`position_state.json`, not inside it -- the same rsync that clobbers
the live directory would clobber the mark too, and the guard would see
two mutually "consistent" stale files and never notice anything was
wrong. A location the deploy mechanism never touches is the only way
this guard can do its job.

Location resolution order: `$AI_STOCK_RADAR_GUARD_DIR` if set,
otherwise `~/.ai_stock_radar_guard`. Plain `Path.home()` alone was
reconsidered after the same 2026-08-14 review noted that a cron
environment does not always populate `$HOME` the same way an
interactive login shell does -- if it differed between an interactive
test run and the actual cron invocation, the guard file a human
verified and the one cron actually reads/writes could silently be two
different paths on disk. Setting `AI_STOCK_RADAR_GUARD_DIR` explicitly
in the crontab/service environment (e.g. `/root/.ai_stock_radar_state`
to match this project's existing `/root/AI-Stock-Radar` deployment
convention) removes that ambiguity; the home-directory fallback stays
for interactive/dev use, where it has already been exercised. This
does mean a fresh server needs no special setup (the file is created
on first successful save, see `save_position_state`) but also that
resetting it after a *deliberate* state reset requires deleting it by
hand at that out-of-repo path -- documented in
`docs/EMERGENCY_STOP_RUNBOOK.md`.

Write discipline: the high-water-mark file is written under a dedicated
lock file (same `O_CREAT|O_EXCL` primitive as `crypto_stop_monitor.py`'s)
to a temp file in the same directory, `fsync`'d, then `os.replace`'d
into place, with the containing directory itself also `fsync`'d -- so a
crash mid-write cannot leave a half-written or missing file, and the
rename is durable against a crash immediately after
(`_atomic_write_text`). A read that finds the file present but corrupt
(bad JSON, missing key) raises `GuardFileCorruptedError` rather than
silently treating it as absent (which would reset the high-water mark
to a permissive 0) -- fail-closed, per the same review.

`save_position_state` writes `position_state.json` with that same
`_atomic_write_text` helper (added 2026-08-15, after a real-crash
resilience test found the state file had no such protection -- a plain
`Path.write_text`, unlike the guard file's already-atomic write). A
crash mid-write of the real ledger is at least as consequential as one
mid-write of the high-water mark, so it gets the identical guarantee.
Not lock-protected the way the guard file is -- see `save_position_state`'s
own docstring for why that's a deliberately separate, not-yet-addressed
question from the crash/corruption one this fixes.

Guard-lock self-healing (added the same day, after a real-SIGKILL test
found a process killed while holding `_guard_write_lock` left it stale
forever, permanently blocking every later write with
`GuardLockTimeoutError` until a human deleted the lock file by hand):
the lock file's content now records the acquiring process's PID and
acquisition time, and a later contender breaks (unlinks) a lock it
finds either older than `_GUARD_LOCK_MAX_AGE_SECONDS` or held by a PID
that is no longer alive. See `_guard_write_lock`'s own docstring for
the race-condition reasoning.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from src.backtest.portfolio_backtest_engine import _MutablePosition, _PendingOrder
from src.backtest.portfolio_backtest_models import PortfolioSignal

# See the module docstring's "High-water-mark rollback guard" section for
# why this deliberately lives outside data/live/ and outside the repo, and
# for the $AI_STOCK_RADAR_GUARD_DIR resolution order.
_GUARD_DIRECTORY = Path(
    os.environ.get("AI_STOCK_RADAR_GUARD_DIR", str(Path.home() / ".ai_stock_radar_guard"))
)
HIGH_WATER_MARK_PATH = _GUARD_DIRECTORY / "high_water_mark.json"
_GUARD_LOCK_PATH = _GUARD_DIRECTORY / "high_water_mark.lock"
_GUARD_LOCK_ATTEMPTS = 5
_GUARD_LOCK_RETRY_SECONDS = 0.5


class RollbackDetectedError(RuntimeError):
    """Raised when a loaded portfolio_bar_index is behind the persisted
    high-water mark -- refuses to proceed rather than silently trading
    against stale/rolled-back state. See HIGH_WATER_MARK_PATH."""


class GuardFileCorruptedError(RuntimeError):
    """Raised when the high-water-mark file exists but cannot be parsed
    (bad JSON, missing field) -- fail-closed: never silently treated as
    "no mark yet" (which would reset protection to a permissive 0)."""


class GuardLockTimeoutError(RuntimeError):
    """Raised when the guard file's write lock could not be acquired
    after all retries by a lock that is neither stale-by-age nor held by
    a dead process -- refuses to write rather than risk a concurrent,
    interleaved write from two live processes."""


# A lock older than this is judged abandoned rather than genuinely still
# in progress -- see _lock_is_stale. Deliberately far longer than any
# real save this module performs (a JSON write + fsync), same spirit as
# order_submission.py's own generously-sized poll windows: the cost of
# waiting a little longer on a real, live holder is trivial next to the
# cost of a false "abandoned" verdict that lets two processes believe
# they both hold the lock.
_GUARD_LOCK_MAX_AGE_SECONDS = 300.0


def _lock_is_stale(lock_path: Path) -> bool:
    """A lock is judged stale (safe to break) if EITHER its recorded PID
    no longer refers to a live process, OR it was acquired more than
    _GUARD_LOCK_MAX_AGE_SECONDS ago -- whichever signal is available.
    An unparseable lock file (e.g. a leftover from before this
    PID+timestamp format existed) is also treated as stale rather than
    blocking forever on a file this code cannot even interpret.

    Neither check is airtight alone -- PID reuse could in principle
    defeat the liveness check, and a genuinely slow holder just under
    the age ceiling would still be waited out -- but requiring both a
    live PID AND a recent timestamp to call a lock "not stale" is a
    reasonable, deliberately generous bar: a real holder's process was
    just observed to exist AND started recently, which a five-minute-old
    abandoned lock file from a killed process cannot satisfy.
    """
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        acquired_at = float(payload["acquired_at"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return True

    if time.time() - acquired_at > _GUARD_LOCK_MAX_AGE_SECONDS:
        return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True  # no such process -- the holder is dead
    except PermissionError:
        pass  # process exists, just not signalable by us -- still alive
    return False


def _try_create_lock_file(lock_path: Path) -> int | None:
    try:
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


@contextmanager
def _guard_write_lock(lock_path: Path = _GUARD_LOCK_PATH) -> Iterator[None]:
    """Mutual-exclusion lock for guard-file writes, same `O_CREAT|O_EXCL`
    primitive as `crypto_stop_monitor.py`'s own lock.

    Self-healing (added after a real incident: a process SIGKILLed while
    holding this lock left it stale forever -- every subsequent write
    failed with GuardLockTimeoutError until a human deleted the lock
    file by hand). The lock file's own content now records the
    acquiring process's PID and acquisition time; on contention, a
    stale lock (see `_lock_is_stale`) is unlinked and acquisition is
    retried immediately, without waiting out the normal contention
    sleep -- a genuinely abandoned lock should never cost a real run
    more than one extra `open()` call.

    Race between two processes that both see the same stale lock: both
    may unlink it and both then attempt the `O_CREAT|O_EXCL` open that
    follows -- the OS still arbitrates that open atomically, so at most
    one of them can succeed. The other sees `FileExistsError` again, now
    against whichever process won, re-evaluates staleness against that
    FRESH lock (a live PID acquired moments ago), correctly finds it not
    stale, and falls back to the normal contention wait/retry below. No
    window exists where two processes both believe they hold the lock;
    the exclusivity guarantee is still the kernel-arbitrated `O_EXCL`
    open, never the staleness heuristic by itself -- the heuristic only
    decides whether it is worth attempting that open again sooner.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    for attempt in range(_GUARD_LOCK_ATTEMPTS):
        descriptor = _try_create_lock_file(lock_path)
        if descriptor is not None:
            break
        if _lock_is_stale(lock_path):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass  # another process already cleared it -- fine, retry below
            descriptor = _try_create_lock_file(lock_path)
            if descriptor is not None:
                break
        if attempt < _GUARD_LOCK_ATTEMPTS - 1:
            time.sleep(_GUARD_LOCK_RETRY_SECONDS)
    if descriptor is None:
        raise GuardLockTimeoutError(
            f"{lock_path} still held after {_GUARD_LOCK_ATTEMPTS} attempts by a lock "
            f"that is neither older than {_GUARD_LOCK_MAX_AGE_SECONDS:g}s nor held by "
            f"a dead process -- refusing to write the high-water mark concurrently."
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _read_high_water_mark(path: Path = HIGH_WATER_MARK_PATH) -> int:
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload["portfolio_bar_index"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
        raise GuardFileCorruptedError(
            f"{path} exists but could not be parsed ({type(error).__name__}: {error}) -- "
            f"refusing to treat this as 'no mark yet' (that would silently reset "
            f"rollback protection to 0). Investigate and repair or deliberately "
            f"delete this file by hand before proceeding."
        ) from error


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` durably: a temp file in the same
    directory, `fsync`'d, then `os.replace`'d into place, with the
    containing directory itself also `fsync`'d -- so a crash mid-write
    cannot leave a half-written or missing file, and the rename is
    durable against a crash immediately after. Shared by
    `_write_high_water_mark` and `save_position_state`, which both need
    exactly this guarantee for their own file."""
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


# Owner rw, group r, other none. _atomic_write_text's own tempfile.mkstemp()
# always produces mode 0600 (owner-only) -- correct/tight for every OTHER
# caller (order_intent.py's journal, issuer_identity_preflight.py,
# session_replay_journal.py/pass_gate.py) but wrong for the two files the
# unprivileged dashboard snapshot producer must read: position_state.json
# and this guard file (snapshot_builder.build_snapshot reads both via
# load_position_state(state_path, guard_path)). Set explicitly at each of
# those two call sites only -- never inside _atomic_write_text itself, so
# every other caller keeps the tight 0600 default. Mirrors the identical
# fix already applied to the dashboard's own snapshot output file, see
# scripts/generate_dashboard_snapshot.py's SNAPSHOT_FILE_MODE.
DASHBOARD_READABLE_FILE_MODE = 0o640


def _write_high_water_mark(value: int, path: Path = HIGH_WATER_MARK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f"{path.name}.lock"
    with _guard_write_lock(lock_path):
        _atomic_write_text(path, json.dumps({"portfolio_bar_index": int(value)}, indent=2) + "\n")
        path.chmod(DASHBOARD_READABLE_FILE_MODE)

SCHEMA_VERSION = 2
DEFAULT_STATE_PATH = Path("data/live/position_state.json")


class LiveRunnerState:
    """In-memory view of what position_state.json persists.

    `last_processed_equity_date`/`last_processed_crypto_date` (added
    2026-08-16, workspace-b research): generic date-string fields
    (ISO "YYYY-MM-DD"), storage-and-schema-only here -- this module
    does not itself decide or interpret what gets written into them.
    The one real caller, `scripts/run_control_arm_decision.py`'s
    same-day double-run guard, stamps them with the WALL-CLOCK UTC date
    its own run completed on (not the processed bar's own date --
    tried first, and rejected after a real test on an actual Sunday
    showed a bar-date stamp fails to match wall-clock "today" on any
    non-trading day, silently defeating the guard on exactly the
    holiday-adjacent cases it most needs to catch; see that wrapper's
    own `_stamp_last_processed_dates` docstring for the full real-test
    account). Both default to `None` and are optional everywhere
    they're read/written specifically so that (a) older
    `position_state.json` files with no such keys still load fine, and
    (b) `run_daily_decision.py` -- which constructs `LiveRunnerState`
    without ever passing these two kwargs, and is not modified by this
    change -- keeps working exactly as before; it simply never
    populates them, and whatever last set them is preserved only by a
    caller that explicitly re-reads and re-saves.

    `last_processed_equity_session_date`/`last_processed_equity_bar_timestamp`
    (added 2026-08-16, same session): a SEPARATE, market-calendar-aware
    pair, deliberately distinct from `last_processed_equity_date` above
    (which this docstring's own instruction says not to touch/repurpose
    -- it stays the plain wall-clock same-day guard it already is).
    These two instead carry the REAL Alpaca trading-session date and the
    literal bar timestamp `run_control_arm_decision.py`'s own market-
    holiday/session-date verification confirmed for the run that last
    set them -- see that wrapper's own module docstring for the full
    calendar-based verification design (real `TradingClient.get_calendar()`
    call, expected-vs-actual-vs-last-processed comparison, fail-closed
    on any anomaly). In today's daily-bar-only design the two hold the
    same value in practice (a daily bar has no finer time component
    meaningfully different from its own date) -- kept as two fields
    anyway because they answer two different questions (a
    calendar-verified session date vs. the literal timestamp on the
    fetched bar) that could diverge under a future, finer-grained bar
    design. Same optional/backward-compatible discipline as the pair
    above.

    `pending_signal_metadata` (added 2026-08-16, workspace-b research,
    Phase 2b): a `f"{ticker}|{kind}"`-keyed dict (`kind` is `"BUY"` or
    `"EXIT"`) of session-based TTL bookkeeping for entries in
    `pending_buys`/`pending_exits` -- storage-and-schema-only here, same
    as the two pairs above; this module does not itself decide or
    interpret the records. The one real caller,
    `src/live/pending_signal_ttl.py` (invoked from
    `scripts/run_control_arm_decision.py`), stamps and reads
    `source_session_date`/`target_execution_session_date`/`created_at_utc`/
    `signal_id`/`status`/`expire_reason` -- see that module's own
    docstring for the full one-shot-TTL design and why it deliberately
    never reads a `PortfolioSignal`'s own `.timestamp` (a real,
    unfixed-because-unfixable-here crypto-date bug in
    `run_daily_decision.py`). Defaults to `{}`, same backward-compatible
    discipline: an older `position_state.json` with no such key still
    loads fine, and `run_daily_decision.py` -- which never passes this
    kwarg and is not modified by this change -- simply never populates
    it; `run_daily_decision()`'s own internal save wipes it to `{}` on
    every run, which is why its one real caller re-applies it in a
    second, deliberate save AFTER that call returns (same discipline as
    `last_processed_equity_date` above)."""

    __slots__ = (
        "portfolio_bar_index",
        "positions",
        "pending_buys",
        "pending_exits",
        "equity_stop_orders",
        "submitted_actions",
        "last_processed_equity_date",
        "last_processed_crypto_date",
        "last_processed_equity_session_date",
        "last_processed_equity_bar_timestamp",
        "pending_signal_metadata",
    )

    def __init__(
        self,
        *,
        portfolio_bar_index: int = 0,
        positions: dict[str, _MutablePosition] | None = None,
        pending_buys: dict[str, _PendingOrder] | None = None,
        pending_exits: dict[str, _PendingOrder] | None = None,
        equity_stop_orders: dict[str, str] | None = None,
        submitted_actions: dict[str, dict[str, Any]] | None = None,
        last_processed_equity_date: str | None = None,
        last_processed_crypto_date: str | None = None,
        last_processed_equity_session_date: str | None = None,
        last_processed_equity_bar_timestamp: str | None = None,
        pending_signal_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.portfolio_bar_index = portfolio_bar_index
        self.positions = positions or {}
        self.pending_buys = pending_buys or {}
        self.pending_exits = pending_exits or {}
        self.equity_stop_orders = equity_stop_orders or {}
        self.submitted_actions = submitted_actions or {}
        self.last_processed_equity_date = last_processed_equity_date
        self.last_processed_crypto_date = last_processed_crypto_date
        self.last_processed_equity_session_date = last_processed_equity_session_date
        self.last_processed_equity_bar_timestamp = last_processed_equity_bar_timestamp
        self.pending_signal_metadata = pending_signal_metadata or {}


def _signal_to_dict(signal: PortfolioSignal) -> dict[str, Any]:
    return signal.to_dict()


def _signal_from_dict(payload: dict[str, Any]) -> PortfolioSignal:
    return PortfolioSignal(**payload)


def _pending_to_dict(pending: _PendingOrder) -> dict[str, Any]:
    return {
        "signal": _signal_to_dict(pending.signal),
        "submitted_portfolio_bar_index": pending.submitted_portfolio_bar_index,
    }


def _pending_from_dict(payload: dict[str, Any]) -> _PendingOrder:
    return _PendingOrder(
        signal=_signal_from_dict(payload["signal"]),
        submitted_portfolio_bar_index=payload["submitted_portfolio_bar_index"],
    )


def _position_to_dict(position: _MutablePosition) -> dict[str, Any]:
    return asdict(position)


def _position_from_dict(payload: dict[str, Any]) -> _MutablePosition:
    return _MutablePosition(**payload)


def _check_high_water_mark(loaded_bar_index: int, guard_path: Path) -> None:
    # Deliberately `<`, not `<=`, considered and rejected on 2026-08-14
    # review: `save_position_state` advances the mark to match
    # `portfolio_bar_index` on every successful save, so the very next
    # normal load always finds loaded_bar_index == high_water_mark --
    # `<=` would make this guard fire on every single legitimate run,
    # not just a real rollback. The scenario `<=` would additionally
    # catch (bar_index unchanged, but `submitted_actions` swapped for a
    # stale version) cannot happen via this module's own write path:
    # `save_position_state` writes bar_index and submitted_actions
    # together, from the same in-memory state, in one atomic file write
    # -- they cannot independently drift within a file this module
    # produced. A ledger that is stale despite a matching bar_index
    # would have to come from something else entirely overwriting or
    # hand-editing the file, a threat model no integer comparison alone
    # defends against; the actual defense for that case is
    # `scripts/run_daily_decision.py`'s broker client_order_id query
    # before every submission (see this module's own docstring).
    high_water_mark = _read_high_water_mark(guard_path)
    if loaded_bar_index < high_water_mark:
        raise RollbackDetectedError(
            f"Rollback detected: loaded bar_index {loaded_bar_index} < "
            f"high-water-mark {high_water_mark} -- refusing to proceed. "
            f"This normally means data/live/ (or its absence) does not "
            f"reflect real progress -- e.g. an rsync deploy overwrote it "
            f"with a stale copy. Investigate before touching "
            f"{guard_path}."
        )


def load_position_state(
    path: Path = DEFAULT_STATE_PATH, *, guard_path: Path = HIGH_WATER_MARK_PATH
) -> LiveRunnerState:
    """Load persisted state, or an empty state if no file exists yet.

    Checks the high-water-mark guard even in the "no file yet" case: a
    missing state file while the mark is already ahead of zero is its
    own kind of rollback/data-loss, not a legitimate fresh start.

    `guard_path` defaults to the real, out-of-repo location (see the
    module docstring) -- only override it in tests, so a synthetic
    `tmp_path`-based state doesn't read or write the real machine-wide
    guard file (confirmed the hard way: a tmp_path-only test tripped a
    stale mark left behind by an earlier, unrelated test run).
    """
    path = Path(path)
    if not path.is_file():
        _check_high_water_mark(0, guard_path)
        return LiveRunnerState()

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported position_state schema_version: {payload.get('schema_version')}"
        )

    _check_high_water_mark(int(payload["portfolio_bar_index"]), guard_path)

    return LiveRunnerState(
        portfolio_bar_index=int(payload["portfolio_bar_index"]),
        positions={
            ticker: _position_from_dict(fields)
            for ticker, fields in payload.get("positions", {}).items()
        },
        pending_buys={
            ticker: _pending_from_dict(fields)
            for ticker, fields in payload.get("pending_buys", {}).items()
        },
        pending_exits={
            ticker: _pending_from_dict(fields)
            for ticker, fields in payload.get("pending_exits", {}).items()
        },
        equity_stop_orders=dict(payload.get("equity_stop_orders", {})),
        submitted_actions={
            key: dict(record)
            for key, record in payload.get("submitted_actions", {}).items()
        },
        last_processed_equity_date=payload.get("last_processed_equity_date"),
        last_processed_crypto_date=payload.get("last_processed_crypto_date"),
        last_processed_equity_session_date=payload.get("last_processed_equity_session_date"),
        last_processed_equity_bar_timestamp=payload.get("last_processed_equity_bar_timestamp"),
        pending_signal_metadata={
            key: dict(record)
            for key, record in payload.get("pending_signal_metadata", {}).items()
        },
    )


def save_position_state(
    state: LiveRunnerState,
    path: Path = DEFAULT_STATE_PATH,
    *,
    guard_path: Path = HIGH_WATER_MARK_PATH,
) -> None:
    """`guard_path` override is for tests only -- see `load_position_state`.

    Written with the same atomic discipline as the high-water-mark file
    (`_atomic_write_text`): a crash mid-write cannot leave a corrupted or
    half-written `position_state.json` behind, only ever the previous
    good file or the new one. This does not add a write-lock the way
    the guard file has one -- `run_daily_decision.py` and
    `run_crypto_stop_monitor.py` are the only two writers, on
    independent schedules, and a genuinely overlapping write from both
    at once (a separate, pre-existing concurrency question, not a
    crash/corruption one) is not what today's fix addresses. A read
    that finds the file present but corrupt still fails loudly with the
    underlying `json.JSONDecodeError` uncaught -- deliberately not
    caught here either, for the same fail-closed reason
    `GuardFileCorruptedError` exists: silently treating unreadable state
    as empty would be far more dangerous than a loud crash + Telegram
    alert. Atomicity narrows how often that can happen; it does not
    replace fail-closed as the backstop for whatever it doesn't cover
    (e.g. external corruption of an already-written file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "portfolio_bar_index": state.portfolio_bar_index,
        "positions": {
            ticker: _position_to_dict(position)
            for ticker, position in state.positions.items()
        },
        "pending_buys": {
            ticker: _pending_to_dict(pending)
            for ticker, pending in state.pending_buys.items()
        },
        "pending_exits": {
            ticker: _pending_to_dict(pending)
            for ticker, pending in state.pending_exits.items()
        },
        "equity_stop_orders": dict(state.equity_stop_orders),
        "submitted_actions": {
            key: dict(record) for key, record in state.submitted_actions.items()
        },
        "last_processed_equity_date": state.last_processed_equity_date,
        "last_processed_crypto_date": state.last_processed_crypto_date,
        "last_processed_equity_session_date": state.last_processed_equity_session_date,
        "last_processed_equity_bar_timestamp": state.last_processed_equity_bar_timestamp,
        "pending_signal_metadata": {
            key: dict(record) for key, record in state.pending_signal_metadata.items()
        },
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(DASHBOARD_READABLE_FILE_MODE)

    # Advance the rollback guard on every successful save -- never move it
    # backward, even if this run's own bar_index is (legitimately) unchanged
    # (e.g. a crypto-monitor-only save that never touches portfolio_bar_index).
    if state.portfolio_bar_index > _read_high_water_mark(guard_path):
        _write_high_water_mark(state.portfolio_bar_index, guard_path)
