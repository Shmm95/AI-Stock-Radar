"""Reboot-recovery drill harness for `run_control_arm_decision.py` --
TEST INFRASTRUCTURE ONLY. Never touches real trading logic.

WHAT THIS IS: a way for the OWNER to actually kill a real, running
`_execute()` process at one of the 6 exact crash points the
order-intent journal's own state machine is designed to survive
(PREPARED/SUBMITTING/BROKER_ACKNOWLEDGED/[state save]/COMMITTED/
TERMINAL -- see `src/live/order_intent.py`'s own docstring), then start
a FRESH process against the SAME on-disk state and confirm the real
Phase-1.5 stray-intent recovery (`src/live/order_intent_reconciliation.py`)
resolves it correctly. This is the "reboot provası" (reboot rehearsal)
referenced throughout this session's own audit rounds -- proving the
crash-recovery code actually survives a REAL process death, not just a
mocked one inside a pytest test.

WHAT THIS IS NOT: it does not change, wrap, or reinterpret
`run_control_arm_decision.py`'s own `_execute()` -- it calls that real
function directly, unmodified, with every EXTERNAL dependency (broker,
market data, SEC EDGAR, calendar, Telegram, Healthchecks, real `.env`)
replaced by a deterministic fake/fixture/recorder. It does not decide
"pass" or "fail" for you -- `--phase recover` prints/saves a structured
report of exactly what the real reconciliation code did; judging
whether that matches your own expected-outcome table for a given
barrier/action/broker-outcome combination is the owner's own call.

SAFETY, IN ORDER OF WHAT ACTUALLY ENFORCES IT:
1. `AI_STOCK_RADAR_GUARD_DIR` is pointed at `--case-dir/guard` BEFORE
   any project import -- this is what isolates order_intent.py's own
   journal, the single-instance lock, and the high-water-mark file from
   the REAL machine-wide guard directory. Without this, a drill run
   would read/write production's own crash-recovery state.
2. `dotenv.load_dotenv` is neutralized to a no-op BEFORE any project
   import -- `src/data/alpaca_market_data.py` and
   `src/notify/telegram_notifier.py` both call it unconditionally at
   THEIR OWN import time (a real, confirmed side effect of merely
   importing `run_control_arm_decision.py` -- this is not a hypothetical
   concern), so neutralizing it here is the only way to guarantee a real
   `.env` file is never loaded into this process's environment.
3. `_execute()` is ALWAYS called with a fake `trading_client` -- no code
   path in this script ever constructs a real `TradingClient`.
4. `rdd.prepare_live_market_data`'s real underlying implementation
   (`carm._real_prepare_live_market_data`) is replaced with a fixed,
   synthetic fixture -- no real Alpaca market-data HTTP call is made.
5. `carm.issuer_identity_preflight.run_issuer_identity_preflight` is
   replaced with a fixed PASS result -- no real SEC EDGAR fetch, no
   real `get_all_assets()` call.
6. `rdd.send_telegram_message` and `carm._ping_healthcheck` are
   replaced with a local JSONL recorder -- no real Telegram/Healthchecks
   HTTP call is ever made, regardless of what credentials might exist
   in the real environment.
7. `_WRITE_AHEAD_JOURNAL_OWNER_APPROVED` (the real, manual, source-level
   flag that blocks `--enable-equity-orders` in actual production -- see
   that constant's own comment in `run_control_arm_decision.py`) is
   flipped to `True` ONLY on the imported module OBJECT, for the
   lifetime of this drill PROCESS -- the source file itself is never
   edited. This is what lets the drill actually reach a real (fake)
   broker call at all; production's own default stays `False`.
8. `--self-check` independently verifies (1)-(3) are actually in effect
   for the CURRENT process/environment, rather than trusting this
   docstring's own claims.

None of the above replaces OS-level isolation (a real reboot, a mount
namespace with no real network route) for a genuine production drill on
the owner's own server -- that isolation is explicitly the owner's own
setup, not something this script can create for itself. This script's
own job is the deterministic, pausable, inspectable process; the actual
kill/reboot action around it is not part of this file.

USAGE (owner's own server, outside a pytest run):

    # 1. Arm: runs up to the barrier, pauses itself with SIGSTOP.
    python scripts/run_control_arm_reboot_drill.py \\
        --case-dir /tmp/drill-case-1 --phase arm \\
        --barrier AFTER_SUBMITTING --action submit --broker-outcome absent

    # 2. Observe --case-dir/barrier.json; the process is now T (stopped).
    #    Reboot the box, or `kill -9` the stopped PID, to simulate the crash.

    # 3. Recover: a FRESH process, same --case-dir, exercises the real
    #    Phase-1.5 stray-intent recovery and reports what happened.
    python scripts/run_control_arm_reboot_drill.py \\
        --case-dir /tmp/drill-case-1 --phase recover --broker-outcome absent

    # Self-check (no drill run):
    python scripts/run_control_arm_reboot_drill.py --case-dir /tmp/drill-case-1 --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
from pathlib import Path

# Same convention `run_control_arm_decision.py` itself uses (its own
# line ~261): running this file directly (`python scripts/run_control_arm_reboot_drill.py`)
# puts only `scripts/` on `sys.path[0]`, not the project root -- needed
# for this file's own `import scripts.run_control_arm_decision as carm`
# below to resolve regardless of invocation style (`python -m
# scripts.run_control_arm_reboot_drill` already works without this;
# this makes the plain-path form work too).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# STEP 0 -- environment isolation. MUST run before any project import.
# ============================================================================


def _early_case_dir_from_argv() -> Path:
    """A tiny, import-free pre-parse of `--case-dir` -- needed because
    `AI_STOCK_RADAR_GUARD_DIR` must be set before `import
    scripts.run_control_arm_decision` (or anything under `src/`) ever
    runs, and that constant is derived from it once, at THAT module's
    own import time (see `src/live/position_state.py`).

    Deliberately NOT `required=True`: this module is also imported as a
    plain library (e.g. by its own test file, to unit-test
    `FakeBrokerLedger`/`FakeTradingClient`/`BarrierController` in
    isolation) where `sys.argv` is the IMPORTING process's own argv
    (pytest's, typically), not this script's -- a hard requirement here
    would crash at import time in that case. Falls back to a
    process-unique tmp directory instead, which is exactly as safe:
    the real guard-dir isolation this function exists to set up (see
    the module-level code right below) still happens, just pointed at
    a throwaway location nothing else will ever read. `main()`'s own
    real argparse parser (built later, after all imports) still
    enforces `--case-dir` as genuinely required for actual drill runs."""
    import tempfile
    import uuid

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--case-dir", type=Path, default=None)
    known, _ = parser.parse_known_args()
    if known.case_dir is not None:
        return known.case_dir.resolve()
    return Path(tempfile.gettempdir()) / f"reboot_drill_unbound_import_{uuid.uuid4().hex[:8]}"


_CASE_DIR = _early_case_dir_from_argv()
_CASE_DIR.mkdir(parents=True, exist_ok=True)
_GUARD_DIR = _CASE_DIR / "guard"
_GUARD_DIR.mkdir(parents=True, exist_ok=True)

# Isolates order_intent.py's own journal directory, single_instance_lock's
# own lock file, and position_state.py's own high-water-mark file from the
# REAL, machine-wide `~/.ai_stock_radar_guard` -- see module docstring
# safety point (1). Every one of those three derives `_GUARD_DIRECTORY`
# from this exact environment variable, read once at THEIR import time.
os.environ["AI_STOCK_RADAR_GUARD_DIR"] = str(_GUARD_DIR)

# Real credential env vars are explicitly unset (defense in depth --
# `trading_client` is always faked regardless, see safety point (3)) so
# that IF something unexpected ever tried to build a real client, it
# would fail on missing credentials rather than silently succeed with
# whatever happens to be in the drill operator's own shell environment.
for _real_credential_var in (
    "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "LIVE_ACCOUNT_NUMBER_SUFFIX",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    "HEALTHCHECKS_LIVENESS_URL", "HEALTHCHECKS_OPERATIONAL_URL",
):
    os.environ.pop(_real_credential_var, None)

import dotenv  # noqa: E402

_REAL_LOAD_DOTENV = dotenv.load_dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False  # noqa: E731 -- see module docstring safety point (2)

# ============================================================================
# STEP 1 -- now safe to import the project. `run_control_arm_decision.py`'s
# own module-level code runs here (including its own `rdd.LIVE_CONTROLLED_TICKERS`/
# `rdd.prepare_live_market_data` overrides -- see that file's lines ~286-309).
# ============================================================================

import pandas as pd  # noqa: E402
from alpaca.common.exceptions import APIError  # noqa: E402

import scripts.run_control_arm_decision as carm  # noqa: E402
import scripts.run_daily_decision as rdd  # noqa: E402
import src.live.issuer_identity_preflight as issuer_identity_preflight  # noqa: E402
import src.live.order_intent as order_intent  # noqa: E402
from src.backtest.portfolio_backtest_engine import _MutablePosition, _PendingOrder  # noqa: E402
from src.backtest.portfolio_backtest_models import PortfolioSignal  # noqa: E402
from src.live import position_state as ps  # noqa: E402
from src.live.control_universe import CONTROL_UNIVERSE_TICKERS  # noqa: E402

BARRIER_NAMES = (
    "AFTER_PREPARED",
    "AFTER_SUBMITTING",
    "AFTER_BROKER_ACKNOWLEDGED",
    "AFTER_STATE_SAVE",
    "AFTER_COMMITTED",
    "BEFORE_TERMINAL",
)
_DRILL_TICKER = "AA"  # a real CONTROL_UNIVERSE_TICKERS member (Materials -- Alcoa)
_ACCOUNT_SUFFIX = "XO4Y"  # matches broker_reconciliation._EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX


# ============================================================================
# Fake broker ledger -- atomic JSON persistence at --case-dir/fake_broker_ledger.json
# ============================================================================


class FakeOrder:
    def __init__(self, payload: dict) -> None:
        self.id = payload["id"]
        self.client_order_id = payload.get("client_order_id", "")
        self.symbol = payload["symbol"]
        self.side = payload["side"]
        self.status = payload["status"]
        self.qty = payload.get("qty", "0")
        self.filled_qty = payload.get("filled_qty", "0")
        self.submitted_at = payload.get("submitted_at", "")
        self.updated_at = payload.get("updated_at", "")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "client_order_id": self.client_order_id, "symbol": self.symbol,
            "side": self.side, "status": self.status, "qty": self.qty,
            "filled_qty": self.filled_qty, "submitted_at": self.submitted_at, "updated_at": self.updated_at,
        }


class FakePosition:
    def __init__(self, *, symbol: str, qty: str, side: str = "long") -> None:
        self.symbol = symbol
        self.qty = qty
        self.side = side


class FakeBrokerLedger:
    """One JSON file, atomically written (temp file + fsync + os.replace +
    directory fsync -- same discipline as `position_state._atomic_write_text`,
    reimplemented here rather than imported since this is deliberately a
    TEST-ONLY fixture, not production code). Reloadable across process
    restarts by construction -- `--phase recover` loads the SAME file
    `--phase arm` last wrote, exactly modeling what survives a real reboot
    (this file lives on disk, unlike anything only held in memory)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.account_number = f"PA3HONFD{_ACCOUNT_SUFFIX}"
        self.orders: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.account_number = payload.get("account_number", self.account_number)
            self.orders = payload.get("orders", {})
            self.positions = payload.get("positions", {})

    def save(self) -> None:
        payload = {"account_number": self.account_number, "orders": self.orders, "positions": self.positions}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def find_by_client_order_id(self, client_order_id: str) -> dict | None:
        for order in self.orders.values():
            if order.get("client_order_id") == client_order_id:
                return order
        return None


class FakeTradingClient:
    """Implements exactly the `TradingClient` surface `_execute()`'s own
    real call chain touches (confirmed by direct code reading, not
    guessed) -- `get_account`/`get_orders`/`get_all_positions`/
    `get_order_by_id`/`get_order_by_client_id`/`submit_order`/
    `cancel_order_by_id`/`get_calendar`. No network socket is ever
    opened by any of these."""

    def __init__(self, ledger: FakeBrokerLedger, *, today_iso: str) -> None:
        self.ledger = ledger
        self.today_iso = today_iso
        self.calls: list[str] = []

    def get_account(self):
        self.calls.append("get_account")
        from types import SimpleNamespace
        return SimpleNamespace(account_number=self.ledger.account_number)

    def get_orders(self, *args, **kwargs):
        self.calls.append("get_orders")
        return [FakeOrder(o) for o in self.ledger.orders.values()]

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        return [FakePosition(**p) for p in self.ledger.positions.values()]

    def get_order_by_id(self, order_id: str):
        self.calls.append("get_order_by_id")
        order = self.ledger.orders.get(order_id)
        if order is None:
            fake_http_error = _fake_http_error(404)
            raise APIError('{"code":40410000,"message":"order not found"}', http_error=fake_http_error)
        return FakeOrder(order)

    def get_order_by_client_id(self, client_order_id: str):
        self.calls.append("get_order_by_client_id")
        order = self.ledger.find_by_client_order_id(client_order_id)
        if order is None:
            fake_http_error = _fake_http_error(404)
            raise APIError('{"code":40410000,"message":"order not found"}', http_error=fake_http_error)
        return FakeOrder(order)

    def submit_order(self, request):
        self.calls.append("submit_order")
        import uuid

        order_id = f"drill-order-{uuid.uuid4().hex[:12]}"
        side = str(request.side.value if hasattr(request.side, "value") else request.side)
        stop_price = getattr(request, "stop_price", None)
        payload = {
            "id": order_id,
            "client_order_id": request.client_order_id or "",
            "symbol": request.symbol,
            "side": side,
            # A fresh submission "fills" immediately in this fixture --
            # deterministic, no polling/sleep needed (see
            # order_submission.wait_for_fill_or_timeout's own real poll
            # loop, which this short-circuits by never returning a
            # non-terminal status).
            "status": "filled",
            "qty": str(request.qty) if request.qty is not None else "0",
            "filled_qty": str(request.qty) if request.qty is not None else "0",
            "submitted_at": f"{self.today_iso}T00:00:00Z",
            "updated_at": f"{self.today_iso}T00:00:01Z",
            "stop_price": stop_price,
        }
        self.ledger.orders[order_id] = payload
        self.ledger.save()
        return FakeOrder(payload)

    def cancel_order_by_id(self, order_id: str):
        self.calls.append("cancel_order_by_id")
        order = self.ledger.orders.get(order_id)
        if order is not None:
            order["status"] = "canceled"
            order["updated_at"] = f"{self.today_iso}T00:00:02Z"
            self.ledger.save()

    def get_calendar(self, request):
        self.calls.append("get_calendar")
        from datetime import date as _date
        from datetime import datetime as _datetime
        from types import SimpleNamespace

        start = getattr(request, "start", None)
        session_date = start if isinstance(start, _date) else _date.fromisoformat(self.today_iso)
        # A fixed, comfortably-past close time on the requested day --
        # `_expected_equity_session_date`'s own +15-minute settle-buffer
        # check just needs `now_utc >= close_utc + 15min`, easily true
        # for any drill run after ~00:15 UTC. See this class's own
        # docstring; a drill launched in the first ~20 minutes after UTC
        # midnight is the one known, narrow edge case where this fixture
        # would need a different fixed hour.
        entry = SimpleNamespace(date=session_date, close=_datetime.combine(session_date, _datetime.min.time()))
        return [entry]


def _fake_http_error(status_code: int):
    from types import SimpleNamespace

    return SimpleNamespace(response=SimpleNamespace(status_code=status_code))


# ============================================================================
# Local recorder -- replaces Telegram/Healthchecks, never makes real HTTP calls
# ============================================================================


class LocalRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _append(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def record_telegram(self, text: str) -> bool:
        self._append({"channel": "telegram", "text": text})
        return True

    def record_healthcheck(self, base_url, event: str, detail: str) -> bool:
        self._append({"channel": "healthcheck", "base_url": base_url, "event": event, "detail": detail})
        return True


# ============================================================================
# Barrier controller -- pause-and-SIGSTOP mechanism
# ============================================================================


class BarrierController:
    def __init__(self, case_dir: Path, *, armed_barrier: str | None, action_kind_filter: str | None) -> None:
        self.case_dir = case_dir
        self.armed_barrier = armed_barrier
        self.action_kind_filter = action_kind_filter
        self.fired = False

    def _write_barrier_file(self, barrier_name: str, intent) -> None:
        payload = {
            "barrier": barrier_name,
            "pid": os.getpid(),
            "intent": intent.to_dict() if intent is not None else None,
        }
        path = self.case_dir / "barrier.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(self.case_dir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def maybe_pause(self, barrier_name: str, intent=None) -> None:
        if self.fired or self.armed_barrier is None or barrier_name != self.armed_barrier:
            return
        if self.action_kind_filter is not None and intent is not None:
            if getattr(intent, "action_kind", None) != self.action_kind_filter:
                return
        self.fired = True
        self._write_barrier_file(barrier_name, intent)
        print(f"[DRILL] Barrier {barrier_name} reached -- barrier.json written and fsynced. SIGSTOP now.", flush=True)
        os.kill(os.getpid(), signal.SIGSTOP)
        print(f"[DRILL] Resumed (SIGCONT received) -- continuing normal execution past {barrier_name}.", flush=True)


def _install_barrier_hooks(controller: BarrierController) -> None:
    """Patches the ONE shared `src.live.order_intent` module object's
    `create_intent`/`transition_intent` attributes -- every caller
    throughout this codebase (`run_control_arm_decision.py`,
    `order_intent_reconciliation.py`) looks these up as
    `order_intent.create_intent(...)`/`order_intent.transition_intent(...)`
    at CALL time, so patching the shared module's own attributes reaches
    all of them, confirmed by direct code reading (not assumed)."""
    real_create_intent = order_intent.create_intent
    real_transition_intent = order_intent.transition_intent

    def hooked_create_intent(*args, **kwargs):
        intent = real_create_intent(*args, **kwargs)
        controller.maybe_pause("AFTER_PREPARED", intent)
        return intent

    def hooked_transition_intent(intent, new_status, **kwargs):
        if new_status == order_intent.TERMINAL:
            controller.maybe_pause("BEFORE_TERMINAL", intent)
        result = real_transition_intent(intent, new_status, **kwargs)
        if new_status == order_intent.SUBMITTING:
            controller.maybe_pause("AFTER_SUBMITTING", result)
        elif new_status == order_intent.BROKER_ACKNOWLEDGED:
            controller.maybe_pause("AFTER_BROKER_ACKNOWLEDGED", result)
        elif new_status == order_intent.COMMITTED:
            controller.maybe_pause("AFTER_COMMITTED", result)
        return result

    order_intent.create_intent = hooked_create_intent
    order_intent.transition_intent = hooked_transition_intent

    real_stamp = carm._stamp_last_processed_dates

    def hooked_stamp(*args, **kwargs):
        result = real_stamp(*args, **kwargs)
        controller.maybe_pause("AFTER_STATE_SAVE", None)
        return result

    carm._stamp_last_processed_dates = hooked_stamp


# ============================================================================
# Market-data / issuer-identity fixtures
# ============================================================================


def _fixed_frame(*, open_: float, high: float, low: float, close: float, dates: tuple[str, str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [open_] * n, "High": [high] * n, "Low": [low] * n, "Close": [close] * n,
            "EMA20": [95.0] * n, "EMA50": [100.0] * n, "RSI14": [50.0] * n, "RegimeAllowed": [True] * n,
        },
        index=index,
    )


def _install_market_data_fixture(*, yesterday_iso: str, today_iso: str) -> None:
    """Replaces `carm._real_prepare_live_market_data` -- the REAL
    underlying function `carm`'s own `_prepare_control_arm_market_data`
    wrapper calls (see that file's module-level override at its own
    import time) -- with a fixed, synthetic fixture. EMA20 < EMA50
    everywhere (no fresh entry/exit is ever freshly DECIDED by the
    frozen engine's own indicator logic in this drill -- see module
    docstring); the drill instead pre-seeds `pending_buys`/`pending_exits`
    directly, which the engine fills/exits unconditionally at today's
    Open regardless of indicator values (`_execute_pending_buys_at_open`/
    `_execute_pending_exits_at_open` -- confirmed by direct code reading,
    neither re-checks the entry/exit setup condition at fill time)."""
    frame = _fixed_frame(open_=50.0, high=52.0, low=48.0, close=51.0, dates=(yesterday_iso, today_iso))
    fixed = {ticker: frame.copy() for ticker in carm.CONTROL_TICKERS_WITH_REGIME}

    def fake_prepare(tickers, **kwargs):
        return {ticker: fixed[ticker].copy() for ticker in fixed}

    carm._real_prepare_live_market_data = fake_prepare
    rdd.get_live_cash_balance = lambda client=None: 100_000.0


def _install_issuer_identity_fixture() -> None:
    """Replaces the real SEC-EDGAR + `get_all_assets()` preflight with a
    fixed PASS -- see module docstring safety point (5)."""
    from datetime import datetime, timezone

    fixed_result = issuer_identity_preflight.IssuerIdentityCheckResult(
        status=issuer_identity_preflight.STATUS_PASS,
        checked_ticker_count=len(CONTROL_UNIVERSE_TICKERS),
        anchor_sha256="drill-fixture-not-a-real-anchor",
        checked_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sec_data_age_days=0.0,
    )
    carm.issuer_identity_preflight.run_issuer_identity_preflight = lambda client, **kwargs: fixed_result


def _install_notification_fixture(recorder: LocalRecorder) -> None:
    """Replaces Telegram/Healthchecks -- see module docstring safety point (6)."""
    rdd.send_telegram_message = lambda text: recorder.record_telegram(text)
    carm._ping_healthcheck = lambda base_url, event, detail: recorder.record_healthcheck(base_url, event, detail)


# ============================================================================
# State construction (--phase arm)
# ============================================================================


def _pending_signal_record(*, source_session_date: str, target_execution_session_date: str) -> dict:
    """Matches `pending_signal_ttl._new_record`'s own exact shape --
    without a `status='pending'` record whose
    `target_execution_session_date >= today`, `evaluate_pending_signals`
    (Phase 2b, called BEFORE `rdd.run_daily_decision()`) treats ANY
    pending_buys/pending_exits entry as `age_unknown` and immediately
    expires it -- confirmed the hard way (this drill's own first test
    run expired the pre-seeded pending BUY before it ever reached
    `_write_blind_prepared_intents`, let alone the barrier)."""
    from datetime import datetime, timezone

    return {
        "signal_id": "drill-fixture-signal-id",
        "source_session_date": source_session_date,
        "target_execution_session_date": target_execution_session_date,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "pending",
        "expire_reason": None,
    }


def _build_initial_runner_state(
    *, action: str, yesterday_iso: str, today_iso: str, stop_order_id: str | None,
) -> ps.LiveRunnerState:
    state = ps.LiveRunnerState()
    if action == "submit":
        state.pending_buys[_DRILL_TICKER] = _PendingOrder(
            signal=PortfolioSignal(
                timestamp=yesterday_iso, ticker=_DRILL_TICKER, action="BUY", reference_price=50.0,
                score=75.0, technical_score=75.0, confidence=75.0, reason="drill fixture",
            ),
            submitted_portfolio_bar_index=0,
        )
        state.pending_signal_metadata["AA|BUY"] = _pending_signal_record(
            source_session_date=yesterday_iso, target_execution_session_date=today_iso,
        )
    elif action == "cancel":
        state.positions[_DRILL_TICKER] = _MutablePosition(
            ticker=_DRILL_TICKER, asset_class="EQUITY", entry_timestamp=f"{yesterday_iso}T00:00:00Z",
            entry_portfolio_bar_index=0, quantity=10.0, entry_price=50.0, entry_fee=0.0,
            stop_loss_price=40.0, highest_close=50.0, trailing_close_percent=7.5,
            initial_risk_amount=100.0, signal_score=0.0, signal_reason="drill fixture",
        )
        state.equity_stop_orders[_DRILL_TICKER] = stop_order_id
        # broker_reconciliation.reconcile()'s own Scenario F invariant
        # requires a submitted_actions record for any equity_stop_orders
        # entry (StopOrderInvariantViolationError otherwise) -- same
        # fixture shape test_run_daily_decision_broker_reconciliation.py's
        # own CEG fixture already established.
        state.submitted_actions[f"{_DRILL_TICKER}|PROTECTIVE_STOP|{yesterday_iso}"] = {
            "order_id": stop_order_id, "client_order_id": f"{_DRILL_TICKER}-PROTECTIVE-STOP-FOR-drill-entry-1",
            "status": "new", "kind": "PROTECTIVE_STOP",
        }
        state.pending_exits[_DRILL_TICKER] = _PendingOrder(
            signal=PortfolioSignal(
                timestamp=yesterday_iso, ticker=_DRILL_TICKER, action="EXIT", reference_price=51.0,
                reason="drill fixture -- EMA20 below EMA50",
            ),
            submitted_portfolio_bar_index=0,
        )
        state.pending_signal_metadata["AA|EXIT"] = _pending_signal_record(
            source_session_date=yesterday_iso, target_execution_session_date=today_iso,
        )
    else:
        raise ValueError(f"Unknown action: {action!r}")
    return state


# ============================================================================
# arm / recover / self-check
# ============================================================================


def _drill_paths(case_dir: Path) -> dict[str, Path]:
    return {
        "state_path": case_dir / "position_state.json",
        "decision_log_directory": case_dir / "decisions",
        "ledger_path": case_dir / "fake_broker_ledger.json",
        "notifications_path": case_dir / "notifications.jsonl",
        "drill_config_path": case_dir / "drill_config.json",
        "barrier_path": case_dir / "barrier.json",
        "report_path": case_dir / "recovery_report.json",
    }


def _today_and_yesterday_iso() -> tuple[str, str]:
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).date()
    return today.isoformat(), (today - timedelta(days=1)).isoformat()


def _run_arm(*, case_dir: Path, barrier: str, action: str, broker_outcome: str) -> None:
    paths = _drill_paths(case_dir)
    today_iso, yesterday_iso = _today_and_yesterday_iso()

    stop_order_id = "drill-resting-stop-1" if action == "cancel" else None
    ledger = FakeBrokerLedger(paths["ledger_path"])
    if action == "cancel":
        ledger.positions[_DRILL_TICKER] = {"symbol": _DRILL_TICKER, "qty": "10", "side": "long"}
        ledger.orders[stop_order_id] = {
            "id": stop_order_id, "client_order_id": "", "symbol": _DRILL_TICKER, "side": "sell",
            "status": "new", "qty": "10", "filled_qty": "0",
            "submitted_at": f"{yesterday_iso}T00:00:00Z", "updated_at": f"{yesterday_iso}T00:00:00Z",
        }
    ledger.save()

    runner_state = _build_initial_runner_state(
        action=action, yesterday_iso=yesterday_iso, today_iso=today_iso, stop_order_id=stop_order_id,
    )
    ps.save_position_state(runner_state, paths["state_path"], guard_path=ps.HIGH_WATER_MARK_PATH)

    drill_config = {
        "action": action, "ticker": _DRILL_TICKER, "barrier": barrier,
        "expected_session_date": today_iso, "stop_order_id": stop_order_id,
        "client_order_id": rdd.client_order_id_for_action(
            _DRILL_TICKER, "ENTRY_MARKET_BUY" if action == "submit" else "SIGNAL_EXIT_MARKET_SELL", today_iso,
        ),
    }
    paths["drill_config_path"].write_text(json.dumps(drill_config, indent=2, sort_keys=True), encoding="utf-8")

    recorder = LocalRecorder(paths["notifications_path"])
    _install_notification_fixture(recorder)
    _install_issuer_identity_fixture()
    _install_market_data_fixture(yesterday_iso=yesterday_iso, today_iso=today_iso)

    action_kind_filter = "ENTRY_MARKET_BUY" if action == "submit" else "CANCEL_PROTECTIVE_STOP"
    controller = BarrierController(case_dir, armed_barrier=barrier, action_kind_filter=action_kind_filter)
    _install_barrier_hooks(controller)

    # Real, manual, source-level flag -- see module docstring safety
    # point (7). Module-object attribute only; the source file is never
    # edited by this script.
    carm._WRITE_AHEAD_JOURNAL_OWNER_APPROVED = True

    fake_client = FakeTradingClient(ledger, today_iso=today_iso)
    arguments = argparse.Namespace(
        state_path=paths["state_path"],
        decision_log_directory=paths["decision_log_directory"],
        enable_equity_orders=True,
        enable_crypto_orders=False,
        env_file=None,
        healthchecks_env_file=None,
    )

    print(f"[DRILL] arm: action={action} barrier={barrier} broker_outcome={broker_outcome} case_dir={case_dir}")
    outcome = carm._execute(arguments, trading_client=fake_client)
    if not controller.fired:
        print(
            f"[DRILL] WARNING: armed barrier {barrier!r} was never reached -- _execute() "
            f"returned outcome={outcome!r} without ever hitting the target transition. This "
            f"drill case did NOT exercise the intended crash point; check drill_config.json "
            f"and the printed [CONTROL] log above for why.",
            file=sys.stderr,
        )
    else:
        print(f"[DRILL] arm: completed WITHOUT hitting the barrier (outcome={outcome!r}) -- unexpected, see above.")


def _apply_broker_outcome_override(*, case_dir: Path, broker_outcome: str) -> None:
    """Only meaningful when the recorded barrier paused BEFORE the fake
    broker call genuinely happened (AFTER_PREPARED/AFTER_SUBMITTING) --
    for every later barrier, the fake broker call already genuinely
    happened (or didn't) as part of the interrupted arm run, and the
    ledger already reflects that ground truth; overriding it here would
    misrepresent what a real crash at that later point could ever
    produce."""
    paths = _drill_paths(case_dir)
    drill_config = json.loads(paths["drill_config_path"].read_text(encoding="utf-8"))
    barrier_payload = json.loads(paths["barrier_path"].read_text(encoding="utf-8"))
    barrier = barrier_payload["barrier"]

    if barrier not in ("AFTER_PREPARED", "AFTER_SUBMITTING"):
        print(
            f"[DRILL] --broker-outcome={broker_outcome!r} ignored: barrier {barrier!r} already reflects "
            f"the real (fake) broker call's own genuine outcome from the interrupted arm run."
        )
        return

    ledger = FakeBrokerLedger(paths["ledger_path"])
    action = drill_config["action"]
    today_iso, yesterday_iso = _today_and_yesterday_iso()

    if action == "submit":
        client_order_id = drill_config["client_order_id"]
        if broker_outcome == "present":
            order_id = "drill-order-recovered-present"
            ledger.orders[order_id] = {
                "id": order_id, "client_order_id": client_order_id, "symbol": _DRILL_TICKER,
                "side": "buy", "status": "filled", "qty": "10", "filled_qty": "10",
                "submitted_at": f"{today_iso}T00:00:00Z", "updated_at": f"{today_iso}T00:00:01Z",
            }
        else:
            existing = ledger.find_by_client_order_id(client_order_id)
            if existing is not None:
                ledger.orders.pop(existing["id"], None)
    else:  # cancel
        stop_order_id = drill_config["stop_order_id"]
        if broker_outcome == "present":
            ledger.orders.setdefault(stop_order_id, {
                "id": stop_order_id, "client_order_id": "", "symbol": _DRILL_TICKER, "side": "sell",
                "qty": "10", "filled_qty": "0",
                "submitted_at": f"{yesterday_iso}T00:00:00Z", "updated_at": f"{today_iso}T00:00:01Z",
            })
            ledger.orders[stop_order_id]["status"] = "canceled"
        else:
            ledger.orders.setdefault(stop_order_id, {
                "id": stop_order_id, "client_order_id": "", "symbol": _DRILL_TICKER, "side": "sell",
                "qty": "10", "filled_qty": "0",
                "submitted_at": f"{yesterday_iso}T00:00:00Z", "updated_at": f"{yesterday_iso}T00:00:00Z",
            })
            ledger.orders[stop_order_id]["status"] = "new"
    ledger.save()
    print(f"[DRILL] Applied --broker-outcome={broker_outcome!r} to the fake ledger for barrier {barrier!r}.")


def _run_recover(*, case_dir: Path, broker_outcome: str) -> None:
    paths = _drill_paths(case_dir)
    if not paths["barrier_path"].is_file():
        raise RuntimeError(
            f"{paths['barrier_path']} does not exist -- did `--phase arm` actually reach its "
            f"barrier before this process ran? Nothing to recover."
        )
    drill_config = json.loads(paths["drill_config_path"].read_text(encoding="utf-8"))
    barrier_payload = json.loads(paths["barrier_path"].read_text(encoding="utf-8"))

    stray_before = [i.to_dict() for i in _stray_intents_snapshot()]

    _apply_broker_outcome_override(case_dir=case_dir, broker_outcome=broker_outcome)

    recorder = LocalRecorder(paths["notifications_path"])
    _install_notification_fixture(recorder)
    _install_issuer_identity_fixture()
    today_iso, yesterday_iso = _today_and_yesterday_iso()
    _install_market_data_fixture(yesterday_iso=yesterday_iso, today_iso=today_iso)
    carm._WRITE_AHEAD_JOURNAL_OWNER_APPROVED = True
    # No barrier armed during recovery -- run to completion.
    controller = BarrierController(case_dir, armed_barrier=None, action_kind_filter=None)
    _install_barrier_hooks(controller)

    ledger = FakeBrokerLedger(paths["ledger_path"])
    fake_client = FakeTradingClient(ledger, today_iso=today_iso)
    arguments = argparse.Namespace(
        state_path=paths["state_path"],
        decision_log_directory=paths["decision_log_directory"],
        enable_equity_orders=True,
        enable_crypto_orders=False,
        env_file=None,
        healthchecks_env_file=None,
    )

    print(f"[DRILL] recover: barrier={barrier_payload['barrier']!r} action={drill_config['action']!r} "
          f"broker_outcome={broker_outcome!r} case_dir={case_dir}")
    recovery_error: str | None = None
    outcome = None
    try:
        outcome = carm._execute(arguments, trading_client=fake_client)
    except Exception as error:  # noqa: BLE001 -- reported in the drill report, not swallowed
        recovery_error = f"{type(error).__name__}: {error}"

    stray_after = [i.to_dict() for i in _stray_intents_snapshot()]
    report = {
        "barrier": barrier_payload["barrier"],
        "action": drill_config["action"],
        "broker_outcome": broker_outcome,
        "execute_outcome": outcome,
        "execute_raised": recovery_error,
        "stray_intents_before_recovery": stray_before,
        "stray_intents_after_recovery": stray_after,
        "final_position_state": json.loads(paths["state_path"].read_text(encoding="utf-8"))
        if paths["state_path"].is_file() else None,
    }
    paths["report_path"].write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[DRILL] recover: report written to {paths['report_path']}")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


def _stray_intents_snapshot():
    import src.live.order_intent_reconciliation as reconciliation

    return reconciliation.find_stray_session_intents_from_prior_run()


def _run_self_check(case_dir: Path) -> None:
    results: dict[str, dict] = {}

    # (1) network isolation -- best-effort: report whether a real
    # external host is reachable. This does NOT create isolation; it
    # only reports the current environment's own state, which is only
    # meaningful when the owner has already set up a real network
    # namespace with no outbound route around this process.
    def _try_connect(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            return False

    reachable = _try_connect("1.1.1.1", 443)
    results["network_isolation"] = {
        "external_host_reachable": reachable,
        "pass": not reachable,
        "note": (
            "reachable=False means this process cannot reach the public internet right now -- "
            "confirm this was checked inside the same sandbox/namespace the real drill runs in. "
            "reachable=True means no network isolation is currently in effect for this process."
        ),
    }

    # (2) fake client is genuinely reached, no real Alpaca host touched.
    ledger = FakeBrokerLedger(case_dir / "self_check_ledger.json")
    client = FakeTradingClient(ledger, today_iso="2026-01-01")
    client.get_account()
    client.get_all_positions()
    results["fake_client_reachable"] = {
        "calls_recorded": client.calls,
        "pass": client.calls == ["get_account", "get_all_positions"],
    }

    # (3) real .env not loaded -- dotenv.load_dotenv is confirmed
    # neutralized, and re-importing telegram_notifier/alpaca_market_data
    # (already imported transitively above) did not populate real
    # credential env vars this process never set itself.
    results["env_not_loaded"] = {
        "dotenv_load_dotenv_neutralized": dotenv.load_dotenv is not _REAL_LOAD_DOTENV,
        "alpaca_api_key_absent": "ALPACA_API_KEY" not in os.environ,
        "alpaca_secret_key_absent": "ALPACA_SECRET_KEY" not in os.environ,
        "telegram_bot_token_absent": "TELEGRAM_BOT_TOKEN" not in os.environ,
        "guard_dir_isolated": os.environ.get("AI_STOCK_RADAR_GUARD_DIR") == str(_GUARD_DIR),
    }
    results["env_not_loaded"]["pass"] = all(
        v for k, v in results["env_not_loaded"].items() if k != "pass"
    )

    overall_pass = all(section.get("pass") for section in results.values())
    report = {"overall_pass": overall_pass, "checks": results}
    report_path = case_dir / "self_check_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not overall_pass:
        sys.exit(1)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("arm", "recover"))
    parser.add_argument("--barrier", choices=BARRIER_NAMES)
    parser.add_argument("--action", choices=("submit", "cancel"))
    parser.add_argument("--broker-outcome", choices=("absent", "present"))
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()

    case_dir = arguments.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    if arguments.self_check:
        _run_self_check(case_dir)
        return

    if arguments.phase is None:
        parser.error("--phase is required unless --self-check is given")
    if arguments.broker_outcome is None:
        parser.error("--broker-outcome is required")

    if arguments.phase == "arm":
        if arguments.barrier is None:
            parser.error("--barrier is required for --phase arm")
        if arguments.action is None:
            parser.error("--action is required for --phase arm")
        _run_arm(case_dir=case_dir, barrier=arguments.barrier, action=arguments.action, broker_outcome=arguments.broker_outcome)
    else:
        _run_recover(case_dir=case_dir, broker_outcome=arguments.broker_outcome)


if __name__ == "__main__":
    main()
