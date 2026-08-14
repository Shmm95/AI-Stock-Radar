"""Pipeline heartbeat test -- exercises the real order-submission path
with a tiny, fixed-dollar-amount BTC/USD round trip, on a schedule
independent of whether the frozen TREND_RSI strategy has generated a
real signal recently (it can go weeks without one by design). This is
NOT a strategy signal and NEVER represents a trading decision -- it
exists purely so kill-switch behavior, idempotency, and real broker
interaction get exercised regularly rather than only whenever an
organic signal happens to fire.

Two independent invocations, meant to be scheduled a few hours apart
(see the report for suggested cron lines -- this script never touches
crontab itself):

    .venv/bin/python scripts/run_heartbeat_test.py --action=buy
    .venv/bin/python scripts/run_heartbeat_test.py --action=sell

Hard separation from the real strategy (all non-negotiable, per the
task this script was built for):
- Own state file (`HEARTBEAT_STATE_PATH`), never touches
  `src/live/position_state.py` or its idempotency ledger.
- Every real order carries `client_order_id` prefixed
  `HEARTBEAT_` (Alpaca's own custom client_order_id field --
  `src/live/order_submission.submit_equity_market_order` was extended,
  never rewritten, to accept it). `scripts/send_status_update.py`'s
  performance report filters any order whose client_order_id starts
  with this prefix out of the real strategy's P&L/win-rate/trade-count
  entirely -- see that script's own docstring for the filter.
- Respects the same `STOP_FLAG_PATH`/`FREEZE_FLAG_PATH` kill switch as
  `run_daily_decision.py` -- imported directly from that module rather
  than redefined here, so there is exactly one source of truth for
  where those flag files live. No exception for either flag: a FREEZE
  is honored as a full stop here (unlike `run_daily_decision.py`, this
  script has no "existing position monitoring" to preserve through a
  freeze -- it only ever opens or closes its own tiny test position,
  which is new-exposure activity either way).
- Sell quantity is `min(this script's own recorded filled_qty,
  crypto_stop_monitor._query_available_crypto_quantity(...))` --
  confirmed necessary with a REAL paper fill: a $15 BTC/USD buy
  recorded `filled_qty=0.000232731`, but selling exactly that amount
  minutes later failed with a real 403 (`insufficient balance ...
  available: 0.000232149`) -- the same in-kind-fee quantity drift
  `crypto_stop_monitor.py`'s own docstring already documents for the
  real strategy's crypto exits. Querying the broker alone (as that
  module does) was ruled out here on its own: if the real strategy
  ever also holds a real BTC-USD position at the same time, "how much
  is available" would return their COMBINED holdings, and heartbeat
  could oversell into the real strategy's position. Taking the
  minimum of the two gets both properties at once: it can never exceed
  what THIS script itself recorded buying (bounded, isolated from the
  real strategy's own holdings even if they coexist), and it never
  requests more than the broker actually confirms is free to sell
  right now (absorbs the real fee drift, exactly like
  `crypto_stop_monitor.py` does for its own case).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402

import run_daily_decision as daily_decision  # noqa: E402 -- single source of truth for the kill-switch flag paths
from src.live import order_submission  # noqa: E402
from src.live.crypto_stop_monitor import _query_available_crypto_quantity  # noqa: E402
from src.notify.telegram_notifier import send_telegram_message  # noqa: E402

SYMBOL = "BTC/USD"
NOTIONAL_USD = 15.0
HEARTBEAT_STATE_PATH = Path("data/live/heartbeat_state.json")
HEARTBEAT_LOCK_PATH = Path("data/live/heartbeat.lock")
SCHEMA_VERSION = 1

_LABEL = "[PIPELINE TEST — HEARTBEAT, gerçek strateji sinyali değil]"


def _notify_safe(text: str) -> bool:
    try:
        return bool(send_telegram_message(text))
    except Exception:
        return False


def _acquire_lock(path: Path = HEARTBEAT_LOCK_PATH, attempts: int = 3, delay_seconds: float = 2.0) -> bool:
    """Same `os.O_CREAT|O_EXCL` primitive as `crypto_stop_monitor._monitor_lock` --
    guards against two overlapping invocations of the SAME action (a
    cron overlap, a manual re-run while one is still in flight), not a
    replacement for the check-then-act state read below."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            return True
        except FileExistsError:
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
    return False


def _release_lock(path: Path = HEARTBEAT_LOCK_PATH) -> None:
    path.unlink(missing_ok=True)


def load_state() -> dict:
    if not HEARTBEAT_STATE_PATH.is_file():
        return {"schema_version": SCHEMA_VERSION, "open": False}
    return json.loads(HEARTBEAT_STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    HEARTBEAT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_kill_switch() -> bool:
    """Returns True if this run must stop immediately. FREEZE is treated
    identically to STOP here -- see module docstring for why this
    script has no "existing position monitoring" case to preserve
    through a freeze the way run_daily_decision.py does."""
    if daily_decision.STOP_FLAG_PATH.exists():
        text = (
            f"{_LABEL} STOP flag detected ({daily_decision.STOP_FLAG_PATH}); "
            f"this run was skipped entirely, no API calls were made."
        )
        print(text)
        _notify_safe(text)
        return True
    if daily_decision.FREEZE_FLAG_PATH.exists():
        text = (
            f"{_LABEL} FREEZE flag detected ({daily_decision.FREEZE_FLAG_PATH}); "
            f"heartbeat has no existing position to protect through a freeze, "
            f"so this counts as a full stop too -- this run was skipped entirely, "
            f"no API calls were made."
        )
        print(text)
        _notify_safe(text)
        return True
    return False


def _client_order_id() -> str:
    return f"HEARTBEAT_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def run_buy() -> None:
    if _check_kill_switch():
        return

    state = load_state()
    if state.get("open"):
        text = (
            f"{_LABEL} buy skipped: a heartbeat position is already open "
            f"(order {state.get('buy_order_id')}, opened {state.get('buy_time')}) -- "
            f"idempotent no-op, no duplicate buy submitted."
        )
        print(text)
        return

    if not _acquire_lock():
        print(f"{HEARTBEAT_LOCK_PATH} already held -- another heartbeat invocation appears to be running. Skipping.")
        return

    try:
        # Re-check state after acquiring the lock -- closes the
        # check-then-act race between two near-simultaneous buy calls.
        state = load_state()
        if state.get("open"):
            print("Heartbeat position opened by a concurrent run while acquiring the lock -- skipping.")
            return

        client = order_submission.get_trading_client()
        client_order_id = _client_order_id()
        order = order_submission.submit_equity_market_order(
            client,
            ticker=SYMBOL,
            side=OrderSide.BUY,
            notional=NOTIONAL_USD,
            time_in_force=TimeInForce.IOC,
            client_order_id=client_order_id,
        )
        status = order_submission.wait_for_fill_or_timeout(client, str(order.id))

        if status != "filled":
            text = (
                f"{_LABEL} BUY order {order.id} did NOT fill (status={status}); "
                f"no heartbeat position recorded as open. Needs manual review."
            )
            print(text)
            _notify_safe(text)
            return

        filled_order = client.get_order_by_id(str(order.id))
        filled_qty = float(filled_order.filled_qty)
        filled_avg_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else None

        new_state = {
            "schema_version": SCHEMA_VERSION,
            "open": True,
            "symbol": SYMBOL,
            "buy_order_id": str(order.id),
            "buy_client_order_id": client_order_id,
            "buy_time": order_submission.now_iso(),
            "notional_usd": NOTIONAL_USD,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
        }
        save_state(new_state)

        text = (
            f"{_LABEL} BUY filled: {filled_qty} {SYMBOL} for ~${NOTIONAL_USD} "
            f"(order {order.id}, client_order_id {client_order_id}, "
            f"avg price {filled_avg_price})."
        )
        print(text)
        _notify_safe(text)
    finally:
        _release_lock()


def run_sell() -> None:
    if _check_kill_switch():
        return

    state = load_state()
    if not state.get("open"):
        print("No open heartbeat position -- nothing to sell. Not an error.")
        return

    if not _acquire_lock():
        print(f"{HEARTBEAT_LOCK_PATH} already held -- another heartbeat invocation appears to be running. Skipping.")
        return

    try:
        state = load_state()
        if not state.get("open"):
            print("Heartbeat position closed by a concurrent run while acquiring the lock -- skipping.")
            return

        recorded_quantity = float(state["filled_qty"])
        client = order_submission.get_trading_client()

        # Real in-kind-fee drift confirmed with a real paper fill (see module
        # docstring) -- selling exactly `recorded_quantity` can 403. Cap at
        # whatever the broker confirms is actually free to sell right now,
        # but never above our own recorded fill (isolation from a real
        # strategy position in the same symbol, see module docstring).
        available_quantity = _query_available_crypto_quantity(client, state["symbol"])
        if available_quantity is None:
            text = (
                f"{_LABEL} SELL skipped: broker reports nothing available for "
                f"{state['symbol']} (buy order {state.get('buy_order_id')}); "
                f"refusing to guess a sell quantity. Needs manual review."
            )
            print(text)
            _notify_safe(text)
            return
        quantity = min(recorded_quantity, available_quantity)

        client_order_id = _client_order_id()
        order = order_submission.submit_equity_market_order(
            client,
            ticker=state["symbol"],
            side=OrderSide.SELL,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
            client_order_id=client_order_id,
        )
        status = order_submission.wait_for_fill_or_timeout(client, str(order.id))

        if status != "filled":
            text = (
                f"{_LABEL} SELL order {order.id} did NOT fill (status={status}) for the "
                f"open heartbeat position (buy order {state.get('buy_order_id')}); "
                f"leaving state open for the next run to retry. Needs manual review "
                f"if this persists."
            )
            print(text)
            _notify_safe(text)
            return

        filled_order = client.get_order_by_id(str(order.id))
        sell_filled_qty = float(filled_order.filled_qty)
        sell_avg_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else None

        save_state({"schema_version": SCHEMA_VERSION, "open": False})

        text = (
            f"{_LABEL} SELL filled: {sell_filled_qty} {state['symbol']} "
            f"(order {order.id}, client_order_id {client_order_id}, "
            f"avg price {sell_avg_price}) -- closed heartbeat position opened by "
            f"buy order {state.get('buy_order_id')}."
        )
        print(text)
        _notify_safe(text)
    finally:
        _release_lock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-dollar-amount pipeline heartbeat test (not a strategy signal).")
    parser.add_argument("--action", choices=("buy", "sell"), required=True)
    arguments = parser.parse_args()

    if arguments.action == "buy":
        run_buy()
    else:
        run_sell()


if __name__ == "__main__":
    main()
