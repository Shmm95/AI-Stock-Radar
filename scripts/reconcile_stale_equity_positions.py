"""One-time cleanup for the class of bug found 2026-08-21: a local
`position_state.json` record for an equity ticker whose PROTECTIVE
STOP has already filled at the broker (closing the position there),
while local state still shows it open -- because
`scripts/run_daily_decision.py` has no independent check that a
tracked resting stop is still genuinely resting at the broker (see
`src/live/broker_reconciliation.py`'s Scenario F, which exists and
covers exactly this case, but was wired only into
`scripts/run_control_arm_decision.py`'s separate 44-ticker control-arm
account, never into this file's own live entrypoint).

`cash_balance_usd` is DELIBERATELY NEVER TOUCHED by this script:
confirmed by reading `run_daily_decision.py`, that value comes from a
real, live `TradingClient.get_account().cash` query
(`src/live/account_state.py`) made FRESH every single run -- it is
never persisted in `position_state.json` and never derived from local
position bookkeeping, so it already reflects the broker's real current
cash (including this stale position's real stop-fill proceeds) with
zero code changes needed. Only `positions`/`equity_stop_orders` --
confirmed genuinely stale by direct read of `open_position_count=
len(state.positions)` in `portfolio_backtest_engine.py` -- are ever
corrected here.

FAIL-CLOSED BY DESIGN, PER THE OWNER'S EXPLICIT REQUIREMENT: a
correction is applied ONLY when ALL of the following match exactly and
unambiguously, confirmed by a REAL broker query, never guessed
(REAL BUGS FOUND AND FIXED in this exact list, 2026-08-22, independent
audit -- the original version claimed "exact match" without actually
enforcing several of these):
- `position.asset_class == "EQUITY"` -- this script never touches a
  crypto position (crypto has no native broker stop at all; see
  `crypto_stop_monitor.py`'s own docstring for why that is a
  structurally different mechanism).
- `equity_stop_orders[ticker]` resolves to EXACTLY ONE
  `submitted_actions` record of kind `PROTECTIVE_STOP` (not orphaned,
  not the wrong kind, and not ambiguous -- multiple local records
  sharing the same `order_id` now refuses rather than silently using
  whichever was found first).
- The broker order actually returned by `get_order_by_id(stop_order_id)`
  has its OWN `id` equal to `stop_order_id` (never assumed just because
  that was the id queried).
- That order's status == "filled".
- The filled order's `symbol` == `ticker`, `side` == SELL, and
  `filled_qty` == the local position's own `quantity` -- compared as
  `Decimal(str(...))`, exact, never `float` +/- a tolerance (the same
  discipline `broker_reconciliation.py` already uses for every other
  quantity comparison in this codebase).
- A real, separate `get_all_positions()` call confirms the ticker is
  NOT an open broker position anymore (the stop order showing "filled"
  is necessary but was never independently confirmed against the
  broker's own current position list).

Any single mismatch, ambiguity, or lookup failure -> refuse, print a
clear diagnostic, change NOTHING. This script makes no attempt to
"figure out" an unclear case -- see `diagnose_and_clean_stale_position`'s
own `AMBIGUOUS` branches.

CONCURRENCY (independent audit finding #5): the `--apply` path now:
1. Runs the whole diagnose+apply operation under `single_instance_lock`
   (KNOWN LIMITATION, disclosed rather than overclaimed:
   `run_daily_decision.py` itself does not currently acquire this same
   lock, so this alone does not fully exclude a concurrently running
   daily runner -- it protects against two invocations of THIS script,
   or a control-arm run on the same state path).
2. Re-hashes `position_state.json` immediately before writing and
   compares against the hash taken at load time -- ANY change (from
   any process, lock or no lock) aborts the write rather than silently
   overwriting a concurrent update. This is the real, effective
   protection regardless of the limitation in (1).
3. Re-queries `get_all_positions()` immediately before writing to
   reconfirm the ticker is STILL absent, minimizing the window between
   "confirmed" and "applied."

DRY-RUN BY DEFAULT: `--apply` must be passed explicitly to actually
write a correction; the default run only diagnoses and prints what it
would do.

Run (dry-run, default):
    .venv/bin/python scripts/reconcile_stale_equity_positions.py --ticker CEG

Run (apply the correction, only after confirming the dry-run output
is exactly the expected one):
    .venv/bin/python scripts/reconcile_stale_equity_positions.py --ticker CEG --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.client import TradingClient

import src.live.position_state as ps
from src.live import order_submission
from src.live.single_instance_lock import SingleInstanceLockError, single_instance_lock


def _hash_state_file(state_path: Path) -> str | None:
    """SHA-256 of the state file's exact bytes, or `None` if it does
    not exist yet. Same convention as
    `run_control_arm_decision.py`'s own `_hash_state_file`."""
    path = Path(state_path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def diagnose_and_clean_stale_position(
    ticker: str,
    state_path: Path,
    client: TradingClient,
    *,
    guard_path: Path = ps.HIGH_WATER_MARK_PATH,
    apply: bool = False,
) -> dict:
    """See module docstring for the full fail-closed contract. Returns
    a plain dict describing what was found/done -- never raises for an
    ambiguous case, since "ambiguous, needs a human" is this function's
    own normal, expected outcome, not an exceptional one. DOES raise
    (real `RuntimeError`) if the concurrency re-checks in the `apply`
    path detect the state changed underneath it -- that is a genuine
    abort condition, not a diagnosis outcome to report and continue
    past."""
    load_hash = _hash_state_file(state_path)
    runner_state = ps.load_position_state(state_path, guard_path=guard_path)

    if ticker not in runner_state.positions:
        return {
            "status": "NO_ACTION",
            "ticker": ticker,
            "reason": f"{ticker!r} is not in local positions -- nothing to clean.",
        }

    position = runner_state.positions[ticker]

    if position.asset_class != "EQUITY":
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"{ticker!r} has asset_class={position.asset_class!r}, not "
                f"'EQUITY' -- this script only handles the equity native-"
                f"broker-stop reconciliation gap. Crypto has no native "
                f"broker stop at all (see crypto_stop_monitor.py); refusing "
                f"to touch it here."
            ),
        }

    stop_order_id = runner_state.equity_stop_orders.get(ticker)
    if stop_order_id is None:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"{ticker!r} has an open local position but no tracked "
                f"equity_stop_orders entry -- cannot auto-verify what closed "
                f"it, if anything. Manual review required."
            ),
        }
    stop_order_id = str(stop_order_id)

    matching_records = [
        (key, record)
        for key, record in runner_state.submitted_actions.items()
        if str(record.get("order_id")) == stop_order_id
    ]
    if not matching_records:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"equity_stop_orders[{ticker!r}]={stop_order_id!r} has no "
                f"matching submitted_actions record at all (orphaned "
                f"reference) -- cannot auto-verify."
            ),
        }
    if len(matching_records) > 1:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"{len(matching_records)} submitted_actions records share "
                f"order_id={stop_order_id!r}: {[key for key, _ in matching_records]}. "
                f"Ambiguous which is authoritative -- refusing to guess by "
                f"taking the first one found."
            ),
        }
    stop_key, stop_record = matching_records[0]
    if stop_record.get("kind") != "PROTECTIVE_STOP":
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"equity_stop_orders[{ticker!r}]={stop_order_id!r}'s local "
                f"record has kind={stop_record.get('kind')!r}, not "
                f"'PROTECTIVE_STOP' -- integrity issue, not auto-correctable."
            ),
        }

    order = client.get_order_by_id(stop_order_id)  # real broker call

    returned_order_id = str(getattr(order, "id", ""))
    if returned_order_id != stop_order_id:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"get_order_by_id({stop_order_id!r}) returned an order whose "
                f"own id is {returned_order_id!r} -- does not match the id "
                f"queried. Refusing to trust this response."
            ),
        }

    status = str(order.status.value if hasattr(order.status, "value") else order.status)
    if status != "filled":
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"broker order {stop_order_id!r} status={status!r}, not "
                f"'filled' -- cannot confirm the position actually closed "
                f"at the broker. Manual review required."
            ),
        }

    order_symbol = str(order.symbol)
    order_side = str(order.side.value if hasattr(order.side, "value") else order.side).upper()
    local_qty = _decimal_or_none(position.quantity)
    broker_filled_qty = _decimal_or_none(order.filled_qty)

    mismatches = []
    if order_symbol != ticker:
        mismatches.append(f"symbol {order_symbol!r} != expected {ticker!r}")
    if order_side != "SELL":
        mismatches.append(f"side {order_side!r} != expected 'SELL'")
    if broker_filled_qty is None or local_qty is None or broker_filled_qty != local_qty:
        mismatches.append(
            f"filled_qty {order.filled_qty!r} != local quantity {position.quantity!r} (Decimal-exact)"
        )
    if mismatches:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": f"broker order {stop_order_id!r} does not exactly match: {'; '.join(mismatches)}.",
        }

    # Real, separate confirmation that the ticker is genuinely no longer
    # an open broker position -- the stop order showing "filled" alone
    # was never independently checked against this before.
    broker_positions = {str(p.symbol): p for p in client.get_all_positions()}
    if ticker in broker_positions:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"broker still reports an open {ticker} position "
                f"(qty={broker_positions[ticker].qty}) despite the stop "
                f"order showing filled -- possible partial fill or a new "
                f"position reopened since. Refusing."
            ),
        }

    # order_id (exact) + symbol + side + quantity (Decimal-exact) +
    # status=filled + confirmed absent at the broker -- the only case
    # this script ever auto-corrects.
    diagnosis = {
        "status": "CONFIRMED_STALE",
        "ticker": ticker,
        "stop_order_id": stop_order_id,
        "broker_fill_price": str(order.filled_avg_price),
        "broker_filled_qty": str(broker_filled_qty),
        "local_quantity": position.quantity,
        "local_entry_price": position.entry_price,
        "local_entry_timestamp": position.entry_timestamp,
    }

    if not apply:
        diagnosis["note"] = "DRY-RUN (--apply not passed) -- no changes made."
        return diagnosis

    # --- Concurrency re-checks (independent audit finding #5) -- ---
    # --- everything above this point never mutates anything.     ---
    current_hash = _hash_state_file(state_path)
    if current_hash != load_hash:
        raise RuntimeError(
            f"position_state.json changed on disk between this diagnosis's "
            f"load and the apply step (hash {load_hash} -> {current_hash}) "
            f"-- another process (the daily runner, crypto monitor, or "
            f"another invocation of this script) wrote to it concurrently. "
            f"Refusing to apply a correction on top of a state snapshot "
            f"that is no longer current -- re-run this script to diagnose "
            f"and apply against the CURRENT state."
        )
    reconfirm_positions = {str(p.symbol): p for p in client.get_all_positions()}
    if ticker in reconfirm_positions:
        raise RuntimeError(
            f"{ticker} reappeared as an open broker position between "
            f"diagnosis and apply -- refusing to write a now-stale "
            f"correction."
        )

    del runner_state.positions[ticker]
    del runner_state.equity_stop_orders[ticker]
    stop_record["status"] = "filled"  # accurate audit trail, same record object already found above
    ps.save_position_state(runner_state, state_path, guard_path=guard_path)
    diagnosis["note"] = "Correction applied and saved to position_state.json."
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Equity ticker to diagnose/clean, e.g. CEG.")
    parser.add_argument("--state-path", type=Path, default=ps.DEFAULT_STATE_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the correction. Omit to dry-run (diagnose only, no changes).",
    )
    arguments = parser.parse_args()

    client = order_submission.get_trading_client()
    try:
        with single_instance_lock(arguments.state_path):
            result = diagnose_and_clean_stale_position(
                arguments.ticker, arguments.state_path, client, apply=arguments.apply
            )
    except SingleInstanceLockError as error:
        print(json.dumps({"status": "LOCK_CONFLICT", "reason": str(error)}, indent=2, default=str))
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
    if result["status"] == "AMBIGUOUS":
        sys.exit(1)


if __name__ == "__main__":
    main()
