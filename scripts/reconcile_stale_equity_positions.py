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
unambiguously, confirmed by a REAL broker query, never guessed:
- `equity_stop_orders[ticker]` resolves to a real `submitted_actions`
  record of kind `PROTECTIVE_STOP` (not orphaned, not the wrong kind).
- That exact `order_id`, queried at the broker, has status == "filled".
- The filled order's `symbol` == `ticker`, `side` == SELL, and
  `filled_qty` == the local position's own `quantity` (exact match,
  not "close enough").

Any single mismatch, ambiguity, or lookup failure -> refuse, print a
clear diagnostic, change NOTHING. This script makes no attempt to
"figure out" an unclear case -- see `diagnose_and_clean_stale_position`'s
own `AMBIGUOUS` branches.

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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.client import TradingClient

import src.live.position_state as ps
from src.live import order_submission


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
    own normal, expected outcome, not an exceptional one."""
    runner_state = ps.load_position_state(state_path, guard_path=guard_path)

    if ticker not in runner_state.positions:
        return {
            "status": "NO_ACTION",
            "ticker": ticker,
            "reason": f"{ticker!r} is not in local positions -- nothing to clean.",
        }

    position = runner_state.positions[ticker]
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

    stop_record = None
    for record in runner_state.submitted_actions.values():
        if str(record.get("order_id")) == stop_order_id:
            stop_record = record
            break
    if stop_record is None:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": (
                f"equity_stop_orders[{ticker!r}]={stop_order_id!r} has no "
                f"matching submitted_actions record at all (orphaned "
                f"reference) -- cannot auto-verify."
            ),
        }
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
    filled_qty = float(order.filled_qty) if order.filled_qty is not None else None

    mismatches = []
    if order_symbol != ticker:
        mismatches.append(f"symbol {order_symbol!r} != expected {ticker!r}")
    if order_side != "SELL":
        mismatches.append(f"side {order_side!r} != expected 'SELL'")
    if filled_qty is None or abs(filled_qty - position.quantity) > 1e-6:
        mismatches.append(f"filled_qty {filled_qty} != local quantity {position.quantity}")
    if mismatches:
        return {
            "status": "AMBIGUOUS",
            "ticker": ticker,
            "reason": f"broker order {stop_order_id!r} does not exactly match: {'; '.join(mismatches)}.",
        }

    # order_id + symbol + side + quantity + status=filled all confirmed --
    # the only case this script ever auto-corrects.
    diagnosis = {
        "status": "CONFIRMED_STALE",
        "ticker": ticker,
        "stop_order_id": stop_order_id,
        "broker_fill_price": str(order.filled_avg_price),
        "broker_filled_qty": filled_qty,
        "local_quantity": position.quantity,
        "local_entry_price": position.entry_price,
        "local_entry_timestamp": position.entry_timestamp,
    }

    if not apply:
        diagnosis["note"] = "DRY-RUN (--apply not passed) -- no changes made."
        return diagnosis

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
    result = diagnose_and_clean_stale_position(
        arguments.ticker, arguments.state_path, client, apply=arguments.apply
    )
    print(json.dumps(result, indent=2, default=str))
    if result["status"] == "AMBIGUOUS":
        sys.exit(1)


if __name__ == "__main__":
    main()
