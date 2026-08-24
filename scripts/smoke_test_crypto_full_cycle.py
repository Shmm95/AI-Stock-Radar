"""Real, controlled integration test: a full crypto entry -> signal-exit
cycle driven through the ACTUAL `run_daily_decision.py` machinery
(the real frozen-engine step functions, the real Alpaca paper API),
using throwaway state/guard files so the real `data/live/position_state.json`
and the real high-water-mark guard file are never touched.

Not a pytest test: makes real (paper) order-submission calls. Crypto
trades 24/7, so this runs immediately, no market-hours wait needed.

Run by hand:

    .venv/bin/python scripts/smoke_test_crypto_full_cycle.py

REQUIRES the same real credentials/env a real live run needs: Alpaca
paper API keys resolvable by `order_submission.get_trading_client()`
(via a loaded `.env` or the process environment) and
`LIVE_ACCOUNT_NUMBER_SUFFIX` set (see `run_daily_decision._require_live_account_suffix`'s
own docstring) -- this script does not set either itself, matching how
a real operator would already have their shell configured before
running the real cron by hand.

What this proves beyond tests/test_live_crypto_order_execution.py's
mocks: that a pending BUY signal, once queued, genuinely flows through
`_execute_pending_buys_at_open` (the real frozen engine, unmodified)
into a real Alpaca market order, that the resulting local position's
`quantity` is what actually gets submitted, and that a subsequent
signal-based EXIT genuinely flows through `_execute_pending_exits_at_open`
into a real market sell that queries the broker's real available
quantity rather than trusting local state.

REWRITTEN 2026-08-24 (independent-audit finding, round covering the
reboot-drill work, item 3): this script used to call the internal
`run_daily_decision(enable_crypto_orders=True, ...)` function directly
-- bypassing `AuthorizedExecutionContext` entirely, exactly the BARE-CALL
gap `src.live.authorized_execution_context`'s own module docstring warns
about, and exactly what the new static test
`tests/test_order_enabled_caller_inventory.py::test_only_blessed_entry_points_call_run_daily_decision_with_order_enabling_behavior`
(item 5a, same round) now permanently flags. Adding
`authorize_order_execution()` to this script directly was explicitly
REJECTED as the fix (that would just reopen the bypass under a different
name -- a real preflight chain has to actually run first, not merely a
bare authorization call). Instead, this script now drives the exact
same real flow through `run_daily_decision.py`'s own `main()` -- one of
this codebase's two documented "blessed callers" (see
`run_daily_decision()`'s own `authorization` parameter docstring) --
in-process, via `sys.argv` + `runner.main()`, the same real-entry-point
discipline `tests/test_missing_session_replay_cron_activation.py`
already established for testing `main()` itself. (The OTHER blessed
caller, `scripts/run_control_arm_decision.py`'s `_execute()`, is not a
fit here: that script's own `_preflight_universe_check` deliberately
refuses to run at all whenever a BTC-USD pending/open position exists
in state -- see its own `--enable-crypto-orders` help text -- which is
exactly the scenario this smoke test needs to create.)

Two real side effects of routing through the real `main()`, both handled
below: (1) `main()` requires a same-day PASS gate before any real
order-enabling run (`_ensure_pass_gate_for_today` writes one, ONLY if
none already exists today, and this script only ever removes the gate
file it itself created -- a real gate written by the real orchestrator
today is left completely alone); (2) `main()` sends real Telegram
notifications on success/failure -- suppressed here (replaced with a
local print) so a smoke-test run on a throwaway ticker/state file
doesn't show up in the real ops channel indistinguishable from genuine
live activity.

Self-cleaning: the exit run sells the entire position back, and this
script deletes its own throwaway state/log/guard files (and its own
PASS gate, if it wrote one) when done, whatever the outcome.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.portfolio_backtest_engine import _PendingOrder
from src.backtest.portfolio_backtest_models import PortfolioSignal
from src.live import order_submission
from src.live import session_replay_pass_gate
from src.live.position_state import LiveRunnerState, save_position_state

import run_daily_decision as runner

SMOKE_STATE_PATH = Path("data/live/_smoke_test_crypto_full_cycle_state.json")
SMOKE_LOG_DIRECTORY = Path("data/live/_smoke_test_crypto_full_cycle_decisions")
SMOKE_GUARD_PATH = Path("data/live/_smoke_test_crypto_full_cycle_guard.json")
SMOKE_TICKER = "BTC-USD"


def _cleanup() -> None:
    SMOKE_STATE_PATH.unlink(missing_ok=True)
    SMOKE_GUARD_PATH.unlink(missing_ok=True)
    if SMOKE_LOG_DIRECTORY.exists():
        for entry in SMOKE_LOG_DIRECTORY.iterdir():
            entry.unlink()
        SMOKE_LOG_DIRECTORY.rmdir()


def _ensure_pass_gate_for_today() -> bool:
    """`main()`'s own real PASS-gate preflight (see module docstring)
    applies to ANY order-enabling run, crypto included, even though the
    gate's own underlying concern is the equity session cursor -- that
    is `main()`'s real, current behavior, not something this script gets
    to opt out of. Returns True only if THIS call wrote a brand new gate
    file -- i.e. only when the caller is responsible for removing it
    again. If a valid gate already exists for today (e.g. the real
    morning orchestrator already ran), it is left completely untouched
    and this returns False."""
    if session_replay_pass_gate.has_valid_pass_gate_for_today():
        return False
    today = datetime.now(UTC).date().isoformat()
    session_replay_pass_gate.write_pass_gate(last_processed_equity_session_date=today, account_number_masked=None)
    return True


def _remove_own_pass_gate() -> None:
    today = datetime.now(UTC).date().isoformat()
    gate_path = session_replay_pass_gate.DEFAULT_PASS_GATE_DIRECTORY / f"PASS_{today}.json"
    gate_path.unlink(missing_ok=True)


def _suppress_real_notifications() -> None:
    def _local_print_instead(text: str) -> bool:
        print(f"\n[SMOKE TEST -- real Telegram notification SUPPRESSED, would have read:]\n{text}\n")
        return True

    runner.send_telegram_message = _local_print_instead


def _run_main_and_load_decision(run_label: str) -> dict:
    """Invokes the REAL, blessed `run_daily_decision.main()` -- never the
    internal `run_daily_decision()` function directly (see module
    docstring). `main()` itself doesn't return the decision dict (it
    only prints it and writes it to `SMOKE_LOG_DIRECTORY`), so this
    reads it back from whichever decision-log file is newest after the
    call -- the same file `main()` itself just wrote and printed the
    path to."""
    before = set(SMOKE_LOG_DIRECTORY.glob("decision_*.json")) if SMOKE_LOG_DIRECTORY.exists() else set()

    sys.argv = [
        "run_daily_decision.py",
        "--state-path", str(SMOKE_STATE_PATH),
        "--decision-log-directory", str(SMOKE_LOG_DIRECTORY),
        "--guard-path", str(SMOKE_GUARD_PATH),
        "--enable-crypto-orders",
    ]
    runner.main()

    after = set(SMOKE_LOG_DIRECTORY.glob("decision_*.json"))
    new_files = sorted(after - before)
    if not new_files:
        raise RuntimeError(
            f"{run_label}: main() returned without writing a new decision log -- it must have "
            f"exited early (STOP flag, FREEZE flag, or a preflight check) rather than actually "
            f"calling run_daily_decision(). Check stdout above for which."
        )
    return json.loads(new_files[-1].read_text(encoding="utf-8"))


def main() -> int:
    _cleanup()  # in case a previous interrupted run left something behind
    _suppress_real_notifications()
    wrote_own_pass_gate = _ensure_pass_gate_for_today()

    try:
        seeded_signal = PortfolioSignal(
            timestamp=datetime.now(UTC).isoformat(),
            ticker=SMOKE_TICKER,
            action="BUY",
            reference_price=0.0,
            score=100.0,
            reason="SMOKE_TEST_SEEDED_SIGNAL",
        )
        seeded_state = LiveRunnerState(
            portfolio_bar_index=0,
            pending_buys={
                SMOKE_TICKER: _PendingOrder(signal=seeded_signal, submitted_portfolio_bar_index=-1)
            },
        )
        save_position_state(seeded_state, SMOKE_STATE_PATH, guard_path=SMOKE_GUARD_PATH)
        print(f"Seeded a pending BUY for {SMOKE_TICKER} in throwaway state: {SMOKE_STATE_PATH}")

        print("\n=== RUN 1: entry (via the real, blessed run_daily_decision.main()) ===")
        decision_1 = _run_main_and_load_decision("RUN 1")
        print(f"portfolio_bar_index after run 1: {decision_1['portfolio_bar_index']}")
        print(f"crypto_order_actions: {decision_1['crypto_order_actions']}")
        print(f"needs_manual_review: {decision_1['needs_manual_review']}")
        print(f"open_positions_after_run: {decision_1['open_positions_after_run']}")

        buy_actions = [a for a in decision_1["crypto_order_actions"] if a["action"] == "BUY"]
        if not buy_actions or buy_actions[0]["status"] != "filled":
            print("\nEntry did not confirm filled -- stopping here without attempting an exit.")
            return 1

        entry_quantity = decision_1["open_positions_after_run"][SMOKE_TICKER]["quantity"]
        entry_price = decision_1["open_positions_after_run"][SMOKE_TICKER]["entry_price"]
        print(f"\nEntry filled: {entry_quantity} {SMOKE_TICKER} @ {entry_price}")

        # Seed a signal-based EXIT for the NEXT run, exactly as
        # `_queue_close_based_exits` would have (a real TREND_RSI exit
        # signal is not guaranteed to occur on demand, so this is seeded
        # the same way the entry was, to exercise the real
        # `_execute_pending_exits_at_open` path deterministically).
        state_after_run_1 = runner.load_position_state(SMOKE_STATE_PATH, guard_path=SMOKE_GUARD_PATH)
        exit_signal = PortfolioSignal(
            timestamp=datetime.now(UTC).isoformat(),
            ticker=SMOKE_TICKER,
            action="EXIT",
            reference_price=entry_price,
            reason="SMOKE_TEST_SEEDED_EXIT_SIGNAL",
        )
        state_after_run_1.pending_exits[SMOKE_TICKER] = _PendingOrder(
            signal=exit_signal,
            submitted_portfolio_bar_index=state_after_run_1.portfolio_bar_index,
        )
        save_position_state(state_after_run_1, SMOKE_STATE_PATH, guard_path=SMOKE_GUARD_PATH)
        print(f"\nSeeded a pending signal-based EXIT for {SMOKE_TICKER}.")

        print("\n=== RUN 2: signal-based exit (via the real, blessed run_daily_decision.main()) ===")
        decision_2 = _run_main_and_load_decision("RUN 2")
        print(f"portfolio_bar_index after run 2: {decision_2['portfolio_bar_index']}")
        print(f"crypto_order_actions: {decision_2['crypto_order_actions']}")
        print(f"needs_manual_review: {decision_2['needs_manual_review']}")
        print(f"executed_today.exits: {decision_2['executed_today']['exits']}")
        print(f"open_positions_after_run: {decision_2['open_positions_after_run']}")

        sell_actions = [a for a in decision_2["crypto_order_actions"] if a["action"] == "SELL"]
        exit_ok = bool(sell_actions) and sell_actions[0]["status"] == "filled"
        print(f"\nExit filled: {exit_ok}")

        print("\nVerifying no BTC-USD position lingers in the real paper account...")
        client = order_submission.get_trading_client()
        try:
            position = client.get_open_position("BTCUSD")
            print(f"  WARNING: a BTCUSD position still exists: {position.qty_available}")
        except Exception as error:  # Alpaca's APIError for "position does not exist"
            print(f"  confirmed clean: {error}")

        return 0 if exit_ok else 1
    finally:
        _cleanup()
        if wrote_own_pass_gate:
            _remove_own_pass_gate()
        print(f"\nThrowaway state/log/guard files removed. Real state at "
              f"{runner.DEFAULT_STATE_PATH} was never touched.")


if __name__ == "__main__":
    raise SystemExit(main())
