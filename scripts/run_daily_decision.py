"""Daily decision runner for the frozen TREND_RSI portfolio strategy.

WHAT THIS SCRIPT DOES: reads persisted position/pending-order state,
reads the real Alpaca paper account's cash balance, fetches today's
live market data, and calls the batch backtest engine's own per-bar
step functions — unmodified, in their documented order — for exactly
one bar (today). It then writes a human-readable log of what those
steps decided and persists the updated state for tomorrow's run.

Real order placement is OPT-IN per asset class, via two independent
flags. By default (neither flag passed) every ticker stays exactly as
in the original dry-run design: nothing is submitted anywhere, the
decision log only describes what would happen.
- `--enable-equity-orders` submits real Alpaca PAPER orders (never
  live/production — `order_submission.py` hardcodes `paper=True`) for
  the equity tickers only (see `src/live/live_universe.LIVE_CONTROLLED_TICKERS`
  for the current live-trading universe, independent of the frozen
  research universe).
- `--enable-crypto-orders` submits real Alpaca PAPER orders for the
  two crypto tickers (BTC-USD, ETH-USD) only. Crypto has no native
  stop order (see the Phase 2 order-type investigation), so real-time
  stop protection lives in the separate `src/live/crypto_stop_monitor.py`
  job, not here — this script only ever submits a crypto market BUY
  for a new entry or a market SELL for a signal-based
  (`EXIT_SIGNAL_NEXT_OPEN`) exit. A crypto STOP_LOSS/GAP_STOP_LOSS
  exit detected by the daily bar-based check NEVER gets a real sell
  from here either (see below) — exactly the same "the intended
  real-time enforcer already handled it, don't guess a duplicate"
  principle as equity's native stop, just with a different
  reconciliation target (the monitor's own activity, not a broker
  order id, since crypto has none to query).
Each flag is fully independent of the other.

Private-API coupling — six functions imported directly from
`src.backtest.portfolio_backtest_engine` and called in the exact order
the batch loop uses them (see that module's `run_portfolio_backtest`,
and docs/BASELINE.md's "Protected portfolio event ordering"):

    _execute_pending_exits_at_open  (step 1)
    _check_gap_stops                (step 2)
    _execute_pending_buys_at_open   (step 3)
    _check_intrabar_stops           (step 4)
    _queue_close_based_exits        (step 5)
    _queue_ranked_entry_signals      (step 6)

Plus `_PortfolioState` itself (the mutable container these six
functions all read/write in place). None of these names are exported
from the module (all are underscore-prefixed); this script treats them
as a stable-enough internal contract to reuse rather than re-implement
the frozen strategy's entry/exit/stop/risk logic a second time, at the
cost of being coupled to `portfolio_backtest_engine.py`'s private
internals. If that module's private functions are ever renamed or
their signatures change, this script breaks loudly (ImportError /
TypeError) rather than silently drifting from the approved baseline —
which is the trade-off intentionally accepted here. `portfolio_backtest_engine.py`
itself is never edited by this script or anything it imports.

Next-available-Open semantics carry over exactly as in the batch
engine: a BUY/EXIT signal generated from TODAY's Close (steps 5-6)
does not execute today. It is persisted as a pending order and will
only execute on the NEXT run, at that day's Open (steps 1-4), once
`submitted_portfolio_bar_index < portfolio_bar_index` holds.

Real-order design, equity-only (see src/live/order_submission.py for
the order-type rationale):
- A newly opened equity position submits a real market BUY. Once that
  order confirms `filled`, a native Alpaca `stop` sell is placed
  immediately at the engine's computed `stop_loss_price`.
- A position closed by `_queue_close_based_exits` (exit_reason
  `EXIT_SIGNAL_NEXT_OPEN`) submits a real market SELL, then cancels
  that ticker's resting protective stop — otherwise the stop would
  keep resting against a position we no longer hold.
- A position closed by `_check_gap_stops` / `_check_intrabar_stops`
  (exit_reason `STOP_LOSS` / `GAP_STOP_LOSS`) NEVER gets a second real
  sell order from here. The resting native stop is the thing that is
  supposed to have already handled that exit at the broker,
  independent of this job's schedule; this script only reconciles
  local bookkeeping against that stop order's real status and surfaces
  a `needs_manual_review` entry if it does not show as filled.
- Idempotency, two independent layers, neither claimed sufficient
  alone (an independent 2026-08-14 review correctly pushed back on an
  earlier version of this note that overclaimed the date key alone as
  "structurally impossible to collide" -- it narrows the collision
  window a lot, but a same-day retry, a crash between a real fill and
  the next save, or a corrupted/never-written ledger entry could still
  create one):
  1. `runner_state.submitted_actions` is keyed by
     `f"{ticker}|{kind}|{date}"` (the real calendar date -- equity's
     own for equity checkpoints, crypto's own for crypto checkpoints,
     see `_bars_today_by_asset_class`) rather than
     `portfolio_bar_index`, which a real incident showed can jump
     backward (`data/live/` overwritten by a stale rsync copy) and
     collide with a stale entry from before the rollback -- confirmed
     via a real-code-path stress test. A calendar date is far less
     likely to recur than an integer counter can, but "far less
     likely" is the honest claim, not "cannot."
  2. The actual authority on whether to submit: every checkpoint now
     builds a deterministic `client_order_id` and queries Alpaca for it
     BEFORE submitting (`_resolve_or_submit_order`). This is what
     closes the gap layer 1 leaves open -- even a lost or rolled-back
     local ledger entry cannot cause a real duplicate, because the
     decision no longer depends on this file's own memory of what
     happened; it depends on what the broker itself confirms.
  `src/live/position_state.py`'s `RollbackDetectedError` is a third,
  separate layer (refuses to run at all on a detected rollback) --
  also real, also partial, not a substitute for either of the above.
- Known limitation, accepted rather than solved with a heavier
  two-phase-commit design: a newly opened position is recorded in
  local state as soon as the engine decides to open it, without
  waiting for real fill confirmation (mirroring the batch engine's own
  assumption of an immediate, guaranteed fill). If the real market
  order does not confirm filled within the short poll window (e.g.
  market closed), the position stays in local state but its
  protective stop is deferred to the next run's reconciliation pass
  (`_reconcile_pending_equity_orders`), which also flags a rejected/
  canceled/expired entry order for manual review rather than silently
  guessing what happened.

Real-order design, crypto-only:
- Entry and signal-based exit both submit a market order with
  `time_in_force=IOC` (crypto rejects `DAY`; only `gtc`/`ioc` are
  valid — Phase 2 order-type investigation). IOC over GTC because this
  job decides "now" — it should not leave a resting order behind for
  a later, unrelated moment to fill.
- No protective-stop placement step (unlike equity): there is nothing
  to place. Real-time stop protection is the separate
  `crypto_stop_monitor.py` job, which reads `stop_loss_price` straight
  from this same persisted state file.
- Sell quantity is queried from the broker
  (`crypto_stop_monitor._query_available_crypto_quantity`, the exact
  same safe-quantity helper the monitor uses), not taken from the
  engine's `trade.quantity` — crypto fees can be deducted in-kind, so
  local bookkeeping can overstate what is actually sellable (confirmed
  with a real paper fill in the monitor's own development). If the
  broker reports nothing sellable, no sell is submitted; it goes to
  `needs_manual_review` instead of guessing a quantity.
- Concurrency with the monitor: both this script and
  `crypto_stop_monitor.py` can decide to touch the same crypto position
  (a new signal here vs. a stop breach there). Before submitting any
  real crypto order, this script takes the monitor's own lock file
  (`data/live/crypto_monitor.lock`, same `os.O_CREAT|O_EXCL` primitive),
  retrying a few times a couple of seconds apart rather than either
  failing outright or blocking indefinitely. The monitor's own
  check-and-execute cycle is short (typically one API call; at most a
  handful of seconds if a trade actually needs to be submitted and
  polled for a fill), so a brief retry resolves the large majority of
  real overlaps without meaningfully delaying this job. If every retry
  still finds the lock held, this script does NOT wait indefinitely or
  fail the whole run (equity is unaffected either way) — it skips real
  crypto order submission for this run only and reports every affected
  ticker in `needs_manual_review`, so a human can submit it by hand or
  simply wait for the next scheduled run. Local bookkeeping (position
  open/closed in `runner_state.positions`) is not reverted in that
  case — it already reflects the engine's decision, same as the
  deferred-fill case above; only the *real order* is what got skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order

from src.backtest.portfolio_backtest_engine import (
    _MutablePosition,
    _PortfolioState,
    _check_gap_stops,
    _check_intrabar_stops,
    _execute_pending_buys_at_open,
    _execute_pending_exits_at_open,
    _is_crypto_ticker,
    _queue_close_based_exits,
    _queue_ranked_entry_signals,
)
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig, PortfolioTrade
import src.live.broker_reconciliation as broker_reconciliation
from src.live import order_submission
from src.live.authorized_execution_context import (
    AuthorizedExecutionContext,
    authorize_order_execution,
    require_authorized_execution_context,
)
from src.live.account_state import get_live_cash_balance
from src.live.crypto_stop_monitor import DEFAULT_LOCK_PATH, _query_available_crypto_quantity
from src.live.data_preparer import prepare_live_market_data
from src.live.live_universe import LIVE_CONTROLLED_TICKERS
import src.live.position_state as ps
from src.live.position_state import (
    DEFAULT_STATE_PATH,
    LiveRunnerState,
    RollbackDetectedError,
    load_position_state,
    save_position_state,
)
from src.live.single_instance_lock import single_instance_lock
from src.notify.telegram_notifier import send_telegram_message

DEFAULT_DECISION_LOG_DIRECTORY = Path("data/live/decisions")

# Emergency two-tier switch: empty files, content never read, only
# existence checked. Both are checked in main() before any Alpaca/data
# API call. See docs/EMERGENCY_STOP_RUNBOOK.md for exactly what to type
# in a crisis -- this exists because SSH + hand-editing crontab is too
# slow and error-prone to trust under pressure.
#
# STOP_FLAG_PATH: full stop. main() returns before touching anything --
# no cash balance, no market data, no client. If both flags are present,
# STOP wins (checked first, unconditionally).
#
# FREEZE_FLAG_PATH: only new-position-opening stops (skipping steps 3 and
# 6 below); data fetch and existing-position exit/stop monitoring (steps
# 1, 2, 4, 5) run exactly as normal, so an open position never loses its
# stop-loss coverage during a freeze. run_crypto_stop_monitor.py is
# deliberately NOT gated on this flag for the same reason (see that
# script's STOP_FLAG_PATH comment).
STOP_FLAG_PATH = Path("data/live/STOP")
FREEZE_FLAG_PATH = Path("data/live/FREEZE")

# Live-only risk-parameter override, deliberately NOT a change to
# PortfolioBacktestConfig's own defaults (portfolio_backtest_models.py).
# Those defaults are hash-locked: run_portfolio_research_baseline_lock.py's
# verify_runtime_config() calls PortfolioBacktestConfig().to_dict() with no
# arguments and compares every field, including these two, against
# config/research_baseline_lock_v1.json's frozen baseline_config -- editing
# the shared dataclass default would break that certification for every
# research module that also relies on PortfolioBacktestConfig() (confirmed
# by reading verify_runtime_config, not assumed). This mirrors the same
# live/research split already established for the ticker universe
# (LIVE_CONTROLLED_TICKERS vs. CONTROLLED_TICKERS): only this script's own
# config construction below uses these values; every research module's
# bare PortfolioBacktestConfig() is completely unaffected.
#
# Values: research (2026-08-13, see the position-sizing formula
# investigation) found maximum_open_positions=4 and
# maximum_total_open_risk_percent=4.0% were already mutually redundant
# under the frozen defaults (risk_per_trade_percent=1% x 4 positions = the
# same 4% ceiling either cap would hit first). Raised together to 6/6.0%
# as a deliberate first step before widening the live ticker universe
# (Phase 4), so the risk-parameter change and the universe-size change are
# never tested confounded together. risk_per_trade_percent is unchanged.
LIVE_MAXIMUM_OPEN_POSITIONS = 6
LIVE_MAXIMUM_TOTAL_OPEN_RISK_PERCENT = 6.0

# Asset class + exit_reason -> recommended live order type, per the
# Phase 2 order-type investigation. Entries are always "market" (the
# engine never uses a limit price on entry, see that investigation's
# finding). Trend/trailing exits (EXIT_SIGNAL_NEXT_OPEN) are a
# scheduled next-Open sell, not a resting-stop scenario, so "market"
# applies there for both asset classes. STOP_LOSS/GAP_STOP_LOSS is the
# case with an asset-class split: equities have a native Alpaca `stop`
# order that mirrors the engine's guaranteed-fill assumption; crypto
# has no plain `stop` order (only `stop_limit`), so the closest
# faithful equivalent is our own monitoring loop + a market order.
_STOP_EXIT_REASONS = {"STOP_LOSS", "GAP_STOP_LOSS"}

# Broker-authoritative idempotency (2026-08-14, responding to an
# independent review's HIGH-severity finding: a date-keyed
# `submitted_actions` entry is not a hard guarantee against collision --
# same-day retries, a crash between a real fill and the next save, or a
# corrupted/lost local ledger could all still let a genuinely-new
# action attempt collide with, or fail to recognize, a real prior
# submission). Every new-order checkpoint below now builds a
# deterministic `client_order_id` and asks Alpaca -- not
# `submitted_actions` -- whether it already exists BEFORE submitting.
# The local ledger is still written (for the decision log and
# `_reconcile_pending_equity_orders`), but it is a record of what
# happened, not the authority on whether to submit -- the broker is.
_CLIENT_ORDER_ID_MAX_LENGTH = 128


def _deterministic_client_order_id(*parts: str) -> str:
    """Builds a stable id from `parts` -- calling this again with the
    exact same parts (a retry, a crash-restart, a re-run for the same
    day) always produces the same string, which is what makes the
    broker query below meaningful: it is asking "has THIS exact intent
    already happened," not merely "has any order for this ticker
    happened." `/` (crypto symbols) is not valid in Alpaca's
    client_order_id, hence the substitution."""
    sanitized = "-".join(part.replace("/", "-") for part in parts)
    return sanitized[:_CLIENT_ORDER_ID_MAX_LENGTH]


# REAL BUG FOUND AND FIXED (2026-08-22, independent audit finding #2):
# scripts/run_control_arm_decision.py's write-ahead intent journal built
# its own "blind" client_order_id using the literal action-kind string
# "ENTRY_MARKET_BUY" (underscore -- the same spelling order_intent.py
# uses for its `action_kind` field), while this file's own real-order
# call sites below built theirs using "ENTRY-MARKET-BUY" (dash) --
# `_deterministic_client_order_id` only substitutes "/", never "_", so
# these produced two DIFFERENT strings for the exact same real
# ticker+action+date. The journal's own client_order_id therefore never
# matched what was actually submitted to Alpaca -- any future broker-side
# correlation (Scenario C's client_order_id match in
# broker_reconciliation.py, or a human cross-checking the journal against
# the broker) would silently fail to find it. ONE canonical mapping from
# the semantic action_kind (the spelling order_intent.py's own state
# machine and `_PENDING_SIGNAL_ACTION_KIND_FAMILIES` already use) to the
# exact literal segment fed into `_deterministic_client_order_id` closes
# this -- every real-order call site below, and every caller building a
# journal id for one of these two action kinds (see
# run_control_arm_decision.py's `_write_blind_prepared_intents`/
# `_run_intent_protocol`/`_verify_write_ahead_evidence_before_broker_call`),
# now goes through `client_order_id_for_action` instead of hand-choosing a
# literal. Only the two action kinds that ever reach a real broker order
# are listed -- QUEUED_ENTRY_SIGNAL/QUEUED_EXIT_SIGNAL (pending-signal
# tracking ids) have no broker-side counterpart to stay consistent with,
# and keep using `_deterministic_client_order_id` directly.
_ACTION_KIND_ORDER_ID_SEGMENT: dict[str, str] = {
    "ENTRY_MARKET_BUY": "ENTRY-MARKET-BUY",
    "SIGNAL_EXIT_MARKET_SELL": "SIGNAL-EXIT-MARKET-SELL",
}


def client_order_id_for_action(ticker: str, action_kind: str, date: str) -> str:
    """The one canonical id-building call for any action kind that can
    reach a real broker order -- see `_ACTION_KIND_ORDER_ID_SEGMENT`'s own
    comment for the real cross-file id-mismatch bug this closes. Raises
    `KeyError` (fail-closed, not a silent fallback to the raw
    `action_kind` string) if `action_kind` is not one of the known,
    broker-reaching kinds -- exactly the same "fail closed on an
    unrecognized kind" discipline `broker_reconciliation._resolve_scenario_c`
    already uses for its own `_KNOWN_ORDER_SIDES_BY_KIND` lookup."""
    segment = _ACTION_KIND_ORDER_ID_SEGMENT[action_kind]
    return _deterministic_client_order_id(ticker, segment, date)


def _deterministic_operation_id(ticker: str, action_kind: str, target_broker_order_id: str) -> str:
    """Task 3 (2026-08-22): the CANCEL-side counterpart to
    `client_order_id_for_action`. A cancel is never itself a broker
    order -- Alpaca's cancel API has no client_order_id of its own to
    generate or accept -- so this builds a LOCAL-only, deterministic
    journal correlation key instead (`OrderIntent.operation_id`, never
    sent to the broker). Keyed off the TARGET order's own real broker
    id (the resting stop being canceled), the same "immutable reference
    that exists the moment the thing it protects/targets is known"
    discipline `_reconcile_pending_equity_orders`'s own protective-stop
    keying already uses -- not a date, so a retry/crash-restart
    targeting the same resting order always reproduces the same id."""
    return _deterministic_client_order_id(ticker, action_kind, target_broker_order_id)


def _order_status_str(order: Order) -> str:
    status = order.status
    return str(status.value if hasattr(status, "value") else status)


# INJECTION-SEAM CALLBACK (2026-08-22, independent audit finding #3,
# approach (B) -- explicitly chosen over a direct `order_intent` import
# here: order_intent.py's own module docstring states "nothing here is
# wired into run_daily_decision.py (never modified) or any real
# order-submission code path" -- a direct import would break that
# documented boundary. This optional callback keeps the same one-directional
# dependency `trading_client: TradingClient | None = None` already
# establishes elsewhere in this file (see run_daily_decision()'s own
# docstring): this module defines the seam and calls it if given one, but
# never imports or knows anything about order_intent.py itself.
#
# THE REAL GAP THIS CLOSES: `scripts/run_control_arm_decision.py`'s
# `_run_intent_protocol` only transitions PREPARED -> SUBMITTING ->
# BROKER_ACKNOWLEDGED AFTER `run_daily_decision()` returns -- i.e. after
# every real broker call inside `_execute_equity_orders`/
# `_execute_crypto_orders` has already happened. A crash DURING one of
# those real broker calls leaves the on-disk journal at PREPARED for
# every candidate, with no way to tell which one (if any) actually
# reached the broker before the crash. Calling this hook immediately
# BEFORE each real broker call/query, and again immediately after it
# resolves, lets a caller (the control arm) durably record SUBMITTING
# right at the moment a broker call is about to happen -- for the exact
# ticker/action about to be attempted, not a blanket pre-run guess.
#
# CONTRACT (extended 2026-08-22, Task 3; extended again 2026-08-23,
# independent-audit-round-2 finding #3 -- see PROTECTIVE_STOP/
# CANCEL_STOP call sites below for why): `hook(phase, *, ticker,
# action_kind, client_order_id, order=None, operation_id=None,
# side=None, order_type=None, quantity=None, notional=None,
# stop_price=None, parent_order_id=None, broker_status=None)`.
# `phase` is `"SUBMITTING"` (called BEFORE the broker is queried/called
# for this client_order_id/operation_id -- may fire even when the order
# turns out to already exist, since the query itself is what
# "SUBMITTING" durably guards against a crash during) or
# `"BROKER_ACKNOWLEDGED"` (called AFTER that query/call resolves
# successfully, `order` is the real `Order` when one exists).
#
# `broker_status` (added 2026-08-23): a plain string status to use when
# NO `Order` object exists to derive one from -- the CANCEL_STOP call
# site below is the one caller that needs this: Alpaca's cancel API
# returns no `Order`, only the plain status string
# `order_submission.cancel_order_and_confirm` already returns
# (`cancel_status`). Before this parameter existed, that real status was
# fetched but never threaded through to the hook, so a cancel intent's
# own `broker_status` field landed `None` even though the real status
# was known and already recorded elsewhere (`submitted_actions`). A
# caller that also has a real `order` object should pass that instead
# and leave this `None` -- see `_order_intent_hook_for_run`'s own
# `hook()` for which one wins when both could theoretically be supplied
# (this one does, since it is the more specific, deliberately-supplied
# value for the callers that have no `Order` at all).
# Never called on a raised exception -- the exception propagates uncaught
# exactly as before this parameter existed; a SUBMITTING-with-no-following-
# BROKER_ACKNOWLEDGED journal entry left behind by a real crash is exactly
# the durable evidence this hook exists to create, not a bug in the hook.
# `None` (the default, every call site in this file that does not pass one
# explicitly) means "no journal integration" -- zero behavior change for
# any caller that does not opt in.
#
# `client_order_id` vs `operation_id`: a real order submission
# (ENTRY_MARKET_BUY/SIGNAL_EXIT_MARKET_SELL/PROTECTIVE_STOP) always has
# a `client_order_id` and leaves `operation_id` `None`. A CANCEL has no
# client_order_id of its own (Alpaca's cancel API doesn't take/generate
# one) -- it passes `client_order_id=""` and a real `operation_id`
# instead (see `_deterministic_operation_id`). A caller must check which
# one is populated, never assume `client_order_id` alone identifies the
# call.
#
# `side`/`order_type`/`quantity`/`notional`/`stop_price`/`parent_order_id`
# -- the real order attributes, passed through so a hook building a
# durable journal entry (see `run_control_arm_decision._order_intent_hook_for_run`)
# has everything needed to create one JUST-IN-TIME, right before this
# call, for an action kind (PROTECTIVE_STOP, CANCEL_PROTECTIVE_STOP)
# that -- unlike ENTRY_MARKET_BUY/SIGNAL_EXIT_MARKET_SELL -- is decided
# reactively, mid-run, with no pre-run "blind" candidate to have already
# journaled these values ahead of time. `parent_order_id` is the
# immutable id of the order this action protects/targets (the ENTRY
# fill a PROTECTIVE_STOP defends, or the resting stop a
# CANCEL_PROTECTIVE_STOP targets) -- the same anchor
# `_reconcile_pending_equity_orders`'s own protective-stop keying
# already uses, never a date.
OrderIntentHook = Callable[..., None]


def _call_order_intent_hook(
    hook: OrderIntentHook | None,
    phase: str,
    *,
    ticker: str,
    action_kind: str,
    client_order_id: str = "",
    order=None,
    operation_id: str | None = None,
    side: str | None = None,
    order_type: str | None = None,
    quantity: float | None = None,
    notional: float | None = None,
    stop_price: float | None = None,
    parent_order_id: str | None = None,
    broker_status: str | None = None,
) -> None:
    if hook is not None:
        hook(
            phase, ticker=ticker, action_kind=action_kind, client_order_id=client_order_id, order=order,
            operation_id=operation_id, side=side, order_type=order_type, quantity=quantity,
            notional=notional, stop_price=stop_price, parent_order_id=parent_order_id,
            broker_status=broker_status,
        )


def _resolve_or_submit_order(
    client: TradingClient,
    *,
    client_order_id: str,
    submit,
    order_intent_hook: OrderIntentHook | None = None,
    ticker: str = "",
    action_kind: str = "",
    authorization: AuthorizedExecutionContext | None = None,
    side: str | None = None,
    order_type: str | None = None,
    quantity: float | None = None,
    notional: float | None = None,
    stop_price: float | None = None,
    parent_order_id: str | None = None,
) -> tuple[Order, bool]:
    """The actual fix: check the broker BEFORE calling `submit`, via
    `order_submission.get_order_by_client_order_id` (a plain,
    monkeypatchable module function, not a bare method call on
    `client` -- see that function's own docstring for the 404-only
    contract). If an order with this `client_order_id` already exists,
    `submit` is never invoked -- a lost, never-written, or rolled-back
    local ledger entry can no longer cause a real duplicate, because
    the decision no longer depends on the local ledger at all. Returns
    (order, already_existed).

    `order_intent_hook`/`ticker`/`action_kind` -- see `OrderIntentHook`'s
    own module-level comment. `SUBMITTING` fires before the broker query
    below; `BROKER_ACKNOWLEDGED` fires after it resolves, whether the
    order already existed or was freshly submitted -- both are a real,
    confirmed broker response, and either way this function's own crash
    window (the one this hook exists to narrow) has closed by the time
    either return path is reached.

    `side`/`order_type`/`quantity`/`notional`/`stop_price`/`parent_order_id`
    (added 2026-08-22, Task 3) -- passed straight through to the hook,
    unused by this function itself; see `OrderIntentHook`'s own
    module-level comment for why a hook building a just-in-time journal
    entry (PROTECTIVE_STOP has no pre-run "blind" candidate) needs
    these.

    `authorization` -- independent audit finding "Madde E" (2026-08-22):
    this is EVERY real order's shared choke point (entry BUY, protective
    stop, equity signal-exit SELL, crypto entry BUY all funnel through
    here), so this is where the second, inner-function-level
    authorization check lives -- a caller that reaches this function
    directly, bypassing `run_daily_decision()`'s own top-level check, is
    still refused here, before any broker call. See
    `src.live.authorized_execution_context`'s own module docstring for
    the full design and why this is scoped to the bare-call bypass, not
    to `run_daily_decision.py`'s own `main()`. (Task 3's own "hook
    required in control-arm real-order mode" fail-closed check
    deliberately does NOT live here -- see
    `run_control_arm_decision.py`'s own preflight for why putting it in
    this shared, caller-agnostic function would have broken the live
    system's own `main()`, which self-authorizes but never supplies a
    hook.)"""
    require_authorized_execution_context(authorization, action_description=f"Submitting order ({action_kind} {ticker})")
    _call_order_intent_hook(
        order_intent_hook, "SUBMITTING", ticker=ticker, action_kind=action_kind, client_order_id=client_order_id,
        side=side, order_type=order_type, quantity=quantity, notional=notional, stop_price=stop_price,
        parent_order_id=parent_order_id,
    )
    existing = order_submission.get_order_by_client_order_id(client, client_order_id)
    if existing is not None:
        _call_order_intent_hook(
            order_intent_hook, "BROKER_ACKNOWLEDGED",
            ticker=ticker, action_kind=action_kind, client_order_id=client_order_id, order=existing,
            side=side, order_type=order_type, quantity=quantity, notional=notional, stop_price=stop_price,
            parent_order_id=parent_order_id,
        )
        return existing, True
    order = submit()
    _call_order_intent_hook(
        order_intent_hook, "BROKER_ACKNOWLEDGED",
        ticker=ticker, action_kind=action_kind, client_order_id=client_order_id, order=order,
        side=side, order_type=order_type, quantity=quantity, notional=notional, stop_price=stop_price,
        parent_order_id=parent_order_id,
    )
    return order, False


def _recommended_exit_order_type(*, exit_reason: str, asset_class: str) -> str:
    if exit_reason not in _STOP_EXIT_REASONS:
        return "market"
    return "stop (native)" if asset_class == "EQUITY" else "monitoring_loop+market"


def _bars_today(prepared: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.Series], str]:
    """Return each ticker's last row, keyed by its own native timestamp.

    Alpaca stamps daily equity bars at ~04:00-05:00 UTC (US market
    midnight, DST-dependent) and daily crypto bars at 00:00 UTC —
    different clock times for the same calendar trading day. Comparing
    exact Timestamps across asset classes therefore fails even when
    the data genuinely lines up; comparing calendar dates is the
    correct check.

    Callers must pass tickers from a single asset class only — see
    `_bars_today_by_asset_class`. Equity and crypto trade on
    independent schedules (crypto is 24/7; equity closes on weekends
    and holidays) and are never expected to share a date with each
    other, only with tickers in their own asset class.
    """
    end_dates = {ticker: frame.index[-1].date() for ticker, frame in prepared.items()}
    if len(set(end_dates.values())) != 1:
        raise ValueError(f"Prepared tickers do not share a common latest calendar date: {end_dates}")
    today_date = next(iter(end_dates.values()))
    bars = {ticker: frame.iloc[-1] for ticker, frame in prepared.items()}
    return bars, today_date.isoformat()


def _bars_today_by_asset_class(
    prepared: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.Series], str, str]:
    """Validate equity and crypto tickers' latest calendar date independently.

    The original design required ALL nine controlled tickers (seven
    equity + two crypto) to share one common latest calendar date. That
    is wrong: US equities close on weekends and holidays while crypto
    trades 24/7, so equity's latest bar is structurally behind crypto's
    on every one of those days — a `ValueError` here on every single
    weekend/holiday run, before any order logic even runs. The correct
    invariant is narrower: equity tickers must agree with EACH OTHER,
    and crypto tickers must agree with EACH OTHER, but the two asset
    classes never need to agree with one another.

    The six per-bar engine step functions do not use `timestamp` for
    any date arithmetic or comparison — it is only ever stored as a
    display label on trade/signal/equity-curve records (confirmed by
    reading `portfolio_backtest_engine.py`). This means the two
    validated groups' bars can be merged back into one combined
    `bars_today` dict — exactly the shape the six steps already expect,
    unchanged from before — and run through a SINGLE combined call per
    step, preserving the engine's existing cross-asset-class entry
    ranking and `maximum_open_positions` cap behavior exactly as-is.
    Splitting into two separate calls (equity's steps, then crypto's)
    would silently change that shared-cap tie-break order — out of
    scope for a data-preparation fix, and never done here.

    Returns (bars_today, equity_date, crypto_date) — the merged bars
    dict for the six step calls, plus each group's own validated date
    string (identical on every non-weekend, non-equity-holiday day).
    """
    equity_prepared = {t: f for t, f in prepared.items() if not _is_crypto_ticker(t)}
    crypto_prepared = {t: f for t, f in prepared.items() if _is_crypto_ticker(t)}
    if not equity_prepared:
        raise ValueError("No equity tickers found in prepared market data.")
    if not crypto_prepared:
        raise ValueError("No crypto tickers found in prepared market data.")

    equity_bars, equity_date = _bars_today(equity_prepared)
    crypto_bars, crypto_date = _bars_today(crypto_prepared)
    return {**equity_bars, **crypto_bars}, equity_date, crypto_date


def _build_state(
    *,
    runner_state: LiveRunnerState,
    cash: float,
    bars_today: dict[str, pd.Series],
) -> _PortfolioState:
    last_prices = {ticker: float(row["Open"]) for ticker, row in bars_today.items()}
    positions_value = sum(
        position.quantity * last_prices.get(ticker, position.entry_price)
        for ticker, position in runner_state.positions.items()
    )
    starting_equity = cash + positions_value
    return _PortfolioState(
        cash=cash,
        positions=dict(runner_state.positions),
        pending_buys=dict(runner_state.pending_buys),
        pending_exits=dict(runner_state.pending_exits),
        last_prices=last_prices,
        trades=[],
        rejections=[],
        equity_curve=[],
        realized_pnl=0.0,
        peak_equity=starting_equity,
        maximum_drawdown_amount=0.0,
        maximum_drawdown_percent=0.0,
        skipped_signals=0,
        maximum_open_positions_observed=len(runner_state.positions),
    )


def _require_live_account_suffix() -> str:
    """MANDATORY, fail-closed (2026-08-22, independent audit finding
    #1): a missing, blank, or malformed `LIVE_ACCOUNT_NUMBER_SUFFIX`
    must refuse to proceed, never silently skip account-identity
    verification. `broker_reconciliation.verify_account_identity`'s own
    `expected_suffix=None`-or-falsy skip-and-warn path is a generic
    library capability for callers that legitimately want it -- this
    caller (the real live trading entrypoint) is not one of them, and
    enforces its own strict policy here rather than relying on that
    library default. Real Alpaca account numbers observed elsewhere in
    this codebase (`_EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX = "XO4Y"`)
    are exactly 4 characters -- enforced here, not assumed to always be
    4 by the underlying library function itself (which accepts any
    non-empty string, deliberately, since a caller could theoretically
    have a different-length real value)."""
    raw = os.environ.get("LIVE_ACCOUNT_NUMBER_SUFFIX")
    suffix = (raw or "").strip()
    if not suffix:
        raise broker_reconciliation.AccountIdentityMismatchError(
            "LIVE_ACCOUNT_NUMBER_SUFFIX is not set (or is blank) in the "
            "environment. Real account-identity verification is MANDATORY "
            "for this live entrypoint -- refusing to proceed rather than "
            "silently skipping the check. Set it to the real account's "
            "masked 4-character suffix before running this script."
        )
    if len(suffix) != 4:
        raise broker_reconciliation.AccountIdentityMismatchError(
            f"LIVE_ACCOUNT_NUMBER_SUFFIX is set but is not exactly 4 "
            f"characters (got length {len(suffix)}). Refusing to proceed -- "
            f"a malformed expected suffix is as dangerous as a missing one "
            f"(it could accidentally match, or never match, an unrelated "
            f"account). The raw value is never logged here."
        )
    return suffix


def _reconcile_pending_equity_orders(
    client: TradingClient,
    runner_state: LiveRunnerState,
    *,
    order_intent_hook: OrderIntentHook | None = None,
    authorization: AuthorizedExecutionContext | None = None,
) -> list[dict]:
    """Catch up on entry orders submitted, but not confirmed, on a previous run.

    The PROTECTIVE_STOP this places used to be keyed by
    `portfolio_bar_index` -- the same rollback-collision exposure the
    other six checkpoints had before the 2026-08-14 fix, flagged as its
    own HIGH-severity finding by an independent review specifically
    because this function runs before `equity_date` even exists yet
    (called ahead of the market-data fetch), so it could not simply
    reuse that fix the way the others did. Fixed here differently, and
    arguably more robustly than a date ever could be: keyed off the
    ENTRY order's own id (`record["order_id"]`) -- an immutable
    reference to "the specific fill this stop protects" that exists
    the moment the entry confirms, with no dependency on what day it
    happens to be reconciled. Also broker-queried by client_order_id
    before submitting, same as every other checkpoint.
    """
    needs_review: list[dict] = []
    for key, record in list(runner_state.submitted_actions.items()):
        if record.get("kind") != "ENTRY_MARKET_BUY":
            continue
        if record.get("status") in order_submission.TERMINAL_STATUSES:
            continue
        ticker = key.split("|", 1)[0]
        status = order_submission.get_order_status(client, record["order_id"])
        record["status"] = status
        if status == "filled":
            position = runner_state.positions.get(ticker)
            if position is not None and ticker not in runner_state.equity_stop_orders:
                stop_key = f"{ticker}|PROTECTIVE_STOP|FOR|{record['order_id']}"
                stop_record = runner_state.submitted_actions.get(stop_key)
                if stop_record is None:
                    stop_client_order_id = _deterministic_client_order_id(
                        ticker, "PROTECTIVE-STOP-FOR", record["order_id"]
                    )
                    stop_order, _ = _resolve_or_submit_order(
                        client,
                        client_order_id=stop_client_order_id,
                        submit=lambda: order_submission.submit_equity_stop_sell(
                            client, ticker=ticker, quantity=position.quantity,
                            stop_price=position.stop_loss_price,
                            client_order_id=stop_client_order_id,
                        ),
                        order_intent_hook=order_intent_hook,
                        ticker=ticker,
                        action_kind="PROTECTIVE_STOP",
                        authorization=authorization,
                        side="SELL",
                        order_type="stop",
                        quantity=position.quantity,
                        stop_price=position.stop_loss_price,
                        parent_order_id=record["order_id"],
                    )
                    stop_record = {
                        "order_id": str(stop_order.id),
                        "status": "new",
                        "kind": "PROTECTIVE_STOP",
                        "submitted_at": order_submission.now_iso(),
                        "client_order_id": stop_client_order_id,
                    }
                    runner_state.submitted_actions[stop_key] = stop_record
                runner_state.equity_stop_orders[ticker] = stop_record["order_id"]
        elif status in {"rejected", "canceled", "expired"}:
            needs_review.append(
                {
                    "ticker": ticker,
                    "issue": "entry order from a previous run did not fill",
                    "status": status,
                    "order_id": record["order_id"],
                }
            )
    return needs_review


def _execute_equity_orders(
    client: TradingClient,
    *,
    runner_state: LiveRunnerState,
    equity_date: str,
    newly_opened: dict[str, _MutablePosition],
    closed_trades: list[PortfolioTrade],
    order_intent_hook: OrderIntentHook | None = None,
    authorization: AuthorizedExecutionContext | None = None,
) -> tuple[list[dict], list[dict]]:
    """Submit real equity orders for today's decisions. Never called for crypto.

    Idempotency, two layers (see the module docstring's fuller
    explanation of why neither is claimed sufficient alone): the local
    `submitted_actions` lookup below uses `equity_date` rather than
    `portfolio_bar_index` (a real incident showed that counter can jump
    backward), which makes a same-key collision far less likely but not
    impossible. The actual authority is `_resolve_or_submit_order`'s
    broker query by deterministic `client_order_id`, called before any
    real submission -- that is what keeps a lost or stale local ledger
    entry from causing a real duplicate, not the key format by itself.
    """
    actions: list[dict] = []
    needs_review: list[dict] = []

    for ticker, position in newly_opened.items():
        if position.asset_class != "EQUITY":
            continue
        action_key = f"{ticker}|ENTRY_MARKET_BUY|{equity_date}"
        record = runner_state.submitted_actions.get(action_key)
        if record is None:
            client_order_id = client_order_id_for_action(ticker, "ENTRY_MARKET_BUY", equity_date)
            order, already_existed = _resolve_or_submit_order(
                client,
                client_order_id=client_order_id,
                submit=lambda: order_submission.submit_equity_market_order(
                    client, ticker=ticker, side=OrderSide.BUY, quantity=position.quantity,
                    client_order_id=client_order_id,
                ),
                order_intent_hook=order_intent_hook,
                ticker=ticker,
                action_kind="ENTRY_MARKET_BUY",
                authorization=authorization,
            )
            status = (
                _order_status_str(order) if already_existed
                else order_submission.wait_for_fill_or_timeout(client, str(order.id))
            )
            record = {
                "order_id": str(order.id),
                "status": status,
                "kind": "ENTRY_MARKET_BUY",
                "submitted_at": order_submission.now_iso(),
                "client_order_id": client_order_id,
            }
            runner_state.submitted_actions[action_key] = record
        else:
            record["status"] = order_submission.get_order_status(client, record["order_id"])
        actions.append({"ticker": ticker, "action": "BUY", **record})

        if record["status"] == "filled" and ticker not in runner_state.equity_stop_orders:
            # Keyed off the ENTRY order's own id, not a date -- the stop's
            # whole identity is "protect THIS fill," an immutable
            # reference that exists the moment the entry confirms, and
            # is a strictly better anchor than any date could be.
            stop_key = f"{ticker}|PROTECTIVE_STOP|{equity_date}"
            stop_record = runner_state.submitted_actions.get(stop_key)
            if stop_record is None:
                stop_client_order_id = _deterministic_client_order_id(
                    ticker, "PROTECTIVE-STOP-FOR", record["order_id"]
                )
                stop_order, _ = _resolve_or_submit_order(
                    client,
                    client_order_id=stop_client_order_id,
                    submit=lambda: order_submission.submit_equity_stop_sell(
                        client, ticker=ticker, quantity=position.quantity,
                        stop_price=position.stop_loss_price,
                        client_order_id=stop_client_order_id,
                    ),
                    order_intent_hook=order_intent_hook,
                    ticker=ticker,
                    action_kind="PROTECTIVE_STOP",
                    authorization=authorization,
                    side="SELL",
                    order_type="stop",
                    quantity=position.quantity,
                    stop_price=position.stop_loss_price,
                    parent_order_id=record["order_id"],
                )
                stop_record = {
                    "order_id": str(stop_order.id),
                    "status": "new",
                    "kind": "PROTECTIVE_STOP",
                    "submitted_at": order_submission.now_iso(),
                    "client_order_id": stop_client_order_id,
                }
                runner_state.submitted_actions[stop_key] = stop_record
            runner_state.equity_stop_orders[ticker] = stop_record["order_id"]
            actions.append({"ticker": ticker, "action": "PLACE_STOP", **stop_record})
        elif record["status"] != "filled":
            needs_review.append(
                {
                    "ticker": ticker,
                    "issue": "entry order not yet confirmed filled; protective stop deferred",
                    "status": record["status"],
                    "order_id": record["order_id"],
                }
            )

    for trade in closed_trades:
        if trade.asset_class != "EQUITY":
            continue
        if trade.exit_reason in _STOP_EXIT_REASONS:
            # The resting native stop should already have filled at the
            # broker. Reconcile against it; never submit a second sell.
            stop_order_id = runner_state.equity_stop_orders.get(trade.ticker)
            if stop_order_id is None:
                continue
            stop_status = order_submission.get_order_status(client, stop_order_id)
            if stop_status == "filled":
                runner_state.equity_stop_orders.pop(trade.ticker, None)
                actions.append(
                    {
                        "ticker": trade.ticker,
                        "action": "RECONCILE_STOP_FILLED",
                        "order_id": stop_order_id,
                        "status": stop_status,
                    }
                )
            else:
                needs_review.append(
                    {
                        "ticker": trade.ticker,
                        "issue": (
                            "local simulation closed this trade via "
                            f"{trade.exit_reason} but the resting stop order "
                            "does not yet show filled"
                        ),
                        "status": stop_status,
                        "order_id": stop_order_id,
                    }
                )
            continue

        # Cancel the resting stop BEFORE selling, and only proceed to
        # the sell once cancellation actually confirms. Order matters:
        # a resting stop SELL holds its shares as collateral
        # (`held_for_orders`), and Alpaca rejects a second sell against
        # them with `insufficient qty available` while the stop is
        # still active -- confirmed with a real paper order. Selling
        # first (the previous ordering here) fails every single time
        # there is still an active protective stop, which is the
        # normal case for any signal-based exit.
        stop_order_id = runner_state.equity_stop_orders.get(trade.ticker)
        if stop_order_id is not None:
            cancel_key = f"{trade.ticker}|CANCEL_STOP|{equity_date}"
            cancel_record = runner_state.submitted_actions.get(cancel_key)
            if cancel_record is None:
                require_authorized_execution_context(
                    authorization, action_description=f"Canceling protective stop ({trade.ticker})"
                )
                # Task 3 (2026-08-22): a CANCEL has no client_order_id
                # of its own (Alpaca's cancel API takes/generates none)
                # -- durable journal coverage uses `operation_id`
                # instead (see `_deterministic_operation_id` and
                # `OrderIntentHook`'s own module-level comment). Does
                # not go through `_resolve_or_submit_order` (a cancel
                # has no "does it already exist" broker query the way
                # a submission does) -- hook calls inlined here, same
                # discipline as the crypto signal-exit's own bespoke
                # inline path.
                cancel_operation_id = _deterministic_operation_id(
                    trade.ticker, "CANCEL_PROTECTIVE_STOP", stop_order_id
                )
                _call_order_intent_hook(
                    order_intent_hook, "SUBMITTING", ticker=trade.ticker, action_kind="CANCEL_PROTECTIVE_STOP",
                    operation_id=cancel_operation_id, side="SELL", order_type="stop",
                    parent_order_id=stop_order_id,
                )
                cancel_status = order_submission.cancel_order_and_confirm(client, stop_order_id)
                _call_order_intent_hook(
                    order_intent_hook, "BROKER_ACKNOWLEDGED", ticker=trade.ticker,
                    action_kind="CANCEL_PROTECTIVE_STOP", operation_id=cancel_operation_id,
                    side="SELL", order_type="stop", parent_order_id=stop_order_id,
                    # independent-audit-round-2 finding #3 (2026-08-23):
                    # a cancel has no Order object to derive broker_status
                    # from (Alpaca's cancel API returns only this plain
                    # status string) -- without threading it through here,
                    # the intent's own broker_status landed None even
                    # though the real status was already known.
                    broker_status=cancel_status,
                )
                cancel_record = {
                    "order_id": stop_order_id,
                    "status": cancel_status,
                    "kind": "CANCEL_STOP",
                    "submitted_at": order_submission.now_iso(),
                }
                runner_state.submitted_actions[cancel_key] = cancel_record
            else:
                cancel_status = cancel_record["status"]
            actions.append(
                {
                    "ticker": trade.ticker,
                    "action": "CANCEL_STOP",
                    "order_id": stop_order_id,
                    "status": cancel_status,
                }
            )

            if cancel_status == "filled":
                # The stop triggered on its own before the cancel
                # request landed -- the position is already closed at
                # the broker. There is nothing left to sell.
                runner_state.equity_stop_orders.pop(trade.ticker, None)
                needs_review.append(
                    {
                        "ticker": trade.ticker,
                        "issue": (
                            "the protective stop filled before this signal-based "
                            "exit's cancel request landed; the position is "
                            "already closed at the broker -- no sell submitted"
                        ),
                        "order_id": stop_order_id,
                    }
                )
                continue
            if cancel_status != "canceled":
                needs_review.append(
                    {
                        "ticker": trade.ticker,
                        "issue": (
                            "could not confirm the protective stop was canceled; "
                            "refusing to sell while it may still be resting"
                        ),
                        "status": cancel_status,
                        "order_id": stop_order_id,
                    }
                )
                continue
            runner_state.equity_stop_orders.pop(trade.ticker, None)

        action_key = f"{trade.ticker}|SIGNAL_EXIT_MARKET_SELL|{equity_date}"
        record = runner_state.submitted_actions.get(action_key)
        if record is None:
            client_order_id = client_order_id_for_action(trade.ticker, "SIGNAL_EXIT_MARKET_SELL", equity_date)
            order, already_existed = _resolve_or_submit_order(
                client,
                client_order_id=client_order_id,
                submit=lambda: order_submission.submit_equity_market_order(
                    client, ticker=trade.ticker, side=OrderSide.SELL, quantity=trade.quantity,
                    client_order_id=client_order_id,
                ),
                order_intent_hook=order_intent_hook,
                ticker=trade.ticker,
                action_kind="SIGNAL_EXIT_MARKET_SELL",
                authorization=authorization,
            )
            status = (
                _order_status_str(order) if already_existed
                else order_submission.wait_for_fill_or_timeout(client, str(order.id))
            )
            record = {
                "order_id": str(order.id),
                "status": status,
                "kind": "SIGNAL_EXIT_MARKET_SELL",
                "submitted_at": order_submission.now_iso(),
                "client_order_id": client_order_id,
            }
            runner_state.submitted_actions[action_key] = record
        else:
            record["status"] = order_submission.get_order_status(client, record["order_id"])
        actions.append({"ticker": trade.ticker, "action": "SELL", **record})

    return actions, needs_review


_CRYPTO_LOCK_RETRY_ATTEMPTS = 3
_CRYPTO_LOCK_RETRY_DELAY_SECONDS = 2.0


def _try_acquire_crypto_lock(
    *,
    attempts: int = _CRYPTO_LOCK_RETRY_ATTEMPTS,
    delay_seconds: float = _CRYPTO_LOCK_RETRY_DELAY_SECONDS,
) -> bool:
    """Try the crypto monitor's own lock file a few times, briefly.

    Uses the same path and the same `os.O_CREAT|O_EXCL` primitive as
    `crypto_stop_monitor._monitor_lock` (imported as `DEFAULT_LOCK_PATH`
    for a single source of truth on the path) rather than importing
    that context manager itself: a retry-before-holding loop does not
    compose cleanly with a plain `@contextmanager`, and duplicating six
    lines here is a smaller risk than touching the already real-money-
    adjacent-tested monitor module for this script's sake.

    Returns True (and leaves the lock held — caller must call
    `_release_crypto_lock()`) or False if every attempt found it held.
    """
    DEFAULT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            descriptor = os.open(DEFAULT_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            return True
        except FileExistsError:
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
    return False


def _release_crypto_lock() -> None:
    DEFAULT_LOCK_PATH.unlink(missing_ok=True)


def _execute_crypto_orders(
    client: TradingClient,
    *,
    runner_state: LiveRunnerState,
    crypto_date: str,
    newly_opened: dict[str, _MutablePosition],
    closed_trades: list[PortfolioTrade],
    order_intent_hook: OrderIntentHook | None = None,
    authorization: AuthorizedExecutionContext | None = None,
) -> tuple[list[dict], list[dict]]:
    """Submit real crypto orders for today's decisions. Never called for equity.

    No protective-stop step (unlike equity): crypto has no native stop
    order. Real-time protection is `crypto_stop_monitor.py`, reading
    `stop_loss_price` from this same persisted state.

    Idempotency keys use `crypto_date` (crypto's own real calendar date,
    which can differ from `equity_date` -- crypto trades 24/7) rather
    than `portfolio_bar_index` -- see `_execute_equity_orders`'s
    docstring for why.
    """
    actions: list[dict] = []
    needs_review: list[dict] = []

    crypto_entries = {t: p for t, p in newly_opened.items() if p.asset_class == "CRYPTO"}
    crypto_exits = [t for t in closed_trades if t.asset_class == "CRYPTO"]
    if not crypto_entries and not crypto_exits:
        return actions, needs_review

    if not _try_acquire_crypto_lock():
        for ticker, position in crypto_entries.items():
            needs_review.append(
                {
                    "ticker": ticker,
                    "action": "BUY",
                    "issue": (
                        "crypto monitor lock still held after retries; entry order "
                        "NOT submitted this run (local position bookkeeping already "
                        "reflects the engine's decision)"
                    ),
                    "quantity": position.quantity,
                }
            )
        for trade in crypto_exits:
            needs_review.append(
                {
                    "ticker": trade.ticker,
                    "action": "SELL",
                    "issue": (
                        "crypto monitor lock still held after retries; exit order "
                        "NOT submitted this run"
                    ),
                    "exit_reason": trade.exit_reason,
                    "quantity": trade.quantity,
                }
            )
        return actions, needs_review

    try:
        for ticker, position in crypto_entries.items():
            alpaca_symbol = ticker.replace("-", "/")
            action_key = f"{ticker}|ENTRY_MARKET_BUY|{crypto_date}"
            record = runner_state.submitted_actions.get(action_key)
            if record is None:
                client_order_id = client_order_id_for_action(ticker, "ENTRY_MARKET_BUY", crypto_date)
                order, already_existed = _resolve_or_submit_order(
                    client,
                    client_order_id=client_order_id,
                    submit=lambda: order_submission.submit_equity_market_order(
                        client,
                        ticker=alpaca_symbol,
                        side=OrderSide.BUY,
                        quantity=position.quantity,
                        time_in_force=TimeInForce.IOC,
                        client_order_id=client_order_id,
                    ),
                    order_intent_hook=order_intent_hook,
                    ticker=ticker,
                    action_kind="ENTRY_MARKET_BUY",
                    authorization=authorization,
                )
                status = (
                    _order_status_str(order) if already_existed
                    else order_submission.wait_for_fill_or_timeout(client, str(order.id))
                )
                record = {
                    "order_id": str(order.id),
                    "status": status,
                    "kind": "ENTRY_MARKET_BUY",
                    "submitted_at": order_submission.now_iso(),
                    "client_order_id": client_order_id,
                }
                runner_state.submitted_actions[action_key] = record
            else:
                record["status"] = order_submission.get_order_status(client, record["order_id"])
            actions.append({"ticker": ticker, "action": "BUY", **record})
            if record["status"] != "filled":
                needs_review.append(
                    {
                        "ticker": ticker,
                        "issue": "crypto entry order not confirmed filled",
                        "status": record["status"],
                        "order_id": record["order_id"],
                    }
                )

        for trade in crypto_exits:
            if trade.exit_reason in _STOP_EXIT_REASONS:
                # The intraday crypto monitor is the intended real-time
                # enforcer for stop-type exits and there is no
                # broker-side order to reconcile against (crypto has no
                # native stop) -- never guess a duplicate sell here.
                needs_review.append(
                    {
                        "ticker": trade.ticker,
                        "issue": (
                            "daily bar-based check closed this CRYPTO position via "
                            f"{trade.exit_reason}; the intraday monitor is the "
                            "intended real-time enforcer for crypto stops -- verify "
                            "manually whether a real sell already occurred"
                        ),
                        "quantity": trade.quantity,
                    }
                )
                continue

            alpaca_symbol = trade.ticker.replace("-", "/")
            action_key = f"{trade.ticker}|SIGNAL_EXIT_MARKET_SELL|{crypto_date}"
            record = runner_state.submitted_actions.get(action_key)
            if record is None:
                require_authorized_execution_context(
                    authorization, action_description=f"Submitting crypto signal-exit SELL ({trade.ticker})"
                )
                client_order_id = client_order_id_for_action(trade.ticker, "SIGNAL_EXIT_MARKET_SELL", crypto_date)
                # Checked before spending a call on available_quantity --
                # if this exact intent already happened, no quantity
                # decision is needed at all. Does not go through
                # _resolve_or_submit_order (the extra available-quantity
                # lookup below does not fit that helper's plain
                # check-then-submit shape) -- the SUBMITTING/
                # BROKER_ACKNOWLEDGED hook calls are inlined here instead,
                # at the same two real moments: immediately before this
                # existence query, and immediately after it resolves.
                _call_order_intent_hook(
                    order_intent_hook, "SUBMITTING",
                    ticker=trade.ticker, action_kind="SIGNAL_EXIT_MARKET_SELL", client_order_id=client_order_id,
                )
                existing_order = order_submission.get_order_by_client_order_id(client, client_order_id)
                if existing_order is not None:
                    record = {
                        "order_id": str(existing_order.id),
                        "status": _order_status_str(existing_order),
                        "kind": "SIGNAL_EXIT_MARKET_SELL",
                        "submitted_at": order_submission.now_iso(),
                        "client_order_id": client_order_id,
                    }
                    runner_state.submitted_actions[action_key] = record
                    _call_order_intent_hook(
                        order_intent_hook, "BROKER_ACKNOWLEDGED",
                        ticker=trade.ticker, action_kind="SIGNAL_EXIT_MARKET_SELL",
                        client_order_id=client_order_id, order=existing_order,
                    )
                else:
                    available_quantity = _query_available_crypto_quantity(client, alpaca_symbol)
                    if available_quantity is None:
                        needs_review.append(
                            {
                                "ticker": trade.ticker,
                                "issue": (
                                    "signal-based exit due but the broker reports no "
                                    "available quantity for this symbol; refusing to "
                                    "guess a sell quantity from local state"
                                ),
                                "local_quantity": trade.quantity,
                            }
                        )
                        continue
                    order = order_submission.submit_equity_market_order(
                        client,
                        ticker=alpaca_symbol,
                        side=OrderSide.SELL,
                        quantity=available_quantity,
                        time_in_force=TimeInForce.IOC,
                        client_order_id=client_order_id,
                    )
                    _call_order_intent_hook(
                        order_intent_hook, "BROKER_ACKNOWLEDGED",
                        ticker=trade.ticker, action_kind="SIGNAL_EXIT_MARKET_SELL",
                        client_order_id=client_order_id, order=order,
                    )
                    status = order_submission.wait_for_fill_or_timeout(client, str(order.id))
                    record = {
                        "order_id": str(order.id),
                        "status": status,
                        "kind": "SIGNAL_EXIT_MARKET_SELL",
                        "submitted_at": order_submission.now_iso(),
                        "local_quantity": trade.quantity,
                        "broker_quantity": available_quantity,
                        "client_order_id": client_order_id,
                    }
                    runner_state.submitted_actions[action_key] = record
            else:
                record["status"] = order_submission.get_order_status(client, record["order_id"])
            actions.append({"ticker": trade.ticker, "action": "SELL", **record})
            if record["status"] != "filled":
                needs_review.append(
                    {
                        "ticker": trade.ticker,
                        "issue": "crypto signal-exit order not confirmed filled",
                        "status": record["status"],
                        "order_id": record["order_id"],
                    }
                )
    finally:
        _release_crypto_lock()

    return actions, needs_review


def run_daily_decision(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    decision_log_directory: Path = DEFAULT_DECISION_LOG_DIRECTORY,
    enable_equity_orders: bool = False,
    enable_crypto_orders: bool = False,
    freeze: bool = False,
    guard_path: Path = ps.HIGH_WATER_MARK_PATH,
    trading_client: TradingClient | None = None,
    skip_broker_reconciliation: bool = False,
    order_intent_hook: OrderIntentHook | None = None,
    authorization: AuthorizedExecutionContext | None = None,
) -> dict:
    """`guard_path` overrides the rollback high-water-mark file's location;
    defaults to the real, out-of-repo path. Only override in tests -- see
    `load_position_state`'s docstring for why a tmp_path-only test must
    not touch the real, machine-wide guard file.

    `authorization` -- MANDATORY whenever `enable_equity_orders`/
    `enable_crypto_orders` is `True` (independent audit finding "Madde
    E", 2026-08-22, "Doğrudan runner bypass'ı hâlâ açık"): checked as
    the FIRST statement in this function's body, before any credential
    load, broker call, or state read/write -- see
    `src.live.authorized_execution_context`'s own module docstring for
    the full design, and this repo's two blessed callers: this file's
    own `main()` (self-authorizes after its STOP/FREEZE preflight -- the
    real live cron's `--enable-equity-orders` usage is UNCHANGED by
    this, by deliberate scope decision) and
    `scripts/run_control_arm_decision.py`'s `_execute()` (authorizes
    only after its full preflight chain: identity, reconciliation, TTL,
    session, write-ahead evidence). A BARE call to this function with an
    order-enabling flag True and no valid `authorization` -- bypassing
    both of those -- now raises immediately, before this function does
    anything else at all. `None` is always accepted when both order
    flags are `False` (the dry-run path, unauthenticated by design --
    dry-run makes no real broker call).

    `order_intent_hook` -- injection-seam callback (2026-08-22, independent
    audit finding #3, approach (B)), same one-directional-dependency
    discipline as `trading_client` above: this function never imports
    `order_intent.py` itself (see that module's own docstring for why),
    it only calls this optional hook, if given one, at the real
    SUBMITTING/BROKER_ACKNOWLEDGED moments around every real broker order
    call this function makes. See `OrderIntentHook`'s own module-level
    comment for the full contract and the real crash-window gap this
    closes. `None` (the default, every real production invocation today)
    means zero behavior change from before this parameter existed.

    `trading_client` -- injection seam, test-harness-only, mirrors
    `scripts/run_control_arm_decision.py`'s own `_execute(...,
    trading_client=None)` pattern (that file already reuses this exact
    function, so the seam existing here benefits both callers). When
    `None` (every real production invocation), the real
    `order_submission.get_trading_client()` factory is used, exactly
    once, unconditionally -- see the broker-reconciliation section
    below for why this is no longer gated behind
    `enable_equity_orders`/`enable_crypto_orders`.

    BROKER RECONCILIATION (added after the 2026-08-21 governance
    finding: `src/live/broker_reconciliation.py`'s Scenario F --
    verifies a resting protective stop for every open local equity
    position is still genuinely open at the broker -- existed and was
    fully tested, but was wired only into
    `scripts/run_control_arm_decision.py`'s separate control-arm
    account, never here. A real position (CEG) whose broker-side stop
    filled went undetected for three consecutive days because nothing
    on this path ever checked). `reconcile()` now runs on EVERY call,
    dry-run or order-enabled, BEFORE any market-data fetch -- a stale
    local position with a stop that no longer genuinely rests at the
    broker must never be allowed to influence a signal-generation-only
    run either, since that is exactly the scenario that went
    undetected. See that function's own docstring for the full
    six-scenario table; any of its fail-closed exceptions propagates
    uncaught out of this function, exactly like any other real failure
    here -- `main()`'s existing broad `except Exception` already
    notifies and re-raises, no new exception handling was added for
    this.

    `skip_broker_reconciliation` -- REAL BUG FOUND AND FIXED (2026-08-22,
    independent audit): `scripts/run_control_arm_decision.py` already
    runs its OWN, correctly-configured `broker_reconciliation.reconcile()`
    (its own account's identity suffix, before this function is even
    called) -- calling this function afterward, unmodified, meant a
    SECOND, redundant reconciliation ran here too, on a freshly built
    client (never the caller's already-verified one) with
    `LIVE_ACCOUNT_NUMBER_SUFFIX` -- the wrong suffix entirely for that
    account, and unset in that deployment anyway, so the second pass's
    identity check was silently skipped. That caller now passes both
    `trading_client=<its own already-verified client>` and
    `skip_broker_reconciliation=True`; every other real caller leaves
    this `False` (the unchanged, default behavior) and gets the real
    reconciliation this function's own docstring describes above."""
    if enable_equity_orders or enable_crypto_orders:
        # MANDATORY, fail-closed, checked FIRST -- before load_position_state,
        # before any credential/broker/state call. See `authorization`'s
        # own docstring above for the two blessed callers this accepts.
        require_authorized_execution_context(
            authorization, action_description="run_daily_decision(enable_equity_orders/enable_crypto_orders=True)"
        )

    config = PortfolioBacktestConfig(
        maximum_open_positions=LIVE_MAXIMUM_OPEN_POSITIONS,
        maximum_total_open_risk_percent=LIVE_MAXIMUM_TOTAL_OPEN_RISK_PERCENT,
    )
    config.validate()

    runner_state = load_position_state(state_path, guard_path=guard_path)
    positions_before = dict(runner_state.positions)
    pending_buys_before = dict(runner_state.pending_buys)
    pending_exits_before = dict(runner_state.pending_exits)

    live_orders_enabled = enable_equity_orders or enable_crypto_orders
    if trading_client is None:
        trading_client = order_submission.get_trading_client()

    if not skip_broker_reconciliation:
        # MANDATORY, fail-closed -- see _require_live_account_suffix's own
        # docstring for why this is no longer an optional skip-and-warn
        # path for this specific caller (independent audit finding #1).
        live_account_suffix = _require_live_account_suffix()

        reconciliation_result = broker_reconciliation.reconcile(
            trading_client,
            runner_state,
            expected_account_suffix=live_account_suffix,
            # Independent, not blended (audit finding: a single boolean
            # contradicted this file's own --enable-equity-orders/
            # --enable-crypto-orders independence contract) -- see
            # reconcile()'s own docstring for the false-alarm/false-pass
            # this fixes.
            equity_orders_enabled=enable_equity_orders,
            crypto_orders_enabled=enable_crypto_orders,
        )
        if reconciliation_result.order_status_updates:
            save_position_state(runner_state, state_path, guard_path=guard_path)

    needs_review: list[dict] = []
    if enable_equity_orders:
        needs_review.extend(
            _reconcile_pending_equity_orders(
                trading_client, runner_state, order_intent_hook=order_intent_hook, authorization=authorization
            )
        )

    # Reuses the SAME already-verified trading_client reconciliation just
    # used, rather than building a second, separate connection -- see
    # get_live_cash_balance's own docstring for the real bug this closes.
    cash = get_live_cash_balance(trading_client)
    prepared = prepare_live_market_data(LIVE_CONTROLLED_TICKERS)
    bars_today, equity_date, crypto_date = _bars_today_by_asset_class(prepared)
    # Crypto is real-time; equity's own date is only ever equal to or
    # behind it (weekends/holidays), never ahead. Used only as a
    # display label (see _bars_today_by_asset_class) -- the precise
    # per-asset-class dates are recorded separately in `decision` below.
    timestamp_string = crypto_date
    local_indices = {ticker: len(frame) - 1 for ticker, frame in prepared.items()}

    today_index = runner_state.portfolio_bar_index + 1
    state = _build_state(runner_state=runner_state, cash=cash, bars_today=bars_today)

    # Steps 1-4: exits and stops execute at today's Open, using
    # positions/pending orders carried over from the previous run.
    _execute_pending_exits_at_open(
        state, bars_today=bars_today, timestamp=timestamp_string,
        portfolio_bar_index=today_index, config=config,
    )
    _check_gap_stops(
        state, bars_today=bars_today, timestamp=timestamp_string,
        portfolio_bar_index=today_index, config=config,
    )
    # Step 3 (position-opening) is the one step FREEZE suppresses here --
    # skipping it leaves any previously-queued buy pending rather than
    # opening it, so no new exposure is taken on while frozen. Steps 1, 2,
    # and 4 (exits/stops, both above and below) are exit-only and always run.
    if not freeze:
        _execute_pending_buys_at_open(
            state, bars_today=bars_today, timestamp=timestamp_string,
            portfolio_bar_index=today_index, config=config,
        )
    _check_intrabar_stops(
        state, bars_today=bars_today, timestamp=timestamp_string,
        portfolio_bar_index=today_index, config=config,
    )

    # Mark to Close before evaluating Close-based conditions, exactly
    # as the batch loop does between its Open-phase and Close-phase.
    for ticker, row in bars_today.items():
        state.last_prices[ticker] = float(row["Close"])

    # Steps 5-6: new signals queued at today's Close, only executable
    # on the NEXT run's Open.
    _queue_close_based_exits(
        state, bars_today=bars_today, timestamp=timestamp_string,
        portfolio_bar_index=today_index,
    )
    # Step 6 (queuing new entry signals for a future run) is the other
    # step FREEZE suppresses -- close-based exits above still queue
    # normally, since those reduce exposure rather than add to it.
    if not freeze:
        _queue_ranked_entry_signals(
            state, data_by_ticker=prepared, local_indices=local_indices,
            timestamp=timestamp_string, portfolio_bar_index=today_index,
        )

    newly_opened = {
        ticker: position
        for ticker, position in state.positions.items()
        if ticker not in positions_before
    }

    equity_order_actions: list[dict] = []
    if enable_equity_orders:
        equity_order_actions, more_review = _execute_equity_orders(
            trading_client,
            runner_state=runner_state,
            equity_date=equity_date,
            newly_opened=newly_opened,
            closed_trades=state.trades,
            order_intent_hook=order_intent_hook,
            authorization=authorization,
        )
        needs_review.extend(more_review)

    crypto_order_actions: list[dict] = []
    if enable_crypto_orders:
        crypto_order_actions, more_review = _execute_crypto_orders(
            trading_client,
            runner_state=runner_state,
            crypto_date=crypto_date,
            newly_opened=newly_opened,
            closed_trades=state.trades,
            order_intent_hook=order_intent_hook,
            authorization=authorization,
        )
        needs_review.extend(more_review)

    decision = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_bar_timestamp": timestamp_string,
        "as_of_bar_timestamp_equity": equity_date,
        "as_of_bar_timestamp_crypto": crypto_date,
        "portfolio_bar_index": today_index,
        "cash_balance_usd": round(cash, 2),
        "live_equity_orders_enabled": enable_equity_orders,
        "live_crypto_orders_enabled": enable_crypto_orders,
        "freeze_active": freeze,
        "note": (
            "executed_today reflects fills at today's actual Open from "
            "signals queued on a previous run. queued_for_next_run "
            "reflects signals generated from today's Close; those only "
            "execute on the NEXT run, never today (next-available-Open rule). "
            "equity_order_actions is empty unless --enable-equity-orders was "
            "passed. crypto_order_actions is empty unless --enable-crypto-orders "
            "was passed. Each flag is independent of the other."
        ),
        "executed_today": {
            "exits": [
                {
                    **trade.to_dict(),
                    "recommended_order_type": _recommended_exit_order_type(
                        exit_reason=trade.exit_reason,
                        asset_class=trade.asset_class,
                    ),
                }
                for trade in state.trades
            ],
            "entries": [
                {
                    "ticker": ticker,
                    "asset_class": position.asset_class,
                    "action": "BUY",
                    "recommended_order_type": "market",
                    "quantity": position.quantity,
                    "fill_price": position.entry_price,
                    "stop_loss_price": position.stop_loss_price,
                    "entry_timestamp": position.entry_timestamp,
                    "signal_score": position.signal_score,
                    "signal_reason": position.signal_reason,
                }
                for ticker, position in newly_opened.items()
            ],
        },
        "equity_order_actions": equity_order_actions,
        "crypto_order_actions": crypto_order_actions,
        "needs_manual_review": needs_review,
        "rejected_today": [rejection.to_dict() for rejection in state.rejections],
        "queued_for_next_run": {
            "buys": [
                {
                    "ticker": ticker,
                    "action": "BUY",
                    "reference_close": pending.signal.reference_price,
                    "signal_score": pending.signal.score,
                    "signal_reason": pending.signal.reason,
                    "submitted_portfolio_bar_index": pending.submitted_portfolio_bar_index,
                }
                for ticker, pending in state.pending_buys.items()
            ],
            "exits": [
                {
                    "ticker": ticker,
                    "action": "EXIT",
                    "reference_close": pending.signal.reference_price,
                    "reason": pending.signal.reason,
                    "submitted_portfolio_bar_index": pending.submitted_portfolio_bar_index,
                }
                for ticker, pending in state.pending_exits.items()
            ],
        },
        "open_positions_after_run": {
            ticker: asdict(position) for ticker, position in state.positions.items()
        },
        "equity_stop_orders_after_run": dict(runner_state.equity_stop_orders),
        "carried_over_pending_consumed": {
            "buys": sorted(set(pending_buys_before) - set(state.pending_buys)),
            "exits": sorted(set(pending_exits_before) - set(state.pending_exits)),
        },
    }

    decision_log_directory = Path(decision_log_directory)
    decision_log_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_path = decision_log_directory / f"decision_{stamp}.json"
    if log_path.exists():
        raise FileExistsError(log_path)
    log_path.write_text(json.dumps(decision, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    new_runner_state = LiveRunnerState(
        portfolio_bar_index=today_index,
        positions=state.positions,
        pending_buys=state.pending_buys,
        pending_exits=state.pending_exits,
        equity_stop_orders=runner_state.equity_stop_orders,
        submitted_actions=runner_state.submitted_actions,
    )
    save_position_state(new_runner_state, state_path, guard_path=guard_path)

    return {"decision": decision, "log_path": log_path}


def _notify_safe(text: str) -> bool:
    """Send a Telegram notification without ever letting it affect the caller.

    `send_telegram_message` already never raises (see
    `src/notify/telegram_notifier.py`), but this wraps the call anyway
    per the explicit "notification must never touch trading behavior
    or exit code" requirement — defense in depth against a bug in the
    notifier itself, not just its documented contract.

    Returns whether the notification appears to have gone out
    (best-effort; False on any failure, including a notifier bug or
    missing/invalid credentials). Callers on the failure path use this
    to print a loud, log-visible fallback so a delivery failure is
    never itself silent — see `main()`.
    """
    try:
        return bool(send_telegram_message(text))
    except Exception:
        return False


def _build_daily_notification_text(decision: dict) -> str:
    """Owner-facing summary: what happened today, or that nothing did.

    Deliberately always sends something (even "no orders today") so
    silence on the owner's phone means "the job didn't run" rather
    than being ambiguous with "it ran and did nothing."
    """
    equity_date = decision.get("as_of_bar_timestamp_equity")
    crypto_date = decision.get("as_of_bar_timestamp_crypto")
    if equity_date and crypto_date and equity_date != crypto_date:
        date = f"equity {equity_date} / crypto {crypto_date}"
    else:
        date = decision.get("as_of_bar_timestamp", "unknown date")
    equity_actions = decision.get("equity_order_actions") or []
    crypto_actions = decision.get("crypto_order_actions") or []
    review = decision.get("needs_manual_review") or []
    cash = decision.get("cash_balance_usd")

    lines = [f"AI-Stock-Radar daily decision — {date}"]
    if not equity_actions and not crypto_actions:
        lines.append("No equity or crypto orders today.")
    else:
        for label, actions in (("Equity", equity_actions), ("Crypto", crypto_actions)):
            if not actions:
                continue
            lines.append(f"{label} actions ({len(actions)}):")
            for action in actions:
                detail = action.get("status") or action.get("issue") or ""
                lines.append(f"  {action.get('ticker')} {action.get('action')}: {detail}")
    lines.append(f"needs_manual_review: {len(review)}")
    if cash is not None:
        lines.append(f"Cash balance: ${cash:,.2f}")
    return "\n".join(lines)


def _build_failure_notification_text(error: BaseException) -> str:
    return (
        f"AI-Stock-Radar daily decision FAILED at {datetime.now(UTC).isoformat()}\n"
        f"{type(error).__name__}: {error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one day of the frozen TREND_RSI strategy against live Alpaca data."
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--decision-log-directory", type=Path, default=DEFAULT_DECISION_LOG_DIRECTORY)
    parser.add_argument(
        "--enable-equity-orders",
        action="store_true",
        help=(
            "Submit real Alpaca PAPER orders for equity tickers only. "
            "Crypto stays dry-run/log-only regardless of this flag. "
            "Omit this flag to keep the original all-dry-run behavior."
        ),
    )
    parser.add_argument(
        "--enable-crypto-orders",
        action="store_true",
        help=(
            "Submit real Alpaca PAPER orders for crypto tickers only. "
            "Equity stays dry-run/log-only regardless of this flag. "
            "Independent of --enable-equity-orders. Omit this flag to "
            "keep the original all-dry-run behavior for crypto."
        ),
    )
    arguments = parser.parse_args()

    # single_instance_lock (added 2026-08-23, independent-audit-round-3
    # finding #4): this cron's own main() previously took no lock at
    # all, while equity_session_orchestrator.py's own
    # run_missing_session_replay() already did -- a delayed/overlapping
    # daily run and an orchestrator invocation could both write
    # arguments.state_path concurrently with no mutual exclusion
    # whatsoever. Shares the EXACT SAME lock (`single_instance_lock`,
    # keyed by `state_path`) that module already uses, closing the real
    # cross-write race. NOT a zero-behavior-change: if this cron's own
    # prior invocation is still genuinely running when cron fires again
    # (e.g. a slow run overlapping its own next scheduled tick), this
    # now fails closed with `SingleInstanceLockError` instead of
    # silently running twice in parallel against the same state file --
    # the correct, safer behavior, but a real, observable change from
    # before this fix for that specific (already-anomalous) case.
    with single_instance_lock(arguments.state_path):
        if STOP_FLAG_PATH.exists():
            text = (
                f"AI-Stock-Radar daily decision runner -- STOP flag detected "
                f"({STOP_FLAG_PATH}); this run was skipped entirely, no API "
                f"calls were made. Remove the file to resume."
            )
            print(text)
            _notify_safe(text)
            return

        freeze = FREEZE_FLAG_PATH.exists()
        if freeze:
            text = (
                f"AI-Stock-Radar daily decision runner -- FREEZE flag detected "
                f"({FREEZE_FLAG_PATH}); this run will still fetch data and "
                f"monitor/exit existing positions as usual, but will NOT open "
                f"any new positions. Remove the file to resume normal entries."
            )
            print(text)
            _notify_safe(text)

        # Self-authorizes AFTER the STOP/FREEZE preflight above (independent
        # audit finding "Madde E", 2026-08-22): this CLI entrypoint is one of
        # this codebase's two blessed callers of run_daily_decision() with a
        # real order-enabling flag -- see run_daily_decision()'s own
        # `authorization` docstring and src.live.authorized_execution_context's
        # module docstring for the full design and why this is a deliberate
        # scope decision (the live cron's real --enable-equity-orders usage
        # is unchanged by this). Only constructed when actually needed --
        # the dry-run path (neither flag set, every non-live-order invocation)
        # never touches this at all.
        authorization = (
            authorize_order_execution() if (arguments.enable_equity_orders or arguments.enable_crypto_orders) else None
        )

        try:
            result = run_daily_decision(
                state_path=arguments.state_path,
                decision_log_directory=arguments.decision_log_directory,
                enable_equity_orders=arguments.enable_equity_orders,
                enable_crypto_orders=arguments.enable_crypto_orders,
                freeze=freeze,
                authorization=authorization,
            )
        except Exception as error:
            # Notify, then re-raise unchanged -- the notification step must
            # never mask a real failure or alter the script's exit code.
            # If the notification itself could not be confirmed sent (bad
            # credentials, network issue, notifier bug), that failure must
            # ALSO be visible in the log -- otherwise a real crash can look
            # identical to "the job silently produced no output at all",
            # which is exactly the gap that let this failure mode go
            # unnoticed in production. print() to stderr rather than the
            # `logging` module: this failure path must be visible in
            # whatever plain stdout/stderr capture the deployment already
            # has, without depending on separate logging configuration.
            notified = _notify_safe(_build_failure_notification_text(error))
            if not notified:
                print(
                    "WARNING: Telegram failure notification could not be confirmed "
                    "sent (check TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID and network "
                    "reachability). The original error causing this run to fail "
                    f"follows: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
            raise

        print(json.dumps(result["decision"], indent=2, sort_keys=True, default=str))
        print()
        print(f"Decision log written to: {result['log_path'].resolve()}")
        print(f"Position state updated at: {arguments.state_path.resolve()}")

        _notify_safe(_build_daily_notification_text(result["decision"]))


if __name__ == "__main__":
    main()
