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
  the seven equity tickers only.
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
- Idempotency: every real submission is recorded in
  `runner_state.submitted_actions`, keyed by
  `f"{ticker}|{kind}|{portfolio_bar_index}"`, and checked before any
  new submission — re-running this script for the same bar re-checks
  status instead of re-submitting.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce

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
from src.backtest.run_portfolio_entry_statistics import CONTROLLED_TICKERS
from src.live import order_submission
from src.live.account_state import get_live_cash_balance
from src.live.crypto_stop_monitor import DEFAULT_LOCK_PATH, _query_available_crypto_quantity
from src.live.data_preparer import prepare_live_market_data
from src.live.position_state import (
    DEFAULT_STATE_PATH,
    LiveRunnerState,
    load_position_state,
    save_position_state,
)
from src.notify.telegram_notifier import send_telegram_message

DEFAULT_DECISION_LOG_DIRECTORY = Path("data/live/decisions")

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


def _reconcile_pending_equity_orders(
    client: TradingClient, runner_state: LiveRunnerState
) -> list[dict]:
    """Catch up on entry orders submitted, but not confirmed, on a previous run."""
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
                stop_key = f"{ticker}|PROTECTIVE_STOP|{runner_state.portfolio_bar_index}"
                stop_order = order_submission.submit_equity_stop_sell(
                    client,
                    ticker=ticker,
                    quantity=position.quantity,
                    stop_price=position.stop_loss_price,
                )
                runner_state.submitted_actions[stop_key] = {
                    "order_id": str(stop_order.id),
                    "status": "new",
                    "kind": "PROTECTIVE_STOP",
                    "submitted_at": order_submission.now_iso(),
                }
                runner_state.equity_stop_orders[ticker] = str(stop_order.id)
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
    today_index: int,
    newly_opened: dict[str, _MutablePosition],
    closed_trades: list[PortfolioTrade],
) -> tuple[list[dict], list[dict]]:
    """Submit real equity orders for today's decisions. Never called for crypto."""
    actions: list[dict] = []
    needs_review: list[dict] = []

    for ticker, position in newly_opened.items():
        if position.asset_class != "EQUITY":
            continue
        action_key = f"{ticker}|ENTRY_MARKET_BUY|{today_index}"
        record = runner_state.submitted_actions.get(action_key)
        if record is None:
            order = order_submission.submit_equity_market_order(
                client, ticker=ticker, side=OrderSide.BUY, quantity=position.quantity
            )
            status = order_submission.wait_for_fill_or_timeout(client, str(order.id))
            record = {
                "order_id": str(order.id),
                "status": status,
                "kind": "ENTRY_MARKET_BUY",
                "submitted_at": order_submission.now_iso(),
            }
            runner_state.submitted_actions[action_key] = record
        else:
            record["status"] = order_submission.get_order_status(client, record["order_id"])
        actions.append({"ticker": ticker, "action": "BUY", **record})

        if record["status"] == "filled" and ticker not in runner_state.equity_stop_orders:
            stop_key = f"{ticker}|PROTECTIVE_STOP|{today_index}"
            stop_record = runner_state.submitted_actions.get(stop_key)
            if stop_record is None:
                stop_order = order_submission.submit_equity_stop_sell(
                    client,
                    ticker=ticker,
                    quantity=position.quantity,
                    stop_price=position.stop_loss_price,
                )
                stop_record = {
                    "order_id": str(stop_order.id),
                    "status": "new",
                    "kind": "PROTECTIVE_STOP",
                    "submitted_at": order_submission.now_iso(),
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
            cancel_key = f"{trade.ticker}|CANCEL_STOP|{today_index}"
            cancel_record = runner_state.submitted_actions.get(cancel_key)
            if cancel_record is None:
                cancel_status = order_submission.cancel_order_and_confirm(client, stop_order_id)
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

        action_key = f"{trade.ticker}|SIGNAL_EXIT_MARKET_SELL|{today_index}"
        record = runner_state.submitted_actions.get(action_key)
        if record is None:
            order = order_submission.submit_equity_market_order(
                client, ticker=trade.ticker, side=OrderSide.SELL, quantity=trade.quantity
            )
            status = order_submission.wait_for_fill_or_timeout(client, str(order.id))
            record = {
                "order_id": str(order.id),
                "status": status,
                "kind": "SIGNAL_EXIT_MARKET_SELL",
                "submitted_at": order_submission.now_iso(),
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
    today_index: int,
    newly_opened: dict[str, _MutablePosition],
    closed_trades: list[PortfolioTrade],
) -> tuple[list[dict], list[dict]]:
    """Submit real crypto orders for today's decisions. Never called for equity.

    No protective-stop step (unlike equity): crypto has no native stop
    order. Real-time protection is `crypto_stop_monitor.py`, reading
    `stop_loss_price` from this same persisted state.
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
            action_key = f"{ticker}|ENTRY_MARKET_BUY|{today_index}"
            record = runner_state.submitted_actions.get(action_key)
            if record is None:
                order = order_submission.submit_equity_market_order(
                    client,
                    ticker=alpaca_symbol,
                    side=OrderSide.BUY,
                    quantity=position.quantity,
                    time_in_force=TimeInForce.IOC,
                )
                status = order_submission.wait_for_fill_or_timeout(client, str(order.id))
                record = {
                    "order_id": str(order.id),
                    "status": status,
                    "kind": "ENTRY_MARKET_BUY",
                    "submitted_at": order_submission.now_iso(),
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
            action_key = f"{trade.ticker}|SIGNAL_EXIT_MARKET_SELL|{today_index}"
            record = runner_state.submitted_actions.get(action_key)
            if record is None:
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
                )
                status = order_submission.wait_for_fill_or_timeout(client, str(order.id))
                record = {
                    "order_id": str(order.id),
                    "status": status,
                    "kind": "SIGNAL_EXIT_MARKET_SELL",
                    "submitted_at": order_submission.now_iso(),
                    "local_quantity": trade.quantity,
                    "broker_quantity": available_quantity,
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
) -> dict:
    config = PortfolioBacktestConfig()
    config.validate()

    runner_state = load_position_state(state_path)
    positions_before = dict(runner_state.positions)
    pending_buys_before = dict(runner_state.pending_buys)
    pending_exits_before = dict(runner_state.pending_exits)

    live_orders_enabled = enable_equity_orders or enable_crypto_orders
    trading_client = order_submission.get_trading_client() if live_orders_enabled else None
    needs_review: list[dict] = []
    if enable_equity_orders and trading_client is not None:
        needs_review.extend(_reconcile_pending_equity_orders(trading_client, runner_state))

    cash = get_live_cash_balance()
    prepared = prepare_live_market_data(CONTROLLED_TICKERS)
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
    if enable_equity_orders and trading_client is not None:
        equity_order_actions, more_review = _execute_equity_orders(
            trading_client,
            runner_state=runner_state,
            today_index=today_index,
            newly_opened=newly_opened,
            closed_trades=state.trades,
        )
        needs_review.extend(more_review)

    crypto_order_actions: list[dict] = []
    if enable_crypto_orders and trading_client is not None:
        crypto_order_actions, more_review = _execute_crypto_orders(
            trading_client,
            runner_state=runner_state,
            today_index=today_index,
            newly_opened=newly_opened,
            closed_trades=state.trades,
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
    save_position_state(new_runner_state, state_path)

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

    try:
        result = run_daily_decision(
            state_path=arguments.state_path,
            decision_log_directory=arguments.decision_log_directory,
            enable_equity_orders=arguments.enable_equity_orders,
            enable_crypto_orders=arguments.enable_crypto_orders,
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
