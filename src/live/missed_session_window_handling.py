"""Missed-execution-window handling -- "Missing-Equity-Session
Remediation" design (workspace-c), section C: what to do when a missed
session's own execution window (the NEXT real Open after it) has
ALREADY passed, so the narrow auto-replay path (section B) does not
apply.

P0 PRINCIPLE (the design's own, restated here since it is this
module's entire reason for existing): a missed historical Open must
NEVER be simulated. Filling a BUY/EXIT against a bygone Open that a
real broker never actually saw would fabricate a local position/trade
that has no real broker counterpart -- the exact failure mode this
whole remediation effort exists to prevent, not reproduce.

WHAT THIS MODULE ACTUALLY DOES, deliberately narrow:
- A stale pending BUY (queued for a session whose own execution window
  has passed) is simply EXPIRED -- removed from `pending_buys`. The
  opportunity is gone; there is nothing to reconfirm for an entry that
  was never taken.
- A stale pending EXIT is NOT reconfirmed by this module recomputing
  the frozen engine's own exit condition (EMA20<EMA50 / trailing stop)
  itself -- that logic lives in `_queue_close_based_exits`
  (`portfolio_backtest_engine.py`, frozen, never reimplemented here).
  Instead, the stale entry is simply CLEARED from `pending_exits`,
  which lets the very next real `run_daily_decision()` call re-evaluate
  the position fresh, from today's real Close, using the engine's own
  unmodified logic -- rather than this module guessing at a decision
  the frozen engine is the sole authority on. This is a deliberate
  simplification: "reconfirmed" here means "handed back to the real
  engine for a fresh, current decision," not "independently
  recomputed by this module."
- A broker-side fill possibly already having happened (e.g. a resting
  native stop that filled during the missed window) is NEVER inferred
  or applied here -- `broker_reconciliation.py`'s own Scenario
  C/equity_stop_orders-invariant checks are the only authoritative
  source for that, and this module does not duplicate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MissedWindowOutcome:
    expired_buy_tickers: tuple[str, ...] = field(default_factory=tuple)
    cleared_for_reconfirmation_exit_tickers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def touched_anything(self) -> bool:
        return bool(self.expired_buy_tickers or self.cleared_for_reconfirmation_exit_tickers)


def handle_missed_execution_window(runner_state: Any, *, missed_session_dates: tuple[str, ...]) -> MissedWindowOutcome:
    """Mutates `runner_state.pending_buys`/`pending_exits` in place --
    same "caller persists afterward" convention as
    `pending_signal_ttl.evaluate_pending_signals`, never saves to disk
    itself. `missed_session_dates` is informational only (included in
    the returned outcome's context for logging/provenance); every
    currently-queued pending BUY/EXIT is treated as stale by this
    function's own caller-established precondition -- this function is
    only ever invoked once the orchestrator has already determined the
    relevant execution window has passed, it does not re-derive that
    determination itself."""
    expired_buys = tuple(sorted(runner_state.pending_buys.keys()))
    for ticker in expired_buys:
        del runner_state.pending_buys[ticker]

    cleared_exits = tuple(sorted(runner_state.pending_exits.keys()))
    for ticker in cleared_exits:
        del runner_state.pending_exits[ticker]

    return MissedWindowOutcome(
        expired_buy_tickers=expired_buys,
        cleared_for_reconfirmation_exit_tickers=cleared_exits,
    )
