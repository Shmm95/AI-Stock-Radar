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
- A stale pending BUY queued FROM `last_processed_session_before` (see
  below for why that, not `missed_session_dates`, is the correct
  discriminator) is simply EXPIRED -- removed from `pending_buys`. The
  opportunity is gone; there is nothing to reconfirm for an entry that
  was never taken.
- A stale pending EXIT (same discriminator) is NOT reconfirmed by this
  module recomputing the frozen engine's own exit condition
  (EMA20<EMA50 / trailing stop) itself -- that logic lives in
  `_queue_close_based_exits` (`portfolio_backtest_engine.py`, frozen,
  never reimplemented here). Instead, the stale entry is simply CLEARED
  from `pending_exits`, which lets the very next real
  `run_daily_decision()` call re-evaluate the position fresh, from
  today's real Close, using the engine's own unmodified logic -- rather
  than this module guessing at a decision the frozen engine is the sole
  authority on. This is a deliberate simplification: "reconfirmed" here
  means "handed back to the real engine for a fresh, current decision,"
  not "independently recomputed by this module."
- A broker-side fill possibly already having happened (e.g. a resting
  native stop that filled during the missed window) is NEVER inferred
  or applied here -- `broker_reconciliation.py`'s own Scenario
  C/equity_stop_orders-invariant checks are the only authoritative
  source for that, and this module does not duplicate them.

SELECTIVITY -- `last_processed_session_before`, NOT `missed_session_dates`
(fixed 2026-08-23, independent audit round 3 finding #5, with a
correction found while writing this fix's own tests): an earlier
version of this function treated `missed_session_dates` as purely
informational and unconditionally cleared EVERY currently-queued
pending BUY/EXIT -- a real, confirmed over-clearing bug (a pending
order queued for a DIFFERENT, unrelated session would have been
silently wiped too). The first attempt at fixing this filtered by
`pending.signal.timestamp in missed_session_dates` -- which is ALSO
wrong, and provably so (a test written against it failed): every
`_PendingOrder.signal.timestamp` is stamped with the session it was
queued FROM (see `_queue_close_based_exits`/`_queue_ranked_entry_signals`
in `portfolio_backtest_engine.py`), one session BEFORE the Open it is
due to fill at -- never a date that is itself in `missed_session_dates`
(which lists the missed sessions themselves, not the session before
them). Since every pending order currently on disk during a gap was, by
construction, queued by the run that processed
`last_processed_session_before` (the cursor value immediately before
the gap began -- there is no other run that could have queued one), the
correct, sufficient discriminator is `pending.signal.timestamp ==
last_processed_session_before`. `missed_session_dates` is kept as a
parameter for logging/provenance context only; it is not, and was never
correctly usable as, the membership test itself.
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


def handle_missed_execution_window(
    runner_state: Any, *, missed_session_dates: tuple[str, ...], last_processed_session_before: str,
) -> MissedWindowOutcome:
    """Mutates `runner_state.pending_buys`/`pending_exits` in place --
    same "caller persists afterward" convention as
    `pending_signal_ttl.evaluate_pending_signals`, never saves to disk
    itself. Only a pending BUY/EXIT whose `signal.timestamp` equals
    `last_processed_session_before` is touched (independent-audit-round-3
    finding #5 -- see module docstring's own "SELECTIVITY" section for
    why that, not `missed_session_dates` membership, is the correct
    test); anything queued from a different session is left exactly as
    it was. This function is only ever invoked once the orchestrator has
    already determined the relevant execution window(s) have passed; it
    does not re-derive that determination itself."""
    expired_buys = tuple(
        sorted(
            ticker for ticker, pending in runner_state.pending_buys.items()
            if pending.signal.timestamp == last_processed_session_before
        )
    )
    for ticker in expired_buys:
        del runner_state.pending_buys[ticker]

    cleared_exits = tuple(
        sorted(
            ticker for ticker, pending in runner_state.pending_exits.items()
            if pending.signal.timestamp == last_processed_session_before
        )
    )
    for ticker in cleared_exits:
        del runner_state.pending_exits[ticker]

    return MissedWindowOutcome(
        expired_buy_tickers=expired_buys,
        cleared_for_reconfirmation_exit_tickers=cleared_exits,
    )
