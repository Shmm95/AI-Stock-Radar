"""Post-crash recovery for `order_intent.py`'s own journal -- Task 3
(2026-08-22), item 6.

WHY THIS EXISTS: `order_intent.py`'s own state machine (PREPARED ->
SUBMITTING -> BROKER_ACKNOWLEDGED -> COMMITTED -> TERMINAL, with an
UNCERTAIN branch from SUBMITTING/BROKER_ACKNOWLEDGED) already durably
records a crash mid-flight -- see that module's own docstring. But
recording the crash is not the same as RESOLVING it: a stray intent
left at PREPARED/SUBMITTING/BROKER_ACKNOWLEDGED by an interrupted PRIOR
run needs to be checked against what the broker actually says happened,
before the NEXT run can safely trust local state again.

`_resolve_or_submit_order`'s own broker-authoritative idempotency check
(query by client_order_id before submitting) already resolves this
narrow case WHEN the exact same action is naturally re-attempted within
a later run's own normal flow. This module exists for the broader case:
a stray intent whose action is NOT naturally re-attempted by anything
(e.g. a CANCEL whose triggering condition no longer holds on the next
run), or one a caller wants to resolve explicitly, up front, before
deciding whether it is even safe to proceed with a new run at all.

SCOPE, DELIBERATELY NARROW: this module resolves the JOURNAL's own
state machine -- is a stray intent's on-disk status still accurate, or
was it left stale by a crash -- against a real, read-only broker query.
It never mutates `position_state.json` (`submitted_actions`/
`equity_stop_orders`/`positions`) itself, and it never auto-resubmits
an order or auto-retries a cancel. "Broker acknowledged but local state
doesn't reflect it yet" is explicitly NOT solved here a second time --
`src/live/broker_reconciliation.py`'s own Scenario C (the narrow,
already-tested "order fully matches and is confirmed filled" automatic
local order-STATUS-only correction) already exists for exactly that
gap; this module hands a resolved BROKER_ACKNOWLEDGED intent back to
the caller, which is expected to let the normal `reconcile()` call
(already part of every real run) pick it up, not reimplement Scenario
C's own exact-match logic a second time here.

Nothing here calls Alpaca on import or at module load -- every function
below takes a real `TradingClient` as an explicit parameter, same
injection-seam discipline the rest of this codebase's live modules use.
"""

from __future__ import annotations

from alpaca.trading.client import TradingClient

from src.live import order_intent
from src.live import order_submission

_NON_TERMINAL_STATUSES = frozenset(
    {order_intent.PREPARED, order_intent.SUBMITTING, order_intent.BROKER_ACKNOWLEDGED, order_intent.UNCERTAIN}
)

# Alpaca order statuses this module treats as "the target order is
# still genuinely resting" -- i.e. not yet acted on. Mirrors
# `order_submission`'s own terminal/non-terminal vocabulary, restated
# here rather than imported, since this module cares about a
# specifically narrower question ("is it still open") than that
# module's own `TERMINAL_STATUSES` set answers.
_STILL_RESTING_STATUSES = frozenset({"new", "accepted", "pending_new", "held", "replaced"})


def find_stray_session_intents_from_prior_run() -> list[order_intent.OrderIntent]:
    """Every intent currently on disk that is NOT in a terminal state
    (PREPARED/SUBMITTING/BROKER_ACKNOWLEDGED/UNCERTAIN) -- a candidate
    a PRIOR, interrupted run left behind. Local-only, makes no broker
    call itself; see `list_intents()`'s own docstring for why this is
    safe to call cheaply and often."""
    return [intent for intent in order_intent.list_intents() if intent.status in _NON_TERMINAL_STATUSES]


def resolve_stray_order_submission_intent(client: TradingClient, intent: order_intent.OrderIntent) -> order_intent.OrderIntent:
    """Resolves a stray intent that represents a real order SUBMISSION
    (has a non-empty `client_order_id` -- ENTRY_MARKET_BUY/
    SIGNAL_EXIT_MARKET_SELL/PROTECTIVE_STOP). Queries the broker by
    `client_order_id` (the same broker-authoritative lookup
    `_resolve_or_submit_order` itself uses) and transitions the intent
    to reflect what is actually true -- never auto-resubmits.

    Real, found order (any status) -> the submission definitely
    happened; transitions through SUBMITTING (if not already past it)
    to BROKER_ACKNOWLEDGED, recording the real broker order id/status.

    Confirmed 404 (`get_order_by_client_order_id` returns `None`) from
    `PREPARED` -> a definitive negative: the broker never received this
    exact submission. Safe to close TERMINAL as abandoned -- the next
    real run's own normal flow (not this module) will attempt the
    action fresh through its own regular idempotency check, with a
    fresh, real decision behind it.

    Confirmed 404 from `SUBMITTING` -> NOT treated as definitive here,
    deliberately more conservative than the PREPARED case: `SUBMITTING`
    means a broker call was already in flight (or about to be) when the
    prior run stopped, and a query racing a real in-flight submission
    can plausibly still 404 briefly. Transitions to `UNCERTAIN` --
    fail-closed, human review required, never auto-resubmitted.

    `intent.status` outside {PREPARED, SUBMITTING} (e.g. already
    BROKER_ACKNOWLEDGED) is returned unchanged -- see this module's own
    docstring for why "broker acknowledged, local state not yet caught
    up" is intentionally left to `broker_reconciliation.py`'s own
    Scenario C, not resolved a second time here."""
    if intent.status not in (order_intent.PREPARED, order_intent.SUBMITTING):
        return intent

    found = order_submission.get_order_by_client_order_id(client, intent.client_order_id)

    if intent.status == order_intent.PREPARED:
        if found is None:
            # PREPARED -> TERMINAL directly (never via SUBMITTING, which
            # `_VALID_TRANSITIONS` does not allow to reach TERMINAL --
            # and semantically correct anyway: claiming SUBMITTING was
            # ever reached would be false for something we are concluding
            # never happened).
            return order_intent.transition_intent(
                intent, order_intent.TERMINAL,
                last_error="stray-recovery: confirmed 404 from PREPARED -- broker never received this submission.",
            )
        intent = order_intent.transition_intent(intent, order_intent.SUBMITTING, increment_attempt=True)
        return order_intent.transition_intent(
            intent, order_intent.BROKER_ACKNOWLEDGED,
            broker_order_id=str(found.id), broker_status=_status_str(found),
        )

    # intent.status == SUBMITTING
    if found is None:
        return order_intent.transition_intent(
            intent, order_intent.UNCERTAIN,
            last_error="stray-recovery: no order found by client_order_id from SUBMITTING -- ambiguous, fail-closed.",
        )
    return order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED,
        broker_order_id=str(found.id), broker_status=_status_str(found),
    )


def resolve_stray_cancel_intent(client: TradingClient, intent: order_intent.OrderIntent) -> order_intent.OrderIntent:
    """Resolves a stray CANCEL_PROTECTIVE_STOP intent (has a real
    `target_broker_order_id` -- the resting stop it was trying to
    cancel; no `client_order_id`, per `OrderIntent`'s own field
    comments). Queries the TARGET order's current broker status and
    transitions to reflect what actually happened -- never auto-retries
    the cancel.

    Target now shows a real terminal, non-`canceled` status (most
    commonly `filled` -- the stop triggered before the cancel could
    take effect, the same real race `_execute_equity_orders`'s own
    `needs_review` path already documents) -> a definitive outcome
    either way; transitions through SUBMITTING (if not already past it)
    to BROKER_ACKNOWLEDGED, recording the real status honestly (never
    silently relabeled as `canceled`).

    Target still genuinely resting (`_STILL_RESTING_STATUSES`) from
    `PREPARED` -> a definitive negative: the cancel request was never
    attempted. Safe to close TERMINAL -- the next real run's own normal
    signal-exit flow will attempt the cancel fresh.

    Target still resting from `SUBMITTING` -> ambiguous (the cancel
    call may have been in flight when the prior run stopped, and
    Alpaca's own cancellation is asynchronous -- see
    `cancel_order_and_confirm`'s own docstring for why a real paper
    stop briefly still read `new` immediately after a real cancel
    request). Transitions to `UNCERTAIN` -- fail-closed, human review
    required, never auto-retried.

    Target shows `canceled` -> the cancel plainly did succeed;
    transitions through SUBMITTING (if needed) to BROKER_ACKNOWLEDGED
    with `broker_status="canceled"`."""
    if intent.status not in (order_intent.PREPARED, order_intent.SUBMITTING):
        return intent

    target_status = order_submission.get_order_status(client, intent.target_broker_order_id)
    still_resting = target_status in _STILL_RESTING_STATUSES

    if intent.status == order_intent.PREPARED:
        if still_resting:
            # PREPARED -> TERMINAL directly, same reasoning as the
            # submission-intent sibling function: SUBMITTING has no
            # valid path to TERMINAL, and claiming it was reached would
            # be false for a cancel we're concluding was never attempted.
            return order_intent.transition_intent(
                intent, order_intent.TERMINAL,
                last_error=(
                    "stray-recovery: target order still resting from PREPARED -- the cancel "
                    "was never attempted; abandoned, the next run's normal flow will retry."
                ),
            )
        intent = order_intent.transition_intent(intent, order_intent.SUBMITTING, increment_attempt=True)
        return order_intent.transition_intent(
            intent, order_intent.BROKER_ACKNOWLEDGED, broker_status=target_status,
        )

    # intent.status == SUBMITTING
    if still_resting:
        return order_intent.transition_intent(
            intent, order_intent.UNCERTAIN,
            last_error=(
                f"stray-recovery: target order still resting ({target_status!r}) from SUBMITTING -- "
                f"ambiguous whether the cancel request was ever sent, fail-closed."
            ),
        )
    return order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_status=target_status,
    )


def resolve_stray_intent(client: TradingClient, intent: order_intent.OrderIntent) -> order_intent.OrderIntent:
    """Dispatches to `resolve_stray_order_submission_intent` or
    `resolve_stray_cancel_intent` based on which correlation key the
    intent actually carries -- a submission-type intent has a real
    `client_order_id`; a cancel-type intent has a real
    `target_broker_order_id` and an empty `client_order_id` (see
    `OrderIntent`'s own field comments). Raises `ValueError` (fail-closed,
    never a silent guess) for an intent carrying neither, which should be
    structurally impossible given how `create_intent` is called
    throughout this codebase, but this function does not assume that
    invariant holds without checking."""
    if intent.client_order_id:
        return resolve_stray_order_submission_intent(client, intent)
    if intent.target_broker_order_id:
        return resolve_stray_cancel_intent(client, intent)
    raise ValueError(
        f"Intent {intent.intent_id} ({intent.ticker} {intent.action_kind}) has neither a "
        f"client_order_id nor a target_broker_order_id -- cannot determine how to resolve it "
        f"against the broker. Fail-closed."
    )


def _status_str(order) -> str:
    status = order.status
    return str(status.value if hasattr(status, "value") else status)
