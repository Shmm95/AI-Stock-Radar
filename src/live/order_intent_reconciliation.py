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

TWO-PHASE RECOVERY, NOT ONE (reboot-drill findings #1/#2, 2026-08-24 --
a real, end-to-end reboot-drill reproduction found two genuine bugs in
what used to be a single, greedy finalization pass; both confirmed
against the real code and a real crash reproduction before being fixed,
not assumed):

  Phase A (THIS module, called early, before any market-data fetch):
  resolve broker TRUTH only. `resolve_stray_intent` may now return
  `PREPARED` (RESUMABLE_PREPARED -- a confirmed 404 whose candidate is
  STILL live this run; see `resolve_stray_order_submission_intent`'s
  own docstring) or `BROKER_ACKNOWLEDGED` (broker truth known, but
  local-state reconciliation has not happened/completed yet; see
  `finalize_resolved_intent`'s own docstring) as LEGITIMATE, non-error
  resting outcomes -- not just `TERMINAL`/`UNCERTAIN` as before. A
  caller that still treats "anything other than TERMINAL" as a hard
  failure will reject these two correct outcomes; see
  `run_control_arm_decision.py`'s own updated Phase-1.5 acceptance set.

  Phase B (the CALLER's own broker/local-state reconciliation --
  `broker_reconciliation.reconcile()`, unmodified, run separately right
  after this module's own preflight): actually catches
  `position_state.json` up (or correctly halts fail-closed if it
  can't -- see that module's own new
  `JournalAcknowledgedButLocalStateMissingError`, which now
  distinguishes "a known, journal-correlated crash-recovery gap" from a
  genuinely mystery broker order).

  Only ONCE local state is genuinely consistent (a state THIS module
  never itself verifies) is it safe for anything to advance a
  BROKER_ACKNOWLEDGED intent further -- and even then, ONLY when
  `broker_status` shows the order never had any real position/cash
  impact (canceled/expired/rejected/done_for_day) does
  `finalize_resolved_intent` do that automatically; anything else
  (filled, or still resting) requires the run's own normal
  decision/finalization flow, or human review, never an automatic
  advance from this module alone.

NEVER DO (workspace-c's own explicit warning, reboot-drill findings
#1/#2): move `broker_reconciliation.reconcile()` earlier so it runs
BEFORE this module (Scenario D still triggers on the same case, this
does not resolve anything, only reorders when the same halt happens);
silently add a recovered client_order_id to some "known" list so
Scenario D skips it (masks a real local-state gap, letting a run
proceed with incomplete state); or leave an intent artificially at
TERMINAL to suppress Scenario D (the single riskiest option -- a
lying journal record next to a state a human now has no reason to
suspect is inconsistent).

Nothing here calls Alpaca on import or at module load -- every function
below takes a real `TradingClient` as an explicit parameter, same
injection-seam discipline the rest of this codebase's live modules use.
"""

from __future__ import annotations

from alpaca.trading.client import TradingClient

from src.live import order_intent
from src.live import order_submission

_NON_TERMINAL_STATUSES = frozenset(
    {
        order_intent.PREPARED, order_intent.SUBMITTING, order_intent.BROKER_ACKNOWLEDGED,
        # COMMITTED (added independent-audit round 2, 2026-08-23): a
        # crash in the narrow window AFTER the COMMITTED write lands but
        # BEFORE the following TERMINAL write -- see
        # `finalize_resolved_intent`'s own docstring -- left such an
        # intent permanently invisible to this stray scan, since
        # COMMITTED was (incorrectly) treated as "as good as done."
        # order_intent.py's own `_VALID_TRANSITIONS` has always allowed
        # COMMITTED -> TERMINAL; nothing was ever calling it for a
        # leftover-from-a-prior-run COMMITTED intent.
        order_intent.COMMITTED,
        order_intent.UNCERTAIN,
    }
)

# Alpaca order statuses this module treats as "the target order is
# still genuinely resting" -- i.e. not yet acted on. Mirrors
# `order_submission`'s own terminal/non-terminal vocabulary, restated
# here rather than imported, since this module cares about a
# specifically narrower question ("is it still open") than that
# module's own `TERMINAL_STATUSES` set answers.
_STILL_RESTING_STATUSES = frozenset({"new", "accepted", "pending_new", "held", "replaced"})

# Reboot-drill finding #2 (2026-08-24): the subset of
# `order_submission.TERMINAL_STATUSES` that has NO position/cash impact
# -- deliberately EXCLUDES "filled" (real position impact) and
# "partially_filled" (not even in that set; still open). Only when a
# recovered BROKER_ACKNOWLEDGED intent's own `broker_status` is one of
# these is it safe for `finalize_resolved_intent` to claim COMMITTED
# ("local state now reflects this") without any real local-state
# catch-up having happened -- there is nothing for local state to catch
# up ON, since the order never took effect.
_NO_POSITION_IMPACT_TERMINAL_STATUSES = frozenset({"canceled", "expired", "rejected", "done_for_day"})


def find_stray_session_intents_from_prior_run() -> list[order_intent.OrderIntent]:
    """Every intent currently on disk that is NOT in a terminal state
    (PREPARED/SUBMITTING/BROKER_ACKNOWLEDGED/COMMITTED/UNCERTAIN) -- a
    candidate a PRIOR, interrupted run left behind. Local-only, makes no
    broker call itself; see `list_intents()`'s own docstring for why
    this is safe to call cheaply and often."""
    return [intent for intent in order_intent.list_intents() if intent.status in _NON_TERMINAL_STATUSES]


def resolve_stray_order_submission_intent(
    client: TradingClient, intent: order_intent.OrderIntent, *, matches_live_candidate: bool = False,
) -> order_intent.OrderIntent:
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
    exact submission.

    `matches_live_candidate` (reboot-drill finding #1, 2026-08-24; the
    CALLER's own responsibility to compute -- see
    `run_control_arm_decision._stray_intent_matches_live_candidate` --
    since it requires `runner_state.pending_buys`/`pending_exits`, which
    this module deliberately never loads itself): `True` when THIS
    EXACT run still has a live, matching candidate for this intent (same
    ticker/action_kind/client_order_id/account_identity/side/
    source_signal_timestamp) queued in `pending_buys`/`pending_exits`.
    In that case the intent is left UNCHANGED, still `PREPARED` --
    RESUMABLE, not abandoned: the next real run step
    (`_write_blind_prepared_intents`) will naturally reuse this exact
    intent for its own idempotent-resume path, exactly as if this were
    a plain crash-before-`rdd.run_daily_decision()` case (which, from
    the broker's honest perspective, it still is -- the broker
    genuinely never received this submission).

    THE REAL BUG THIS PARAMETER CLOSES: this function used to close
    TERMINAL unconditionally on a confirmed 404 from PREPARED,
    regardless of whether the SAME run still had a live candidate for
    it -- `_write_blind_prepared_intents`'s own conflict check
    (`StrayPreparedIntentConflictError`) then fired on the very next
    step of the SAME run, since it found a TERMINAL record for the
    client_order_id it was about to (idempotently) reuse, not the
    PREPARED it expected. The whole run deadlocked. Confirmed via a
    real, end-to-end reboot-drill reproduction
    (`scripts/run_control_arm_reboot_drill.py`) before this fix, not
    assumed.

    `matches_live_candidate=False` (the default -- no live candidate,
    or the caller has none to check against) -> the ORIGINAL behavior:
    safe to close TERMINAL as `ABANDONED_NO_SUBMISSION` -- the next
    real run's own normal flow will attempt the action fresh through
    its own regular idempotency check, with a fresh, real decision
    behind it, if a new candidate is ever generated again.

    Confirmed 404 from `SUBMITTING` -> NOT treated as definitive here,
    deliberately more conservative than the PREPARED case: `SUBMITTING`
    means a broker call was already in flight (or about to be) when the
    prior run stopped, and a query racing a real in-flight submission
    can plausibly still 404 briefly. Transitions to `UNCERTAIN` --
    fail-closed, human review required, never auto-resubmitted.
    `matches_live_candidate` is irrelevant here -- ambiguity, not a
    definitive negative, so there is nothing safe to resume.

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
            if matches_live_candidate:
                # RESUMABLE_PREPARED: left unchanged, still PREPARED --
                # see this function's own docstring for the full "why."
                return intent
            # PREPARED -> TERMINAL directly (never via SUBMITTING, which
            # `_VALID_TRANSITIONS` does not allow to reach TERMINAL --
            # and semantically correct anyway: claiming SUBMITTING was
            # ever reached would be false for something we are concluding
            # never happened).
            return order_intent.transition_intent(
                intent, order_intent.TERMINAL,
                last_error=(
                    "ABANDONED_NO_SUBMISSION: stray-recovery: confirmed 404 from PREPARED -- broker "
                    "never received this submission, and no live candidate remains for it this run."
                ),
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


def finalize_resolved_intent(intent: order_intent.OrderIntent) -> order_intent.OrderIntent:
    """Closes a stray intent's journal record once (and only once) it is
    actually SAFE to claim done -- COMMITTED (the prior run's own second
    save landed but the final bookkeeping transition to TERMINAL did
    not -- e.g. a crash in the narrow window between those two writes;
    independent-audit-round-2 finding #2, 2026-08-23) always advances to
    TERMINAL, unconditionally: COMMITTED's own defined meaning
    (`order_intent.py`'s own docstring: "local state has been durably
    updated to reflect this intent's outcome") already guarantees local
    state agrees, nothing further to verify.

    BROKER_ACKNOWLEDGED is DIFFERENT, and is where reboot-drill finding
    #2 (2026-08-24) found a real bug: this function used to advance
    BROKER_ACKNOWLEDGED -> COMMITTED -> TERMINAL unconditionally too --
    but for a JUST-recovered stray (this run's own `resolve_stray_*`
    call moved SUBMITTING -> BROKER_ACKNOWLEDGED moments ago because the
    broker turned out to have the order), `position_state.json`'s
    `submitted_actions` has definitely NOT been updated yet -- that
    claim of "COMMITTED" was false. The run would then reach
    `broker_reconciliation.reconcile()`'s own Scenario D, correctly
    detect the broker order as "unknown to local state," and correctly
    halt -- but the journal had ALREADY claimed TERMINAL/done, a
    misleading record sitting next to a correctly-fail-closed halt.
    Confirmed via a real, end-to-end reboot-drill reproduction, not
    assumed.

    Fixed: a BROKER_ACKNOWLEDGED intent advances to COMMITTED here ONLY
    when `broker_status` is one of `_NO_POSITION_IMPACT_TERMINAL_STATUSES`
    (canceled/expired/rejected/done_for_day) -- i.e. the order
    definitively never took effect, so there is genuinely nothing for
    local state to catch up ON, and claiming "local state reflects this"
    is trivially true (there is nothing to reflect). Any OTHER
    `broker_status` (new/accepted/held/replaced/partially_filled/filled
    -- anything that could carry real position/cash impact) leaves the
    intent AT BROKER_ACKNOWLEDGED, deliberately not advanced -- the
    caller's own broker/local-state reconciliation
    (`broker_reconciliation.reconcile()`, run separately, unmodified,
    right after this module's own preflight) is what determines whether
    local state genuinely catches up; ONLY once that has happened (a
    SEPARATE, later call, not made by this function) is it safe for
    something else to advance BROKER_ACKNOWLEDGED -> COMMITTED ->
    TERMINAL. This function itself never re-checks reconciliation state
    -- see `run_control_arm_decision.py`'s own Phase 1.5/Phase 2a
    ordering for where that boundary actually is.

    JOURNAL-ONLY, same scope discipline as every other function in this
    module: never touches `position_state.json` itself.

    A no-op (returns the intent unchanged) for any other status --
    TERMINAL is already done; PREPARED/SUBMITTING are not this
    function's job (that is `resolve_stray_*`'s own job, called before
    this); UNCERTAIN is fail-closed, human-review-only, never
    auto-advanced by anything in this module."""
    if intent.status == order_intent.BROKER_ACKNOWLEDGED:
        if intent.broker_status in _NO_POSITION_IMPACT_TERMINAL_STATUSES:
            intent = order_intent.transition_intent(intent, order_intent.COMMITTED)
        else:
            return intent
    if intent.status == order_intent.COMMITTED:
        intent = order_intent.transition_intent(intent, order_intent.TERMINAL)
    return intent


def resolve_stray_intent(
    client: TradingClient, intent: order_intent.OrderIntent, *, matches_live_candidate: bool = False,
) -> order_intent.OrderIntent:
    """Dispatches to `resolve_stray_order_submission_intent` or
    `resolve_stray_cancel_intent` based on which correlation key the
    intent actually carries -- a submission-type intent has a real
    `client_order_id`; a cancel-type intent has a real
    `target_broker_order_id` and an empty `client_order_id` (see
    `OrderIntent`'s own field comments). Raises `ValueError` (fail-closed,
    never a silent guess) for an intent carrying neither, which should be
    structurally impossible given how `create_intent` is called
    throughout this codebase, but this function does not assume that
    invariant holds without checking.

    `matches_live_candidate` (reboot-drill finding #1, 2026-08-24) is
    forwarded to `resolve_stray_order_submission_intent` only -- see its
    own docstring; irrelevant for a cancel-type intent (no "pending
    candidate" concept applies to those, see
    `resolve_stray_cancel_intent`'s own docstring).

    Always finalizes through `finalize_resolved_intent` before returning
    (independent-audit-round-2 findings #1/#2, 2026-08-23) -- a caller
    of this function never needs to remember to call that separately.
    The possible returned statuses are now: `TERMINAL` (fully resolved,
    safe to treat as done), `PREPARED` (RESUMABLE_PREPARED --
    `matches_live_candidate=True` hit a confirmed 404; reuse it),
    `BROKER_ACKNOWLEDGED` (reboot-drill finding #2 -- broker truth is
    known, but local-state reconciliation has not yet run/completed;
    the caller must let `broker_reconciliation.reconcile()` run next,
    never treat this as "done" on its own), or `UNCERTAIN` (genuinely
    ambiguous, human review required)."""
    if intent.client_order_id:
        resolved = resolve_stray_order_submission_intent(client, intent, matches_live_candidate=matches_live_candidate)
    elif intent.target_broker_order_id:
        resolved = resolve_stray_cancel_intent(client, intent)
    else:
        raise ValueError(
            f"Intent {intent.intent_id} ({intent.ticker} {intent.action_kind}) has neither a "
            f"client_order_id nor a target_broker_order_id -- cannot determine how to resolve it "
            f"against the broker. Fail-closed."
        )
    return finalize_resolved_intent(resolved)


def _status_str(order) -> str:
    status = order.status
    return str(status.value if hasattr(status, "value") else status)
