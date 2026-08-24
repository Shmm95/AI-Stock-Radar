"""Durable, per-order intent journal -- Phase 1 (infrastructure only).

WHY THIS EXISTS: an order submission has a window where the caller does
not yet know whether the broker actually received it (a network error,
a timeout, a crash mid-call) -- `submitted_actions` (the existing ledger
inside `position_state.json`, see that module's own docstring) is only
ever written AFTER a submission attempt completes, so a crash DURING
the attempt leaves no record at all of "we were about to do this." This
journal exists to close that gap: an intent is durably recorded in
PREPARED state BEFORE any broker call is attempted, so a crash mid-flight
leaves real, on-disk evidence of what was about to happen, in a
well-defined incomplete state (`PREPARED` or `SUBMITTING`) rather than
silence.

PHASE 1 SCOPE, EXPLICITLY: this module provides the durable
state-machine and storage only. It does NOT decide what to do with an
`UNCERTAIN` or stuck `PREPARED`/`SUBMITTING` intent found on disk --
that is broker-reconciliation logic (Phase 2, a separate, later task).
Nothing here calls Alpaca, and nothing here is wired into
`run_daily_decision.py` (never modified) or any real order-submission
code path -- see `scripts/run_control_arm_decision.py` for the one
current integration, which uses this journal around its own decision
commit, not around a real per-order broker call (the control arm submits
no real orders today; this is infrastructure built ahead of that, so it
already exists and is already tested once real order submission is
ever enabled here).

STORAGE: one JSON file per intent, `$AI_STOCK_RADAR_GUARD_DIR/order_intents/<intent_id>.json`
-- same guard directory `position_state.py` already resolves
`$AI_STOCK_RADAR_GUARD_DIR`/`~/.ai_stock_radar_guard` from (imported
directly from there, not re-implemented, so the two can never drift).
Written with the exact same atomic discipline as the high-water-mark
file and `position_state.json` -- `_atomic_write_text` (temp file in
the same directory, `fsync`, `os.replace`, directory `fsync`), imported
from `position_state.py` rather than duplicated.

`intent_id` IS PERSISTENT -- generated once (`uuid.uuid4()`) at
`create_intent()` time and never recomputed from inputs like a date or
ticker; the file's own name IS its identity for the rest of that
intent's lifecycle, and every subsequent read/write for it goes through
the same id.

STATE MACHINE:

    PREPARED --> SUBMITTING --> BROKER_ACKNOWLEDGED --> COMMITTED --> TERMINAL
       |              |                  |
       v              v                  v
   TERMINAL       UNCERTAIN          UNCERTAIN
                       \\                /
                        `-----> TERMINAL (Phase 2 resolves this, not here)

- PREPARED: the intent has been decided locally and durably recorded,
  but no broker call has been attempted yet.
- SUBMITTING: a broker call is in flight (or about to be). If the
  process dies here, a PREPARED-or-SUBMITTING file surviving on disk is
  exactly the "we don't know what happened" evidence this journal exists
  to preserve -- see this module's own docstring intro.
- BROKER_ACKNOWLEDGED: the broker call returned a definite response
  (Phase 1: since no real broker call exists yet in this codebase's
  current control-arm usage, this transition currently represents "the
  local decision this intent tracks is confirmed," a scaffold for where
  a real Alpaca acknowledgment will plug in once Phase 2/real order
  submission exists here -- documented explicitly, not silently
  overclaimed as a real broker ack today).
- COMMITTED: local state (`position_state.json`) has been durably
  updated to reflect this intent's outcome.
- TERMINAL: nothing further will happen to this intent -- either a
  normal successful end (from COMMITTED) or an abandoned/never-submitted
  one (from PREPARED, e.g. a crash-recovery decision made by a human or
  by Phase 2 logic that does not exist yet).
- UNCERTAIN: reachable from SUBMITTING or BROKER_ACKNOWLEDGED when
  something ambiguous happened (a network error with no clear
  success/failure signal). Phase 1 deliberately does NOT auto-resolve
  this -- an intent left in UNCERTAIN stays there until Phase 2's
  broker-reconciliation logic (or a human) looks at it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.live.position_state import _GUARD_DIRECTORY, _atomic_write_text

INTENT_DIRECTORY = _GUARD_DIRECTORY / "order_intents"

PREPARED = "PREPARED"
SUBMITTING = "SUBMITTING"
BROKER_ACKNOWLEDGED = "BROKER_ACKNOWLEDGED"
COMMITTED = "COMMITTED"
TERMINAL = "TERMINAL"
UNCERTAIN = "UNCERTAIN"

_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    PREPARED: frozenset({SUBMITTING, TERMINAL}),
    SUBMITTING: frozenset({BROKER_ACKNOWLEDGED, UNCERTAIN}),
    BROKER_ACKNOWLEDGED: frozenset({COMMITTED, UNCERTAIN}),
    COMMITTED: frozenset({TERMINAL}),
    UNCERTAIN: frozenset({TERMINAL}),
    TERMINAL: frozenset(),
}

# The single, canonical definition of "not yet resolved" -- every status
# except TERMINAL. Independent-audit finding (2026-08-24): two other
# modules (`order_intent_reconciliation.py`,
# `scripts/run_control_arm_decision.py`) each maintained their own
# hand-copied non-terminal status set, and one omitted COMMITTED (the
# narrow crash window between a COMMITTED write and the following
# TERMINAL write -- see `_VALID_TRANSITIONS` above, COMMITTED's only
# next state is TERMINAL). That specific omission never caused a bug
# (nothing yet queried run_control_arm_decision.py's copy for a
# COMMITTED intent), but two modules independently defining "unresolved
# intent" is itself the defect: the next person to touch either copy
# has no signal that a sibling definition exists to keep in sync. Both
# callers now import `NON_TERMINAL_STATUSES`/`is_non_terminal` from here
# instead of maintaining their own frozenset.
NON_TERMINAL_STATUSES = frozenset({PREPARED, SUBMITTING, BROKER_ACKNOWLEDGED, COMMITTED, UNCERTAIN})


def is_non_terminal(status: str) -> bool:
    """True for any status other than TERMINAL. See `NON_TERMINAL_STATUSES`
    above for why this is the one place that decision is made."""
    return status in NON_TERMINAL_STATUSES


class InvalidTransitionError(RuntimeError):
    """Raised when a transition is attempted that `_VALID_TRANSITIONS`
    does not allow -- fail-closed, same discipline as this codebase's
    other state-machine guards (e.g. `RollbackDetectedError`)."""


@dataclass
class OrderIntent:
    intent_id: str
    client_order_id: str
    account_identity: str
    ticker: str
    side: str  # "BUY" | "SELL"
    order_type: str  # "market" | "stop" | ...
    quantity: float | None
    notional: float | None
    stop_price: float | None
    action_kind: str  # e.g. "ENTRY_MARKET_BUY", "PROTECTIVE_STOP", "SIGNAL_EXIT_MARKET_SELL"
    source_signal_timestamp: str
    parent_intent_id: str | None
    status: str
    broker_order_id: str | None = None
    broker_status: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    last_reconciled_at: str | None = None
    attempt_count: int = 0
    pre_state_hash: str | None = None
    last_error: str | None = None
    # Added 2026-08-22 (Task 3 -- protective-stop/cancel journal
    # coverage): `client_order_id` is meaningless for a CANCEL -- Alpaca's
    # cancel API takes a broker order id, never generates or accepts a
    # client_order_id of its own. `operation_id` is this journal's own
    # LOCAL-only deterministic correlation key for a non-order-submission
    # action (today: only CANCEL_PROTECTIVE_STOP), built by
    # `run_daily_decision._deterministic_operation_id` -- never sent to
    # the broker, never compared against anything broker-side.
    # `target_broker_order_id` is the id of the REST-ING order a cancel
    # intent targets (e.g. the protective stop being canceled) -- distinct
    # from `broker_order_id` above, which (once set) is the id of the
    # cancel CONFIRMATION itself. A submission-type intent leaves both
    # of these `None`; a cancel-type intent leaves `client_order_id`
    # empty/unused and populates these two instead.
    operation_id: str | None = None
    target_broker_order_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OrderIntent":
        return cls(**payload)


def _intent_path(intent_id: str) -> Path:
    return INTENT_DIRECTORY / f"{intent_id}.json"


def _write_intent(intent: OrderIntent) -> None:
    INTENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        _intent_path(intent.intent_id),
        json.dumps(intent.to_dict(), indent=2, sort_keys=True) + "\n",
    )


def create_intent(
    *,
    client_order_id: str,
    account_identity: str,
    ticker: str,
    side: str,
    order_type: str,
    action_kind: str,
    source_signal_timestamp: str,
    quantity: float | None = None,
    notional: float | None = None,
    stop_price: float | None = None,
    parent_intent_id: str | None = None,
    pre_state_hash: str | None = None,
    operation_id: str | None = None,
    target_broker_order_id: str | None = None,
) -> OrderIntent:
    """Durably record a new intent in PREPARED state BEFORE any broker
    call is attempted. `intent_id` is generated here, once, and is what
    the rest of this intent's lifecycle is addressed by.

    `operation_id`/`target_broker_order_id` -- see `OrderIntent`'s own
    field comments (added 2026-08-22, Task 3). For a real order
    submission, leave both `None` and pass `client_order_id` as usual.
    For a CANCEL (no client_order_id exists), pass `client_order_id=""`
    and supply both of these instead."""
    intent = OrderIntent(
        intent_id=str(uuid.uuid4()),
        client_order_id=client_order_id,
        account_identity=account_identity,
        ticker=ticker,
        side=side,
        order_type=order_type,
        quantity=quantity,
        notional=notional,
        stop_price=stop_price,
        action_kind=action_kind,
        source_signal_timestamp=source_signal_timestamp,
        parent_intent_id=parent_intent_id,
        status=PREPARED,
        pre_state_hash=pre_state_hash,
        operation_id=operation_id,
        target_broker_order_id=target_broker_order_id,
    )
    _write_intent(intent)
    return intent


def load_intent(intent_id: str) -> OrderIntent:
    path = _intent_path(intent_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OrderIntent.from_dict(payload)


def list_intents() -> list[OrderIntent]:
    """All intents currently on disk, any status -- used by tests and
    (in Phase 2) by broker-reconciliation to find PREPARED/SUBMITTING/
    UNCERTAIN intents left over from an interrupted prior run."""
    if not INTENT_DIRECTORY.is_dir():
        return []
    intents = []
    for path in sorted(INTENT_DIRECTORY.glob("*.json")):
        intents.append(OrderIntent.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return intents


class MultipleActiveIntentsForClientOrderIdError(RuntimeError):
    """Raised by `find_intent_by_client_order_id` when MORE THAN ONE
    non-TERMINAL intent shares a `client_order_id` -- independent-audit
    finding, 2026-08-24 (the second reboot-drill gap): this should be
    structurally impossible (one live candidate, one active record) and
    signals real duplicate/divergent journal state. Fail-closed rather
    than silently picking one and hiding the anomaly from the caller."""


def find_intent_by_client_order_id(client_order_id: str) -> OrderIntent | None:
    """LOCAL lookup only -- searches every on-disk intent for a matching
    `client_order_id`. `client_order_id` is generated deterministically
    per ticker+action_kind+session-date (see
    `run_control_arm_decision._compute_pending_signal_id`/
    `_run_intent_protocol`), so in the common case at most one intent
    ever matches -- but `intent_id` (a UUID), not `client_order_id`, is
    this journal's real storage key, and `_write_blind_prepared_intents`
    deliberately creates a fresh intent when a leftover record for the
    same `client_order_id` is TERMINAL (reboot-drill finding #1's own
    fix) rather than reusing or deleting it, as durable historical
    evidence. So TWO records sharing one `client_order_id` -- one
    TERMINAL, one live -- is an expected, legitimate state, not a bug.

    Independent-audit finding, 2026-08-24 (the second reboot-drill gap):
    the ORIGINAL version of this function returned the first match in
    on-disk (UUID-sorted) order, silently assuming uniqueness. Once two
    records could legitimately share a `client_order_id`, that could
    return the WRONG one -- e.g. an old TERMINAL record instead of the
    genuinely active one a caller like `_write_blind_prepared_intents`
    or the write-ahead-evidence check needs. Reproduced for real before
    this fix: a lookup for a client_order_id with one TERMINAL and one
    fresh PREPARED record returned the TERMINAL one whenever its
    `intent_id` happened to sort first.

    Disambiguation rule, now explicit: among all matches,
      - exactly one non-TERMINAL match -> return it (the one genuinely
        active record; TERMINAL matches are dead ends, never confused
        with it);
      - more than one non-TERMINAL match -> `MultipleActiveIntentsForClientOrderIdError`,
        fail-closed (a real anomaly: two live journal records for one
        candidate should never happen);
      - zero non-TERMINAL matches (every match is TERMINAL -- the
        normal end state once a superseded client_order_id's newer
        record also completes) -> the most recently created one,
        purely informational since nothing here is "active" for a
        caller to mistake for a live record.

    Added for reboot-recovery: after a real crash/restart, a caller who
    already knows a specific `client_order_id` (e.g. one just fetched
    from the broker via `order_submission.get_order_by_client_order_id`
    -- the SAME `TradingClient.get_order_by_client_id` SDK method,
    called from THAT module, never duplicated here) can use this
    function to find which local intent, if any, that broker order
    corresponds to, and pick up its state machine from wherever it was
    left (PREPARED/SUBMITTING/BROKER_ACKNOWLEDGED/UNCERTAIN).

    Deliberately makes NO Alpaca call itself -- this module's own
    documented scope (see module docstring: "Nothing here calls
    Alpaca") stays intact; correlating a broker-side lookup with this
    local one is the CALLER's job, not this function's."""
    matches = [intent for intent in list_intents() if intent.client_order_id == client_order_id]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    non_terminal = [intent for intent in matches if intent.status != TERMINAL]
    if len(non_terminal) == 1:
        return non_terminal[0]
    if len(non_terminal) > 1:
        raise MultipleActiveIntentsForClientOrderIdError(
            f"client_order_id={client_order_id!r} has {len(non_terminal)} non-TERMINAL "
            f"intents on disk: {[intent.intent_id for intent in non_terminal]} -- exactly "
            f"one active record is expected. Fail-closed."
        )
    return max(matches, key=lambda intent: intent.created_at)


def find_intent_by_operation_id(operation_id: str) -> OrderIntent | None:
    """The cancel-intent counterpart to `find_intent_by_client_order_id`
    above -- see `OrderIntent.operation_id`'s own field comment (added
    2026-08-22, Task 3) for why a CANCEL needs a separate correlation
    key (Alpaca's cancel API has no client_order_id of its own to key
    off). LOCAL lookup only, same as the sibling function -- makes no
    Alpaca call itself."""
    for intent in list_intents():
        if intent.operation_id == operation_id:
            return intent
    return None


def transition_intent(
    intent: OrderIntent,
    new_status: str,
    *,
    broker_order_id: str | None = None,
    broker_status: str | None = None,
    last_error: str | None = None,
    increment_attempt: bool = False,
    quantity: float | None = None,
    notional: float | None = None,
    stop_price: float | None = None,
) -> OrderIntent:
    """Validates the transition against `_VALID_TRANSITIONS`, updates
    the intent in place, and re-writes it durably. Raises
    `InvalidTransitionError` rather than silently allowing an
    out-of-order state change (e.g. PREPARED -> COMMITTED directly) --
    fail-closed, same as this journal's other guarantees.

    `quantity`/`notional`/`stop_price` -- added for the write-ahead
    "blind" intent case (see `scripts/run_control_arm_decision.py`'s
    `_write_blind_prepared_intents`): a blind PREPARED intent is written
    before the real fill quantity/price are known, so this lets the
    SAME on-disk intent be updated with the real numbers at the same
    time it transitions PREPARED -> SUBMITTING, instead of requiring a
    second, separate write. Same `is not None` convention as
    `broker_order_id`/`broker_status`/`last_error` above -- only set if
    a caller actually passes a value."""
    allowed = _VALID_TRANSITIONS.get(intent.status, frozenset())
    if new_status not in allowed:
        raise InvalidTransitionError(
            f"Intent {intent.intent_id}: {intent.status} -> {new_status} is not "
            f"a valid transition (allowed from {intent.status}: {sorted(allowed) or 'none, terminal'})."
        )
    intent.status = new_status
    if broker_order_id is not None:
        intent.broker_order_id = broker_order_id
    if broker_status is not None:
        intent.broker_status = broker_status
    if last_error is not None:
        intent.last_error = last_error
    if quantity is not None:
        intent.quantity = quantity
    if notional is not None:
        intent.notional = notional
    if stop_price is not None:
        intent.stop_price = stop_price
    if increment_attempt:
        intent.attempt_count += 1
    if new_status in (COMMITTED, TERMINAL):
        intent.last_reconciled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_intent(intent)
    return intent
