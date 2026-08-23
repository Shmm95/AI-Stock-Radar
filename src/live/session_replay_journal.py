"""Durable session-level replay journal -- "Missing-Equity-Session
Remediation" design (workspace-c), section D.

WHY A NEW JOURNAL TYPE, NOT AN EXTENSION OF `order_intent.py`: that
module's own `OrderIntent` dataclass is structurally order-shaped
(`client_order_id`, `side`, `quantity`, `stop_price`, `broker_order_id`/
`broker_status`...) -- a session replay has no broker order at all, it
represents "did the market/data timeline get durably advanced by
exactly one session, with the right evidence recorded." Session
replay's own `_VALID_TRANSITIONS` are also genuinely different in
shape (see below) -- there is no broker acknowledgment step, and
"COMMITTED" means something different (the state save actually landed,
not "a broker confirmed an order"). Reusing `OrderIntent` would force
every session record to carry a pile of always-`None` order fields and
would blur two genuinely different kinds of "did this thing durably
happen" journal. This module instead reuses `order_intent.py`'s own
GENERIC storage pattern -- `position_state._atomic_write_text`/
`_GUARD_DIRECTORY`, one JSON file per record -- without reusing its
order-shaped dataclass or state machine.

STATE MACHINE:

    PREPARED --> VERIFIED --> COMMITTED --> TERMINAL
       |
       v
   TERMINAL

- PREPARED: a session replay attempt has been decided and durably
  recorded, but `rdd.run_daily_decision()` has not been called yet for
  this session.
- VERIFIED: `rdd.run_daily_decision()` returned, and the orchestrator
  has confirmed the ACTUAL processed session
  (`decision["as_of_bar_timestamp_equity"]`) matches the session this
  intent targets -- the same "did the fetch really produce what we
  expected" check `run_control_arm_decision._verify_actual_equity_session`
  already established as the right verification point, mirrored here
  for the session-replay path.
- COMMITTED: the orchestrator's own "second save" (re-stamping
  `last_processed_equity_session_date`/`last_processed_equity_date`/
  `last_processed_crypto_date`, the same pattern
  `run_control_arm_decision._stamp_last_processed_dates` already
  established) has durably landed on disk.
- TERMINAL: this session's replay is fully done, one way or another --
  either a successful PREPARED->VERIFIED->COMMITTED->TERMINAL chain, or
  an abandoned/failed attempt closed TERMINAL directly from PREPARED or
  VERIFIED (never silently deleted -- crash evidence stays on disk,
  same discipline as `order_intent.py`).

`outcome` (added 2026-08-23, independent audit round 3 finding #3b): a
TERMINAL record alone does not say WHICH of the two paths above
produced it -- before this field existed, a caller finding an existing
TERMINAL intent for a given `session_id` had no way to tell "this
session was already successfully replayed" (safe to treat as a no-op)
apart from "a PRIOR attempt for this exact session_id already FAILED"
(never safe to silently treat as done -- the real bug this closes: a
failed attempt's own TERMINAL record was being read by
`equity_session_orchestrator._replay_one_session` as proof of success
on every subsequent invocation, permanently masking the failure).
`SUCCEEDED`/`FAILED` are the only two values ever written; `None` only
for a record created before this field existed (defensive, not expected
in practice -- see `from_dict`'s own handling).

`session_id` is the deterministic identity
`EQUITY_SESSION:<YYYY-MM-DD>:<bar_set_sha256>` -- built by
`build_session_id` below, never hand-assembled elsewhere. Including the
bar-set hash (not just the date) means a session whose bar set later
turns out to have been INCOMPLETE at first attempt (a partial fetch
that got a different hash) is treated as a genuinely different replay
attempt, not silently conflated with an earlier, different one for the
same calendar date.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.live.position_state import _GUARD_DIRECTORY, _atomic_write_text

SESSION_REPLAY_DIRECTORY = _GUARD_DIRECTORY / "session_replay_intents"

PREPARED = "PREPARED"
VERIFIED = "VERIFIED"
COMMITTED = "COMMITTED"
TERMINAL = "TERMINAL"

# `outcome` values -- ONLY meaningful once `status == TERMINAL` (see
# `SessionReplayIntent.outcome`'s own field comment above).
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"

_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    PREPARED: frozenset({VERIFIED, TERMINAL}),
    VERIFIED: frozenset({COMMITTED, TERMINAL}),
    COMMITTED: frozenset({TERMINAL}),
    TERMINAL: frozenset(),
}


class InvalidSessionTransitionError(RuntimeError):
    """Raised when a transition is attempted that `_VALID_TRANSITIONS`
    does not allow -- fail-closed, same discipline as
    `order_intent.InvalidTransitionError`."""


def build_session_id(session_date: str, bar_set_sha256: str) -> str:
    """The ONE canonical session-id builder -- never hand-assembled at
    any call site. `EQUITY_SESSION:<date>:<hash>`, per the design's own
    section D, item 1."""
    return f"EQUITY_SESSION:{session_date}:{bar_set_sha256}"


@dataclass
class SessionReplayIntent:
    intent_id: str
    session_id: str
    session_date: str
    bar_set_sha256: str
    status: str
    replay_sequence_index: int
    replay_sequence_total: int
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    broker_snapshot_fetched_at: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    last_reconciled_at: str | None = None
    last_error: str | None = None
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionReplayIntent":
        return cls(**payload)


def _intent_path(intent_id: str) -> Path:
    return SESSION_REPLAY_DIRECTORY / f"{intent_id}.json"


def _write_intent(intent: SessionReplayIntent) -> None:
    SESSION_REPLAY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(_intent_path(intent.intent_id), json.dumps(intent.to_dict(), indent=2, sort_keys=True) + "\n")


def create_session_intent(
    *,
    session_id: str,
    session_date: str,
    bar_set_sha256: str,
    replay_sequence_index: int,
    replay_sequence_total: int,
    state_hash_before: str | None = None,
) -> SessionReplayIntent:
    """Durably records a new session replay attempt in PREPARED state
    BEFORE `rdd.run_daily_decision()` is ever called for it. `intent_id`
    (a plain, ascending-safe identifier built from `session_id` itself,
    since a session's own id is already globally unique per
    `build_session_id`'s own construction) is what the rest of this
    intent's lifecycle is addressed by."""
    intent = SessionReplayIntent(
        intent_id=session_id.replace(":", "_"),
        session_id=session_id,
        session_date=session_date,
        bar_set_sha256=bar_set_sha256,
        status=PREPARED,
        replay_sequence_index=replay_sequence_index,
        replay_sequence_total=replay_sequence_total,
        state_hash_before=state_hash_before,
    )
    _write_intent(intent)
    return intent


def load_session_intent(intent_id: str) -> SessionReplayIntent:
    payload = json.loads(_intent_path(intent_id).read_text(encoding="utf-8"))
    return SessionReplayIntent.from_dict(payload)


def list_session_intents() -> list[SessionReplayIntent]:
    """All session-replay intents currently on disk, any status -- used
    by tests and by crash-recovery callers to find leftover
    PREPARED/VERIFIED intents from an interrupted prior run."""
    if not SESSION_REPLAY_DIRECTORY.is_dir():
        return []
    intents = []
    for path in sorted(SESSION_REPLAY_DIRECTORY.glob("*.json")):
        intents.append(SessionReplayIntent.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return intents


def find_session_intent_by_id(session_id: str) -> SessionReplayIntent | None:
    """LOCAL lookup only, makes no Alpaca call -- searches on-disk
    intents for a matching `session_id` (deterministic per
    `build_session_id`, so at most one should ever match)."""
    for intent in list_session_intents():
        if intent.session_id == session_id:
            return intent
    return None


def transition_session_intent(
    intent: SessionReplayIntent,
    new_status: str,
    *,
    state_hash_after: str | None = None,
    broker_snapshot_fetched_at: str | None = None,
    last_error: str | None = None,
    outcome: str | None = None,
) -> SessionReplayIntent:
    """Validates the transition against `_VALID_TRANSITIONS`, updates
    the intent in place, and re-writes it durably. Raises
    `InvalidSessionTransitionError` rather than silently allowing an
    out-of-order state change -- fail-closed, same discipline as
    `order_intent.transition_intent`.

    `outcome` (`SUCCEEDED`/`FAILED`, added independent-audit-round-3
    finding #3b): MANDATORY whenever `new_status == TERMINAL` -- a
    caller closing an intent TERMINAL without saying which of the two
    reasons it is closing for is exactly the bug this field exists to
    prevent (see `SessionReplayIntent.outcome`'s own field comment).
    Rejected (never silently defaulted) for any other `new_status`,
    since it is meaningless before TERMINAL."""
    allowed = _VALID_TRANSITIONS.get(intent.status, frozenset())
    if new_status not in allowed:
        raise InvalidSessionTransitionError(
            f"Session intent {intent.intent_id}: {intent.status} -> {new_status} is not a valid "
            f"transition (allowed from {intent.status}: {sorted(allowed) or 'none, terminal'})."
        )
    if new_status == TERMINAL:
        if outcome not in (SUCCEEDED, FAILED):
            raise ValueError(
                f"transition_session_intent(..., TERMINAL) requires outcome=SUCCEEDED or "
                f"outcome=FAILED -- got {outcome!r}. Fail-closed: a TERMINAL record must always "
                f"say which of the two it is (independent-audit-round-3 finding #3b)."
            )
    elif outcome is not None:
        raise ValueError(f"outcome is only meaningful for new_status=TERMINAL, not {new_status!r}.")
    intent.status = new_status
    if state_hash_after is not None:
        intent.state_hash_after = state_hash_after
    if broker_snapshot_fetched_at is not None:
        intent.broker_snapshot_fetched_at = broker_snapshot_fetched_at
    if last_error is not None:
        intent.last_error = last_error
    if outcome is not None:
        intent.outcome = outcome
    if new_status in (COMMITTED, TERMINAL):
        intent.last_reconciled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_intent(intent)
    return intent
