"""Phase 2a: read-only broker-vs-local reconciliation for the control arm.

WHY THIS EXISTS: `scripts/run_control_arm_decision.py`'s Phase 1 write-ahead
journal (`src/live/order_intent.py`) durably records what this process
*intended* to do, but nothing in Phase 1 ever checks that intention against
what the broker's own records actually say happened. A crash between a real
order acknowledgment and this control arm's next run would leave local state
(`position_state.json`, `submitted_actions`) silently diverged from the
broker's own ledger -- exactly the gap Phase 2a closes, BEFORE
`rdd.run_daily_decision()` computes a new decision on top of possibly-stale
local state.

SCOPE, EXPLICITLY (per this task's own instruction): reconciliation only.
Read-only broker queries plus, in exactly one narrow case (Scenario C, see
below), a local order-STATUS-only correction. Nothing here ever creates or
deletes a `position_state.json` position, cancels or submits a real order,
or adopts an unknown broker order. Pending-signal TTL and any broader
auto-repair logic are separate, later, out of scope here.

ACCOUNT IDENTITY CHECK: the very first broker call this module makes is
`get_account()`. Its `account_number` must end in `_EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX`
-- the last 4 digits of the control account previously verified on the
deploy server, per this task's own explicit instruction. The full
account_number is NEVER logged or included in any exception message; only
the masked form (`...XO4Y`) ever appears. This exists to catch a
credential-file mix-up (e.g. this process accidentally picking up the LIVE
system's Alpaca keys) before any other broker call is trusted.

CONSISTENT SNAPSHOT: broker state can change between two API calls (an
order fills mid-reconciliation). To avoid reconciling against a
torn/inconsistent view, open orders are read twice -- once before and once
after `get_all_positions()` -- and compared on id/status/filled_qty/updated_at.
If the two reads disagree, the whole three-call sequence is retried exactly
once; if it disagrees again, `BrokerSnapshotInconsistentError` is raised
(fail-closed) rather than reconciling against a snapshot that was actively
changing underneath it.

COMPARISON SOURCES: the LOCAL side of every comparison is `submitted_actions`,
`equity_stop_orders`, and `positions` from `position_state.py`'s
`LiveRunnerState` -- never `pending_buys`/`pending_exits` (queued-but-not-yet-
broker-submitted intents, structurally not yet at the broker at all, so
comparing them against broker state would be comparing two different
things). All quantity comparisons use `Decimal(str(x))`, never `float`, and
always the position's TOTAL `qty`, never `qty_available` (a partially
locked/pending-sell quantity is still real size that must reconcile). Any
broker position reported short is an automatic, unconditional block --
this control arm's strategy is long-only; a short position here is never a
legitimate state to reconcile around.

THE FOUR SCENARIOS (exact case table this task specified):

  A. Broker holds a position local state does not know about.
     -> Fail-closed, `UnknownBrokerPositionError`. No automatic local
        position is created. Framed as possible crash recovery (a fill or
        state save interrupted on a prior run) -- but always requires human
        investigation, never auto-resolved.

  B. Local state holds a position the broker does not have.
     -> Fail-closed, `MissingBrokerPositionError`. No automatic local
        position deletion. If an open (non-terminal) ENTRY_MARKET_BUY
        order still exists locally for that ticker, the ticker is
        classified "in-flight" in the error detail (fill not yet
        broker-confirmed) rather than lumped in with an unexplained
        absence -- still fail-closed either way, just a better diagnostic.
        EXCEPT when `orders_enabled=False` (`reconcile()`'s own
        parameter) -- see "ORDERS-DISABLED / DRY-RUN AWARENESS" in
        `reconcile()`'s own docstring: this scenario is EXPECTED, not
        raised, while real order submission is disabled.

  C. A local order record is non-terminal (still "new"/"accepted"/etc.)
     but no longer appears in the broker's open-orders list.
     -> `client.get_order_by_id()` is queried directly. An automatic
        order-STATUS-only update (never a position create/delete) is
        allowed ONLY if ALL of the following hold simultaneously:
          - the order is found (not a 404)
          - its client_order_id matches the local record's (when both
            sides have one)
          - its symbol matches the local record's ticker
          - its side matches the local record's expected side (BUY for
            ENTRY_MARKET_BUY, SELL otherwise)
          - its broker status is exactly "filled"
          - the ticker has a position on BOTH the local and broker side,
            and local qty == broker qty == the order's own filled_qty
            (Decimal-exact)
        Any other outcome (404, client_order_id/symbol/side mismatch,
        canceled/expired/rejected/replaced/any non-"filled" terminal
        status, a quantity mismatch, or a missing position on either
        side) raises `UnreconciledLocalOrderError` -- fail-closed, no
        automatic change.

  D. The broker has an open order local state has no record of at all
     (matched on neither order id nor client_order_id against anything
     in `submitted_actions`).
     -> Fail-closed, `UnknownBrokerOrderError`, full order detail
        included in the exception for manual review. No automatic
        cancellation, no automatic adoption into local state.

TWO ADDITIONAL INVARIANT CHECKS (added after an independent code audit
found the original A-D table alone left these unchecked):

  E. A local `submitted_actions` record is marked terminal, but its
     `order_id` still appears in the broker's own open-orders list.
     -> Fail-closed, `LocalTerminalButBrokerOpenError`. The original
        Scenario C/D loop only ever inspected NON-terminal local
        records, so this contradiction (local bookkeeping says "done,"
        broker says "still live") was never checked at all.

  F. `equity_stop_orders[ticker]` invariant: the referenced `order_id`
     must correspond to a real `submitted_actions` record of kind
     `PROTECTIVE_STOP`, and -- while `ticker` still holds an open local
     position -- that stop must still be genuinely open at the broker.
     -> Fail-closed, `StopOrderInvariantViolationError`. `equity_stop_orders`
        was never read or compared anywhere in the original `reconcile()`;
        a missing/inconsistent protective stop on an open position is a
        real risk-management gap.

SCENARIO C, STRICTENED (after the same audit): the narrow automatic
order-status update now additionally requires (1) the broker order's own
`id` to literally equal the `order_id` that was queried (never trust a
resolution keyed on the wrong order), (2) `client_order_id` present on
BOTH the local record and the broker order, not just "matching when both
happen to have one," and (3) the local record's `kind` to be one of a
closed whitelist (`_KNOWN_ORDER_SIDES_BY_KIND`) mapped to its expected
order side -- an unrecognized or missing `kind` now fails closed instead
of silently defaulting to "sell".

API FAILURE HANDLING: any broker API/auth/network error (an `APIError`,
a connection error, anything) during `get_account`, the open-orders
reads, or `get_all_positions` is NEVER caught and reinterpreted as "empty" --
it propagates as-is. The one deliberate exception is `get_order_by_id`
inside Scenario C's narrow resolution path, where a confirmed 404 (and
ONLY a confirmed 404) is treated as "order not found," itself still
fail-closed (`UnreconciledLocalOrderError`), never as a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

# Last 4 digits of the control-arm Alpaca paper account's masked
# fingerprint (cash $100,000, zero positions, created 2026-08-15).
# Independently verified twice: once by the user directly via a real
# `curl` call from their own Terminal, and once from this codebase via
# `get_account()` called with the correct production import order (rdd
# import, THEN the .env.control override -- see
# `run_control_arm_decision._load_env_file_fail_closed`'s own docstring
# for why that order matters). An earlier version of this constant
# briefly held a different suffix -- that was never a real
# control-account reading; it was this account's own live-system
# sibling, reached only because an ad hoc test script (not the real
# production entry point) loaded .env.control BEFORE importing
# `order_submission`, and that import's own transitive
# `load_dotenv(MAIN .env, override=True)` (see
# `src/data/alpaca_market_data.py`) silently re-overrode the control
# credentials afterward. Root-caused and corrected same-session.
#
# The full account_number is intentionally NEVER written in this file --
# not in this comment, not in any docstring, not in any log/exception
# string anywhere in this module -- only this 4-character masked suffix,
# and only ever compared/displayed in masked form (see
# `_mask_account_number`). If you are about to paste the full
# account_number into this file to "document" it, don't -- mask it first.
_EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX = "XO4Y"

# Local order-record statuses treated as terminal -- an order in one of
# these states is done, one way or another, and is never a Scenario C/D
# candidate. Mirrors `order_submission.TERMINAL_STATUSES` plus "replaced"
# (an Alpaca status this module's own broker reads can surface even though
# `order_submission.py` itself never triggers a replace).
_LOCAL_TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
)


class ReconciliationError(RuntimeError):
    """Base class for every fail-closed reconciliation failure in this module."""


class AccountIdentityMismatchError(ReconciliationError):
    """The broker's own account_number does not match the expected control-arm
    account fingerprint -- see module docstring's "ACCOUNT IDENTITY CHECK"."""


class BrokerSnapshotInconsistentError(ReconciliationError):
    """Two consecutive attempts at a consistent open-orders/positions read
    both disagreed -- see module docstring's "CONSISTENT SNAPSHOT" section."""


class UnexpectedShortPositionError(ReconciliationError):
    """A broker position was reported short -- never expected under this
    control arm's long-only strategy. Unconditional block."""


class UnknownBrokerPositionError(ReconciliationError):
    """Scenario A: the broker holds a position local state has no record of."""


class MissingBrokerPositionError(ReconciliationError):
    """Scenario B: local state holds a position the broker does not have."""


class UnreconciledLocalOrderError(ReconciliationError):
    """Scenario C's fail-closed branch: a local non-terminal order could not
    be safely resolved against the broker's own order record."""


class UnknownBrokerOrderError(ReconciliationError):
    """Scenario D: the broker has an open order local state has no record of."""


class LocalTerminalButBrokerOpenError(ReconciliationError):
    """A local `submitted_actions` record is marked terminal (filled/
    canceled/expired/rejected/done_for_day/replaced) but its `order_id`
    STILL appears in the broker's own open-orders list -- local
    bookkeeping contradicts the broker's live state. Fail-closed; never
    silently trusted either way (an audit gap independently flagged: the
    original Scenario C/D loop only ever inspected NON-terminal local
    records, so a stale "already done" local status masking a genuinely
    still-open broker order was never checked at all)."""


class StopOrderInvariantViolationError(ReconciliationError):
    """`equity_stop_orders[ticker]` must reference a real, findable
    `submitted_actions` record of kind `PROTECTIVE_STOP`, and -- while
    the ticker still holds an open local position -- that stop's
    `order_id` must still be genuinely resting (open) at the broker.
    Fail-closed on any violation; a missing or inconsistent protective
    stop on an open position is a real risk-management gap, never
    silently ignored (an audit gap independently flagged:
    `equity_stop_orders` was never read/compared anywhere in the
    original `reconcile()`)."""


def _mask_account_number(account_number: str) -> str:
    """Only the last 4 characters are ever shown -- see module docstring.
    Python slicing already makes this correct for a short input (`s[-4:]`
    on a string shorter than 4 characters returns the whole string, which
    IS its own last 4-or-fewer characters) -- no separate short-input
    branch is needed, and having had one before was misleading (it
    suggested the two cases behave differently, when they don't)."""
    return "..." + str(account_number)[-4:]


def verify_account_identity(
    client: TradingClient,
    *,
    expected_suffix: str | None = _EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX,
) -> str:
    """Real `client.get_account()` call. Raises `AccountIdentityMismatchError`
    (fail-closed) if the account_number does not end in `expected_suffix`.
    Returns the masked account number (e.g. "...XO4Y") on success -- callers
    should use this masked value in any log/notification, never the raw
    `account.account_number` field.

    `expected_suffix=None` -- deliberate, explicit opt-out (added for
    `run_daily_decision.py`'s own live-account caller, which has no
    known-safe suffix value hardcoded anywhere in this codebase, and
    must never silently reuse `_EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX`
    -- that would compare the LIVE account against the CONTROL ARM's
    own suffix and fail every real run). Still makes the real
    `get_account()` call and still returns the masked value for
    logging; only the comparison/raise is skipped. Prints a loud,
    unmissable notice every time this path is taken, so an unconfigured
    check is never silently permanent -- see the caller for how to
    supply a real value once known."""
    account = client.get_account()
    account_number = str(account.account_number)
    masked = _mask_account_number(account_number)

    if expected_suffix is None:
        print(
            f"[CONTROL] WARNING: account-identity verification SKIPPED (no "
            f"expected suffix configured) -- connected account: {masked}. "
            f"This safety check is not yet active for this caller; supply "
            f"an expected suffix to enable it."
        )
        return masked

    if not account_number.endswith(expected_suffix):
        raise AccountIdentityMismatchError(
            f"Broker account identity mismatch: connected account ends "
            f"{masked}, expected an account ending ...{expected_suffix}. "
            f"Refusing to proceed -- this usually means a credential "
            f"mix-up (e.g. the live system's Alpaca keys loaded instead "
            f"of the control arm's own). Full account_number is never "
            f"logged."
        )
    print(f"[CONTROL] Broker account identity verified: {masked}")
    return masked


@dataclass(frozen=True)
class BrokerSnapshot:
    orders: tuple[Any, ...]  # open orders (alpaca.trading.models.Order), consistency-checked
    positions: tuple[Any, ...]  # alpaca.trading.models.Position
    fetched_at: str  # ISO UTC timestamp of the successful (consistent) read


def _fetch_open_orders(client: TradingClient) -> list[Any]:
    return list(
        client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
        )
    )


def _order_fingerprint(order: Any) -> tuple[str, str, str | None, str | None]:
    status = order.status
    status_str = str(status.value if hasattr(status, "value") else status)
    filled_qty = str(order.filled_qty) if order.filled_qty is not None else None
    updated_at = str(order.updated_at) if order.updated_at is not None else None
    return (str(order.id), status_str, filled_qty, updated_at)


def fetch_consistent_broker_snapshot(
    client: TradingClient, *, max_attempts: int = 2
) -> BrokerSnapshot:
    """Three real calls per attempt: open-orders (before), all-positions,
    open-orders (after). If before/after disagree (id/status/filled_qty/
    updated_at), retries the whole three-call sequence once more
    (`max_attempts=2` total). Raises `BrokerSnapshotInconsistentError`
    (fail-closed) if the retry is also inconsistent. See module docstring's
    "CONSISTENT SNAPSHOT" section."""
    last_before: list[Any] = []
    last_after: list[Any] = []
    for _ in range(max_attempts):
        orders_before = _fetch_open_orders(client)
        positions = list(client.get_all_positions())
        orders_after = _fetch_open_orders(client)
        if {_order_fingerprint(o) for o in orders_before} == {
            _order_fingerprint(o) for o in orders_after
        }:
            return BrokerSnapshot(
                orders=tuple(orders_after),
                positions=tuple(positions),
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        last_before, last_after = orders_before, orders_after
    raise BrokerSnapshotInconsistentError(
        f"Broker open-orders snapshot changed between the before/after reads "
        f"on {max_attempts} consecutive attempts (order state is actively "
        f"changing mid-reconciliation) -- fail-closed. "
        f"before={sorted(_order_fingerprint(o) for o in last_before)} "
        f"after={sorted(_order_fingerprint(o) for o in last_after)}"
    )


def _has_open_entry_buy(runner_state: Any, ticker: str) -> bool:
    prefix = f"{ticker}|ENTRY_MARKET_BUY|"
    for action_key, record in runner_state.submitted_actions.items():
        if action_key.startswith(prefix) and record.get("status") not in _LOCAL_TERMINAL_ORDER_STATUSES:
            return True
    return False


# Explicit, closed whitelist: every real `kind` value `run_daily_decision.py`
# ever writes into a `submitted_actions` record (confirmed by reading that
# file directly), mapped to the order side it must correspond to at the
# broker. An unrecognized/missing `kind` has NO entry here -- see
# `_resolve_scenario_c`'s own lookup, which fails closed rather than
# guessing (the original code defaulted anything not exactly
# "ENTRY_MARKET_BUY" to "sell", which would have silently treated a typo'd
# or future/unknown kind as a sell-side match).
_KNOWN_ORDER_SIDES_BY_KIND: dict[str, str] = {
    "ENTRY_MARKET_BUY": "buy",
    "PROTECTIVE_STOP": "sell",
    "SIGNAL_EXIT_MARKET_SELL": "sell",
    "CANCEL_STOP": "sell",  # re-records the SAME stop order_id's own (sell) side
}


def _resolve_scenario_c(
    client: TradingClient,
    action_key: str,
    record: dict,
    local_positions: dict,
    broker_positions: dict,
) -> str:
    """Scenario C's narrow resolution path -- see module docstring's case
    table. Returns the broker's confirmed "filled" status string on the one
    fully-matched success path; raises `UnreconciledLocalOrderError`
    (fail-closed) for every other outcome. Never mutates anything itself --
    the caller applies the returned status to the local record, and only
    after EVERY candidate in the same reconciliation pass has independently
    resolved successfully (see `reconcile()`'s own two-pass discipline)."""
    order_id = str(record.get("order_id"))
    ticker = action_key.split("|", 1)[0]
    try:
        broker_order = client.get_order_by_id(order_id)
    except APIError as error:
        if getattr(error, "status_code", None) == 404:
            raise UnreconciledLocalOrderError(
                f"Order {order_id} ({action_key}) is not in the broker's open "
                f"list and a direct lookup returned 404 (not found) -- "
                f"fail-closed, no automatic order-status change."
            ) from error
        raise

    # The broker's OWN id on the returned order must match what we asked
    # for -- defense against ever trusting a resolution keyed on the
    # wrong order (e.g. a client implementation, real or test-double,
    # that does not actually honor the id it was queried with).
    broker_order_id = str(getattr(broker_order, "id", ""))
    if broker_order_id != order_id:
        raise UnreconciledLocalOrderError(
            f"Order lookup mismatch for {action_key}: queried order_id="
            f"{order_id!r} but the broker returned an order whose own id "
            f"is {broker_order_id!r}. Fail-closed -- never resolve against "
            f"an order that isn't provably the one being asked about."
        )

    # client_order_id must be present on BOTH sides and match -- not
    # "match only if both happen to have one." A record with no
    # client_order_id on either side carries strictly weaker identity
    # evidence (id/symbol/side/status/qty alone), which this narrow,
    # automatic path no longer accepts.
    broker_client_order_id = getattr(broker_order, "client_order_id", None)
    local_client_order_id = record.get("client_order_id")
    if not local_client_order_id or not broker_client_order_id:
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}): client_order_id missing on "
            f"the local record ({local_client_order_id!r}) or the broker "
            f"order ({broker_client_order_id!r}) -- both sides must carry "
            f"one and it must match. Fail-closed."
        )
    if str(local_client_order_id) != str(broker_client_order_id):
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}): client_order_id mismatch -- "
            f"local={local_client_order_id!r}, broker={broker_client_order_id!r}. "
            f"Fail-closed."
        )

    broker_symbol = str(getattr(broker_order, "symbol", ""))
    if broker_symbol != ticker:
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}): symbol mismatch -- local "
            f"ticker={ticker!r}, broker symbol={broker_symbol!r}. Fail-closed."
        )

    local_kind = record.get("kind")
    expected_side = _KNOWN_ORDER_SIDES_BY_KIND.get(local_kind)
    if expected_side is None:
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}): local record has an "
            f"unrecognized or missing kind {local_kind!r} -- cannot "
            f"safely determine its expected order side. Fail-closed "
            f"rather than defaulting to a guess (e.g. 'sell')."
        )
    broker_side = broker_order.side
    broker_side_str = str(broker_side.value if hasattr(broker_side, "value") else broker_side).lower()
    if broker_side_str != expected_side:
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}): side mismatch -- expected "
            f"{expected_side!r} (from local kind={local_kind!r}), "
            f"broker side={broker_side_str!r}. Fail-closed."
        )

    broker_status = broker_order.status
    broker_status_str = str(broker_status.value if hasattr(broker_status, "value") else broker_status).lower()
    if broker_status_str != "filled":
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}) resolved to broker status "
            f"{broker_status_str!r}, not 'filled' -- fail-closed. Any "
            f"non-'filled' terminal status (canceled/expired/rejected/"
            f"replaced/...) always requires human review, never an "
            f"automatic order-status change."
        )

    if ticker not in local_positions or ticker not in broker_positions:
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}) shows broker status 'filled', "
            f"but a position for {ticker} is missing on the local side "
            f"({ticker in local_positions}), the broker side "
            f"({ticker in broker_positions}), or both -- fail-closed, no "
            f"automatic order-status change."
        )

    local_qty = Decimal(str(local_positions[ticker].quantity))
    broker_qty = Decimal(str(broker_positions[ticker].qty))
    filled_qty_raw = getattr(broker_order, "filled_qty", None)
    filled_qty = Decimal(str(filled_qty_raw)) if filled_qty_raw is not None else None
    if filled_qty is None or not (local_qty == broker_qty == filled_qty):
        raise UnreconciledLocalOrderError(
            f"Order {order_id} ({action_key}): quantity mismatch resolving a "
            f"filled order -- local position qty={local_qty}, broker "
            f"position qty={broker_qty}, order filled_qty={filled_qty}. All "
            f"three must match exactly (Decimal). Fail-closed, no automatic "
            f"order-status change."
        )
    return broker_status_str


@dataclass(frozen=True)
class ReconciliationResult:
    account_number_masked: str
    broker_open_order_count: int
    broker_position_count: int
    local_position_count: int
    order_status_updates: dict[str, str]  # action_key -> new (terminal) status applied
    checked_at: str


def reconcile(
    client: TradingClient,
    runner_state: Any,
    *,
    expected_account_suffix: str | None = _EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX,
    orders_enabled: bool = True,
) -> ReconciliationResult:
    """Full reconciliation pass. Mutates `runner_state.submitted_actions[...]
    ["status"]` in place ONLY for the narrow Scenario C success case (see
    `_resolve_scenario_c`) -- `runner_state.positions` is never mutated by
    this function under any outcome. Raises one of this module's
    `ReconciliationError` subclasses (fail-closed) for every other anomaly;
    on a raise, `runner_state` has NOT been mutated. This holds even when
    MULTIPLE Scenario C candidates exist in the same call: every candidate
    is validated first, with zero mutation, and mutations are only ever
    applied in a second pass after every candidate has independently
    resolved successfully (see the two-pass block below) -- a later
    candidate's failure can never leave an earlier candidate's status
    already changed in memory. Every earlier-raising check (account
    identity, snapshot inconsistency, short position, Scenario A/B,
    position-quantity mismatch, local-terminal-but-broker-open, the
    equity_stop_orders invariant) also exits before any mutation is
    possible.

    `orders_enabled` -- ORDERS-DISABLED / DRY-RUN AWARENESS (added after
    an independent audit found a real, latent false-alarm bug): when
    real order submission is disabled (the control arm's current, only
    real-world mode -- `--enable-equity-orders`/`--enable-crypto-orders`
    never passed), `run_daily_decision()`'s frozen per-bar engine STILL
    populates `runner_state.positions` with its own purely SIMULATED
    bar-by-bar bookkeeping -- that is what makes the control arm's local
    state a genuine, continuous forward-test track record at all, not
    reimplementable or suppressible from this wrapper without either
    editing the frozen engine (forbidden) or freezing `position_state.json`
    entirely (which would silently defeat the whole point of running the
    control arm -- no continuous simulated portfolio, no forward-test
    signal, just one-off dry-run log lines). No real order was EVER
    submitted for these positions, so their absence at the broker is
    EXPECTED, not an anomaly -- comparing them against the broker at all
    is a category error, not a legitimate reconciliation question. When
    `orders_enabled=False`, Scenario B (`MissingBrokerPositionError`) is
    never raised; a local-only position is logged as an expected,
    simulated-only entry instead. Scenario A (broker has something local
    doesn't) is DELIBERATELY left fully active even in this mode -- a
    real position unexpectedly appearing on what should be an
    orders-disabled paper account is still a genuine anomaly worth
    stopping for. Every other check (account identity, snapshot
    consistency, short-position guard, Scenario C/D order comparisons,
    the equity_stop_orders invariant, local-terminal-but-broker-open) is
    completely unaffected by this parameter -- `submitted_actions`/
    `equity_stop_orders` are never populated at all while orders are
    disabled (see `run_daily_decision.py`'s own `_execute_equity_orders`
    gate), so those checks are already natural no-ops in this mode, not
    ones that needed a special case."""
    account_number_masked = verify_account_identity(client, expected_suffix=expected_account_suffix)
    snapshot = fetch_consistent_broker_snapshot(client)

    broker_positions = {str(p.symbol): p for p in snapshot.positions}
    local_positions = dict(runner_state.positions)

    for symbol, position in broker_positions.items():
        side_str = str(getattr(position, "side", "")).lower()
        qty = Decimal(str(position.qty))
        if "short" in side_str or qty < 0:
            raise UnexpectedShortPositionError(
                f"Broker position {symbol} is short (side={side_str!r}, "
                f"qty={qty}) -- this control arm's strategy is long-only; a "
                f"short position is never expected here. Fail-closed, no "
                f"automatic action taken."
            )

    unknown_at_broker = sorted(set(broker_positions) - set(local_positions))
    if unknown_at_broker:
        raise UnknownBrokerPositionError(
            f"Broker holds position(s) not present in local state: "
            f"{unknown_at_broker}. Fail-closed -- no automatic local "
            f"position record created. This commonly indicates possible "
            f"crash recovery (a fill or state save interrupted mid-run on a "
            f"previous invocation); investigate broker order history for "
            f"these tickers before proceeding."
        )

    missing_at_broker = sorted(set(local_positions) - set(broker_positions))
    if missing_at_broker and not orders_enabled:
        # Expected, not an anomaly -- see this function's own docstring,
        # "ORDERS-DISABLED / DRY-RUN AWARENESS". No automatic mutation
        # either way; this is purely informational.
        print(
            f"[CONTROL] Dry-run mode (orders disabled): {len(missing_at_broker)} "
            f"local-only simulated position(s) not compared against the "
            f"broker -- no real order was ever submitted for these, so "
            f"their absence at the broker is expected: {missing_at_broker}"
        )
    elif missing_at_broker:
        in_flight = sorted(t for t in missing_at_broker if _has_open_entry_buy(runner_state, t))
        unexplained = sorted(t for t in missing_at_broker if t not in in_flight)
        raise MissingBrokerPositionError(
            f"Local state holds position(s) not present at the broker: "
            f"{missing_at_broker}. Fail-closed -- no automatic local "
            f"position deletion. In-flight (open ENTRY BUY order still "
            f"outstanding, fill not yet broker-confirmed): {in_flight}. "
            f"Unexplained (no open entry order accounts for the absence): "
            f"{unexplained}. Investigate before proceeding."
        )

    for ticker in sorted(set(local_positions) & set(broker_positions)):
        local_qty = Decimal(str(local_positions[ticker].quantity))
        broker_qty = Decimal(str(broker_positions[ticker].qty))
        if local_qty != broker_qty:
            raise ReconciliationError(
                f"Quantity mismatch for {ticker}: local position={local_qty}, "
                f"broker position={broker_qty} (total qty, not "
                f"qty_available). Fail-closed."
            )

    broker_open_order_ids = {str(o.id) for o in snapshot.orders}
    known_order_ids: set[str] = set()
    known_client_order_ids: set[str] = set()
    order_id_to_record: dict[str, dict] = {}
    for record in runner_state.submitted_actions.values():
        if record.get("order_id"):
            order_id_str = str(record["order_id"])
            known_order_ids.add(order_id_str)
            order_id_to_record.setdefault(order_id_str, record)
        if record.get("client_order_id"):
            known_client_order_ids.add(str(record["client_order_id"]))

    # Local-terminal-but-broker-still-open check: the ORIGINAL Scenario
    # C/D loop only ever inspected NON-terminal local records, so a
    # locally "already done" status that contradicts the broker's own,
    # currently-open view of that same order_id was never checked at
    # all. Runs BEFORE any Scenario C resolution attempt.
    for action_key, record in runner_state.submitted_actions.items():
        if record.get("status") not in _LOCAL_TERMINAL_ORDER_STATUSES:
            continue
        if str(record.get("order_id")) in broker_open_order_ids:
            raise LocalTerminalButBrokerOpenError(
                f"Local record {action_key} (order_id={record.get('order_id')}, "
                f"status={record.get('status')!r}) is marked terminal, but "
                f"the broker STILL shows this order as open. Fail-closed -- "
                f"local bookkeeping may be stale or wrong; investigate "
                f"before trusting either side."
            )

    # equity_stop_orders invariant: every ticker's claimed protective-stop
    # order_id must (a) correspond to a real, findable submitted_actions
    # record of kind PROTECTIVE_STOP, and (b) while that ticker still
    # holds an open local position, still be genuinely resting (open) at
    # the broker. Never checked at all in the original reconcile().
    for ticker, stop_order_id in runner_state.equity_stop_orders.items():
        stop_order_id = str(stop_order_id)
        stop_record = order_id_to_record.get(stop_order_id)
        if stop_record is None:
            raise StopOrderInvariantViolationError(
                f"equity_stop_orders[{ticker!r}] references order_id "
                f"{stop_order_id!r}, but no submitted_actions record for "
                f"that order_id exists at all -- an orphaned stop-order "
                f"reference. Fail-closed."
            )
        if stop_record.get("kind") != "PROTECTIVE_STOP":
            raise StopOrderInvariantViolationError(
                f"equity_stop_orders[{ticker!r}] references order_id "
                f"{stop_order_id!r}, whose submitted_actions record has "
                f"kind {stop_record.get('kind')!r}, not 'PROTECTIVE_STOP'. "
                f"Fail-closed -- integrity violation."
            )
        if ticker in local_positions and stop_order_id not in broker_open_order_ids:
            raise StopOrderInvariantViolationError(
                f"Ticker {ticker!r} holds an open local position and "
                f"equity_stop_orders claims its protective stop is "
                f"order_id {stop_order_id!r}, but the broker does NOT "
                f"show that order as currently open -- this position's "
                f"downside protection may be missing. Fail-closed; "
                f"investigate before proceeding."
            )

    # Scenario C, two-pass: EVERY stale (non-terminal, no-longer-open)
    # candidate is validated FIRST, with zero mutation of `runner_state`.
    # Mutations are applied only in a second pass, and only if every
    # candidate in this call independently resolved successfully -- so a
    # later candidate's failure can never leave an earlier candidate's
    # in-memory status already changed (the original single-pass loop
    # mutated `record["status"]` immediately upon each success, so a
    # LATER candidate raising left EARLIER candidates' mutations applied
    # in memory despite the overall call raising -- contradicting this
    # function's own "on a raise, nothing is mutated" contract).
    resolutions: dict[str, str] = {}
    for action_key, record in runner_state.submitted_actions.items():
        if record.get("status") in _LOCAL_TERMINAL_ORDER_STATUSES:
            continue
        if str(record.get("order_id")) in broker_open_order_ids:
            continue  # still genuinely open at the broker -- consistent, nothing to do
        resolutions[action_key] = _resolve_scenario_c(
            client, action_key, record, local_positions, broker_positions
        )

    order_status_updates: dict[str, str] = {}
    for action_key, resolved_status in resolutions.items():
        runner_state.submitted_actions[action_key]["status"] = resolved_status
        order_status_updates[action_key] = resolved_status

    unknown_broker_orders = [
        order
        for order in snapshot.orders
        if str(order.id) not in known_order_ids
        and (not order.client_order_id or str(order.client_order_id) not in known_client_order_ids)
    ]
    if unknown_broker_orders:
        details = [
            {
                "order_id": str(order.id),
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": str(order.side.value if hasattr(order.side, "value") else order.side),
                "qty": str(order.qty),
                "status": str(order.status.value if hasattr(order.status, "value") else order.status),
                "submitted_at": str(order.submitted_at),
            }
            for order in unknown_broker_orders
        ]
        raise UnknownBrokerOrderError(
            f"Broker has {len(details)} open order(s) not known to local "
            f"state: {details}. Fail-closed -- no automatic cancellation or "
            f"adoption. Full detail reported for manual review."
        )

    return ReconciliationResult(
        account_number_masked=account_number_masked,
        broker_open_order_count=len(snapshot.orders),
        broker_position_count=len(snapshot.positions),
        local_position_count=len(local_positions),
        order_status_updates=order_status_updates,
        checked_at=snapshot.fetched_at,
    )
