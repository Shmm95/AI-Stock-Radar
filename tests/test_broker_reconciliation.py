"""Tests for src/live/broker_reconciliation.py -- Phase 2a broker-vs-local
reconciliation for the control arm.

Deterministic and network-free, same discipline as
`test_live_equity_order_execution.py`: every test uses a `FakeClient`
(never a real `TradingClient`), so these exercise all four documented
mismatch scenarios (A/B/C/D), the two additional invariant checks (E/F),
and the three strictness fixes from the independent code-audit round
(order-id-must-match, client_order_id-required-on-both-sides,
kind-must-be-whitelisted) without depending on network access, market
hours, or a live account.

Real, live-account evidence (an actual `get_account()` call returning the
control account's real masked identity, a real 3-call consistent-snapshot
fetch, real wall-clock timing) was gathered separately this session
against the real `.env.control` paper account -- see the session's own
report for that real-API timing (~0.5s / 4 calls for a clean-pass
reconciliation). This file's job is to real-verify, deterministically and
on every future run, the LOGIC those real calls feed into -- plus the
exact real call-count `reconcile()` makes in a clean pass, printed with a
live timestamp below (`test_clean_pass_reports_expected_broker_call_count`)
so that count itself is continuously re-verified, not just asserted once
by hand.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from src.backtest.portfolio_backtest_engine import _MutablePosition
from src.live import broker_reconciliation as br
from src.live import order_intent
from src.live.position_state import LiveRunnerState


@pytest.fixture(autouse=True)
def _isolate_order_intent_directory(tmp_path, monkeypatch):
    """Reboot-drill finding #2 (2026-08-24): Scenario D now consults
    `order_intent.list_intents()` (to distinguish a journal-correlated
    crash-recovery gap from a genuinely mystery broker order) --
    isolated here to a per-test tmp_path, same discipline
    `test_write_ahead_intent_journal.py` already established, never the
    real out-of-repo guard directory."""
    monkeypatch.setattr(order_intent, "INTENT_DIRECTORY", tmp_path / "order_intents")


# ---------------------------------------------------------------------------
# Fakes -- no real Alpaca client anywhere in this file.
# ---------------------------------------------------------------------------


class FakeOrder:
    def __init__(
        self,
        *,
        id: str,
        client_order_id: str | None,
        symbol: str,
        side: str,
        status: str,
        qty: str,
        filled_qty: str,
        submitted_at: str = "2026-08-16T00:00:00Z",
        updated_at: str = "2026-08-16T00:00:00Z",
    ) -> None:
        self.id = id
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = side
        self.status = status
        self.qty = qty
        self.filled_qty = filled_qty
        self.submitted_at = submitted_at
        self.updated_at = updated_at


class FakePosition:
    def __init__(self, *, symbol: str, qty: str, side: str = "long") -> None:
        self.symbol = symbol
        self.qty = qty
        self.side = side


class FakeAccount:
    def __init__(self, account_number: str) -> None:
        self.account_number = account_number


_REAL_SUFFIX = br._EXPECTED_CONTROL_ACCOUNT_NUMBER_SUFFIX


class FakeClient:
    """`open_orders`/`positions` are returned by reference on every call
    (the SAME list each time) -- a stable, consistent broker view.
    `order_by_id` looks up by the exact id queried; a `WrongOrderIdClient`
    subclass below overrides this to return a mismatched order instead.
    `AlwaysChangingClient` (in the snapshot-consistency test) overrides
    `get_orders` to simulate a genuinely inconsistent before/after read."""

    def __init__(
        self,
        *,
        open_orders: list[FakeOrder] | None = None,
        positions: list[FakePosition] | None = None,
        order_by_id: dict[str, FakeOrder] | None = None,
        account_number: str = f"PA3HONFD{_REAL_SUFFIX}",
    ) -> None:
        self._open_orders = open_orders or []
        self._positions = positions or []
        self._order_by_id = order_by_id or {}
        self._account_number = account_number
        self.get_orders_call_count = 0
        self.get_all_positions_call_count = 0
        self.get_account_call_count = 0
        self.get_order_by_id_call_count = 0

    def get_account(self):
        self.get_account_call_count += 1
        return FakeAccount(self._account_number)

    def get_orders(self, filter=None):
        self.get_orders_call_count += 1
        return list(self._open_orders)

    def get_all_positions(self):
        self.get_all_positions_call_count += 1
        return list(self._positions)

    def get_order_by_id(self, order_id):
        self.get_order_by_id_call_count += 1
        return self._order_by_id[order_id]


class WrongOrderIdClient(FakeClient):
    """Simulates a client (real or buggy) whose `get_order_by_id` does NOT
    actually honor the id it was asked for -- used to prove the new
    id-must-match check in `_resolve_scenario_c`."""

    def __init__(self, *, returned_order: FakeOrder, **kwargs) -> None:
        super().__init__(**kwargs)
        self._returned_order = returned_order

    def get_order_by_id(self, order_id):
        self.get_order_by_id_call_count += 1
        return self._returned_order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_position(ticker: str, quantity: float) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker,
        asset_class="EQUITY",
        entry_timestamp="2026-08-14T00:00:00Z",
        entry_portfolio_bar_index=1,
        quantity=quantity,
        entry_price=100.0,
        entry_fee=0.0,
        stop_loss_price=90.0,
        highest_close=100.0,
        trailing_close_percent=0.0,
        initial_risk_amount=50.0,
        signal_score=0.0,
        signal_reason="test",
    )


def entry_buy_record(*, order_id: str, client_order_id: str, status: str) -> dict:
    return {
        "order_id": order_id,
        "status": status,
        "kind": "ENTRY_MARKET_BUY",
        "submitted_at": "2026-08-16T00:00:00Z",
        "client_order_id": client_order_id,
    }


# ---------------------------------------------------------------------------
# Account identity + masking
# ---------------------------------------------------------------------------


def test_mask_account_number_shows_only_last_four():
    assert br._mask_account_number("PA3HONFDXO4Y") == "...XO4Y"


def test_mask_account_number_short_input_stays_last_four_only():
    # A short input's "last 4 characters" is itself, per plain slicing --
    # the same definition, no separate branch.
    assert br._mask_account_number("AB") == "...AB"
    assert br._mask_account_number("") == "..."


def test_verify_account_identity_accepts_matching_suffix():
    client = FakeClient(account_number=f"PA3HONFD{_REAL_SUFFIX}")
    masked = br.verify_account_identity(client)
    assert masked == f"...{_REAL_SUFFIX}"


def test_verify_account_identity_rejects_mismatched_suffix():
    client = FakeClient(account_number="PA3HONFDWRONG")
    with pytest.raises(br.AccountIdentityMismatchError) as excinfo:
        br.verify_account_identity(client)
    # The exception message shows the MASKED mismatched value (last 4:
    # "RONG"), never the full account_number ("PA3HONFDWRONG").
    assert "...RONG" in str(excinfo.value)
    assert "PA3HONFDWRONG" not in str(excinfo.value)


def test_no_raw_account_number_ever_appears_in_module_source():
    import inspect

    source = inspect.getsource(br)
    assert "PA3HONFDXO4Y" not in source


# ---------------------------------------------------------------------------
# Consistent snapshot
# ---------------------------------------------------------------------------


def test_fetch_consistent_broker_snapshot_single_attempt_when_stable():
    client = FakeClient(open_orders=[], positions=[])
    snapshot = br.fetch_consistent_broker_snapshot(client)
    assert snapshot.orders == ()
    assert client.get_orders_call_count == 2  # before + after, one attempt


def test_fetch_consistent_broker_snapshot_raises_when_orders_change_mid_read():
    order_v1 = FakeOrder(id="o1", client_order_id="c1", symbol="AAPL", side="buy", status="new", qty="1", filled_qty="0")
    order_v2 = FakeOrder(id="o1", client_order_id="c1", symbol="AAPL", side="buy", status="filled", qty="1", filled_qty="1")

    class AlwaysChangingClient(FakeClient):
        """Alternates between two different open-orders snapshots on EVERY
        call, so before/after disagree within every attempt (not just the
        first) -- genuinely never settles, proving the retry-once-then-
        fail-closed path rather than accidentally happening to agree on
        a later attempt."""

        def get_orders(self, filter=None):
            self.get_orders_call_count += 1
            return [order_v1] if self.get_orders_call_count % 2 == 1 else [order_v2]

    client = AlwaysChangingClient(positions=[])
    with pytest.raises(br.BrokerSnapshotInconsistentError):
        br.fetch_consistent_broker_snapshot(client)
    # 2 attempts x 2 get_orders calls each = 4
    assert client.get_orders_call_count == 4


# ---------------------------------------------------------------------------
# Scenario A: broker has a position local state doesn't know about
# ---------------------------------------------------------------------------


def test_scenario_a_unknown_broker_position_fails_closed():
    client = FakeClient(positions=[FakePosition(symbol="ZZZ", qty="10")])
    runner_state = LiveRunnerState()
    with pytest.raises(br.UnknownBrokerPositionError):
        br.reconcile(client, runner_state)
    assert runner_state.positions == {}  # nothing auto-created


def test_short_broker_position_always_blocked():
    client = FakeClient(positions=[FakePosition(symbol="ZZZ", qty="10", side="short")])
    runner_state = LiveRunnerState()
    with pytest.raises(br.UnexpectedShortPositionError):
        br.reconcile(client, runner_state)


# ---------------------------------------------------------------------------
# Scenario B: local state has a position broker doesn't have
# ---------------------------------------------------------------------------


def test_scenario_b_missing_broker_position_classifies_in_flight_vs_unexplained():
    runner_state = LiveRunnerState()
    runner_state.positions["UNEXPLAINED"] = make_position("UNEXPLAINED", 5.0)
    runner_state.positions["INFLIGHT"] = make_position("INFLIGHT", 3.0)
    runner_state.submitted_actions["INFLIGHT|ENTRY_MARKET_BUY|2026-08-14"] = entry_buy_record(
        order_id="still-open-order", client_order_id="cid-inflight", status="new"
    )
    client = FakeClient(positions=[])
    with pytest.raises(br.MissingBrokerPositionError) as excinfo:
        br.reconcile(client, runner_state)
    message = str(excinfo.value)
    assert "INFLIGHT" in message and "UNEXPLAINED" in message
    assert runner_state.positions["UNEXPLAINED"].quantity == 5.0  # nothing deleted


def test_orders_disabled_suppresses_scenario_b_false_alarm():
    """The frozen engine populates `positions` with its own purely
    simulated bookkeeping even while real orders are disabled -- no real
    order was ever submitted for these, so their absence at the broker
    must NOT be a Scenario B anomaly when `equity_orders_enabled=False`."""
    runner_state = LiveRunnerState()
    runner_state.positions["AAA"] = make_position("AAA", 10.0)
    client = FakeClient(positions=[])
    result = br.reconcile(client, runner_state, equity_orders_enabled=False, crypto_orders_enabled=False)
    assert result.local_position_count == 1
    assert runner_state.positions["AAA"].quantity == 10.0  # untouched either way


def test_orders_enabled_still_raises_the_same_gap():
    """Contrast case: the EXACT same local-only-position scenario, but
    with orders actually enabled -- must still fail closed exactly as
    before this fix (both flags default True)."""
    runner_state = LiveRunnerState()
    runner_state.positions["AAA"] = make_position("AAA", 10.0)
    client = FakeClient(positions=[])
    with pytest.raises(br.MissingBrokerPositionError):
        br.reconcile(client, runner_state, equity_orders_enabled=True, crypto_orders_enabled=True)
    with pytest.raises(br.MissingBrokerPositionError):
        br.reconcile(client, runner_state)  # default is both True


def test_real_crypto_symbol_slash_normalizes_to_local_dash_convention():
    """REAL bug reproduction: before the fix, a real open BTC-USD
    position would have appeared as BOTH unknown_at_broker (broker's
    real "BTC/USD" symbol never matched local "BTC-USD") AND
    missing_at_broker (local "BTC-USD" never matched broker's real
    symbol set) -- a guaranteed false alarm on every run holding any
    real crypto position, making this module unusable against the live
    universe. Must reconcile cleanly once normalized."""
    runner_state = LiveRunnerState()
    crypto_position = make_position("BTC-USD", 1.0)
    crypto_position.asset_class = "CRYPTO"
    runner_state.positions["BTC-USD"] = crypto_position
    client = FakeClient(positions=[FakePosition(symbol="BTC/USD", qty="1.0")])

    result = br.reconcile(client, runner_state)  # both flags default True -- must NOT raise
    assert result.local_position_count == 1
    assert result.broker_position_count == 1


def test_equity_disabled_crypto_enabled_only_suppresses_equity_anomaly():
    """REAL bug reproduction: a single blended orders_enabled boolean
    would have either wrongly suppressed a genuine crypto anomaly, or
    wrongly raised for a legitimately dry-run equity position. Each
    ticker's own asset class must be judged against its own flag."""
    runner_state = LiveRunnerState()
    runner_state.positions["AAA"] = make_position("AAA", 10.0)  # EQUITY, dry-run -- expected absence
    crypto_position = make_position("BTC-USD", 1.0)
    crypto_position.asset_class = "CRYPTO"
    runner_state.positions["BTC-USD"] = crypto_position  # CRYPTO, orders enabled -- real anomaly
    client = FakeClient(positions=[])

    with pytest.raises(br.MissingBrokerPositionError, match="BTC-USD"):
        br.reconcile(client, runner_state, equity_orders_enabled=False, crypto_orders_enabled=True)


def test_position_quantity_mismatch_fails_closed():
    runner_state = LiveRunnerState()
    runner_state.positions["AAPL"] = make_position("AAPL", 10.0)
    client = FakeClient(positions=[FakePosition(symbol="AAPL", qty="7")])
    with pytest.raises(br.ReconciliationError):
        br.reconcile(client, runner_state)


# ---------------------------------------------------------------------------
# Scenario C: success path + all strictness fixes
# ---------------------------------------------------------------------------


def test_scenario_c_success_updates_status_only_never_positions():
    runner_state = LiveRunnerState()
    runner_state.positions["QQQ"] = make_position("QQQ", 20.0)
    runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-1", client_order_id="cid-1", status="new"
    )
    positions_before = dict(runner_state.positions)
    client = FakeClient(
        positions=[FakePosition(symbol="QQQ", qty="20")],
        order_by_id={
            "order-1": FakeOrder(
                id="order-1", client_order_id="cid-1", symbol="QQQ",
                side="buy", status="filled", qty="20", filled_qty="20",
            )
        },
    )
    result = br.reconcile(client, runner_state)
    assert result.order_status_updates == {"QQQ|ENTRY_MARKET_BUY|2026-08-16": "filled"}
    assert runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"]["status"] == "filled"
    assert runner_state.positions is not None
    assert set(runner_state.positions) == set(positions_before)
    assert runner_state.positions["QQQ"] is positions_before["QQQ"]


def test_scenario_c_rejects_wrong_order_id_even_with_matching_fields():
    """Fix #1: the broker's returned order.id must equal the queried
    order_id. Previously accepted if ticker/side/qty/status matched
    regardless of the returned order's own id."""
    runner_state = LiveRunnerState()
    runner_state.positions["QQQ"] = make_position("QQQ", 20.0)
    runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-1", client_order_id="cid-1", status="new"
    )
    mismatched_order = FakeOrder(
        id="SOME-OTHER-ORDER-ID", client_order_id="cid-1", symbol="QQQ",
        side="buy", status="filled", qty="20", filled_qty="20",
    )
    client = WrongOrderIdClient(
        returned_order=mismatched_order,
        positions=[FakePosition(symbol="QQQ", qty="20")],
    )
    with pytest.raises(br.UnreconciledLocalOrderError, match="lookup mismatch"):
        br.reconcile(client, runner_state)
    assert runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"]["status"] == "new"


def test_scenario_c_requires_client_order_id_on_both_sides():
    """Fix #2: previously, a missing client_order_id on EITHER side
    skipped the check entirely ("match only if both present"). Now both
    sides must carry one."""
    runner_state = LiveRunnerState()
    runner_state.positions["QQQ"] = make_position("QQQ", 20.0)
    record = entry_buy_record(order_id="order-1", client_order_id="cid-1", status="new")
    record["client_order_id"] = None  # local side missing it
    runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"] = record
    client = FakeClient(
        positions=[FakePosition(symbol="QQQ", qty="20")],
        order_by_id={
            "order-1": FakeOrder(
                id="order-1", client_order_id="broker-side-cid", symbol="QQQ",
                side="buy", status="filled", qty="20", filled_qty="20",
            )
        },
    )
    with pytest.raises(br.UnreconciledLocalOrderError, match="client_order_id missing"):
        br.reconcile(client, runner_state)


def test_scenario_c_rejects_unknown_kind_instead_of_defaulting_to_sell():
    """Fix #3: an unrecognized/missing `kind` must fail closed, never
    silently default to 'sell'."""
    runner_state = LiveRunnerState()
    runner_state.positions["QQQ"] = make_position("QQQ", 20.0)
    record = entry_buy_record(order_id="order-1", client_order_id="cid-1", status="new")
    record["kind"] = "SOME_FUTURE_KIND_NOT_YET_WHITELISTED"
    runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"] = record
    client = FakeClient(
        positions=[FakePosition(symbol="QQQ", qty="20")],
        order_by_id={
            "order-1": FakeOrder(
                id="order-1", client_order_id="cid-1", symbol="QQQ",
                side="sell", status="filled", qty="20", filled_qty="20",
            )
        },
    )
    with pytest.raises(br.UnreconciledLocalOrderError, match="unrecognized or missing kind"):
        br.reconcile(client, runner_state)


def test_scenario_c_multi_candidate_failure_leaves_zero_partial_mutation():
    """Fix (atomicity): two Scenario C candidates in the same call; the
    first would resolve successfully, the second fails. Neither may end
    up mutated -- validate-all-then-apply-all, not apply-as-you-go."""
    runner_state = LiveRunnerState()
    runner_state.positions["AAA"] = make_position("AAA", 5.0)
    runner_state.positions["BBB"] = make_position("BBB", 9.0)
    runner_state.submitted_actions["AAA|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-aaa", client_order_id="cid-aaa", status="new"
    )
    runner_state.submitted_actions["BBB|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-bbb", client_order_id="cid-bbb", status="new"
    )
    client = FakeClient(
        positions=[FakePosition(symbol="AAA", qty="5"), FakePosition(symbol="BBB", qty="9")],
        order_by_id={
            "order-aaa": FakeOrder(  # this one WOULD resolve successfully
                id="order-aaa", client_order_id="cid-aaa", symbol="AAA",
                side="buy", status="filled", qty="5", filled_qty="5",
            ),
            "order-bbb": FakeOrder(  # this one fails -- quantity mismatch
                id="order-bbb", client_order_id="cid-bbb", symbol="BBB",
                side="buy", status="filled", qty="9", filled_qty="1",
            ),
        },
    )
    with pytest.raises(br.UnreconciledLocalOrderError):
        br.reconcile(client, runner_state)
    # Neither candidate's status may have been mutated -- not even AAA,
    # which on its own would have resolved cleanly.
    assert runner_state.submitted_actions["AAA|ENTRY_MARKET_BUY|2026-08-16"]["status"] == "new"
    assert runner_state.submitted_actions["BBB|ENTRY_MARKET_BUY|2026-08-16"]["status"] == "new"


def test_scenario_c_404_fails_closed():
    from types import SimpleNamespace

    from alpaca.common.exceptions import APIError

    runner_state = LiveRunnerState()
    runner_state.positions["QQQ"] = make_position("QQQ", 20.0)
    runner_state.submitted_actions["QQQ|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-missing", client_order_id="cid-1", status="new"
    )

    # A real APIError whose `status_code` property resolves to 404 via a
    # fake `http_error.response.status_code` -- see APIError's own
    # `status_code` property implementation.
    fake_http_error = SimpleNamespace(response=SimpleNamespace(status_code=404))
    real_404_error = APIError('{"message": "not found"}', http_error=fake_http_error)
    assert real_404_error.status_code == 404  # sanity-check the fixture itself

    class FourOhFourClient(FakeClient):
        def get_order_by_id(self, order_id):
            self.get_order_by_id_call_count += 1
            raise real_404_error

    client = FourOhFourClient(positions=[FakePosition(symbol="QQQ", qty="20")])
    with pytest.raises(br.UnreconciledLocalOrderError, match="404"):
        br.reconcile(client, runner_state)


# ---------------------------------------------------------------------------
# Scenario D: broker has an unknown open order
# ---------------------------------------------------------------------------


def test_scenario_d_unknown_broker_order_fails_closed():
    unknown_order = FakeOrder(
        id="unknown-1", client_order_id="MYST-cid", symbol="MYST",
        side="buy", status="accepted", qty="10", filled_qty="0",
    )
    client = FakeClient(open_orders=[unknown_order], positions=[])
    runner_state = LiveRunnerState()
    with pytest.raises(br.UnknownBrokerOrderError, match="unknown-1"):
        br.reconcile(client, runner_state)


def test_scenario_d_journal_correlated_order_raises_the_specific_error(monkeypatch: pytest.MonkeyPatch):
    """Reboot-drill finding #2 (2026-08-24): a "known to local state"
    check failing must NOT always mean a genuine mystery order -- if the
    order correlates with an order_intent.py journal entry sitting at
    BROKER_ACKNOWLEDGED (the exact shape of a crash between a broker
    acknowledgment and this run's own local submitted_actions update),
    the more specific, more actionable exception must fire instead of
    the generic one. Still fail-closed either way -- only the
    diagnostic differs."""
    acknowledged_order = FakeOrder(
        id="broker-order-1", client_order_id="AA-cid", symbol="AA",
        side="buy", status="filled", qty="10", filled_qty="10",
    )
    client = FakeClient(open_orders=[acknowledged_order], positions=[])
    runner_state = LiveRunnerState()

    intent = order_intent.create_intent(
        client_order_id="AA-cid", account_identity="control_arm_v1", ticker="AA",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY",
        source_signal_timestamp="2026-08-24",
    )
    intent = order_intent.transition_intent(intent, order_intent.SUBMITTING)
    order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="broker-order-1", broker_status="filled",
    )

    with pytest.raises(br.JournalAcknowledgedButLocalStateMissingError, match="broker-order-1"):
        br.reconcile(client, runner_state)


def test_scenario_d_mixes_journal_correlated_and_genuinely_unknown_orders(monkeypatch: pytest.MonkeyPatch):
    """Both kinds of "unknown to local state" order present at once --
    the more specific, more actionable error must still fire (never
    silently swallow the journal-correlated one into the generic
    message), and the genuinely-unknown one must still be visible in
    its own detail."""
    acknowledged_order = FakeOrder(
        id="broker-order-1", client_order_id="AA-cid", symbol="AA",
        side="buy", status="filled", qty="10", filled_qty="10",
    )
    mystery_order = FakeOrder(
        id="unknown-1", client_order_id="MYST-cid", symbol="MYST",
        side="buy", status="accepted", qty="10", filled_qty="0",
    )
    client = FakeClient(open_orders=[acknowledged_order, mystery_order], positions=[])
    runner_state = LiveRunnerState()

    intent = order_intent.create_intent(
        client_order_id="AA-cid", account_identity="control_arm_v1", ticker="AA",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY",
        source_signal_timestamp="2026-08-24",
    )
    intent = order_intent.transition_intent(intent, order_intent.SUBMITTING)
    order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="broker-order-1", broker_status="filled",
    )

    with pytest.raises(br.JournalAcknowledgedButLocalStateMissingError, match="unknown-1") as exc_info:
        br.reconcile(client, runner_state)
    assert "broker-order-1" in str(exc_info.value)


def test_scenario_d_journal_entry_not_yet_broker_acknowledged_does_not_correlate():
    """A PREPARED/SUBMITTING journal entry (broker truth NOT yet known)
    must never suppress or soften Scenario D -- only a resolved
    BROKER_ACKNOWLEDGED entry is a legitimate correlation."""
    unknown_order = FakeOrder(
        id="unknown-1", client_order_id="AA-cid", symbol="AA",
        side="buy", status="accepted", qty="10", filled_qty="0",
    )
    client = FakeClient(open_orders=[unknown_order], positions=[])
    runner_state = LiveRunnerState()

    order_intent.create_intent(
        client_order_id="AA-cid", account_identity="control_arm_v1", ticker="AA",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY",
        source_signal_timestamp="2026-08-24",
    )  # left at PREPARED -- broker truth not yet known

    with pytest.raises(br.UnknownBrokerOrderError, match="unknown-1"):
        br.reconcile(client, runner_state)


# ---------------------------------------------------------------------------
# Scenario E (new): local-terminal-but-broker-open
# ---------------------------------------------------------------------------


def test_local_terminal_but_broker_open_fails_closed():
    order = FakeOrder(
        id="order-1", client_order_id="cid-1", symbol="AAPL",
        side="buy", status="new", qty="10", filled_qty="0",
    )
    runner_state = LiveRunnerState()
    runner_state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-1", client_order_id="cid-1", status="filled",  # LOCAL says filled...
    )
    client = FakeClient(open_orders=[order], positions=[])  # ...but broker STILL shows it open
    with pytest.raises(br.LocalTerminalButBrokerOpenError):
        br.reconcile(client, runner_state)


def test_local_terminal_and_broker_also_closed_is_fine():
    runner_state = LiveRunnerState()
    runner_state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-1", client_order_id="cid-1", status="filled",
    )
    client = FakeClient(open_orders=[], positions=[])  # broker agrees it's no longer open
    result = br.reconcile(client, runner_state)
    assert result.order_status_updates == {}


# ---------------------------------------------------------------------------
# Scenario F (new): equity_stop_orders invariant
# ---------------------------------------------------------------------------


def test_equity_stop_orders_orphaned_reference_fails_closed():
    runner_state = LiveRunnerState()
    runner_state.equity_stop_orders["AAPL"] = "stop-order-nowhere-in-submitted-actions"
    client = FakeClient(positions=[])
    with pytest.raises(br.StopOrderInvariantViolationError, match="orphaned"):
        br.reconcile(client, runner_state)


def test_equity_stop_orders_wrong_kind_fails_closed():
    runner_state = LiveRunnerState()
    runner_state.equity_stop_orders["AAPL"] = "order-1"
    runner_state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-16"] = entry_buy_record(
        order_id="order-1", client_order_id="cid-1", status="filled",
    )
    client = FakeClient(positions=[])
    with pytest.raises(br.StopOrderInvariantViolationError, match="not 'PROTECTIVE_STOP'"):
        br.reconcile(client, runner_state)


def test_equity_stop_orders_missing_at_broker_while_position_open_fails_closed():
    runner_state = LiveRunnerState()
    runner_state.positions["AAPL"] = make_position("AAPL", 10.0)
    runner_state.equity_stop_orders["AAPL"] = "stop-order-1"
    runner_state.submitted_actions["AAPL|PROTECTIVE_STOP|2026-08-16"] = {
        "order_id": "stop-order-1", "status": "new", "kind": "PROTECTIVE_STOP",
        "submitted_at": "2026-08-16T00:00:00Z", "client_order_id": "stop-cid-1",
    }
    # Broker shows the position, but NOT the stop order as open -- gap.
    client = FakeClient(positions=[FakePosition(symbol="AAPL", qty="10")], open_orders=[])
    with pytest.raises(br.StopOrderInvariantViolationError, match="downside protection"):
        br.reconcile(client, runner_state)


def test_equity_stop_orders_present_and_consistent_passes():
    runner_state = LiveRunnerState()
    runner_state.positions["AAPL"] = make_position("AAPL", 10.0)
    runner_state.equity_stop_orders["AAPL"] = "stop-order-1"
    runner_state.submitted_actions["AAPL|PROTECTIVE_STOP|2026-08-16"] = {
        "order_id": "stop-order-1", "status": "new", "kind": "PROTECTIVE_STOP",
        "submitted_at": "2026-08-16T00:00:00Z", "client_order_id": "stop-cid-1",
    }
    stop_order = FakeOrder(
        id="stop-order-1", client_order_id="stop-cid-1", symbol="AAPL",
        side="sell", status="new", qty="10", filled_qty="0",
    )
    client = FakeClient(positions=[FakePosition(symbol="AAPL", qty="10")], open_orders=[stop_order])
    result = br.reconcile(client, runner_state)
    assert result.order_status_updates == {}


# ---------------------------------------------------------------------------
# Real API call-count proof (deterministic; real wall-clock latency against
# the actual paper account was measured separately this session -- see
# module docstring)
# ---------------------------------------------------------------------------


def test_clean_pass_reports_expected_broker_call_count():
    client = FakeClient(open_orders=[], positions=[])
    runner_state = LiveRunnerState()
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    br.reconcile(client, runner_state)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[broker_reconciliation call-count] {started_at} -- "
        f"get_account={client.get_account_call_count}, "
        f"get_orders={client.get_orders_call_count}, "
        f"get_all_positions={client.get_all_positions_call_count}, "
        f"get_order_by_id={client.get_order_by_id_call_count}, "
        f"local elapsed={elapsed:.6f}s (fake client; real paper-account "
        f"latency for this exact call sequence was measured separately "
        f"this session at ~0.5s for 4 calls, zero retries needed)."
    )
    assert client.get_account_call_count == 1
    assert client.get_orders_call_count == 2  # before + after, one attempt
    assert client.get_all_positions_call_count == 1
    assert client.get_order_by_id_call_count == 0  # nothing stale to resolve in a clean pass


# ---------------------------------------------------------------------------
# Independent audit finding #5 (2026-08-22): bare (separator-less) crypto
# symbol form -- a theoretical risk (alpaca-py issue #537 referenced by the
# audit; no open crypto position exists on the real server today, so this
# has never actually been observed against production data). Closed
# defensively before the first real crypto position exists.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broker_symbol,expected",
    [
        ("BTC/USD", "BTC-USD"),
        ("ETH/USD", "ETH-USD"),
        ("BTCUSD", "BTC-USD"),  # bare form, no separator
        ("ETHUSD", "ETH-USD"),
        ("UNIUSD", "UNI-USD"),
        ("BTCEUR", "BTC-EUR"),
        ("BTCGBP", "BTC-GBP"),
    ],
)
def test_normalize_broker_symbol_handles_bare_and_slash_crypto_forms(broker_symbol, expected):
    assert br._normalize_broker_symbol(broker_symbol) == expected


@pytest.mark.parametrize("equity_symbol", ["AAPL", "PINS", "UNH", "JPM", "XOM", "A", "AA"])
def test_normalize_broker_symbol_is_a_no_op_for_real_equity_tickers(equity_symbol):
    """Guards against the bare-crypto-form regex ever misfiring on a real
    equity ticker -- none in either universe ends in USD/EUR/GBP, but this
    proves the narrow match, not just asserts the absence of a collision
    in today's ticker lists."""
    assert br._normalize_broker_symbol(equity_symbol) == equity_symbol


def test_bare_crypto_position_reconciles_cleanly_end_to_end():
    """Real reproduction, one level up from the unit test above: an open
    BTC-USD position reported by the broker with NO separator at all
    (`BTCUSD`) must reconcile cleanly, exactly like the already-covered
    slash form (BTC/USD) does above."""
    runner_state = LiveRunnerState()
    crypto_position = make_position("BTC-USD", 1.0)
    crypto_position.asset_class = "CRYPTO"
    runner_state.positions["BTC-USD"] = crypto_position
    client = FakeClient(positions=[FakePosition(symbol="BTCUSD", qty="1.0")])
    result = br.reconcile(client, runner_state, equity_orders_enabled=True, crypto_orders_enabled=True)
    assert result.broker_position_count == 1
