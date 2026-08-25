"""Tests for src/live/read_only_portfolio_snapshot.py -- the shared,
read-only P&L data layer for the P&L Telegram notification and (later)
the dashboard.

A fake `TradingClient` (account/positions/orders only, no order-mutating
methods at all -- their absence is itself part of the proof that this
module never calls them) drives every test; no real network/broker
access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import scripts.generate_performance_report as generate_performance_report
from src.live import read_only_portfolio_snapshot as ros


class _FakeAccount:
    def __init__(self, *, cash: str = "100000.00", equity: str = "100000.00", last_equity: str = "100000.00") -> None:
        self.cash = cash
        self.equity = equity
        self.last_equity = last_equity


class _FakePosition:
    def __init__(
        self, *, symbol: str, asset_class: str = "us_equity", side: str = "long",
        qty: str, avg_entry_price: str, current_price: str, market_value: str,
        unrealized_pl: str, unrealized_plpc: str,
    ) -> None:
        self.symbol = symbol
        self.asset_class = asset_class
        self.side = side
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price
        self.market_value = market_value
        self.unrealized_pl = unrealized_pl
        self.unrealized_plpc = unrealized_plpc


class _FakeOrder:
    def __init__(
        self, *, symbol: str, side: str, status: str, filled_qty: str, filled_avg_price: str,
        filled_at: datetime, asset_class: str = "us_equity", client_order_id: str = "AAPL-ENTRY-1",
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.filled_at = filled_at
        self.asset_class = asset_class
        self.client_order_id = client_order_id


class _FakeClient:
    """Deliberately has NO submit/cancel/replace method at all -- any
    accidental call to one would raise AttributeError, which no test
    below ever catches, so a passing test suite is itself proof this
    module never attempts one."""

    def __init__(self, *, account: _FakeAccount, positions: list, orders: list) -> None:
        self._account = account
        self._positions = positions
        self._orders = orders

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def get_orders(self, *, filter=None):
        return list(self._orders)


def _at(day: int) -> datetime:
    return datetime(2026, 8, day, 20, 0, 0, tzinfo=timezone.utc)


def test_empty_account_has_zeroed_pnl_and_complete_history():
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=[])

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.positions == []
    assert snapshot.cash == Decimal("100000.00")
    assert snapshot.portfolio_value == Decimal("100000.00")
    assert snapshot.day_pnl == Decimal("0")
    assert snapshot.broker_unrealized_pnl_total == Decimal("0")
    assert snapshot.realized_closed_trade_pnl == Decimal("0")
    assert snapshot.realized_history_complete is True
    assert snapshot.combined_strategy_pnl == Decimal("0")


def test_day_pnl_is_the_whole_account_equity_change_never_folded_into_combined():
    client = _FakeClient(
        account=_FakeAccount(cash="50000", equity="102500.00", last_equity="100000.00"),
        positions=[], orders=[],
    )

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.day_pnl == Decimal("2500.00")
    assert snapshot.combined_strategy_pnl == Decimal("0")  # unaffected by day_pnl -- separate concept


def test_long_position_fields_map_directly_from_broker_and_integrity_check_agrees():
    position = _FakePosition(
        symbol="AAPL", side="long", qty="10", avg_entry_price="150.00", current_price="160.00",
        market_value="1600.00", unrealized_pl="100.00", unrealized_plpc="0.0667",
    )
    client = _FakeClient(account=_FakeAccount(), positions=[position], orders=[])

    snapshot = ros.fetch_portfolio_pnl(client)

    assert len(snapshot.positions) == 1
    p = snapshot.positions[0]
    assert p.ticker == "AAPL"
    assert p.asset_class == "us_equity"
    assert p.side == "long"
    assert p.quantity == Decimal("10")
    assert p.avg_entry_price == Decimal("150.00")
    assert p.current_price == Decimal("160.00")
    assert p.market_value == Decimal("1600.00")
    assert p.broker_unrealized_pnl == Decimal("100.00")  # the authoritative number
    # LONG: (current - entry) * qty = (160-150)*10 = 100 -- agrees with broker here.
    assert p.calculated_unrealized_pnl == Decimal("100.00")
    assert p.integrity_discrepancy == Decimal("0")
    assert snapshot.broker_unrealized_pnl_total == Decimal("100.00")


def test_short_position_uses_the_short_formula_and_broker_value_stays_authoritative():
    position = _FakePosition(
        symbol="TSLA", side="short", qty="-5", avg_entry_price="200.00", current_price="180.00",
        market_value="-900.00", unrealized_pl="99.50", unrealized_plpc="0.0995",
    )
    client = _FakeClient(account=_FakeAccount(), positions=[position], orders=[])

    snapshot = ros.fetch_portfolio_pnl(client)

    p = snapshot.positions[0]
    # SHORT: (entry - current) * abs(qty) = (200-180)*5 = 100.
    assert p.calculated_unrealized_pnl == Decimal("100.00")
    # Broker's own 99.50 (e.g. a fee/swap this module doesn't model) is
    # still what's reported as broker_unrealized_pnl -- never overridden.
    assert p.broker_unrealized_pnl == Decimal("99.50")
    assert p.integrity_discrepancy == Decimal("0.50")


def test_realized_pnl_reuses_reconstruct_round_trips_for_a_real_closed_buy_sell_pair():
    orders = [
        _FakeOrder(symbol="MSFT", side="buy", status="filled", filled_qty="10", filled_avg_price="300.00", filled_at=_at(1)),
        _FakeOrder(symbol="MSFT", side="sell", status="filled", filled_qty="10", filled_avg_price="310.00", filled_at=_at(5)),
    ]
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=orders)

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.realized_closed_trade_pnl == Decimal("100.00")  # (310-300)*10
    assert snapshot.realized_history_complete is True
    assert snapshot.combined_strategy_pnl == Decimal("100.00")


def test_combined_pnl_adds_realized_and_broker_unrealized_and_carries_the_exact_required_label():
    position = _FakePosition(
        symbol="AAPL", side="long", qty="10", avg_entry_price="150.00", current_price="160.00",
        market_value="1600.00", unrealized_pl="100.00", unrealized_plpc="0.0667",
    )
    orders = [
        _FakeOrder(symbol="MSFT", side="buy", status="filled", filled_qty="10", filled_avg_price="300.00", filled_at=_at(1)),
        _FakeOrder(symbol="MSFT", side="sell", status="filled", filled_qty="10", filled_avg_price="310.00", filled_at=_at(5)),
    ]
    client = _FakeClient(account=_FakeAccount(), positions=[position], orders=orders)

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.combined_strategy_pnl == Decimal("200.00")  # 100 realized + 100 unrealized
    assert snapshot.combined_strategy_pnl_label == "Strategy fill-based P&L (realized + broker unrealized)"
    assert "lifetime" not in snapshot.combined_strategy_pnl_label.lower()


def test_realized_history_complete_is_false_when_the_order_lookback_limit_is_hit(monkeypatch):
    """Independent-audit spec: hitting the 500-record cap must never be
    silently presented as a complete/definitive number. Monkeypatches
    the limit down to 2 so this test doesn't need 500 fake orders."""
    monkeypatch.setattr(generate_performance_report, "DEFAULT_ORDER_LOOKBACK_LIMIT", 2)
    orders = [
        _FakeOrder(symbol="MSFT", side="buy", status="filled", filled_qty="10", filled_avg_price="300.00", filled_at=_at(1)),
        _FakeOrder(symbol="MSFT", side="sell", status="filled", filled_qty="10", filled_avg_price="310.00", filled_at=_at(5)),
    ]
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=orders)

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.realized_history_complete is False
    # The number itself is still computed from whatever WAS fetched --
    # "don't show a definitive number" is the caller's own display
    # concern (a footnote/caveat), not this module silently omitting data.
    assert snapshot.realized_closed_trade_pnl == Decimal("100.00")


def test_realized_history_complete_is_true_when_strictly_under_the_limit(monkeypatch):
    monkeypatch.setattr(generate_performance_report, "DEFAULT_ORDER_LOOKBACK_LIMIT", 5)
    orders = [
        _FakeOrder(symbol="MSFT", side="buy", status="filled", filled_qty="10", filled_avg_price="300.00", filled_at=_at(1)),
        _FakeOrder(symbol="MSFT", side="sell", status="filled", filled_qty="10", filled_avg_price="310.00", filled_at=_at(5)),
    ]
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=orders)

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.realized_history_complete is True


def test_heartbeat_and_non_filled_orders_are_excluded_from_realized_pnl():
    """Delegated entirely to fetch_filled_orders's own established
    filtering (heartbeat client_order_id prefix, non-filled statuses) --
    this test just confirms the delegation actually happens end to end,
    not reimplemented/bypassed here."""
    orders = [
        _FakeOrder(
            symbol="BTC-USD", side="buy", status="filled", filled_qty="1", filled_avg_price="15.00",
            filled_at=_at(1), client_order_id="HEARTBEAT_abc123",
        ),
        _FakeOrder(
            symbol="BTC-USD", side="sell", status="filled", filled_qty="1", filled_avg_price="20.00",
            filled_at=_at(2), client_order_id="HEARTBEAT_abc123",
        ),
        _FakeOrder(symbol="AAPL", side="buy", status="canceled", filled_qty="0", filled_avg_price="0", filled_at=_at(3)),
    ]
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=orders)

    snapshot = ros.fetch_portfolio_pnl(client)

    assert snapshot.realized_closed_trade_pnl == Decimal("0")
