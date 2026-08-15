"""Tests for scripts/generate_performance_report.py.

Deterministic and network-free: fake Order-like objects stand in for
Alpaca's real `Order` model, and a fake client stands in for
`TradingClient`. No real API call is made.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_performance_report as report  # noqa: E402


def fake_order(
    *,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    filled_at: str,
    status: str = "filled",
    asset_class: str = "us_equity",
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        status=status,
        asset_class=asset_class,
        filled_qty=quantity,
        filled_avg_price=price,
        filled_at=datetime.fromisoformat(filled_at).replace(tzinfo=timezone.utc),
    )


class FakeClient:
    def __init__(self, orders: list, positions: list | None = None):
        self._orders = orders
        self._positions = positions or []

    def get_orders(self, filter=None):
        return list(self._orders)

    def get_all_positions(self):
        return list(self._positions)


def fake_position(
    *, symbol: str, quantity: float, entry_price: float, asset_class: str = "crypto"
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=quantity,
        avg_entry_price=entry_price,
        asset_class=asset_class,
    )


def test_fetch_filled_orders_excludes_non_filled_and_zero_quantity():
    orders = [
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100, filled_at="2026-01-01T00:00:00", status="filled"),
        fake_order(symbol="AAPL", side="buy", quantity=0, price=100, filled_at="2026-01-01T00:01:00", status="filled"),
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100, filled_at="2026-01-01T00:02:00", status="canceled"),
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100, filled_at="2026-01-01T00:03:00", status="rejected"),
    ]
    client = FakeClient(orders)

    result = report.fetch_filled_orders(client)

    assert len(result) == 1
    assert result[0].status == "filled"


def test_simple_buy_then_sell_produces_one_closed_trade():
    orders = [
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100.0, filled_at="2026-01-01T00:00:00"),
        fake_order(symbol="AAPL", side="sell", quantity=1, price=110.0, filled_at="2026-01-02T00:00:00"),
    ]

    closed, open_lots = report.reconstruct_round_trips(orders)

    assert open_lots == []
    assert len(closed) == 1
    trade = closed[0]
    assert trade.symbol == "AAPL"
    assert trade.entry_price == 100.0
    assert trade.exit_price == 110.0
    assert trade.quantity == 1
    assert trade.pnl == pytest.approx(10.0)


def test_unmatched_buy_is_reported_as_an_open_lot_not_a_trade():
    orders = [
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100.0, filled_at="2026-01-01T00:00:00"),
    ]

    closed, open_lots = report.reconstruct_round_trips(orders)

    assert closed == []
    assert len(open_lots) == 1
    assert open_lots[0].symbol == "AAPL"
    assert open_lots[0].quantity == 1
    assert open_lots[0].entry_price == 100.0


def test_partial_sell_closes_part_of_a_lot_and_leaves_the_rest_open():
    orders = [
        fake_order(symbol="BTC/USD", side="buy", quantity=1.0, price=60000.0, filled_at="2026-01-01T00:00:00", asset_class="crypto"),
        fake_order(symbol="BTC/USD", side="sell", quantity=0.4, price=65000.0, filled_at="2026-01-02T00:00:00", asset_class="crypto"),
    ]

    closed, open_lots = report.reconstruct_round_trips(orders)

    assert len(closed) == 1
    assert closed[0].quantity == pytest.approx(0.4)
    assert closed[0].pnl == pytest.approx((65000.0 - 60000.0) * 0.4)
    assert len(open_lots) == 1
    assert open_lots[0].quantity == pytest.approx(0.6)


def test_crypto_in_kind_fee_dust_is_removed_when_broker_is_flat():
    orders = [
        fake_order(
            symbol="BTC/USD", side="buy", quantity=0.001, price=60000.0,
            filled_at="2026-01-01T00:00:00", asset_class="crypto",
        ),
        fake_order(
            symbol="BTC/USD", side="sell", quantity=0.000999, price=61000.0,
            filled_at="2026-01-02T00:00:00", asset_class="crypto",
        ),
    ]

    _, fifo_open_lots = report.reconstruct_round_trips(orders)
    reconciled, details = report.reconcile_crypto_open_lots(fifo_open_lots, [])

    assert sum(lot.quantity for lot in fifo_open_lots) == pytest.approx(0.000001)
    assert reconciled == []
    assert len(details) == 1
    assert details[0].fifo_quantity == pytest.approx(0.000001)
    assert details[0].broker_quantity == 0.0
    assert "in-kind fee dust" in details[0].resolution


def test_crypto_open_quantity_and_entry_price_use_broker_ground_truth():
    fifo_open_lots = [report.OpenLot("BTC/USD", "crypto", 0.6, 60000.0, "t1")]
    broker_positions = [
        fake_position(symbol="BTCUSD", quantity=0.599, entry_price=60010.0)
    ]

    reconciled, details = report.reconcile_crypto_open_lots(
        fifo_open_lots, broker_positions
    )

    assert len(reconciled) == 1
    assert reconciled[0].symbol == "BTC/USD"
    assert reconciled[0].quantity == pytest.approx(0.599)
    assert reconciled[0].entry_price == pytest.approx(60010.0)
    assert details[0].fifo_quantity == pytest.approx(0.6)
    assert details[0].broker_quantity == pytest.approx(0.599)
    assert "broker quantity used" in details[0].resolution


def test_equity_open_lots_are_unchanged_by_crypto_reconciliation():
    equity_lot = report.OpenLot("AAPL", "us_equity", 2.0, 100.0, "t1")

    reconciled, details = report.reconcile_crypto_open_lots([equity_lot], [])

    assert reconciled == [equity_lot]
    assert details == []


def test_symbols_are_matched_independently():
    orders = [
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100.0, filled_at="2026-01-01T00:00:00"),
        fake_order(symbol="AAPL", side="sell", quantity=1, price=90.0, filled_at="2026-01-02T00:00:00"),
        fake_order(symbol="MSFT", side="buy", quantity=2, price=200.0, filled_at="2026-01-01T00:00:00"),
        fake_order(symbol="MSFT", side="sell", quantity=2, price=210.0, filled_at="2026-01-02T00:00:00"),
    ]

    closed, open_lots = report.reconstruct_round_trips(orders)

    assert open_lots == []
    symbols = {trade.symbol for trade in closed}
    assert symbols == {"AAPL", "MSFT"}
    aapl_trade = next(trade for trade in closed if trade.symbol == "AAPL")
    msft_trade = next(trade for trade in closed if trade.symbol == "MSFT")
    assert aapl_trade.pnl == pytest.approx(-10.0)
    assert msft_trade.pnl == pytest.approx(20.0)


def test_fifo_matches_oldest_buy_lot_first():
    orders = [
        fake_order(symbol="AAPL", side="buy", quantity=1, price=100.0, filled_at="2026-01-01T00:00:00"),
        fake_order(symbol="AAPL", side="buy", quantity=1, price=120.0, filled_at="2026-01-02T00:00:00"),
        fake_order(symbol="AAPL", side="sell", quantity=1, price=130.0, filled_at="2026-01-03T00:00:00"),
    ]

    closed, open_lots = report.reconstruct_round_trips(orders)

    assert len(closed) == 1
    assert closed[0].entry_price == 100.0, "must consume the OLDEST lot first, not the cheapest/newest"
    assert len(open_lots) == 1
    assert open_lots[0].entry_price == 120.0


def test_compute_metrics_separates_by_asset_class_and_computes_win_rate():
    closed_trades = [
        report.ClosedTrade("AAPL", "us_equity", 100, 110, 1, "t1", "t2", 10.0),
        report.ClosedTrade("MSFT", "us_equity", 200, 190, 1, "t1", "t2", -10.0),
        report.ClosedTrade("BTC/USD", "crypto", 60000, 65000, 0.1, "t1", "t2", 500.0),
    ]

    metrics = report.compute_metrics(closed_trades)

    assert metrics["us_equity"]["total_trades"] == 2
    assert metrics["us_equity"]["win_rate_percent"] == 50.0
    assert metrics["us_equity"]["cumulative_pnl"] == pytest.approx(0.0)
    assert metrics["crypto"]["total_trades"] == 1
    assert metrics["crypto"]["win_rate_percent"] == 100.0
    assert metrics["crypto"]["cumulative_pnl"] == pytest.approx(500.0)


def test_compute_metrics_on_empty_input_does_not_crash():
    assert report.compute_metrics([]) == {}


def test_scan_needs_manual_review_reads_and_flattens_entries(tmp_path: Path):
    log_directory = tmp_path / "decisions"
    log_directory.mkdir()
    (log_directory / "decision_1.json").write_text(
        '{"as_of_bar_timestamp": "2026-08-10", "needs_manual_review": '
        '[{"ticker": "AAPL", "issue": "not confirmed filled"}]}'
    )
    (log_directory / "decision_2.json").write_text(
        '{"as_of_bar_timestamp": "2026-08-11", "needs_manual_review": []}'
    )

    entries = report.scan_needs_manual_review(log_directory)

    assert len(entries) == 1
    assert entries[0]["ticker"] == "AAPL"
    assert entries[0]["as_of_bar_timestamp"] == "2026-08-10"
    assert entries[0]["decision_log"] == "decision_1.json"


def test_scan_needs_manual_review_missing_directory_returns_empty():
    assert report.scan_needs_manual_review(Path("/nonexistent/path/for/sure")) == []


def test_scan_needs_manual_review_skips_malformed_json_without_crashing(tmp_path: Path):
    log_directory = tmp_path / "decisions"
    log_directory.mkdir()
    (log_directory / "decision_bad.json").write_text("{ not valid json")

    assert report.scan_needs_manual_review(log_directory) == []


def test_build_report_text_with_no_data_is_graceful():
    text = report.build_report_text(
        closed_trades=[], open_lots=[], metrics={}, review_entries=[], generated_at="2026-08-10"
    )

    assert "No filled orders found yet" in text
    assert "None (0 open positions)" in text
    assert "No needs_manual_review entries" in text


def test_build_report_text_explains_removed_crypto_dust():
    reconciliation = [
        report.CryptoOpenLotReconciliation(
            symbol="BTC/USD",
            fifo_quantity=0.000779697,
            broker_quantity=0.0,
            resolution=(
                "removed FIFO-only residual; broker reports no position "
                "(consistent with in-kind fee dust)"
            ),
        )
    ]

    text = report.build_report_text(
        closed_trades=[],
        open_lots=[],
        metrics={},
        review_entries=[],
        generated_at="2026-08-15",
        crypto_reconciliation=reconciliation,
    )

    assert "FIFO qty 0.000779697" in text
    assert "broker qty 0" in text
    assert "in-kind fee dust" in text


def test_build_report_text_includes_trades_open_lots_and_review():
    closed_trades = [report.ClosedTrade("AAPL", "us_equity", 100, 110, 1, "t1", "t2", 10.0)]
    open_lots = [report.OpenLot("MSFT", "us_equity", 2, 200.0, "t1")]
    metrics = report.compute_metrics(closed_trades)
    review_entries = [
        {"as_of_bar_timestamp": "2026-08-10", "ticker": "ETH-USD", "issue": "no available quantity", "decision_log": "decision_1.json"}
    ]

    text = report.build_report_text(
        closed_trades=closed_trades, open_lots=open_lots, metrics=metrics,
        review_entries=review_entries, generated_at="2026-08-10",
    )

    assert "us_equity" in text
    assert "AAPL" in text
    assert "MSFT" in text
    assert "ETH-USD" in text
    assert "no available quantity" in text
