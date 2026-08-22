"""Regression tests for run_daily_decision.py's equity/crypto date handling.

Production incident (2026-08-10, weekend): the daily cron crashed with
`ValueError: Prepared tickers do not share a common latest calendar
date` because equity's latest bar was Friday (market closed on
weekends) while crypto's was today (crypto trades 24/7). The original
`_bars_today` required ALL nine controlled tickers to share one common
latest calendar date -- an assumption that breaks on every weekend and
every equity holiday. These tests pin down the fix: equity tickers must
only agree with each other, crypto tickers must only agree with each
other, and the two asset classes are never required to agree with one
another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402

EQUITY_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]


def _bar_only_frame(*dates: str) -> pd.DataFrame:
    """Minimal frame with just what `_bars_today` reads (index + OHLC)."""
    index = pd.DatetimeIndex([pd.Timestamp(date) for date in dates])
    n = len(dates)
    return pd.DataFrame(
        {"Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n, "Close": [1.0] * n},
        index=index,
    )


def _decision_ready_frame(*dates: str) -> pd.DataFrame:
    """Frame with the columns the six engine steps actually touch.

    EMA20 < EMA50 guarantees `_is_entry_setup` is always False (see
    `portfolio_backtest_engine._is_entry_setup`), so no new position is
    opened -- these tests are only about the date-handling fix, not
    entry/exit decision logic.
    """
    index = pd.DatetimeIndex([pd.Timestamp(date) for date in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.0] * n,
            "EMA20": [95.0] * n,
            "EMA50": [100.0] * n,
            "RSI14": [50.0] * n,
            "RegimeAllowed": [True] * n,
        },
        index=index,
    )


def test_weekend_scenario_equity_closed_crypto_open_does_not_raise():
    """The exact production failure shape: equity stuck on Friday, crypto on today."""
    prepared = {
        **{ticker: _bar_only_frame("2026-08-07") for ticker in EQUITY_TICKERS},
        **{ticker: _bar_only_frame("2026-08-10") for ticker in CRYPTO_TICKERS},
    }

    bars_today, equity_date, crypto_date = runner._bars_today_by_asset_class(prepared)

    assert equity_date == "2026-08-07"
    assert crypto_date == "2026-08-10"
    assert set(bars_today) == set(prepared)


def test_matching_dates_across_asset_classes_still_works_as_before():
    prepared = {
        **{ticker: _bar_only_frame("2026-08-06") for ticker in EQUITY_TICKERS},
        **{ticker: _bar_only_frame("2026-08-06") for ticker in CRYPTO_TICKERS},
    }

    bars_today, equity_date, crypto_date = runner._bars_today_by_asset_class(prepared)

    assert equity_date == crypto_date == "2026-08-06"
    assert set(bars_today) == set(prepared)


def test_equity_tickers_disagreeing_among_themselves_still_raises():
    """The narrowed invariant is still enforced WITHIN each asset class."""
    prepared = {
        "AAPL": _bar_only_frame("2026-08-07"),
        "MSFT": _bar_only_frame("2026-08-06"),
        "BTC-USD": _bar_only_frame("2026-08-10"),
        "ETH-USD": _bar_only_frame("2026-08-10"),
    }

    with pytest.raises(ValueError, match="do not share a common latest calendar date"):
        runner._bars_today_by_asset_class(prepared)


def test_crypto_tickers_disagreeing_among_themselves_still_raises():
    prepared = {
        "AAPL": _bar_only_frame("2026-08-07"),
        "MSFT": _bar_only_frame("2026-08-07"),
        "BTC-USD": _bar_only_frame("2026-08-10"),
        "ETH-USD": _bar_only_frame("2026-08-09"),
    }

    with pytest.raises(ValueError, match="do not share a common latest calendar date"):
        runner._bars_today_by_asset_class(prepared)


class _FakeAccount:
    account_number = "PA3HONFDTEST"


class _FakeReconciliationClient:
    """Minimal fake satisfying broker_reconciliation.reconcile()'s real
    calls for a clean pass against EMPTY local state (fresh
    LiveRunnerState -- no submitted_actions, no equity_stop_orders, so
    neither the Scenario C loop nor the Scenario F invariant loop has
    anything to iterate, and get_order_by_id is never reached)."""

    def get_account(self):
        return _FakeAccount()

    def get_orders(self, *args, **kwargs):
        return []

    def get_all_positions(self):
        return []


def test_run_daily_decision_end_to_end_does_not_crash_on_weekend_dates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Full call chain through the real crash site: run_daily_decision()
    itself, with equity stuck on Friday and crypto on today, must
    complete without raising and must record both dates."""
    prepared = {
        **{
            ticker: _decision_ready_frame("2026-08-06", "2026-08-07")
            for ticker in EQUITY_TICKERS
        },
        **{
            ticker: _decision_ready_frame("2026-08-09", "2026-08-10")
            for ticker in CRYPTO_TICKERS
        },
    }

    monkeypatch.setattr(runner, "prepare_live_market_data", lambda tickers: prepared)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda: 100_000.0)

    result = runner.run_daily_decision(
        state_path=tmp_path / "position_state.json",
        decision_log_directory=tmp_path / "decisions",
        # Isolate the rollback high-water-mark to this test's own tmp_path --
        # the real default lives outside the repo (see position_state.py)
        # specifically so it is NOT test-isolated by tmp_path alone; a
        # synthetic run here must not read or write the real, machine-wide
        # guard file (confirmed the hard way: it collided with state left
        # behind by an earlier, unrelated test run in the same session).
        guard_path=tmp_path / "high_water_mark.json",
        # Injection seam (added alongside the broker-reconciliation wiring
        # fix, 2026-08-21) -- without this, run_daily_decision() now
        # builds a real TradingClient and calls broker_reconciliation.reconcile()
        # unconditionally, which would need real Alpaca credentials this
        # test environment does not have.
        trading_client=_FakeReconciliationClient(),
    )

    decision = result["decision"]
    assert decision["as_of_bar_timestamp_equity"] == "2026-08-07"
    assert decision["as_of_bar_timestamp_crypto"] == "2026-08-10"
    assert decision["as_of_bar_timestamp"] == "2026-08-10"


def test_no_equity_tickers_raises_a_clear_error():
    prepared = {ticker: _bar_only_frame("2026-08-10") for ticker in CRYPTO_TICKERS}

    with pytest.raises(ValueError, match="No equity tickers"):
        runner._bars_today_by_asset_class(prepared)


def test_no_crypto_tickers_raises_a_clear_error():
    prepared = {ticker: _bar_only_frame("2026-08-07") for ticker in EQUITY_TICKERS}

    with pytest.raises(ValueError, match="No crypto tickers"):
        runner._bars_today_by_asset_class(prepared)
