"""Tests for scripts/run_control_arm_decision.py's
`_guard_against_premature_order_activation` -- the hard, zero-API-call
guard added after an independent integrated-chain audit found the intent
journal currently records PREPARED after `rdd.run_daily_decision()`
returns, not before its real broker call. Blocks
`--enable-equity-orders`/`--enable-crypto-orders` entirely until the
full write-ahead ordering fix (designed, not yet built -- see that
module's own docstring) exists.

Deterministic, no API dependency -- the guard itself never touches
Alpaca; these tests call it directly with a plain namespace.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_control_arm_decision as carm
import src.live.broker_reconciliation as broker_reconciliation
import src.live.pending_signal_ttl as pending_signal_ttl
import src.live.position_state as ps


def test_equity_orders_flag_is_blocked():
    args = SimpleNamespace(enable_equity_orders=True, enable_crypto_orders=False)
    with pytest.raises(RuntimeError, match="BLOCKED"):
        carm._guard_against_premature_order_activation(args)


def test_crypto_orders_flag_is_blocked():
    args = SimpleNamespace(enable_equity_orders=False, enable_crypto_orders=True)
    with pytest.raises(RuntimeError, match="BLOCKED"):
        carm._guard_against_premature_order_activation(args)


def test_both_flags_set_is_blocked():
    args = SimpleNamespace(enable_equity_orders=True, enable_crypto_orders=True)
    with pytest.raises(RuntimeError, match="BLOCKED"):
        carm._guard_against_premature_order_activation(args)


def test_neither_flag_set_does_not_raise():
    args = SimpleNamespace(enable_equity_orders=False, enable_crypto_orders=False)
    carm._guard_against_premature_order_activation(args)  # must not raise


def test_execute_calls_the_guard_first_zero_api_calls():
    """Proves the guard is actually wired into _execute() as the FIRST
    thing it does -- a poison client whose any method call raises must
    NEVER be touched."""
    from pathlib import Path

    class PoisonClient:
        def __getattr__(self, name):
            def _boom(*a, **k):
                raise AssertionError(f"{name}() was called -- guard did not fire first")
            return _boom

    args = SimpleNamespace(
        state_path=Path("data/live/position_state.json"),
        decision_log_directory=Path("data/live/decisions"),
        enable_equity_orders=True,
        enable_crypto_orders=False,
    )
    with pytest.raises(RuntimeError, match="BLOCKED"):
        carm._execute(args, trading_client=PoisonClient())


# --- Independent audit finding #1 (2026-08-22): reconcile() call-site regression ---


def test_reconcile_call_site_kwargs_bind_against_the_real_signature():
    """REAL REGRESSION FOUND AND FIXED (2026-08-22, independent audit):
    broker_reconciliation.reconcile()'s signature moved from a single
    blended `orders_enabled` to independent `equity_orders_enabled`/
    `crypto_orders_enabled` when run_daily_decision.py's own caller was
    fixed for the same reason -- this file's own call site was never
    updated to match, so every real control-arm run raised a real
    TypeError (unexpected keyword argument 'orders_enabled') before
    ever reaching rdd.run_daily_decision(). Extracts the real kwargs
    used at the call site from source text (never a hand-copied guess)
    and binds them against reconcile()'s REAL, current signature -- a
    future rename/signature drift on either side fails this test
    loudly instead of only failing on the real server."""
    source = inspect.getsource(carm._execute)
    call_start = source.index("broker_reconciliation.reconcile(")
    call_text = source[call_start:call_start + 1400]

    assert "equity_orders_enabled=arguments.enable_equity_orders" in call_text
    assert "crypto_orders_enabled=arguments.enable_crypto_orders" in call_text
    # The old, removed blended kwarg (bare "orders_enabled=", not the
    # "equity_"/"crypto_"-prefixed forms above) must never come back.
    assert not re.search(r"(?<![a-z_])orders_enabled=", call_text)

    signature = inspect.signature(broker_reconciliation.reconcile)
    signature.bind(
        object(), object(),
        expected_account_suffix="XXXX",
        equity_orders_enabled=True,
        crypto_orders_enabled=True,
    )  # raises TypeError on any kwarg-name mismatch -- the real regression's exact failure mode


def test_reconcile_call_site_no_longer_uses_the_removed_blended_kwarg():
    """Direct reproduction of the real failure: calling reconcile() the
    way this file's call site used to (before the fix) must raise
    TypeError -- confirms the old call shape is genuinely incompatible
    with the current signature, not just cosmetically different."""
    with pytest.raises(TypeError):
        inspect.signature(broker_reconciliation.reconcile).bind(
            object(), object(), orders_enabled=True,
        )


# --- independent-audit finding, 2026-08-24: the missing intersection test --
# "LiveRunnerState metadata is preserved across the core save." Both
# `pending_signal_ttl.py`'s own module docstring and
# `_finalize_pending_signals_after_run`'s own docstring ASSERT this is
# true (`rdd.run_daily_decision()`'s internal save wipes
# `pending_signal_metadata` to `{}`, and this function's own second save
# restores it) -- but until now nothing actually exercised that claim
# against the REAL `rdd.run_daily_decision()`, only against fakes/mocks
# standing in for it.


class _CalendarFakeClient:
    """Only what a queued-signal-free run_daily_decision() call plus
    `_finalize_pending_signals_after_run`'s own `next_equity_session_date`
    call (in case anything ever does get queued) need."""

    def get_account(self):
        return SimpleNamespace(account_number="PA3HONFDTEST")

    def get_orders(self, *args, **kwargs):
        return []

    def get_all_positions(self):
        return []

    def get_calendar(self, request):
        return [SimpleNamespace(date=request.start)]


def _no_entry_frame(*dates: str) -> pd.DataFrame:
    """EMA20 < EMA50 -- guarantees no entry signal, so `queued_for_next_run`
    stays empty and this test's own `pending_signal_metadata` restoration
    is the only thing being proven, not entry/exit decision logic (same
    trick `test_missing_session_replay_cron_activation.py`'s own
    `_integration_frame` uses)."""
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [100.0] * n, "High": [101.0] * n, "Low": [99.0] * n, "Close": [100.0] * n,
            "EMA20": [95.0] * n, "EMA50": [100.0] * n, "RSI14": [50.0] * n, "RegimeAllowed": [True] * n,
        },
        index=index,
    )


def test_pending_signal_metadata_survives_the_real_run_daily_decision_core_save(tmp_path, monkeypatch):
    """The full, real round-trip: seed a `pending_signal_metadata` record
    (as `evaluate_pending_signals` would leave one THIS run), confirm the
    GENUINE `rdd.run_daily_decision()` really does wipe it to `{}` on its
    own internal save (the documented claim, now actually checked), then
    confirm `_finalize_pending_signals_after_run` -- the real, second save
    -- genuinely restores it. Never a fake standing in for either
    function under test."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"

    seeded_record = {
        "signal_id": "AAPL-QUEUED_ENTRY_SIGNAL-2026-08-19",
        "source_session_date": "2026-08-19",
        "target_execution_session_date": "2026-08-20",
        "created_at_utc": "2026-08-19T21:15:00Z",
        "status": pending_signal_ttl.STATUS_EXPIRED,
        "expire_reason": "SESSION_TTL_EXPIRED",
    }
    state = ps.LiveRunnerState(pending_signal_metadata={"AAPL|BUY": dict(seeded_record)})
    ps.save_position_state(state, state_path, guard_path=guard_path)

    fake_client = _CalendarFakeClient()
    tickers_seen: list[tuple[str, ...]] = []

    def _prepare(tickers):
        tickers_seen.append(tuple(tickers))
        return {t: _no_entry_frame("2026-08-19", "2026-08-20") for t in tickers}

    monkeypatch.setattr(carm.rdd, "prepare_live_market_data", _prepare)
    monkeypatch.setattr(carm.rdd, "get_live_cash_balance", lambda client=None: 100_000.0)
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")

    result = carm.rdd.run_daily_decision(
        state_path=state_path, decision_log_directory=tmp_path / "decisions",
        guard_path=guard_path, trading_client=fake_client,
    )
    decision = result["decision"]
    assert decision["queued_for_next_run"] == {"buys": [], "exits": []}, (
        "this test's own EMA setup must produce zero new queued signals -- "
        "otherwise it would be proving something about entry/exit logic, "
        "not about metadata preservation"
    )

    # The documented claim, now actually verified: the REAL core save
    # really does wipe pending_signal_metadata.
    after_core_save = ps.load_position_state(state_path, guard_path=guard_path)
    assert after_core_save.pending_signal_metadata == {}

    # The REAL second save restores it.
    ttl_outcome = pending_signal_ttl.PendingSignalTTLOutcome(expired_buys={"AAPL": dict(seeded_record)})
    carm._finalize_pending_signals_after_run(
        state_path, client=fake_client, decision=decision, ttl_outcome=ttl_outcome,
    )

    after_second_save = ps.load_position_state(state_path, guard_path=guard_path)
    assert after_second_save.pending_signal_metadata == {"AAPL|BUY": seeded_record}
