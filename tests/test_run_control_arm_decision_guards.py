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

import pytest

import scripts.run_control_arm_decision as carm
import src.live.broker_reconciliation as broker_reconciliation


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
