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

from types import SimpleNamespace

import pytest

import scripts.run_control_arm_decision as carm


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
