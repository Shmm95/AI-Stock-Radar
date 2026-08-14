"""JSON persistence for the live daily runner's portfolio state.

Dry-run research tooling only — no broker order is placed here.

Cash is intentionally NOT persisted. Every run reads the real Alpaca
paper account's cash balance live (see `src/live/account_state.py`),
so our own bookkeeping can never silently drift from the broker's
authoritative balance.

Why this persists more than "just positions": the batch engine's
`_execute_pending_buys_at_open` / `_execute_pending_exits_at_open`
only execute a queued signal once `submitted_portfolio_bar_index <
portfolio_bar_index` — i.e. a signal queued on day N is only allowed
to fill on day N+1 or later, never the same day it was generated.
A daily runner that persisted only filled positions would silently
drop every signal generated "today" (it would be queued in memory and
then discarded when the process exits), which is not a faithful
reuse of the engine's own next-available-Open execution rule. This
module therefore also persists `pending_buys`, `pending_exits`, and
the running `portfolio_bar_index` counter across runs.

Private-API coupling (documented per CLAUDE.md's "protected areas"
policy — never modified, only imported and reconstructed as-is):
- `src.backtest.portfolio_backtest_engine._MutablePosition`
- `src.backtest.portfolio_backtest_engine._PendingOrder`
Both are reconstructed directly from the persisted JSON instead of via
an independently-defined, drift-prone equivalent, so whatever the six
per-bar step functions actually read/write is exactly what gets saved
and loaded.

Real-order tracking (schema v2, equity-only): `equity_stop_orders`
maps a ticker to the Alpaca order id of its currently resting
protective stop, so it can be canceled the moment a signal-based exit
closes that position. `submitted_actions` is the idempotency ledger,
keyed per checkpoint by the real calendar date the action belongs to
(not `portfolio_bar_index` — see the high-water-mark note below for
why), checked before any real order submission so re-running the
script for the same day never re-submits an order that was already
sent. Crypto tickers never appear in either dict.

High-water-mark rollback guard (added after a real incident: an rsync
deploy overwrote the server's `data/live/` with a stale local copy,
making `portfolio_bar_index` jump backward). `load_position_state`
refuses to proceed if the loaded `portfolio_bar_index` is behind the
persisted high-water mark, raising `RollbackDetectedError` rather than
silently continuing with stale state.

Deliberately stored OUTSIDE `data/live/` (and outside the repo
entirely), at `HIGH_WATER_MARK_PATH` under the running user's home
directory: the whole point is to survive the exact failure mode that
caused the incident. If the mark lived inside `data/live/` -- even in
a sibling file next to `position_state.json`, not inside it -- the
same rsync that clobbers the live directory would clobber the mark
too, and the guard would see two mutually "consistent" stale files and
never notice anything was wrong. A location the deploy mechanism never
touches is the only way this guard can do its job. This does mean a
fresh server needs no special setup (the file is created on first
successful save, see `save_position_state`) but also that resetting it
after a *deliberate* state reset requires deleting it by hand at that
out-of-repo path -- documented in `docs/EMERGENCY_STOP_RUNBOOK.md`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.backtest.portfolio_backtest_engine import _MutablePosition, _PendingOrder
from src.backtest.portfolio_backtest_models import PortfolioSignal

# See the module docstring's "High-water-mark rollback guard" section for
# why this deliberately lives outside data/live/ and outside the repo.
HIGH_WATER_MARK_PATH = Path.home() / ".ai_stock_radar_guard" / "high_water_mark.json"


class RollbackDetectedError(RuntimeError):
    """Raised when a loaded portfolio_bar_index is behind the persisted
    high-water mark -- refuses to proceed rather than silently trading
    against stale/rolled-back state. See HIGH_WATER_MARK_PATH."""


def _read_high_water_mark(path: Path = HIGH_WATER_MARK_PATH) -> int:
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["portfolio_bar_index"])


def _write_high_water_mark(value: int, path: Path = HIGH_WATER_MARK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"portfolio_bar_index": int(value)}, indent=2) + "\n",
        encoding="utf-8",
    )

SCHEMA_VERSION = 2
DEFAULT_STATE_PATH = Path("data/live/position_state.json")


class LiveRunnerState:
    """In-memory view of what position_state.json persists."""

    __slots__ = (
        "portfolio_bar_index",
        "positions",
        "pending_buys",
        "pending_exits",
        "equity_stop_orders",
        "submitted_actions",
    )

    def __init__(
        self,
        *,
        portfolio_bar_index: int = 0,
        positions: dict[str, _MutablePosition] | None = None,
        pending_buys: dict[str, _PendingOrder] | None = None,
        pending_exits: dict[str, _PendingOrder] | None = None,
        equity_stop_orders: dict[str, str] | None = None,
        submitted_actions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.portfolio_bar_index = portfolio_bar_index
        self.positions = positions or {}
        self.pending_buys = pending_buys or {}
        self.pending_exits = pending_exits or {}
        self.equity_stop_orders = equity_stop_orders or {}
        self.submitted_actions = submitted_actions or {}


def _signal_to_dict(signal: PortfolioSignal) -> dict[str, Any]:
    return signal.to_dict()


def _signal_from_dict(payload: dict[str, Any]) -> PortfolioSignal:
    return PortfolioSignal(**payload)


def _pending_to_dict(pending: _PendingOrder) -> dict[str, Any]:
    return {
        "signal": _signal_to_dict(pending.signal),
        "submitted_portfolio_bar_index": pending.submitted_portfolio_bar_index,
    }


def _pending_from_dict(payload: dict[str, Any]) -> _PendingOrder:
    return _PendingOrder(
        signal=_signal_from_dict(payload["signal"]),
        submitted_portfolio_bar_index=payload["submitted_portfolio_bar_index"],
    )


def _position_to_dict(position: _MutablePosition) -> dict[str, Any]:
    return asdict(position)


def _position_from_dict(payload: dict[str, Any]) -> _MutablePosition:
    return _MutablePosition(**payload)


def _check_high_water_mark(loaded_bar_index: int, guard_path: Path) -> None:
    high_water_mark = _read_high_water_mark(guard_path)
    if loaded_bar_index < high_water_mark:
        raise RollbackDetectedError(
            f"Rollback detected: loaded bar_index {loaded_bar_index} < "
            f"high-water-mark {high_water_mark} -- refusing to proceed. "
            f"This normally means data/live/ (or its absence) does not "
            f"reflect real progress -- e.g. an rsync deploy overwrote it "
            f"with a stale copy. Investigate before touching "
            f"{guard_path}."
        )


def load_position_state(
    path: Path = DEFAULT_STATE_PATH, *, guard_path: Path = HIGH_WATER_MARK_PATH
) -> LiveRunnerState:
    """Load persisted state, or an empty state if no file exists yet.

    Checks the high-water-mark guard even in the "no file yet" case: a
    missing state file while the mark is already ahead of zero is its
    own kind of rollback/data-loss, not a legitimate fresh start.

    `guard_path` defaults to the real, out-of-repo location (see the
    module docstring) -- only override it in tests, so a synthetic
    `tmp_path`-based state doesn't read or write the real machine-wide
    guard file (confirmed the hard way: a tmp_path-only test tripped a
    stale mark left behind by an earlier, unrelated test run).
    """
    path = Path(path)
    if not path.is_file():
        _check_high_water_mark(0, guard_path)
        return LiveRunnerState()

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported position_state schema_version: {payload.get('schema_version')}"
        )

    _check_high_water_mark(int(payload["portfolio_bar_index"]), guard_path)

    return LiveRunnerState(
        portfolio_bar_index=int(payload["portfolio_bar_index"]),
        positions={
            ticker: _position_from_dict(fields)
            for ticker, fields in payload.get("positions", {}).items()
        },
        pending_buys={
            ticker: _pending_from_dict(fields)
            for ticker, fields in payload.get("pending_buys", {}).items()
        },
        pending_exits={
            ticker: _pending_from_dict(fields)
            for ticker, fields in payload.get("pending_exits", {}).items()
        },
        equity_stop_orders=dict(payload.get("equity_stop_orders", {})),
        submitted_actions={
            key: dict(record)
            for key, record in payload.get("submitted_actions", {}).items()
        },
    )


def save_position_state(
    state: LiveRunnerState,
    path: Path = DEFAULT_STATE_PATH,
    *,
    guard_path: Path = HIGH_WATER_MARK_PATH,
) -> None:
    """`guard_path` override is for tests only -- see `load_position_state`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "portfolio_bar_index": state.portfolio_bar_index,
        "positions": {
            ticker: _position_to_dict(position)
            for ticker, position in state.positions.items()
        },
        "pending_buys": {
            ticker: _pending_to_dict(pending)
            for ticker, pending in state.pending_buys.items()
        },
        "pending_exits": {
            ticker: _pending_to_dict(pending)
            for ticker, pending in state.pending_exits.items()
        },
        "equity_stop_orders": dict(state.equity_stop_orders),
        "submitted_actions": {
            key: dict(record) for key, record in state.submitted_actions.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Advance the rollback guard on every successful save -- never move it
    # backward, even if this run's own bar_index is (legitimately) unchanged
    # (e.g. a crypto-monitor-only save that never touches portfolio_bar_index).
    if state.portfolio_bar_index > _read_high_water_mark(guard_path):
        _write_high_water_mark(state.portfolio_bar_index, guard_path)
