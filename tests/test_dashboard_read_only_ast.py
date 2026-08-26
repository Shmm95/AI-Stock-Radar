"""AST-based static guards for the dashboard subsystem (owner's own
spec, 2026-08-25): every file under `src/dashboard/` plus the producer
script `scripts/generate_dashboard_snapshot.py` must never import or
reference any live-trading-engine write path, any mutation function,
or any generic code-execution/network-write primitive.

No real `.env` is loaded and no network call is made anywhere in this
file -- every check here is pure source-text/AST inspection.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PACKAGE_FILES = sorted((PROJECT_ROOT / "src" / "dashboard").glob("*.py"))
PRODUCER_SCRIPT = PROJECT_ROOT / "scripts" / "generate_dashboard_snapshot.py"
HTTP_APP_FILE = PROJECT_ROOT / "src" / "dashboard" / "http_app.py"

ALL_SCANNED_FILES = [*DASHBOARD_PACKAGE_FILES, PRODUCER_SCRIPT]

# Every banned module (never imported, in any form: `import X`,
# `import X as Y`, `from X import ...`, `from pkg import X`).
BANNED_MODULES = {
    "src.live.order_intent",
    "order_intent",
    "src.live.order_intent_reconciliation",
    "order_intent_reconciliation",
    "src.live.order_submission",
    "order_submission",
    "src.live.crypto_stop_monitor",
    "crypto_stop_monitor",
    "run_daily_decision",
    "scripts.run_daily_decision",
    "src.live.equity_session_orchestrator",
    "equity_session_orchestrator",
    "src.live.missed_session_window_handling",
    "missed_session_window_handling",
    "src.live.session_replay_journal",
    "session_replay_journal",
    "subprocess",
}

# Every banned NAME -- as a bare identifier (`ast.Name`), an attribute
# access (`ast.Attribute.attr`), or an import binding -- regardless of
# which module it's reached through. Order-submission request types,
# mutation functions, and generic code-exec/dynamic-import primitives.
BANNED_IDENTIFIERS = {
    "submit_order",
    "cancel_order_by_id",
    "cancel_orders",
    "replace_order_by_id",
    "close_position",
    "close_all_positions",
    "MarketOrderRequest",
    "LimitOrderRequest",
    "StopOrderRequest",
    "ReplaceOrderRequest",
    "create_session_intent",
    "transition_session_intent",
    "create_intent",
    "transition_intent",
    "resolve_stray_intent",
    "run_missing_session_replay",
    "save_position_state",
    "write_pass_gate",
    "handle_missed_execution_window",
    "eval",
    "exec",
    "import_module",  # importlib.import_module -- dynamic import
    "__import__",
}

# requests.post/put/patch/delete -- checked as an Attribute access on
# anything named `requests` (never imported here anyway, checked as
# defense in depth) or as a bare identifier if `from requests import
# post` style were ever used.
BANNED_REQUESTS_METHODS = {"post", "put", "patch", "delete"}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(alias.name)
    return names


def _referenced_identifiers(tree: ast.Module) -> set[str]:
    """Every bare name, attribute-access suffix, and import binding
    anywhere in the file -- a deliberately broad net (a banned name
    used as a local variable would also be flagged, which is fine: a
    file that legitimately needs a symbol with one of these exact names
    for something unrelated should rename it, not weaken this test)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[-1])
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
                names.add(alias.name)
    return names


def _requests_write_calls(tree: ast.Module) -> list[str]:
    """Any call shaped like `requests.post(...)` / `requests.put(...)`
    / etc, or a bare `post(...)`/`put(...)`/... after a `from requests
    import post` style import -- either shape is flagged."""
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in BANNED_REQUESTS_METHODS:
            offenders.append(func.attr)
        elif isinstance(func, ast.Name) and func.id in BANNED_REQUESTS_METHODS:
            offenders.append(func.id)
    return offenders


@pytest.mark.parametrize("path", ALL_SCANNED_FILES, ids=lambda p: p.name)
def test_no_banned_module_imported(path: Path):
    tree = _parse(path)
    imported = _imported_module_names(tree)
    offending = imported & BANNED_MODULES
    assert not offending, f"{path.name} imports banned module(s): {offending}"


@pytest.mark.parametrize("path", ALL_SCANNED_FILES, ids=lambda p: p.name)
def test_no_banned_identifier_referenced(path: Path):
    tree = _parse(path)
    referenced = _referenced_identifiers(tree)
    offending = referenced & BANNED_IDENTIFIERS
    assert not offending, f"{path.name} references banned identifier(s): {offending}"


@pytest.mark.parametrize("path", ALL_SCANNED_FILES, ids=lambda p: p.name)
def test_no_requests_write_method_called(path: Path):
    tree = _parse(path)
    offenders = _requests_write_calls(tree)
    assert not offenders, f"{path.name} calls a write-style requests method: {offenders}"


@pytest.mark.parametrize("path", ALL_SCANNED_FILES, ids=lambda p: p.name)
def test_no_os_system_or_dynamic_code_execution(path: Path):
    tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "system":
                assert False, f"{path.name} calls os.system(...)"
            if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
                assert False, f"{path.name} calls {func.id}(...)"


def test_http_app_never_imports_alpaca_or_a_trading_client():
    """Stricter, HTTP-specific rule (owner's own explicit instruction):
    the HTTP package must never import Alpaca at all, not even
    read-only client classes -- it only ever reads the already-sanitized
    snapshot file the producer wrote."""
    tree = _parse(HTTP_APP_FILE)
    imported = _imported_module_names(tree)
    alpaca_imports = {name for name in imported if name == "alpaca" or name.startswith("alpaca.")}
    assert not alpaca_imports, f"http_app.py imports Alpaca module(s): {alpaca_imports}"
    assert "TradingClient" not in _referenced_identifiers(tree)


def test_http_app_never_imports_snapshot_builder_or_order_intent_family():
    """The HTTP process is a separate system user from the producer --
    it must never be ABLE to build a snapshot or touch broker/journal
    state itself, only read the file the producer already wrote."""
    tree = _parse(HTTP_APP_FILE)
    imported = _imported_module_names(tree)
    assert "snapshot_builder" not in imported
    assert "src.dashboard.snapshot_builder" not in imported


class _StrictReadOnlyFakeClient:
    """Exposes ONLY get_account/get_all_positions/get_orders -- any
    other attribute access raises immediately. Used to prove, at
    runtime (not just via static AST inspection), that
    `snapshot_builder.build_snapshot` never touches anything else on
    the client object it's given."""

    _ALLOWED = frozenset({"get_account", "get_all_positions", "get_orders"})

    def __init__(self, *, account, positions, orders):
        self._account = account
        self._positions = positions
        self._orders = orders

    def __getattr__(self, name):
        if name in self._ALLOWED:
            raise AttributeError(f"'{name}' should be handled by an explicit method below, not __getattr__")
        raise AssertionError(
            f"snapshot_builder attempted to access client.{name} -- not in the allowed read-only "
            f"surface {sorted(self._ALLOWED)}"
        )

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def get_orders(self, *, filter=None):
        return list(self._orders)


def test_snapshot_builder_never_touches_a_client_method_outside_the_read_only_surface(tmp_path, monkeypatch):
    """Runtime proof, not just AST: build a full snapshot against a
    client that raises on any method call outside get_account/
    get_all_positions/get_orders, and confirm it completes without
    tripping that trap."""
    from src.dashboard import snapshot_builder as sb

    class _FakeAccount:
        cash = "1000.00"
        equity = "1000.00"
        last_equity = "1000.00"
        buying_power = "2000.00"
        account_number = "PA000TEST"
        status = "ACTIVE"
        trading_blocked = False
        transfers_blocked = False
        account_blocked = False

    monkeypatch.setattr(sb, "_build_trading_client", lambda: _StrictReadOnlyFakeClient(
        account=_FakeAccount(), positions=[], orders=[],
    ))

    snapshot = sb.build_snapshot(
        state_path=tmp_path / "position_state.json",
        guard_path=tmp_path / "guard.json",
        decision_log_directory=tmp_path / "decisions",
        pass_gate_directory=tmp_path / "pass_gate",
    )
    assert snapshot.account is not None
