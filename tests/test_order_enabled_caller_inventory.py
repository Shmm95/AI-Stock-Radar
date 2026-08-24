"""Static (AST-based) guards over WHO is allowed to call
`run_daily_decision()` with order-enabling behavior -- independent-audit
finding, 2026-08-24 (items 5a/5b of that round).

WHY A STATIC TEST, ON TOP OF THE RUNTIME GATE THAT ALREADY EXISTS:
`run_daily_decision()` itself already raises at call time if an
order-enabling flag is `True` without a valid `AuthorizedExecutionContext`
(see `src.live.authorized_execution_context`'s own module docstring, and
`run_daily_decision()`'s own `authorization` parameter docstring for "this
repo's two blessed callers"). That is a real, load-bearing runtime check --
but it only fires the moment someone actually EXECUTES the offending
script, which for a hand-run smoke-test script may not happen in routine
CI. This module re-asserts the SAME invariant at the source-code level,
so a new unauthorized call site is caught by `pytest` alone, without
anyone having to run the offending script for real (and, for the
`enable_..._orders=True` case, without a live broker connection at all).

Both scans below are intentionally conservative: they flag a call site
unless its value for the relevant keyword is the literal, unambiguous
`False` -- an attribute/name reference (e.g. `arguments.enable_equity_orders`,
whose actual runtime value this static pass cannot know) is treated as
"could be True" and must therefore live in the allow-list too. This is
why the allow-lists below name entire FILES, not just the two literal-True
call sites -- the two blessed callers pass a `Namespace` attribute, not a
literal `True`, and still need to be allow-listed for this scan to have
any signal at all.

Scope: `scripts/` and `src/` only (production code). `tests/` is
deliberately excluded -- test fixtures and fakes calling
`run_daily_decision()` with `enable_equity_orders=True` are the entire
point of those tests (proving the runtime gate itself), not a live
order-enabling code path.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SCAN_ROOTS = ("scripts", "src")

# The two blessed callers documented in `run_daily_decision()`'s own
# `authorization` parameter docstring -- the ONLY files allowed to call
# `run_daily_decision()` with a non-literal-False `enable_equity_orders`/
# `enable_crypto_orders`.
_ORDER_ENABLING_ALLOWLIST = frozenset(
    {"scripts/run_daily_decision.py", "scripts/run_control_arm_decision.py"}
)

# The two documented, justified `skip_broker_reconciliation=True` call
# sites -- see each site's own inline comment for why: run_control_arm_decision.py
# reuses its own already-identity-verified reconciliation client (avoiding
# a redundant, wrong-account-suffix second reconcile() call), and
# equity_session_orchestrator.py's replay path never submits real orders
# in the first place (skip is safe because there is nothing broker-side
# for reconciliation to protect against).
_SKIP_RECONCILIATION_ALLOWLIST = frozenset(
    {"scripts/run_control_arm_decision.py", "src/live/equity_session_orchestrator.py"}
)


def _production_python_files() -> list[Path]:
    files = []
    for root_name in _SCAN_ROOTS:
        root = PROJECT_ROOT / root_name
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _is_call_named(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == name
    if isinstance(func, ast.Attribute):
        return func.attr == name
    return False


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == keyword_name:
            return kw.value
    return None


def _is_literal_false(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is False


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _find_run_daily_decision_calls_with_non_false_keyword(keyword_name: str) -> list[tuple[str, int]]:
    """Every `run_daily_decision(...)` call site under `scripts/`/`src/`
    whose `keyword_name=` value is not the literal constant `False` --
    i.e. every call site that could possibly pass a truthy value at
    runtime, whether via a literal `True` or an opaque expression this
    static pass cannot evaluate."""
    offenders: list[tuple[str, int]] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not _is_call_named(node, "run_daily_decision"):
                continue
            value = _keyword_value(node, keyword_name)
            if value is not None and not _is_literal_false(value):
                offenders.append((_relative_path(path), node.lineno))
    return offenders


def test_only_blessed_entry_points_call_run_daily_decision_with_order_enabling_behavior():
    """Independent-audit finding 5a, 2026-08-24. This is a STRICT,
    intentionally un-relaxed test: it will fail for ANY new production
    call site that passes a possibly-true `enable_equity_orders`/
    `enable_crypto_orders` outside the two blessed entry points --
    including `scripts/smoke_test_crypto_full_cycle.py`, per the SAME
    round's item 3 finding (that script calls `run_daily_decision(
    enable_crypto_orders=True)` directly, with no `AuthorizedExecutionContext`,
    and is not yet resolved -- see that finding's own two proposed options).
    That failure is not a bug in this test: it is this test doing its
    job, and should only go away once item 3 is actually resolved (by
    rewriting the script through an approved preflight entry point, or
    retiring it), not by adding it to the allow-list below."""
    for keyword_name in ("enable_equity_orders", "enable_crypto_orders"):
        offenders = _find_run_daily_decision_calls_with_non_false_keyword(keyword_name)
        unauthorized = [(f, line) for f, line in offenders if f not in _ORDER_ENABLING_ALLOWLIST]
        assert unauthorized == [], (
            f"run_daily_decision({keyword_name}=...) called with a possibly-true value "
            f"outside the two blessed entry points ({sorted(_ORDER_ENABLING_ALLOWLIST)}): "
            f"{unauthorized}. See this test's own docstring before adding a new file to "
            f"the allow-list -- that is a real authorization-scope decision, not a test fix."
        )


def test_skip_broker_reconciliation_true_is_limited_to_the_documented_allowlist():
    """Independent-audit finding 5b, 2026-08-24. `skip_broker_reconciliation=True`
    disables `run_daily_decision()`'s own internal broker-truth check --
    every production call site that does so must have an equally strong
    reconciliation guarantee from elsewhere in its own call chain (see
    each allow-listed site's own inline comment). A new call site
    passing this without landing in that documented allow-list is a
    silent way to skip the one check that catches broker/local-state
    drift, so this test fails closed on anything unrecognized."""
    offenders = _find_run_daily_decision_calls_with_non_false_keyword("skip_broker_reconciliation")
    unauthorized = [(f, line) for f, line in offenders if f not in _SKIP_RECONCILIATION_ALLOWLIST]
    assert unauthorized == [], (
        f"run_daily_decision(skip_broker_reconciliation=...) called with a possibly-true "
        f"value outside the documented allow-list ({sorted(_SKIP_RECONCILIATION_ALLOWLIST)}): "
        f"{unauthorized}. A new call site here must carry the same kind of documented, "
        f"equally-strong reconciliation guarantee the two existing sites do -- not just a "
        f"test-suite exemption."
    )


def test_the_order_enabling_allowlist_files_actually_exist():
    """Guards the allow-list itself against drift (a renamed/moved file
    would otherwise silently make the scan above vacuously pass)."""
    for relative in _ORDER_ENABLING_ALLOWLIST | _SKIP_RECONCILIATION_ALLOWLIST:
        assert (PROJECT_ROOT / relative).is_file(), f"allow-listed path no longer exists: {relative}"
