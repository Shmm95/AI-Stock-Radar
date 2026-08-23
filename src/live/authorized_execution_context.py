"""Closes an independent audit finding (2026-08-22, "Madde E" -- "Doğrudan
runner bypass'ı hâlâ açık"): scripts/run_daily_decision.py's real-order
functions (`_execute_equity_orders`, `_execute_crypto_orders`,
`_reconcile_pending_equity_orders`, and the shared `_resolve_or_submit_order`
choke point they all funnel through) could be reached by a BARE call to
`run_daily_decision(enable_equity_orders=True)` that never went through
either of this codebase's two blessed preflight sequences --
`run_daily_decision.py`'s own `main()` (STOP/FREEZE checks, then that
function's own internal `_require_live_account_suffix()`/`reconcile()`) or
`scripts/run_control_arm_decision.py`'s `_execute()` (identity,
reconciliation, pending-signal TTL, market-session verification, and the
write-ahead intent journal, in that order). This is exactly the same class
of bug this codebase has hit before -- see `run_control_arm_decision.py`'s
own module docstring, "found in an external audit of this session's
earlier scratchpad dry-run script, which called run_daily_decision()
directly" -- just for the real-order path instead of the STOP/FREEZE path.

DESIGN, DELIBERATELY SCOPED (owner's own explicit decision, 2026-08-22):
this closes the BARE-CALL bypass -- something that skips BOTH `main()` and
the control-arm wrapper's `_execute()` -- not "any direct CLI use of
--enable-equity-orders." `run_daily_decision.py`'s own `main()` is treated
as an existing, already-adequate blessed entry point (it is the real
live system's own production cron path, with no separate wrapper of its
own) and self-authorizes right before calling `run_daily_decision()`,
AFTER its own STOP/FREEZE preflight -- the live cron's real,
`--enable-equity-orders` production usage is UNCHANGED by this module,
by deliberate choice. `run_control_arm_decision.py`'s `_execute()`
self-authorizes only after its own full preflight chain (identity,
reconciliation, TTL, session, write-ahead evidence) has already passed --
see that file's own `_order_intent_hook_for_run`-adjacent call site for
where.

NOT A SECURITY BOUNDARY: a plain Python object cannot stop a determined,
source-reading, root-access operator from calling `authorize_order_execution()`
themselves -- nothing in the language enforces that. What this DOES
prevent is the accidental/structural bypass class described above: a new
scratchpad script, a test missing proper setup, or a future refactor that
calls `run_daily_decision(enable_equity_orders=True)` without going
through either blessed entry point now gets an immediate, loud
`RuntimeError` instead of silently reaching a real broker call.
"""

from __future__ import annotations

_AUTHORIZATION_SENTINEL = object()


class AuthorizedExecutionContext:
    """Opaque authorization token. Constructible ONLY via
    `authorize_order_execution()` below -- directly instantiating this
    class with anything other than the module-private sentinel raises.
    `None`, `object()`, `True`, an env-var string, or a file-marker path
    can never satisfy the `isinstance` check `require_authorized_execution_context`
    performs; only a real instance of this exact class does."""

    __slots__ = ("_token",)

    def __init__(self, _token: object) -> None:
        if _token is not _AUTHORIZATION_SENTINEL:
            raise RuntimeError(
                "AuthorizedExecutionContext must never be constructed directly -- "
                "use authorize_order_execution() instead, called only after your "
                "own full preflight sequence has already passed."
            )
        self._token = _token


def authorize_order_execution() -> AuthorizedExecutionContext:
    """The one legitimate factory. Callers must call this ONLY after
    their own preflight guards have already passed -- see this module's
    own docstring for exactly which guards `main()` and
    `run_control_arm_decision.py`'s `_execute()` each require first."""
    return AuthorizedExecutionContext(_AUTHORIZATION_SENTINEL)


def require_authorized_execution_context(context: object, *, action_description: str) -> None:
    """Fail-closed: raises `RuntimeError` unless `context` is a real
    `AuthorizedExecutionContext` instance. Called at every real-order
    entry point in `scripts/run_daily_decision.py` -- both the top-level
    `run_daily_decision()` function and, separately, the inner
    `_resolve_or_submit_order` choke point (and its two bespoke
    non-choke-point call sites: the crypto signal-exit submission, and
    the equity CANCEL_STOP path) -- so a caller that reaches any of
    those directly, bypassing the top-level check, is still refused
    before any broker call.

    `action_description` is purely for the error message -- no
    behavior branches on it."""
    if not isinstance(context, AuthorizedExecutionContext):
        raise RuntimeError(
            f"{action_description} requires a valid AuthorizedExecutionContext, "
            f"obtained only via authorize_order_execution() AFTER a full "
            f"preflight sequence has already passed (see this module's own "
            f"docstring for run_daily_decision.py's main() and "
            f"run_control_arm_decision.py's _execute()). Got "
            f"{type(context).__name__}={context!r} -- refusing to proceed. "
            f"No broker or state call has been made."
        )
