"""Best-effort Healthchecks.io dead-man's-switch pings for the control arm.

TWO INDEPENDENT CHECKS (per the approved two-check spec): LIVENESS answers
"did this cron invocation run at all, and did it not crash uncaught" --
pinged success/fail purely on whether the runner raised, independent of
what it actually decided to do. OPERATIONAL STATE answers "was today's
outcome the intended one" -- pinged success/fail based on a domain-level
outcome classification (STOP active, FREEZE active, a lock conflict, a
Telegram delivery failure, an expected market-closed no-op, or a genuine
successful run). See `scripts/run_control_arm_decision.py`'s own
`_execute()`/`main()` for the full outcome-to-ping mapping -- this module
only provides the mechanical `ping_healthcheck()` primitive, it makes no
domain decisions itself.

CRITICAL RULE: a Healthchecks ping failure or misconfiguration must NEVER
affect, mask, or delay the underlying runner's own exception/exit code --
the exact same "notification is best-effort, never load-bearing"
discipline as `src/notify/telegram_notifier.py`. `ping_healthcheck` never
raises for a network/response failure; it returns `False`.

URL/UUID NEVER LOGGED: a Healthchecks.io check URL embeds a bearer-token-
like UUID (anyone holding the URL can ping or spoof the check) -- the same
secrecy class as Telegram's bot-token-embedded API URL in
`telegram_notifier.py`. This module never prints, logs, or includes the
URL itself in any message; only the event name, attempt number, and
response/exception TYPE (never its string, which for some HTTP client
exceptions embeds the request URL) are ever logged.

POST BODY: only a short, non-sensitive status-code string is ever sent
(e.g. `"RUN_OK"`, `"STOP_ACTIVE"`, `"TELEGRAM_DELIVERY_FAILED"`) -- never
the full decision, never any account/position detail.

RESPONSE VALIDATION: `curl -f` (a common Healthchecks-ping shell pattern)
only checks the HTTP status code, which is NOT sufficient on its own --
Healthchecks.io's own documented contract is a `200` status AND a literal
`"OK"` response body; a proxy, captive portal, or misconfigured endpoint
can return `200` with an unrelated body. This module checks both.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0

LIVENESS_URL_ENV_VAR = "HEALTHCHECKS_LIVENESS_URL"
OPERATIONAL_URL_ENV_VAR = "HEALTHCHECKS_OPERATIONAL_URL"

_VALID_EVENTS = ("start", "success", "fail")


def load_healthchecks_env_file_fail_closed(env_file: str) -> None:
    """Same fail-closed loading discipline as
    `run_control_arm_decision._load_env_file_fail_closed`: the path must
    resolve to a real file (`resolve(strict=True)`, raises
    `FileNotFoundError` otherwise -- never a silent no-op), and
    `load_dotenv()` reporting nothing loaded raises `RuntimeError` rather
    than silently continuing. A SEPARATE function from `--env-file`'s own
    loader (not reused) because the two have different CALLER contracts:
    `--env-file`'s credentials must always be correct when the flag is
    given (a bad path there is a hard stop for the whole run); this
    Healthchecks config is optional best-effort infrastructure, and it is
    the CALLER's job (see `run_control_arm_decision._load_healthchecks_config`)
    to catch this function's own exceptions and downgrade them to
    "Healthchecks disabled this run" rather than blocking the trading
    pipeline -- this function itself stays strict/fail-closed so that
    downgrade decision is explicit and auditable at the call site, not
    silently baked in here."""
    resolved = Path(env_file).resolve(strict=True)
    loaded = load_dotenv(resolved, override=True)
    if not loaded:
        raise RuntimeError(
            f"load_dotenv() reported nothing loaded from {resolved} -- "
            f"refusing to silently continue. Fail-closed by design; see "
            f"this module's docstring."
        )
    print(f"[CONTROL] Healthchecks config override applied from: {resolved}")


def ping_healthcheck(base_url: str | None, event: str, detail: str) -> bool:
    """`event`: `"start"` | `"success"` | `"fail"`. URL is `base_url +
    "/start"`, `base_url` itself, or `base_url + "/fail"` respectively
    (Healthchecks.io's own URL convention). `detail` is POSTed as the
    request body -- keep it a short, non-sensitive status code (see
    module docstring).

    Returns `True` only on a CONFIRMED `200` status AND a response body
    that is exactly `"OK"` after stripping whitespace. Returns `False`
    for a missing/empty `base_url` (Healthchecks not configured -- a
    normal, expected case, not logged as an error), any network/timeout
    exception, or a response that doesn't meet both conditions above --
    retried up to `MAX_ATTEMPTS` times with a short backoff between
    attempts before giving up. NEVER raises for a delivery failure (a
    programming error, e.g. an unrecognized `event`, still raises
    `ValueError` -- that is a caller bug, not a delivery failure, and
    should not be swallowed the same way)."""
    if event not in _VALID_EVENTS:
        raise ValueError(f"Unknown healthcheck event: {event!r} (expected one of {_VALID_EVENTS})")
    if not base_url:
        return False

    if event == "start":
        url = base_url.rstrip("/") + "/start"
    elif event == "fail":
        url = base_url.rstrip("/") + "/fail"
    else:
        url = base_url

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, data=detail.encode("utf-8"), timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200 and response.text.strip() == "OK":
                return True
            logger.warning(
                "Healthchecks ping (%s) attempt %d/%d: unexpected response "
                "(status=%s) -- treated as failure. URL never logged.",
                event, attempt, MAX_ATTEMPTS, response.status_code,
            )
        except Exception as error:  # noqa: BLE001 - deliberately blanket, see module docstring
            logger.warning(
                "Healthchecks ping (%s) attempt %d/%d failed (%s); URL never logged.",
                event, attempt, MAX_ATTEMPTS, type(error).__name__,
            )
        if attempt < MAX_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_SECONDS)
    return False
