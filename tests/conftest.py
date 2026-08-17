"""Repo-wide pytest safety net: no test may ever perform a real network
call, regardless of what credentials happen to be sitting in the process
environment or loaded by a test.

WHY THIS EXISTS (real, not hypothetical): `src/notify/telegram_notifier.py`
calls `requests.post(...)` directly (`import requests`, not `from requests
import post`, so patching `requests.post` at the module level -- not
`telegram_notifier.send_telegram_message` -- is the one interception point
every call path funnels through regardless of how many layers of
`from telegram_notifier import send_telegram_message` re-exporting sit on
top of it; confirmed by reading that file directly). Every existing
Telegram-touching test in this suite (`test_telegram_notifier.py`,
`test_daily_decision_notifications.py`, etc.) already, individually,
monkeypatches this correctly -- this fixture does not change their
behavior. What it closes is the gap for any test that does NOT think to
do so itself: this session's own ad hoc (non-pytest, Bash-run) regression
scripts loaded `.env.control`'s REAL Telegram credentials
(`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`, confirmed 46/10 characters
respectively) via `_load_env_file_fail_closed`, and real
`send_telegram_message()` calls were made against the real Telegram API
during that testing (confirmed by real `HTTPError` responses observed in
that session's own tool output). A future *pytest* test that loads real
`.env`/`.env.control` credentials (directly, or transitively by importing
`scripts.run_control_arm_decision` and calling `_load_env_file_fail_closed`)
and then exercises a code path that calls `send_telegram_message` would
have the exact same real-network exposure without this fixture -- this
autouse fixture makes that structurally impossible in the pytest suite,
not just conventionally avoided.

If a future test genuinely needs to exercise a real HTTP round trip
against something that is NOT Telegram (e.g. a real localhost test
server), it can locally re-monkeypatch `requests.post` back within its
own test body -- a test-scoped `monkeypatch.setattr` always wins over
this session-level default for the duration of that one test.
"""

from __future__ import annotations

import pytest
import requests


class RealNetworkCallBlocked(RuntimeError):
    """Raised instead of ever letting `requests.post` reach a real socket
    during a pytest run -- see this module's own docstring."""


# Every blocked attempt is recorded here (url only, never headers/body,
# so a blocked call can never leak a credential into a log either) --
# lets a test distinguish "returned False because the guard blocked a
# real call" from "returned False for some unrelated reason (e.g. no
# credentials configured)" by checking whether this actually grew,
# rather than trusting a return value alone. Cleared at the start of
# every test by the fixture below, so counts never leak across tests.
BLOCKED_CALL_LOG: list[str] = []


def _blocked_post(*args, **kwargs):
    url = args[0] if args else kwargs.get("url", "<unknown>")
    host = url.split("/")[2] if "://" in str(url) else str(url)
    BLOCKED_CALL_LOG.append(host)
    raise RealNetworkCallBlocked(
        f"BLOCKED a real requests.post() call during the pytest suite "
        f"(target host: {host}). No test may perform real network I/O by "
        f"default -- see tests/conftest.py's own module docstring. If "
        f"this test genuinely needs a real HTTP call (e.g. to a real "
        f"local test server, never Telegram/Alpaca), monkeypatch "
        f"requests.post again within the test itself to override this "
        f"default."
    )


@pytest.fixture(autouse=True)
def block_real_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    BLOCKED_CALL_LOG.clear()
    monkeypatch.setattr(requests, "post", _blocked_post)
    # Defense in depth: never let a real credential already sitting in
    # the shell environment (or loaded by an EARLIER, unrelated import in
    # the same process) silently flow into a test. Does not protect a
    # test that explicitly re-loads a real .env file mid-test -- the
    # requests.post block above is what makes that harmless regardless.
    for _var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(_var, raising=False)
