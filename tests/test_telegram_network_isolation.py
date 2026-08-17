"""Proves the `tests/conftest.py` autouse network guard actually blocks a
real Telegram delivery attempt -- including the exact real-world scenario
that happened during this session's own ad hoc (non-pytest) testing:
`.env.control`'s REAL `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` loaded, then
`send_telegram_message()` called, completely unmodified, with no
additional per-test monkeypatching beyond the autouse fixture.

Deliberately does NOT re-monkeypatch `requests.post` anywhere in this
file -- the whole point is proving the conftest-level default alone is
sufficient, not that a test author remembered to add their own guard.

Distinguishes "returned False because the guard blocked a real call"
from "returned False for some unrelated reason" (e.g. missing
credentials, which `send_telegram_message` also silently returns False
for) by checking `conftest.BLOCKED_CALL_LOG`, not just the return value
-- a return value of False alone would NOT be sufficient proof that a
real network call was actually attempted-and-blocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

import src.notify.telegram_notifier as telegram_notifier

_ENV_CONTROL_PATH = Path(__file__).resolve().parents[1] / ".env.control"

# NOTE: deliberately NOT `from tests.conftest import BLOCKED_CALL_LOG`.
# Without a tests/__init__.py, pytest's default rootless import mode
# loads conftest.py as a bare top-level module `conftest` (for fixture
# registration) while an explicit `from tests.conftest import ...` here
# resolves it as the PACKAGE-QUALIFIED `tests.conftest` -- two distinct
# module objects, each with its OWN `BLOCKED_CALL_LOG` list (confirmed
# empirically: importing it that way and asserting on it after a real
# blocked call showed the imported list staying empty even though the
# fixture's own copy demonstrably grew). `caplog` sidesteps this import-
# identity ambiguity entirely by reading the actual log record the
# blocked call emits, regardless of which module object produced it.


def test_requests_post_is_replaced_by_the_autouse_guard():
    """The most direct possible check: `requests.post` is not the real
    library function during a pytest run -- it is this suite's own
    blocking stand-in, for every test, without that test asking for it."""
    assert requests.post.__name__ == "_blocked_post"


@pytest.mark.skipif(not _ENV_CONTROL_PATH.is_file(), reason=".env.control not present in this checkout")
def test_real_env_control_credentials_plus_unmodified_send_function_makes_zero_real_calls(caplog):
    """Loads the REAL `.env.control` file (the exact credentials this
    session's own ad hoc regression testing used, and which really did
    reach api.telegram.org during that testing) and calls the REAL,
    completely unmodified `send_telegram_message` -- relying ONLY on the
    autouse conftest fixture, no test-local monkeypatch. Proves the
    guard itself, not a well-behaved test author, is what keeps this
    safe."""
    from dotenv import load_dotenv

    loaded = load_dotenv(_ENV_CONTROL_PATH, override=True)
    assert loaded, "expected .env.control to actually load something"

    import os
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    assert token and chat_id, "expected real-looking credentials to now be present"
    assert len(token) == 46 and len(chat_id) == 10, "expected the real .env.control shape (46/10 chars)"

    with caplog.at_level("WARNING", logger="src.notify.telegram_notifier"):
        result = telegram_notifier.send_telegram_message(
            "THIS MUST NEVER ACTUALLY BE SENT -- tests/test_telegram_network_isolation.py"
        )

    assert result is False
    # The decisive check: `send_telegram_message`'s own except-block logged
    # a warning naming RealNetworkCallBlocked as the caught exception type
    # -- proving requests.post really was invoked with these real
    # credentials and was intercepted by the guard, not that this
    # returned False for some unrelated, harmless reason (e.g. missing
    # credentials -- see the contrast test below, which shows NO such
    # warning is logged in that case).
    assert any("RealNetworkCallBlocked" in record.message for record in caplog.records)


def test_missing_credentials_is_a_different_false_no_call_even_attempted(caplog):
    """Contrast case: when credentials are genuinely absent,
    `send_telegram_message` returns False WITHOUT ever attempting
    `requests.post` at all -- no warning is logged. Distinguishes this
    from the guard-blocked case above, where a real attempt WAS made and
    intercepted (and DID log a warning)."""
    import os

    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("TELEGRAM_CHAT_ID", None)

    with caplog.at_level("WARNING", logger="src.notify.telegram_notifier"):
        result = telegram_notifier.send_telegram_message("should short-circuit before any network attempt")

    assert result is False
    assert caplog.records == []
