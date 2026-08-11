"""Fail-safe Telegram notifications for the live daily/monitoring scripts.

Credentials are read from `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
environment variables only — same pattern as `ALPACA_API_KEY` /
`ALPACA_SECRET_KEY` in `src/data/alpaca_market_data.py`. Never
hardcoded, never logged, never printed.

Critical rule this module exists to guarantee: a notification failure
must NEVER affect the caller's trading flow. `send_telegram_message`
catches everything itself — a missing token, a network error, a bad
response, anything — and returns `False` rather than raising.
Notification is a best-effort side effect that happens *after*
trading logic has already done its job, never a dependency trading
logic waits on or is gated by.

Deliberately never logs `str(exception)` on failure: Telegram's API
URL embeds the bot token (`.../bot<TOKEN>/sendMessage`), and
`requests`' own `HTTPError` message includes the request URL — logging
that string would leak the token into logs. Only the exception's class
name is logged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
# override=True: python-dotenv's own default (override=False) skips a key
# that is ALREADY present in os.environ, even if its value is an empty
# string -- e.g. a stale `TELEGRAM_BOT_TOKEN=` left in a crontab's own
# env-var block, a systemd EnvironmentFile, or anything else the process
# inherits before Python starts. Confirmed on the real server: a manual
# `python -c` test (no such inherited var) sent successfully, while
# send_status_update.py returned False with no error -- .env is this
# project's authoritative credential source (see alpaca_market_data.py's
# module docstring), so it must always win over a blank inherited value.
load_dotenv(ENV_PATH, override=True)

TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 10


def send_telegram_message(text: str) -> bool:
    """Best-effort send. Returns True on confirmed delivery, False otherwise.

    Never raises. Missing `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` is a
    silent no-op (notifications are optional; the owner may not have
    configured them yet), not an error.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info(
            "Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID missing); "
            "skipping notification."
        )
        return False

    try:
        response = requests.post(
            f"{TELEGRAM_API_BASE}/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        return bool(payload.get("ok"))
    except Exception as error:  # noqa: BLE001 - deliberately blanket, see module docstring
        logger.warning(
            "Telegram notification failed (%s); trading flow is unaffected.",
            type(error).__name__,
        )
        return False
