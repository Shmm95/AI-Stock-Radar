"""One-shot crypto stop-loss check.

Run by hand, once per invocation:

    .venv/bin/python scripts/run_crypto_stop_monitor.py

This does NOT loop or schedule itself — see src/live/crypto_stop_monitor.py
for why, and for the polling-interval recommendation this script's
report accompanies. Wiring this up to run periodically (cron, a
scheduler) is a separate, not-yet-done deployment task.

Submits real Alpaca PAPER market sell orders for any open crypto
position whose latest trade price has reached or breached its
recorded `stop_loss_price`. Never touches equities.

Notifications are deliberately silent on a routine, nothing-triggered
check: at a 5-minute cadence, a message every run would be spam, and
would drown out the runs that actually matter. A Telegram message is
sent only when there is something the owner could not have predicted
from "the job ran on schedule" alone — a stop that actually fired
(`SELL`), or a `NEEDS_REVIEW` entry (a real anomaly, e.g. the broker
reporting no sellable quantity or the lock being held). Both are a
judgment call beyond this task's literal "stop tetiklendi, satış
gönderildi" wording, extended here because staying silent on a
NEEDS_REVIEW would hide exactly the kind of problem this notification
exists to surface; see the accompanying report if that scope
expansion should be narrowed back.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live import order_submission
from src.live.crypto_stop_monitor import (
    DEFAULT_LOCK_PATH,
    MonitorAlreadyRunningError,
    _monitor_lock,
    check_and_execute_crypto_stops,
)
from src.live.position_state import DEFAULT_STATE_PATH, load_position_state, save_position_state
from src.notify.telegram_notifier import send_telegram_message

_NOTIFY_WORTHY_ACTIONS = {"SELL", "NEEDS_REVIEW"}

# Emergency full-stop switch: an empty file at this path (content is never
# read, only existence checked) makes this script exit before ANY API call
# -- no Alpaca client, no price fetch, no lock. This is deliberately the
# only thing checked before anything else runs. See
# docs/EMERGENCY_STOP_RUNBOOK.md for exactly what to type in a crisis.
# NOT gated on FREEZE: that flag only suppresses new-position-opening in
# run_daily_decision.py -- an open crypto position's stop-loss watch must
# keep running through a freeze, or the position would sit unprotected.
STOP_FLAG_PATH = Path("data/live/STOP")


def _notify_safe(text: str) -> None:
    try:
        send_telegram_message(text)
    except Exception:
        pass


def _build_notification_text(actions: list[dict]) -> str | None:
    """Return notification text, or None if nothing notify-worthy happened."""
    notable = [a for a in actions if a.get("action") in _NOTIFY_WORTHY_ACTIONS]
    if not notable:
        return None
    lines = [f"AI-Stock-Radar crypto stop monitor — {datetime.now(UTC).isoformat()}"]
    for action in notable:
        if action["action"] == "SELL":
            lines.append(
                f"STOP TRIGGERED: sold {action.get('ticker')} — "
                f"order {action.get('order_id')}, status {action.get('status')}"
            )
        else:
            lines.append(f"NEEDS REVIEW: {action.get('ticker')} — {action.get('issue')}")
    return "\n".join(lines)


def main() -> int:
    if STOP_FLAG_PATH.exists():
        text = (
            f"AI-Stock-Radar crypto stop monitor -- STOP flag detected "
            f"({STOP_FLAG_PATH}); this run was skipped entirely, no API "
            f"calls were made. Remove the file to resume monitoring."
        )
        print(text)
        _notify_safe(text)
        return 0

    check_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        with _monitor_lock(DEFAULT_LOCK_PATH):
            runner_state = load_position_state(DEFAULT_STATE_PATH)
            client = order_submission.get_trading_client()
            actions = check_and_execute_crypto_stops(client, runner_state, check_id=check_id)
            save_position_state(runner_state, DEFAULT_STATE_PATH)
    except MonitorAlreadyRunningError as error:
        print(f"SKIPPED: {error}")
        return 0

    print(json.dumps({"check_id": check_id, "actions": actions}, indent=2, sort_keys=True, default=str))

    notification_text = _build_notification_text(actions)
    if notification_text is not None:
        _notify_safe(notification_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
