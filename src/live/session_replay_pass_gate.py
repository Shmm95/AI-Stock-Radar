"""Dated PASS gate between `equity_session_orchestrator.py` and the
real 21:15 live daily-decision job -- closes the last gap workspace-c
flagged (2026-08-24) before missing-session-replay's own cron can be
activated: a bare `CLEAN_NO_OP` orchestrator outcome is not, by
itself, proof it is safe for the real order-enabled job to proceed
(`CLEAN_NO_OP` also covers "the STOP flag was set," "there is no prior
cursor yet," and "the single expected session is still missing data"
-- none of those mean "verified caught up"). See
`equity_session_orchestrator._cursor_is_current` for the actual
self-verifying safety check; this module is a small, mechanical
read/write layer only, no policy of its own.

ONE JSON FILE PER CALENDAR DAY, `PASS_<YYYY-MM-DD>.json`, atomic write
(`position_state._atomic_write_text`, reused, not reimplemented) in
`DEFAULT_PASS_GATE_DIRECTORY` (or an injected override, test-harness
seam, same discipline as every other path parameter in this
codebase's live modules).

WHO WRITES: only `equity_session_orchestrator.run_missing_session_replay`,
only after its own `_cursor_is_current` check confirms the session
cursor is genuinely caught up -- this module never decides that on its
own, it only durably records a decision already made elsewhere.

WHO READS/ENFORCES: `scripts/run_daily_decision.py`'s own `main()` (the
real 21:15 job) -- and ONLY when either real order-enabling flag is
set, mirroring this codebase's own established
`authorize_order_execution()`-gated convention (dry-run invocations,
including every test that calls `main()` without those flags, are
unaffected). `has_valid_pass_gate_for_today` itself makes no policy
judgment; the caller decides what "no valid gate" means for it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.live.position_state import _atomic_write_text

DEFAULT_PASS_GATE_DIRECTORY = Path("data/live/session_replay_pass_gate")


def _gate_path(directory: Path, *, date_iso: str) -> Path:
    return directory / f"PASS_{date_iso}.json"


def write_pass_gate(
    directory: Path = DEFAULT_PASS_GATE_DIRECTORY,
    *,
    last_processed_equity_session_date: str,
    account_number_masked: str | None,
) -> Path:
    """Durably records that, as of right now, the session cursor is
    genuinely caught up. `directory` defaults to the real path but is
    always the caller's own resolved value, not re-read from this
    module's constant -- see `equity_session_orchestrator.py`'s own
    `pass_gate_directory` parameter for the test-injection seam."""
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": today,
        "verified_at_utc": now.isoformat(),
        "last_processed_equity_session_date": last_processed_equity_session_date,
        "broker_account_masked": account_number_masked,
    }
    path = _gate_path(directory, date_iso=today)
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def has_valid_pass_gate_for_today(directory: Path = DEFAULT_PASS_GATE_DIRECTORY) -> bool:
    """True iff a real, on-disk PASS gate file exists for TODAY's real
    UTC calendar date. Read-only, makes no judgment about whether it
    SHOULD exist -- that policy lives entirely in the caller."""
    today = datetime.now(timezone.utc).date().isoformat()
    return _gate_path(directory, date_iso=today).is_file()
