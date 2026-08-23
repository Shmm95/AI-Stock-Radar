"""CLI entry point for `src.live.equity_session_orchestrator.run_missing_session_replay`
-- "Missing-Equity-Session Remediation" design (workspace-c), the last
item of the approved scope (item 5). A thin wrapper only: all real
logic lives in `src/live/equity_session_orchestrator.py` (never
reimplemented here), the same "CLI script just parses arguments, calls
the real function, prints/notifies" discipline
`scripts/run_daily_decision.py`'s own `main()` and
`scripts/run_control_arm_decision.py` already established.

Intended to run once per day, AFTER the real daily decision cron
(`run_daily_decision.py --enable-equity-orders ...`) has already run for
today -- this script's own job is to catch up exactly one missed
session BEFORE today's next run, not to replace or race against it.
Scheduling/cron wiring is explicitly the owner's own task (not done by
this script or by Claude), per this session's own explicit scope
instruction.

Exit code is 0 for every outcome that is not itself a failure --
`CLEAN_NO_OP`, `REPLAYED_ONE_SESSION`, and `MISSED_WINDOW_HANDLED` are
all legitimate, successful outcomes of a normal run (see
`OrchestratorResult.outcome`'s own docstring). A non-zero exit code
means a fail-closed exception was raised (`SessionReplayFailClosedError`
and its subclasses) or some other unexpected error occurred -- both
cases are notified via the same Telegram channel
`run_daily_decision.py` already uses, then re-raised so the exit code
reflects the real failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scripts.run_daily_decision as rdd
from src.live.equity_session_orchestrator import (
    DEFAULT_PROVENANCE_LOG_DIRECTORY,
    OrchestratorResult,
    SessionReplayFailClosedError,
    run_missing_session_replay,
)


def _result_to_json_dict(result: OrchestratorResult) -> dict:
    """JSON-safe summary of an `OrchestratorResult` -- deliberately
    excludes `detection.raw_bars_by_ticker` (a dict of `pandas.DataFrame`,
    not JSON-serializable and not useful in a CLI summary; the real
    per-ticker bars are already durably captured in the provenance log's
    own `bar_set_sha256` and, for a replay, in the real decision log
    `run_daily_decision()` itself wrote)."""
    detection_summary = None
    if result.detection is not None:
        detection_summary = {
            "expected_sessions": list(result.detection.expected_sessions),
            "complete_sessions": list(result.detection.complete_sessions),
            "missing_by_session": {k: list(v) for k, v in result.detection.missing_by_session.items()},
            "unexpected_future_bars": {k: list(v) for k, v in result.detection.unexpected_future_bars.items()},
        }
    missed_window_summary = None
    if result.missed_window_outcome is not None:
        missed_window_summary = {
            "expired_buy_tickers": list(result.missed_window_outcome.expired_buy_tickers),
            "cleared_for_reconfirmation_exit_tickers": list(
                result.missed_window_outcome.cleared_for_reconfirmation_exit_tickers
            ),
        }
    return {
        "outcome": result.outcome,
        "replayed_session_date": result.replayed_session_date,
        "detection": detection_summary,
        "missed_window_outcome": missed_window_summary,
        "provenance_log_path": str(result.provenance_log_path) if result.provenance_log_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and, only in the narrow single-session eligible case, replay exactly one "
            "missed equity session -- never submits real orders."
        )
    )
    parser.add_argument("--state-path", type=Path, default=rdd.DEFAULT_STATE_PATH)
    parser.add_argument("--decision-log-directory", type=Path, default=rdd.DEFAULT_DECISION_LOG_DIRECTORY)
    parser.add_argument("--guard-path", type=Path, default=rdd.ps.HIGH_WATER_MARK_PATH)
    parser.add_argument("--provenance-log-directory", type=Path, default=DEFAULT_PROVENANCE_LOG_DIRECTORY)
    arguments = parser.parse_args()

    if rdd.STOP_FLAG_PATH.exists():
        text = (
            f"AI-Stock-Radar missing-session-replay runner -- STOP flag detected "
            f"({rdd.STOP_FLAG_PATH}); this run was skipped entirely, no API calls were made. "
            f"Remove the file to resume."
        )
        print(text)
        rdd._notify_safe(text)
        return

    try:
        result = run_missing_session_replay(
            state_path=arguments.state_path,
            decision_log_directory=arguments.decision_log_directory,
            guard_path=arguments.guard_path,
            provenance_log_directory=arguments.provenance_log_directory,
        )
    except Exception as error:
        # Same discipline as run_daily_decision.py's own main(): notify,
        # then re-raise unchanged -- the notification step must never
        # mask a real failure or alter the exit code. A
        # SessionReplayFailClosedError here is not a bug, it is this
        # module's own fail-closed design working as intended -- but it
        # still needs a human to see it and decide next steps, which is
        # exactly what this notification is for.
        prefix = "FAIL-CLOSED" if isinstance(error, SessionReplayFailClosedError) else "UNEXPECTED ERROR"
        text = (
            f"AI-Stock-Radar missing-session-replay runner -- {prefix}: "
            f"{type(error).__name__}: {error}"
        )
        notified = rdd._notify_safe(text)
        if not notified:
            print(
                "WARNING: Telegram failure notification could not be confirmed sent (check "
                "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID and network reachability). The original "
                f"error causing this run to fail follows: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        raise

    summary = _result_to_json_dict(result)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    if result.outcome != "CLEAN_NO_OP":
        rdd._notify_safe(
            f"AI-Stock-Radar missing-session-replay runner -- outcome: {result.outcome}"
            + (f" (session {result.replayed_session_date})" if result.replayed_session_date else "")
        )


if __name__ == "__main__":
    main()
