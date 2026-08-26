"""Dashboard V1 snapshot producer -- one-shot CLI, invoked by
`deploy/systemd/ai-stock-radar-dashboard-snapshot.timer` (every 5
minutes; the timer owns the schedule, this script never sleeps or
loops itself).

Runs under the privileged `ai-dashboard-producer` system user: real
Alpaca credentials, read-only access to `data/live/` and the `.guard`
directory. Writes ONE file, atomically (temp file in the same
directory, fsync'd, then `os.replace`'d into place, directory itself
also fsync'd -- reuses `position_state._atomic_write_text`, the same
durability guarantee `order_intent.py`'s own journal already relies
on, never reimplemented here).

FAIL-CLOSED ON A CORE FAILURE: if the snapshot builder's own
`collection_status` comes back `ERROR` (the P&L/account section itself
could not be read at all -- see `snapshot_builder.build_snapshot`'s own
docstring for exactly when that happens), this script does NOT write
anything -- the previous, still-good snapshot file is left completely
untouched, and this process exits non-zero so systemd/cron surfaces the
failure. A `PARTIAL` result (the core succeeded, some secondary section
did not) IS written -- partial-but-honest data (with its own `errors[]`
list) is more useful to the owner than no update at all, and the
frontend's own staleness indicator is what protects against a truly
broken producer going silent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard import snapshot_builder  # noqa: E402
from src.dashboard.http_app import DEFAULT_SNAPSHOT_PATH  # noqa: E402
from src.dashboard.models import COLLECTION_STATUS_ERROR  # noqa: E402
from src.live import position_state as ps  # noqa: E402
from src.live import session_replay_pass_gate  # noqa: E402
from src.live.position_state import _atomic_write_text  # noqa: E402

# Owner rw, group r, other none -- see the chmod call below for the
# full "why." Requires the snapshot directory's group to actually be
# `ai-dashboard` (docs/DASHBOARD_V1_RUNBOOK.md's own setup steps).
SNAPSHOT_FILE_MODE = 0o640


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--state-path", type=Path, default=ps.DEFAULT_STATE_PATH)
    parser.add_argument("--guard-path", type=Path, default=ps.HIGH_WATER_MARK_PATH)
    parser.add_argument("--decision-log-directory", type=Path, default=snapshot_builder.DEFAULT_DECISION_LOG_DIRECTORY)
    parser.add_argument(
        "--pass-gate-directory", type=Path, default=session_replay_pass_gate.DEFAULT_PASS_GATE_DIRECTORY,
    )
    arguments = parser.parse_args()

    snapshot = snapshot_builder.build_snapshot(
        state_path=arguments.state_path,
        guard_path=arguments.guard_path,
        decision_log_directory=arguments.decision_log_directory,
        pass_gate_directory=arguments.pass_gate_directory,
    )

    if snapshot.collection_status == COLLECTION_STATUS_ERROR:
        error_types = [error.error_type for error in snapshot.errors]
        print(
            f"ERROR: snapshot core (broker P&L/account) failed ({error_types}) -- "
            f"refusing to overwrite {arguments.output_path} with a broken snapshot. "
            f"Previous snapshot, if any, is left untouched.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n"
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(arguments.output_path, payload)
    # Independent-audit finding, 2026-08-26: _atomic_write_text's own
    # tempfile.mkstemp() creates (and os.replace preserves) mode 0600 --
    # owner (ai-dashboard-producer) read/write only. The UNPRIVILEGED
    # ai-dashboard user (a different Unix user, running the HTTP
    # service -- see deploy/systemd/) could never read that file. 0640
    # (owner rw, GROUP r) lets ai-dashboard read it via the shared
    # group set up in docs/DASHBOARD_V1_RUNBOOK.md's own `chown ...:
    # ai-dashboard` + setgid step -- explicitly set here rather than
    # relying on mkstemp's own default, which is correct/tight for
    # every OTHER caller of _atomic_write_text (the real guard
    # directory, order_intent.py's journal) but wrong for a file that
    # is deliberately meant to be read by a second, unprivileged user.
    arguments.output_path.chmod(SNAPSHOT_FILE_MODE)
    print(f"Dashboard snapshot written to {arguments.output_path} (collection_status={snapshot.collection_status}).")


if __name__ == "__main__":
    main()
