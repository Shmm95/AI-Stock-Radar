"""Single-instance lock for `position_state.json` writers -- a
DIFFERENT mechanism from `position_state.py`'s own `_guard_write_lock`,
deliberately, per this task's own explicit instruction.

WHY A NEW, SEPARATE MECHANISM (not a reuse of `_guard_write_lock`):
`_guard_write_lock` protects a single, short JSON write (the high-water
mark) and self-heals via an AGE-based staleness heuristic (a lock older
than 300s, or held by a dead PID, is unlinked and re-acquired -- see
that function's own docstring). This lock protects something
structurally different: the ENTIRE lifetime of one control-arm run
(STOP check through intent commit through lock release), and this
task's own explicit policy is the opposite of `_guard_write_lock`'s:
**no blind age-based deletion at all** -- if a process holding this
lock dies (however it dies, including SIGKILL), the KERNEL releases the
lock the instant that process's file descriptors close, with zero
staleness heuristics needed. `fcntl.flock()` provides exactly this
guarantee natively -- an advisory lock tied to an open file description,
never a separate lock-file-existence convention a second process has to
reason about the age of.

Because the OS itself is the only thing that can ever release this
lock (via fd closure), failure to acquire it means fail-closed,
unconditionally -- there is no staleness override, no retry-past-a-
timeout, no manual unlink path built into this module. If two
processes are both really alive and one holds this lock, the second
MUST NOT proceed; if the reported holder is not really alive, the
kernel has ALREADY released the lock (by the time a second process
could observe the first's PID as dead, the OS has already closed its
descriptors) -- there is no window where a genuinely dead holder's
lock can still block a new acquisition, which is exactly why no
staleness-override logic is needed or provided here.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.live.position_state import _GUARD_DIRECTORY

LOCK_PATH = _GUARD_DIRECTORY / "position_state.writer.lock"


class SingleInstanceLockError(RuntimeError):
    """Raised when the lock cannot be acquired -- another real process
    currently holds it. Fail-closed: no retry, no override, no age-based
    bypass built into this module. See module docstring for why none is
    needed."""


def _boot_id() -> str:
    """Real, best-effort, platform-appropriate boot identifier -- Linux's
    `/proc/sys/kernel/random/boot_id` (a real random UUID minted at
    every boot) tried first (the real production server is Linux);
    macOS's `sysctl kern.bootsessionuuid` (the same concept, confirmed
    present on this dev machine) as a fallback for local testing;
    `"unknown"` if neither is available -- diagnostic metadata only,
    never used for lock logic itself, so a missing value degrades
    gracefully rather than failing the lock."""
    proc_path = Path("/proc/sys/kernel/random/boot_id")
    if proc_path.is_file():
        try:
            return proc_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.bootsessionuuid"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        value = result.stdout.strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _process_start_time() -> str:
    """Real, best-effort process start time via `ps -o lstart=` (POSIX,
    present on both the Linux production server and macOS dev
    machines -- confirmed on this machine). Diagnostic metadata only,
    same fallback discipline as `_boot_id`."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=5, check=True,
        )
        value = result.stdout.strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@contextmanager
def single_instance_lock(state_path: Path, lock_path: Path = LOCK_PATH) -> Iterator[dict]:
    """Hold an exclusive `fcntl.flock()` on `lock_path` for the duration
    of the `with` block. The file descriptor is kept open for exactly
    that duration (matching this task's own requirement that the
    descriptor stay open process-lifetime, not just around one write) --
    closed only when the `with` block exits, which is also the only
    thing (besides real process death) that releases the lock.

    Writes real metadata into the lock file the moment it is acquired
    (PID, hostname, boot id, process start time, a fresh run id,
    the absolute state path this run is protecting, and acquisition
    time) -- for a human to read if investigating contention, never
    read back programmatically by this module itself (that would
    reintroduce exactly the staleness-heuristic problem this design
    deliberately avoids).

    Raises `SingleInstanceLockError` immediately (`LOCK_NB`, non-blocking)
    if another real process already holds it -- no retry loop, no
    waiting, fail-closed per this module's own docstring.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise SingleInstanceLockError(
                f"Could not acquire single-instance lock at {lock_path} -- "
                f"another process is already holding it (fcntl.flock "
                f"non-blocking exclusive lock failed: {error}). Refusing to "
                f"proceed; no override exists. If you are certain no real "
                f"process is running, the OS would already have released "
                f"this lock on its own -- investigate rather than bypass."
            ) from error

        metadata = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "boot_id": _boot_id(),
            "process_start_time": _process_start_time(),
            "run_id": str(uuid.uuid4()),
            "state_path": str(Path(state_path).resolve()),
            "acquired_at": time.time(),
            "last_heartbeat_at": time.time(),
        }
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, (json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
        os.fsync(descriptor)

        yield metadata
    finally:
        # Closing the descriptor releases the flock -- this is the ONLY
        # release path this module implements, by design (see module
        # docstring). A SIGKILL between acquisition and this line still
        # releases the lock, just via the kernel closing the descriptor
        # on process exit rather than via this line running at all.
        os.close(descriptor)
