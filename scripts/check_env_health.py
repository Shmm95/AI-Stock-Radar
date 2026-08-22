"""Four-layer credential health check for `.env.control` (and, by
parameterizing the path, any similarly-shaped control-arm credential
file).

WHY THIS EXISTS: `.env.control` broke three separate times in one real
session -- a truncated Telegram token, an import-order bug that silently
swapped in the live system's Alpaca credentials instead of the control
account's, and two field values concatenated into one line. The existing
fail-closed loader (`_load_env_file_fail_closed` in
`scripts/run_control_arm_decision.py`) only ever asked "does this file
exist and did `load_dotenv()` report loading something" -- it never asked
"are the values in it actually correct." This module closes that gap
with four independent layers, each strictly stronger than the last:

  Layer 1 -- FILE INTEGRITY: is this a real, regular file (never a
    symlink) at a sane permission mode, valid UTF-8, pure LF line
    endings with a final newline, free of CRLF/NUL/control characters,
    every non-blank line in exact KEY=VALUE form, no duplicate keys, no
    unrecognized/misspelled keys. Answers "is this file even
    well-formed," independent of what any value actually says.

  Layer 2 -- FIELD SCHEMA: are the four required fields
    (ALPACA_API_KEY, ALPACA_SECRET_KEY, TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID) present, non-empty, and shaped correctly (real,
    measured lengths -- see the constants below and their own
    docstrings for exactly how each was established; a length this
    module cannot confidently justify is never silently invented).
    Catches a truncated token or two concatenated values WITHOUT
    needing a single network call.

  Layer 3 -- CROSS IDENTITY: real, read-only calls -- Alpaca
    `get_account()` (does it succeed, and does the account number end
    in the expected control-arm suffix), Telegram `getMe()` (does the
    token resolve to a real bot), Telegram `getChat()` (is the
    configured chat target actually reachable with this token).
    DELIBERATELY NEVER calls Telegram's `sendMessage` -- this layer
    verifies identity, it does not deliver anything.

  Layer 4 -- SYNC VERIFICATION: local file's SHA-256 vs. the same file's
    SHA-256 on a remote host (via SSH), so a local fix that was never
    actually pushed -- or a remote copy nobody re-synced after a local
    edit -- shows up explicitly. A mismatch is reported as exactly that,
    a mismatch -- this layer never guesses which side is the "healthy"
    one; only Layers 1-3, run independently against each side, can say
    that.

A credential's raw VALUE is never printed, logged, or included in any
finding message anywhere in this module -- only lengths, hashes,
booleans, and masked last-4-character forms (for the Alpaca account
number, matching `broker_reconciliation.py`'s own masking discipline)
ever appear in output.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The keys this file is allowed to contain -- anything else is a
# typo or a leftover from a bad edit, not a legitimate field.
#
# LIVE_ACCOUNT_NUMBER_SUFFIX added 2026-08-22 (independent audit
# finding #1): run_daily_decision.py's mandatory account-identity check
# (src/live/broker_reconciliation.py) reads this from the live system's
# own .env -- without it in KNOWN_KEYS, this script would flag it as an
# unrecognized/leftover key the first time it's actually set, a false
# positive purely from this schema not knowing about it yet. NOT in
# REQUIRED_KEYS: this script is shared between the live .env and
# .env.control (see --expected-account-suffix's own default/override),
# and .env.control never needs this specific key.
KNOWN_KEYS = frozenset(
    {
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "LIVE_ACCOUNT_NUMBER_SUFFIX",
    }
)
REQUIRED_KEYS = frozenset({"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"})
EXPECTED_LIVE_ACCOUNT_NUMBER_SUFFIX_LENGTH = 4

# PROFILE DISTINCTION (2026-08-22, independent audit finding #4): this
# script previously treated LIVE_ACCOUNT_NUMBER_SUFFIX as always-optional
# (not in REQUIRED_KEYS) and Layer 3 always compared against the
# CONTROL arm's own hardcoded EXPECTED_ACCOUNT_SUFFIX ("XO4Y") --
# regardless of which .env this was run against. But
# scripts/run_daily_decision.py's own `_require_live_account_suffix()`
# treats LIVE_ACCOUNT_NUMBER_SUFFIX as MANDATORY for the LIVE entrypoint
# -- so this health check could report PASS on the live .env (the field
# is merely optional here) while the real runner would FAIL closed on
# the exact same file (the field is missing there). Two named profiles
# close that gap: PROFILE_CONTROL (default, unchanged behavior) checks
# .env.control's own shape; PROFILE_LIVE additionally requires
# LIVE_ACCOUNT_NUMBER_SUFFIX in Layer 2, and in Layer 3 compares the
# connected Alpaca account against THAT FILE'S OWN LIVE_ACCOUNT_NUMBER_SUFFIX
# value (never the control arm's XO4Y constant -- the live account's
# real suffix is deliberately not hardcoded anywhere in this codebase;
# see run_daily_decision.py's own docstring).
PROFILE_CONTROL = "control"
PROFILE_LIVE = "live"
LIVE_PROFILE_REQUIRED_KEYS = REQUIRED_KEYS | {"LIVE_ACCOUNT_NUMBER_SUFFIX"}

# Real, MEASURED lengths -- never a guess. ALPACA_API_KEY (26) and
# ALPACA_SECRET_KEY (44) were confirmed this session from TWO
# independent real files agreeing exactly: `.env.control` itself and the
# live system's own, completely separate `.env` -- both currently
# measure 26/44 respectively (checked with a length-only read, values
# never printed). A "62-character secret" figure was proposed at the
# start of this task and explicitly NOT used here after this
# cross-check contradicted it and the user confirmed proceeding with the
# measured value instead. TELEGRAM_BOT_TOKEN (46) and the
# numeric_bot_id:token shape were confirmed the same way, independently,
# in `tests/test_telegram_network_isolation.py` earlier this session.
EXPECTED_ALPACA_API_KEY_LENGTH = 26
EXPECTED_ALPACA_SECRET_KEY_LENGTH = 44
EXPECTED_TELEGRAM_BOT_TOKEN_LENGTH = 46

EXPECTED_ACCOUNT_SUFFIX = "XO4Y"

_TELEGRAM_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]+$")
_TELEGRAM_CHAT_ID_RE = re.compile(r"^-?\d+$")
_KEY_LINE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=.*$")
_CONTROL_CHAR_RE = re.compile(rb"[\x00-\x08\x0B-\x1F\x7F]")


@dataclass
class Finding:
    layer: int
    severity: str  # "FAIL" | "WARN" | "INFO" -- never a credential value
    message: str

    def __str__(self) -> str:
        return f"[L{self.layer} {self.severity}] {self.message}"


@dataclass
class HealthCheckReport:
    path: Path
    findings: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "FAIL" for f in self.findings)

    def print_report(self) -> None:
        print(f"\n=== Credential health check: {self.path} ===")
        for f in self.findings:
            print(str(f))
        print(f"\nRESULT: {'PASS' if self.passed else 'FAIL'}")


# ---------------------------------------------------------------------------
# Layer 1 -- file integrity
# ---------------------------------------------------------------------------


def check_layer1_file_integrity(path: Path) -> list[Finding]:
    findings: list[Finding] = []

    if path.is_symlink():
        findings.append(Finding(1, "FAIL", f"{path} is a symlink -- refusing to trust an indirect credential path"))
        return findings
    if not path.exists():
        findings.append(Finding(1, "FAIL", f"{path} does not exist"))
        return findings
    if not path.is_file():
        findings.append(Finding(1, "FAIL", f"{path} exists but is not a regular file"))
        return findings

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        findings.append(
            Finding(
                1, "WARN",
                f"{path} permissions are {oct(mode)}, expected 0600 -- not auto-fixing; "
                f"fix by hand: chmod 600 {path}",
            )
        )

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        findings.append(Finding(1, "FAIL", f"{path} is not valid UTF-8: {error}"))
        return findings

    if b"\r" in raw:
        findings.append(Finding(1, "FAIL", f"{path} contains CRLF (\\r) line endings -- expected pure LF"))
    if b"\x00" in raw:
        findings.append(Finding(1, "FAIL", f"{path} contains a NUL byte"))
    control_chars = _CONTROL_CHAR_RE.findall(raw)
    if control_chars:
        findings.append(
            Finding(1, "FAIL", f"{path} contains {len(control_chars)} control character(s) outside LF")
        )
    if raw and not raw.endswith(b"\n"):
        findings.append(Finding(1, "FAIL", f"{path} does not end with a final newline"))

    seen_keys: dict[str, int] = {}
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # the element after the final newline's split
    for line_number, line in enumerate(lines, start=1):
        if line == "":
            continue
        if not _KEY_LINE_RE.match(line):
            findings.append(Finding(1, "FAIL", f"{path}:{line_number} is not in KEY=VALUE format"))
            continue
        key = line.split("=", 1)[0]
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if key not in KNOWN_KEYS:
            findings.append(
                Finding(1, "FAIL", f"{path}:{line_number} has an unrecognized key {key!r} -- expected one of {sorted(KNOWN_KEYS)}")
            )

    for key, count in seen_keys.items():
        if count > 1:
            findings.append(Finding(1, "FAIL", f"{path} defines {key!r} {count} times -- duplicate key"))

    return findings


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# Layer 2 -- field schema
# ---------------------------------------------------------------------------


def check_layer2_field_schema(path: Path, *, profile: str = PROFILE_CONTROL) -> list[Finding]:
    findings: list[Finding] = []
    values = _parse_env_file(path)

    required_keys = LIVE_PROFILE_REQUIRED_KEYS if profile == PROFILE_LIVE else REQUIRED_KEYS
    for key in sorted(required_keys):
        if key not in values or values[key] == "":
            findings.append(Finding(2, "FAIL", f"{key} is missing or empty"))

    if values.get("ALPACA_API_KEY"):
        length = len(values["ALPACA_API_KEY"])
        if length != EXPECTED_ALPACA_API_KEY_LENGTH:
            findings.append(
                Finding(2, "FAIL", f"ALPACA_API_KEY length is {length}, expected {EXPECTED_ALPACA_API_KEY_LENGTH}")
            )

    if values.get("ALPACA_SECRET_KEY"):
        length = len(values["ALPACA_SECRET_KEY"])
        if length != EXPECTED_ALPACA_SECRET_KEY_LENGTH:
            findings.append(
                Finding(2, "FAIL", f"ALPACA_SECRET_KEY length is {length}, expected {EXPECTED_ALPACA_SECRET_KEY_LENGTH}")
            )

    if values.get("TELEGRAM_BOT_TOKEN"):
        token = values["TELEGRAM_BOT_TOKEN"]
        if len(token) != EXPECTED_TELEGRAM_BOT_TOKEN_LENGTH:
            findings.append(
                Finding(2, "FAIL", f"TELEGRAM_BOT_TOKEN length is {len(token)}, expected {EXPECTED_TELEGRAM_BOT_TOKEN_LENGTH}")
            )
        if not _TELEGRAM_TOKEN_RE.match(token):
            findings.append(Finding(2, "FAIL", "TELEGRAM_BOT_TOKEN does not match <numeric_bot_id>:<token> format"))

    if "LIVE_ACCOUNT_NUMBER_SUFFIX" in values:
        suffix = values["LIVE_ACCOUNT_NUMBER_SUFFIX"]
        if len(suffix) != EXPECTED_LIVE_ACCOUNT_NUMBER_SUFFIX_LENGTH:
            findings.append(
                Finding(
                    2, "FAIL",
                    f"LIVE_ACCOUNT_NUMBER_SUFFIX length is {len(suffix)}, expected exactly "
                    f"{EXPECTED_LIVE_ACCOUNT_NUMBER_SUFFIX_LENGTH} -- run_daily_decision.py's own "
                    f"mandatory account-identity check requires exactly this length (see "
                    f"scripts/run_daily_decision.py's _require_live_account_suffix)",
                )
            )

    if values.get("TELEGRAM_CHAT_ID"):
        chat_id = values["TELEGRAM_CHAT_ID"]
        if not _TELEGRAM_CHAT_ID_RE.match(chat_id):
            findings.append(Finding(2, "FAIL", "TELEGRAM_CHAT_ID is not purely numeric"))

    if values.get("ALPACA_BASE_URL") and not values["ALPACA_BASE_URL"].startswith("https://"):
        findings.append(Finding(2, "WARN", "ALPACA_BASE_URL does not start with https://"))

    return findings


# ---------------------------------------------------------------------------
# Layer 3 -- cross identity (real, read-only API calls)
# ---------------------------------------------------------------------------


def check_layer3_cross_identity(
    path: Path, *, expected_account_suffix: str = EXPECTED_ACCOUNT_SUFFIX, profile: str = PROFILE_CONTROL
) -> list[Finding]:
    findings: list[Finding] = []
    values = _parse_env_file(path)

    if profile == PROFILE_LIVE:
        # Never compare a live .env against the control arm's own XO4Y
        # constant -- the live account's real suffix is deliberately
        # not hardcoded anywhere in this codebase (see
        # run_daily_decision.py's own docstring); the only correct
        # expectation for THIS file is its own declared value, the
        # exact same value _require_live_account_suffix() will read at
        # runtime.
        live_suffix = values.get("LIVE_ACCOUNT_NUMBER_SUFFIX")
        if not live_suffix:
            findings.append(
                Finding(
                    3, "FAIL",
                    "profile=live but LIVE_ACCOUNT_NUMBER_SUFFIX is missing/empty -- "
                    "cannot perform the account-identity cross-check this profile "
                    "requires (run_daily_decision.py's own mandatory check would "
                    "also fail closed on this same file).",
                )
            )
        else:
            expected_account_suffix = live_suffix

    api_key = values.get("ALPACA_API_KEY")
    secret_key = values.get("ALPACA_SECRET_KEY")
    if api_key and secret_key:
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
            account = client.get_account()
            account_number = str(account.account_number)
            masked = "..." + account_number[-4:]
            if account_number.endswith(expected_account_suffix):
                findings.append(Finding(3, "INFO", f"Alpaca get_account() succeeded, account ends {masked} (matches expected suffix)"))
            else:
                findings.append(
                    Finding(3, "FAIL", f"Alpaca get_account() succeeded but account ends {masked}, expected ...{expected_account_suffix}")
                )
        except Exception as error:  # noqa: BLE001 - report, never crash the health check itself
            findings.append(Finding(3, "FAIL", f"Alpaca get_account() failed: {type(error).__name__}: {error}"))
    else:
        findings.append(Finding(3, "FAIL", "Skipped Alpaca cross-identity check -- ALPACA_API_KEY/ALPACA_SECRET_KEY missing"))

    token = values.get("TELEGRAM_BOT_TOKEN")
    chat_id = values.get("TELEGRAM_CHAT_ID")
    if token:
        try:
            import requests

            response = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            payload = response.json()
            if response.status_code == 200 and payload.get("ok"):
                username = payload.get("result", {}).get("username", "<unknown>")
                findings.append(Finding(3, "INFO", f"Telegram getMe() succeeded, bot username: @{username}"))
            else:
                findings.append(Finding(3, "FAIL", f"Telegram getMe() failed: status={response.status_code}, ok={payload.get('ok')}"))
        except Exception as error:  # noqa: BLE001
            findings.append(Finding(3, "FAIL", f"Telegram getMe() failed: {type(error).__name__}: {error}"))
    else:
        findings.append(Finding(3, "FAIL", "Skipped Telegram getMe() -- TELEGRAM_BOT_TOKEN missing"))

    if token and chat_id:
        try:
            import requests

            # NEVER sendMessage here -- getChat only, per this module's
            # own explicit scope (see module docstring, Layer 3).
            response = requests.get(
                f"https://api.telegram.org/bot{token}/getChat", params={"chat_id": chat_id}, timeout=10
            )
            payload = response.json()
            if response.status_code == 200 and payload.get("ok"):
                findings.append(Finding(3, "INFO", "Telegram getChat() succeeded -- configured chat target is reachable"))
            else:
                findings.append(
                    Finding(3, "FAIL", f"Telegram getChat() failed: status={response.status_code}, ok={payload.get('ok')}, description={payload.get('description')}")
                )
        except Exception as error:  # noqa: BLE001
            findings.append(Finding(3, "FAIL", f"Telegram getChat() failed: {type(error).__name__}: {error}"))
    elif token:
        findings.append(Finding(3, "FAIL", "Skipped Telegram getChat() -- TELEGRAM_CHAT_ID missing"))

    return findings


# ---------------------------------------------------------------------------
# Layer 4 -- sync verification (local vs. remote SHA-256)
# ---------------------------------------------------------------------------


def check_layer4_sync(local_path: Path, deploy_host: str | None, remote_path: str) -> list[Finding]:
    findings: list[Finding] = []
    if not deploy_host:
        findings.append(Finding(4, "INFO", "No --deploy-host given -- sync check skipped"))
        return findings

    local_sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
    remote_cmd = f"sha256sum {shlex.quote(remote_path)} 2>/dev/null || shasum -a 256 {shlex.quote(remote_path)}"
    try:
        result = subprocess.run(
            ["ssh", deploy_host, remote_cmd],
            capture_output=True, text=True, timeout=15, check=True,
        )
        remote_sha256 = result.stdout.split()[0]
    except Exception as error:  # noqa: BLE001
        findings.append(Finding(4, "FAIL", f"Could not compute remote SHA-256 via SSH: {type(error).__name__}: {error}"))
        return findings

    if local_sha256 == remote_sha256:
        findings.append(Finding(4, "INFO", f"Local and remote SHA-256 match ({local_sha256[:12]}...)"))
    else:
        findings.append(
            Finding(
                4, "WARN",
                f"Local SHA-256 ({local_sha256[:12]}...) != remote SHA-256 ({remote_sha256[:12]}...) -- "
                f"files differ; WHICH side is healthy is unknown from this check alone, run Layers 1-3 "
                f"against each side independently to find out.",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_health_check(
    path: Path,
    *,
    expected_account_suffix: str = EXPECTED_ACCOUNT_SUFFIX,
    profile: str = PROFILE_CONTROL,
    run_layer3: bool = True,
    deploy_host: str | None = None,
    remote_path: str | None = None,
) -> HealthCheckReport:
    report = HealthCheckReport(path=path)
    report.findings.extend(check_layer1_file_integrity(path))
    layer1_ok = not any(f.severity == "FAIL" and f.layer == 1 for f in report.findings)

    if layer1_ok:
        report.findings.extend(check_layer2_field_schema(path, profile=profile))
        layer2_ok = not any(f.severity == "FAIL" and f.layer == 2 for f in report.findings)
        if layer2_ok and run_layer3:
            report.findings.extend(
                check_layer3_cross_identity(path, expected_account_suffix=expected_account_suffix, profile=profile)
            )
        elif not layer2_ok:
            report.findings.append(Finding(3, "INFO", "Layer 3 skipped -- Layer 2 field-schema checks failed first"))
    else:
        report.findings.append(Finding(2, "INFO", "Layer 2 skipped -- Layer 1 file-integrity checks failed first"))
        report.findings.append(Finding(3, "INFO", "Layer 3 skipped -- Layer 1 file-integrity checks failed first"))

    if deploy_host:
        report.findings.extend(check_layer4_sync(path, deploy_host, remote_path or str(path)))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("env_file", type=Path, help="Path to the .env-style credential file to check")
    parser.add_argument(
        "--expected-account-suffix", default=EXPECTED_ACCOUNT_SUFFIX,
        help=(
            f"Last 4 characters the Alpaca account_number must end in "
            f"(default: {EXPECTED_ACCOUNT_SUFFIX}). Ignored for --profile live, "
            f"which always compares against the checked file's own "
            f"LIVE_ACCOUNT_NUMBER_SUFFIX value instead."
        ),
    )
    parser.add_argument(
        "--profile", choices=(PROFILE_CONTROL, PROFILE_LIVE), default=PROFILE_CONTROL,
        help=(
            "'control' (default): checks .env.control's own shape, "
            "LIVE_ACCOUNT_NUMBER_SUFFIX optional. 'live': additionally "
            "requires LIVE_ACCOUNT_NUMBER_SUFFIX (Layer 2) and cross-checks "
            "the connected account against the FILE'S OWN declared suffix "
            "(Layer 3), matching run_daily_decision.py's own mandatory "
            "_require_live_account_suffix() check on the same file."
        ),
    )
    parser.add_argument("--skip-layer3", action="store_true", help="Skip real Alpaca/Telegram API calls (Layer 3)")
    parser.add_argument("--deploy-host", default=None, help="If given, also runs Layer 4 (SSH SHA-256 sync check) against this host")
    parser.add_argument("--remote-path", default=None, help="Remote path to compare against for Layer 4 (defaults to the same path as env_file)")
    arguments = parser.parse_args()

    report = run_health_check(
        arguments.env_file,
        expected_account_suffix=arguments.expected_account_suffix,
        profile=arguments.profile,
        run_layer3=not arguments.skip_layer3,
        deploy_host=arguments.deploy_host,
        remote_path=arguments.remote_path,
    )
    report.print_report()
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
