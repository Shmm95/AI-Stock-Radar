#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

die() {
    printf 'ENV DEPLOY FAILED: %s\n' "$*" >&2
    exit 1
}

if [ "$#" -ne 1 ]; then
    die "Usage: DEPLOY_HOST=root@host $0 <local-candidate-env-file>"
fi

: "${DEPLOY_HOST:?Set DEPLOY_HOST, for example root@control-server}"

CANDIDATE="$1"
REMOTE_ENV_DIR="${REMOTE_ENV_DIR:-/root/AI-Stock-Radar-Control}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-${REMOTE_ENV_DIR}/deploy/current}"
LOCAL_PYTHON_BIN="${LOCAL_PYTHON_BIN:-python3}"
LOCAL_CHECK_SCRIPT="${LOCAL_CHECK_SCRIPT:-$(dirname "$0")/check_env_health.py}"

[ -f "$CANDIDATE" ] || die "Candidate file does not exist: $CANDIDATE"
[ ! -L "$CANDIDATE" ] || die "Candidate must not be a symlink: $CANDIDATE"
[ -f "$LOCAL_CHECK_SCRIPT" ] || die "Health-check script not found: $LOCAL_CHECK_SCRIPT"

printf '\n=== Step 1: local candidate health check (Layers 1-3, real Alpaca+Telegram calls) ===\n'
"$LOCAL_PYTHON_BIN" "$LOCAL_CHECK_SCRIPT" "$CANDIDATE" \
    || die "Local candidate failed health check -- zero transfer attempted"

CANDIDATE_SHA256="$(shasum -a 256 "$CANDIDATE" | awk '{print $1}')"
printf 'Candidate SHA256: %s\n' "$CANDIDATE_SHA256"

UPLOAD_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REMOTE_PARTIAL="${REMOTE_ENV_DIR}/.env.next.${UPLOAD_TOKEN}"

printf '\n=== Step 2: transfer candidate to server as .env.next (NEVER over the active .env) ===\n'
scp "$CANDIDATE" "${DEPLOY_HOST}:${REMOTE_PARTIAL}"

printf '\n=== Step 3-5: remote re-check + SHA256 match + atomic swap (all-or-nothing) ===\n'
ssh "$DEPLOY_HOST" bash -s -- \
    "$REMOTE_PARTIAL" \
    "$CANDIDATE_SHA256" \
    "$REMOTE_ENV_DIR" \
    "$REMOTE_APP_DIR" <<'REMOTE_BASH'
set -Eeuo pipefail
umask 077

die() {
    printf 'REMOTE ENV DEPLOY FAILED: %s\n' "$*" >&2
    exit 1
}

REMOTE_PARTIAL="$1"
EXPECTED_SHA256="$2"
REMOTE_ENV_DIR="$3"
REMOTE_APP_DIR="$4"

ACTIVE_ENV="${REMOTE_ENV_DIR}/.env"
PREVIOUS_ENV="${REMOTE_ENV_DIR}/.env.previous"
CANONICAL_NEXT="${REMOTE_ENV_DIR}/.env.next"

case "$REMOTE_PARTIAL" in
    "${REMOTE_ENV_DIR}/.env.next."*) ;;
    *) die "Unexpected incoming candidate path: $REMOTE_PARTIAL" ;;
esac

test -f "$REMOTE_PARTIAL" || die "Transferred candidate is missing: $REMOTE_PARTIAL"

REMOTE_SHA256="$(sha256sum "$REMOTE_PARTIAL" | awk '{print $1}')"
[ "$REMOTE_SHA256" = "$EXPECTED_SHA256" ] \
    || die "Transferred candidate SHA-256 mismatch -- refusing to proceed"

chmod 600 "$REMOTE_PARTIAL"
mv -f -- "$REMOTE_PARTIAL" "$CANONICAL_NEXT"

REMOTE_PYTHON_BIN="${REMOTE_APP_DIR}/.venv/bin/python3"
if [ ! -x "$REMOTE_PYTHON_BIN" ]; then
    REMOTE_PYTHON_BIN="${REMOTE_ENV_DIR}/.venv/bin/python3"
fi
if [ ! -x "$REMOTE_PYTHON_BIN" ]; then
    REMOTE_PYTHON_BIN="python3"
fi

REMOTE_CHECK_SCRIPT="${REMOTE_APP_DIR}/scripts/check_env_health.py"
test -f "$REMOTE_CHECK_SCRIPT" \
    || die "Remote health-check script not found: $REMOTE_CHECK_SCRIPT (deploy the app release first)"

"$REMOTE_PYTHON_BIN" "$REMOTE_CHECK_SCRIPT" "$CANONICAL_NEXT" \
    || die "Remote .env.next failed health check -- active .env left completely untouched"

FINAL_SHA256="$(sha256sum "$CANONICAL_NEXT" | awk '{print $1}')"
[ "$FINAL_SHA256" = "$EXPECTED_SHA256" ] \
    || die "Post-check SHA-256 changed unexpectedly between transfer and re-check -- refusing to swap"

if [ -f "$ACTIVE_ENV" ] && [ ! -L "$ACTIVE_ENV" ]; then
    cp -f -- "$ACTIVE_ENV" "$PREVIOUS_ENV"
    chmod 600 "$PREVIOUS_ENV"
elif [ -L "$ACTIVE_ENV" ]; then
    die "Active .env is unexpectedly a symlink -- refusing to back it up/replace it blindly"
fi

mv -f -- "$CANONICAL_NEXT" "$ACTIVE_ENV"
chmod 600 "$ACTIVE_ENV"

printf '\nENV DEPLOY VERIFIED\n'
printf 'Active file: %s\n' "$ACTIVE_ENV"
printf 'SHA256:      %s\n' "$FINAL_SHA256"
if [ -f "$PREVIOUS_ENV" ]; then
    printf 'Previous version backed up at: %s (single copy, overwritten on next deploy)\n' "$PREVIOUS_ENV"
fi
REMOTE_BASH
