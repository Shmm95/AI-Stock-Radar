#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

die() {
    printf 'DEPLOY FAILED: %s\n' "$*" >&2
    exit 1
}

if [ "$#" -ne 1 ]; then
    die "Usage: LOCAL_REPO=/path DEPLOY_HOST=root@host $0 <approved-commit>"
fi

: "${LOCAL_REPO:?Set LOCAL_REPO to the local repository/worktree path}"
: "${DEPLOY_HOST:?Set DEPLOY_HOST, for example root@control-server}"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-workspace/b-forward-research}"
ALLOW_DIRTY_SOURCE="${ALLOW_DIRTY_SOURCE:-0}"

RUNTIME_ROOT="/root/AI-Stock-Radar-Control"
DEPLOY_ROOT="${RUNTIME_ROOT}/deploy"

REQUESTED_COMMIT="$1"

git -C "$LOCAL_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "LOCAL_REPO is not a Git worktree: $LOCAL_REPO"

SOURCE_BRANCH="$(git -C "$LOCAL_REPO" branch --show-current)"
[ "$SOURCE_BRANCH" = "$EXPECTED_BRANCH" ] \
    || die "Expected branch $EXPECTED_BRANCH, found $SOURCE_BRANCH"

FULL_COMMIT="$(
    git -C "$LOCAL_REPO" rev-parse --verify "${REQUESTED_COMMIT}^{commit}"
)" || die "Commit does not exist: $REQUESTED_COMMIT"

git -C "$LOCAL_REPO" merge-base --is-ancestor "$FULL_COMMIT" HEAD \
    || die "Approved commit is not an ancestor of current branch HEAD"

if [ -n "$(git -C "$LOCAL_REPO" diff --name-only --diff-filter=U)" ]; then
    die "Source worktree contains unresolved merge conflicts"
fi

SOURCE_WORKTREE_CLEAN="true"
if [ -n "$(git -C "$LOCAL_REPO" status --porcelain --untracked-files=normal)" ]; then
    SOURCE_WORKTREE_CLEAN="false"
    if [ "$ALLOW_DIRTY_SOURCE" != "1" ]; then
        git -C "$LOCAL_REPO" status --short >&2
        die "Source worktree is dirty. Commit/clean it, or deliberately set ALLOW_DIRTY_SOURCE=1"
    fi
fi

TREE_SHA="$(git -C "$LOCAL_REPO" rev-parse "${FULL_COMMIT}^{tree}")"
COMMIT_TIME="$(git -C "$LOCAL_REPO" show -s --format=%cI "$FULL_COMMIT")"
DEPLOYED_BY="$(id -un)"

LOCAL_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/control-arm-deploy.XXXXXX")"
cleanup() {
    if [ -n "${LOCAL_TMP_DIR:-}" ] && [ -d "$LOCAL_TMP_DIR" ]; then
        rm -rf -- "$LOCAL_TMP_DIR"
    fi
}
trap cleanup EXIT

LOCAL_ARTIFACT="${LOCAL_TMP_DIR}/control-arm-${FULL_COMMIT}.tar"

git -C "$LOCAL_REPO" archive \
    --format=tar \
    --prefix=app/ \
    "$FULL_COMMIT" > "$LOCAL_ARTIFACT"

ARCHIVE_COMMIT="$(git get-tar-commit-id < "$LOCAL_ARTIFACT")"
[ "$ARCHIVE_COMMIT" = "$FULL_COMMIT" ] \
    || die "git archive commit header does not match approved commit"

ARTIFACT_SHA256="$(
    shasum -a 256 "$LOCAL_ARTIFACT" | awk '{print $1}'
)"

printf '\nControl-arm release approval\n'
printf '  Branch:          %s\n' "$SOURCE_BRANCH"
printf '  Commit:          %s\n' "$FULL_COMMIT"
printf '  Tree:            %s\n' "$TREE_SHA"
printf '  Commit time:     %s\n' "$COMMIT_TIME"
printf '  Artifact SHA256: %s\n' "$ARTIFACT_SHA256"
printf '  Worktree clean:  %s\n\n' "$SOURCE_WORKTREE_CLEAN"

printf 'Type the FULL commit hash to approve this deployment: '
IFS= read -r CONFIRMED_COMMIT

[ "$CONFIRMED_COMMIT" = "$FULL_COMMIT" ] \
    || die "Typed approval did not match the full commit hash"

UPLOAD_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REMOTE_PARTIAL="${DEPLOY_ROOT}/incoming/${FULL_COMMIT}.${UPLOAD_TOKEN}.tar.part"

ssh "$DEPLOY_HOST" "
    set -eu
    test \"\$(id -u)\" = 0
    install -d -m 0700 \
        '${DEPLOY_ROOT}/incoming' \
        '${DEPLOY_ROOT}/artifacts' \
        '${DEPLOY_ROOT}/releases'
"

scp "$LOCAL_ARTIFACT" "${DEPLOY_HOST}:${REMOTE_PARTIAL}"

ssh "$DEPLOY_HOST" bash -s -- \
    "$FULL_COMMIT" \
    "$TREE_SHA" \
    "$ARTIFACT_SHA256" \
    "$SOURCE_BRANCH" \
    "$COMMIT_TIME" \
    "$SOURCE_WORKTREE_CLEAN" \
    "$DEPLOYED_BY" \
    "$REMOTE_PARTIAL" <<'REMOTE_BASH'
set -Eeuo pipefail
umask 077

die() {
    printf 'REMOTE DEPLOY FAILED: %s\n' "$*" >&2
    exit 1
}

FULL_COMMIT="$1"
TREE_SHA="$2"
EXPECTED_ARTIFACT_SHA="$3"
SOURCE_BRANCH="$4"
COMMIT_TIME="$5"
SOURCE_WORKTREE_CLEAN="$6"
DEPLOYED_BY="$7"
REMOTE_PARTIAL="$8"

RUNTIME_ROOT="/root/AI-Stock-Radar-Control"
DEPLOY_ROOT="${RUNTIME_ROOT}/deploy"
INCOMING_DIR="${DEPLOY_ROOT}/incoming"
ARTIFACTS_DIR="${DEPLOY_ROOT}/artifacts"
RELEASES_DIR="${DEPLOY_ROOT}/releases"
RELEASE_DIR="${RELEASES_DIR}/${FULL_COMMIT}"
CURRENT_LINK="${DEPLOY_ROOT}/current"
ROOT_MANIFEST="${RUNTIME_ROOT}/DEPLOYED_VERSION.txt"

case "$FULL_COMMIT" in
    *[!0-9a-f]*|'') die "Invalid commit hash" ;;
esac
[ "${#FULL_COMMIT}" -eq 40 ] || die "Commit hash must contain 40 hexadecimal characters"

case "$TREE_SHA" in
    *[!0-9a-f]*|'') die "Invalid tree hash" ;;
esac
[ "${#TREE_SHA}" -eq 40 ] || die "Tree hash must contain 40 hexadecimal characters"

case "$EXPECTED_ARTIFACT_SHA" in
    *[!0-9a-f]*|'') die "Invalid artifact SHA-256" ;;
esac
[ "${#EXPECTED_ARTIFACT_SHA}" -eq 64 ] \
    || die "Artifact SHA-256 must contain 64 hexadecimal characters"

case "$REMOTE_PARTIAL" in
    "${INCOMING_DIR}/"*.tar.part) ;;
    *) die "Unexpected incoming artifact path: $REMOTE_PARTIAL" ;;
esac

test "$(id -u)" = 0 || die "Remote deploy must run as root"
test -f "${RUNTIME_ROOT}/.env" || die "Control .env is missing"
test -x "${RUNTIME_ROOT}/.venv/bin/python3" || die "Control virtualenv is missing"
test -d "${RUNTIME_ROOT}/data/live" || die "Persistent data/live directory is missing"
test -d "${RUNTIME_ROOT}/logs" || die "Persistent logs directory is missing"
test -d "${RUNTIME_ROOT}/.guard" || die "Persistent guard directory is missing"
test -f "$REMOTE_PARTIAL" || die "Transferred artifact is missing"

REMOTE_ARTIFACT_SHA="$(sha256sum "$REMOTE_PARTIAL" | awk '{print $1}')"
[ "$REMOTE_ARTIFACT_SHA" = "$EXPECTED_ARTIFACT_SHA" ] \
    || die "Transferred artifact SHA-256 mismatch"

ARCHIVE_COMMIT="$(git get-tar-commit-id < "$REMOTE_PARTIAL")"
[ "$ARCHIVE_COMMIT" = "$FULL_COMMIT" ] \
    || die "Transferred archive identifies a different commit"

ARTIFACT_PATH="${ARTIFACTS_DIR}/${FULL_COMMIT}-${EXPECTED_ARTIFACT_SHA}.tar"

if [ -L "$ARTIFACT_PATH" ] || { [ -e "$ARTIFACT_PATH" ] && [ ! -f "$ARTIFACT_PATH" ]; }; then
    die "Artifact destination is not a regular file"
fi

sync "$REMOTE_PARTIAL"
mv -f -- "$REMOTE_PARTIAL" "$ARTIFACT_PATH"
sync "$ARTIFACT_PATH"
sync "$ARTIFACTS_DIR"

if [ -L "$RELEASE_DIR" ]; then
    die "Release destination must not be a symbolic link"
fi

if [ -d "$RELEASE_DIR" ]; then
    test -f "${RELEASE_DIR}/DEPLOYED_VERSION.txt" \
        || die "Existing release has no manifest"
    grep -Fqx "commit_sha=${FULL_COMMIT}" "${RELEASE_DIR}/DEPLOYED_VERSION.txt" \
        || die "Existing release commit mismatch"
    grep -Fqx "git_tree_sha=${TREE_SHA}" "${RELEASE_DIR}/DEPLOYED_VERSION.txt" \
        || die "Existing release tree mismatch"
    grep -Fqx "artifact_sha256=${EXPECTED_ARTIFACT_SHA}" "${RELEASE_DIR}/DEPLOYED_VERSION.txt" \
        || die "Existing release artifact mismatch"
else
    STAGING_DIR="${RELEASES_DIR}/.staging-${FULL_COMMIT}-$$"
    [ ! -e "$STAGING_DIR" ] && [ ! -L "$STAGING_DIR" ] \
        || die "Staging directory already exists"

    install -d -m 0755 "$STAGING_DIR"
    tar --extract \
        --file="$ARTIFACT_PATH" \
        --directory="$STAGING_DIR" \
        --no-same-owner

    APP_DIR="${STAGING_DIR}/app"

    test -f "${APP_DIR}/scripts/run_control_arm_decision.py" \
        || die "Runner missing from archive"
    test -f "${APP_DIR}/src/live/control_universe.py" \
        || die "Control universe missing from archive"
    test -f "${APP_DIR}/src/live/order_intent.py" \
        || die "Order-intent module missing from archive"
    test -f "${APP_DIR}/src/live/single_instance_lock.py" \
        || die "Single-instance lock missing from archive"
    test -f "${APP_DIR}/docs/control_arm_crontab_v3.draft" \
        || die "Control cron draft missing from archive"

    for RUNTIME_ENTRY in .env .venv .guard logs; do
        if [ -e "${APP_DIR}/${RUNTIME_ENTRY}" ] || [ -L "${APP_DIR}/${RUNTIME_ENTRY}" ]; then
            die "Archive unexpectedly contains runtime path: ${RUNTIME_ENTRY}"
        fi
    done

    if [ -e "${APP_DIR}/data/live" ] || [ -L "${APP_DIR}/data/live" ]; then
        die "Archive unexpectedly contains persistent data/live"
    fi

    if [ -e "${APP_DIR}/DEPLOYED_VERSION.txt" ] || [ -L "${APP_DIR}/DEPLOYED_VERSION.txt" ]; then
        die "Archive unexpectedly contains DEPLOYED_VERSION.txt"
    fi

    mkdir -p "${APP_DIR}/data"
    ln -s "${RUNTIME_ROOT}/.env" "${APP_DIR}/.env"
    ln -s "${RUNTIME_ROOT}/.venv" "${APP_DIR}/.venv"
    ln -s "${RUNTIME_ROOT}/.guard" "${APP_DIR}/.guard"
    ln -s "${RUNTIME_ROOT}/logs" "${APP_DIR}/logs"
    ln -s "${RUNTIME_ROOT}/data/live" "${APP_DIR}/data/live"

    DEPLOYED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    MANIFEST_TMP="${APP_DIR}/.DEPLOYED_VERSION.txt.tmp.$$"

    {
        printf 'schema_version=1\n'
        printf 'component=control-arm\n'
        printf 'commit_sha=%s\n' "$FULL_COMMIT"
        printf 'git_tree_sha=%s\n' "$TREE_SHA"
        printf 'source_branch=%s\n' "$SOURCE_BRANCH"
        printf 'commit_time=%s\n' "$COMMIT_TIME"
        printf 'artifact_sha256=%s\n' "$EXPECTED_ARTIFACT_SHA"
        printf 'artifact_path=%s\n' "$ARTIFACT_PATH"
        printf 'artifact_format=git-archive-tar\n'
        printf 'archive_commit_sha=%s\n' "$ARCHIVE_COMMIT"
        printf 'deploy_method=git-archive-immutable-release-v1\n'
        printf 'deployed_at_utc=%s\n' "$DEPLOYED_AT_UTC"
        printf 'deployed_by=%s\n' "$DEPLOYED_BY"
        printf 'source_worktree_clean=%s\n' "$SOURCE_WORKTREE_CLEAN"
        printf 'source_working_tree_used=false\n'
        printf 'runtime_root=%s\n' "$RUNTIME_ROOT"
        printf 'release_path=%s\n' "$RELEASE_DIR"
    } > "$MANIFEST_TMP"

    chmod 0444 "$MANIFEST_TMP"
    sync "$MANIFEST_TMP"
    mv -f -- "$MANIFEST_TMP" "${APP_DIR}/DEPLOYED_VERSION.txt"
    sync "$APP_DIR"

    mv -T -- "$APP_DIR" "$RELEASE_DIR"
    rmdir "$STAGING_DIR"
    sync "$RELEASES_DIR"
fi

CURRENT_TMP="${DEPLOY_ROOT}/.current.$$"
[ ! -e "$CURRENT_TMP" ] && [ ! -L "$CURRENT_TMP" ] \
    || die "Temporary current link already exists"

ln -s "releases/${FULL_COMMIT}" "$CURRENT_TMP"
mv -Tf -- "$CURRENT_TMP" "$CURRENT_LINK"
sync "$DEPLOY_ROOT"

EXPECTED_ROOT_MANIFEST_LINK="deploy/current/DEPLOYED_VERSION.txt"

if [ -L "$ROOT_MANIFEST" ]; then
    [ "$(readlink "$ROOT_MANIFEST")" = "$EXPECTED_ROOT_MANIFEST_LINK" ] \
        || die "DEPLOYED_VERSION.txt points to an unexpected target"
elif [ -e "$ROOT_MANIFEST" ]; then
    die "Legacy DEPLOYED_VERSION.txt is not a symbolic link; migrate it manually"
else
    ROOT_MANIFEST_TMP="${RUNTIME_ROOT}/.DEPLOYED_VERSION.txt.$$"
    ln -s "$EXPECTED_ROOT_MANIFEST_LINK" "$ROOT_MANIFEST_TMP"
    mv -Tf -- "$ROOT_MANIFEST_TMP" "$ROOT_MANIFEST"
    sync "$RUNTIME_ROOT"
fi

ACTIVE_RELEASE="$(readlink -f "$CURRENT_LINK")"
[ "$ACTIVE_RELEASE" = "$RELEASE_DIR" ] \
    || die "Active release path does not match the approved commit"

MANIFEST_TARGET="$(readlink -f "$ROOT_MANIFEST")"
[ "$MANIFEST_TARGET" = "${RELEASE_DIR}/DEPLOYED_VERSION.txt" ] \
    || die "Root manifest does not resolve into the active release"

grep -Fqx "commit_sha=${FULL_COMMIT}" "$ROOT_MANIFEST" \
    || die "Active manifest commit mismatch"
grep -Fqx "git_tree_sha=${TREE_SHA}" "$ROOT_MANIFEST" \
    || die "Active manifest tree mismatch"
grep -Fqx "artifact_sha256=${EXPECTED_ARTIFACT_SHA}" "$ROOT_MANIFEST" \
    || die "Active manifest artifact mismatch"

FINAL_ARTIFACT_SHA="$(sha256sum "$ARTIFACT_PATH" | awk '{print $1}')"
[ "$FINAL_ARTIFACT_SHA" = "$EXPECTED_ARTIFACT_SHA" ] \
    || die "Stored artifact no longer matches the manifest"

printf '\nDEPLOY VERIFIED\n'
printf 'Active release: %s\n' "$ACTIVE_RELEASE"
cat "$ROOT_MANIFEST"
REMOTE_BASH
