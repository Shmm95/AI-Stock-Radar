#!/usr/bin/env bash
# Live-decision-system release deploy -- adapted from
# scripts/deploy_control_arm_release.sh (already verified against
# RUNTIME_ROOT=/root/AI-Stock-Radar-Control for the control arm), same
# principle applied to the MAIN live system's own runtime root:
# /root/AI-Stock-Radar. Same safety properties, unchanged:
#   - git archive of an exact, ancestor-verified commit (never a
#     working tree, never HEAD unless HEAD IS the approved commit)
#   - SHA-256 verified at every hop (local artifact -> transfer ->
#     stored artifact -> final manifest)
#   - .env / .venv / .guard / logs / data/live are NEVER shipped in the
#     archive -- always symlinked from the persistent runtime root, so
#     a release can never silently overwrite live credentials, the
#     provisioned virtualenv, or live position/order state
#   - atomic release activation via a single symlink swap
#     (deploy/current -> releases/<commit>), with the manifest itself
#     re-verified to resolve into the active release before declaring
#     success
#   - idempotent: re-running with the same commit reuses the existing
#     release directory (verified against its own manifest) rather
#     than re-extracting
#   - BYTE-FOR-BYTE verified (2026-08-29 addition): after extraction
#     (or reuse), every tracked file in RELEASE_DIR is checked against
#     a full-tree sha256 manifest built locally from the SAME approved
#     artifact, via `sha256sum -c` -- not a curated subset, and not
#     something that trusts DEPLOYED_VERSION.txt's own field values
#     alone. Runs before the deploy is ever declared VERIFIED.
#   - RUNTIME_ROOT code sync (2026-08-29 addition, owner-approved --
#     supersedes the note that used to be here): the REAL crontab (raw
#     `crontab -l`, confirmed directly by the owner) executes from
#     RUNTIME_ROOT's own top level, `cd /root/AI-Stock-Radar && ...
#     scripts/...`, NEVER through deploy/current -- this predates
#     every deploy script built this session. So, strictly AFTER the
#     byte-for-byte RELEASE_DIR check above passes, this script now
#     ALSO rsyncs (`-a --delete`) exactly scripts/, src/, config/ (and
#     nothing else -- never docs/, tests/, or any persistent path) from
#     RELEASE_DIR into RUNTIME_ROOT, then re-verifies THAT sync
#     byte-for-byte too, before ever declaring VERIFIED. The
#     deploy/releases/<commit> + deploy/current symlink structure is
#     kept regardless -- this sync is additive, for the audit trail
#     and rollback capability, never a replacement for it.
#

# NOT LOCATED: a previously-used "deploy_method=git-archive-rsync-
# checksum-v2" script was searched for across all four sibling
# worktrees (workspace-a, workspace-b, workspace-c, the main
# checkpoint checkout) and not found anywhere on this machine. This
# script is a fresh adaptation of the one git-archive deploy script
# that DOES exist and IS already verified
# (deploy_control_arm_release.sh, deploy_method=git-archive-immutable-
# release-v1) -- labeled deploy_method=git-archive-immutable-release-v1
# below for the same reason, not v2, since that is the mechanism this
# script actually implements.
#
# REQUIRED-FILE CHECKLIST -- auto-derived from the REAL remote crontab,
# never hand-maintained (2026-08-29 revision, independent-audit
# finding). A hand-built list is exactly the kind of thing that
# silently drifts out of sync: an earlier draft of this checklist was
# traced once, by hand, against one specific commit, and was already
# wrong for THIS release the moment a cherry-picked commit changed
# send_status_update.py's own import chain (it started importing
# src/live/read_only_portfolio_snapshot.py, which the old hand-built
# list never re-traced to pick up). Rather than re-trace by hand every
# time code changes, this script now calls
# scripts/derive_required_files_from_crontab.py, which:
#   1. fetches the REAL crontab from the deploy target itself
#      (`ssh $DEPLOY_HOST crontab -l`) -- the actual, authoritative
#      list of entry points, never a copy that can go stale,
#   2. statically parses (Python `ast`, never grep/regex-on-code) every
#      entry point's own import statements, transitively, at the EXACT
#      release commit's own git tree (`git show <commit>:<path>`, never
#      the working tree) until no new project module is discovered.
# The resulting list is transferred to the remote host alongside the
# release artifact and is what the remote extraction step below
# actually checks -- see REQUIRED_FILES_PARTIAL/REQUIRED_FILES_SHA256
# below for the transfer, and the remote `while read -r REQUIRED_FILE`
# loop for the check itself.
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
: "${DEPLOY_HOST:?Set DEPLOY_HOST, for example root@live-server}"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-workspace/b-forward-research}"
ALLOW_DIRTY_SOURCE="${ALLOW_DIRTY_SOURCE:-0}"

RUNTIME_ROOT="/root/AI-Stock-Radar"
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

LOCAL_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/live-decision-deploy.XXXXXX")"
cleanup() {
    if [ -n "${LOCAL_TMP_DIR:-}" ] && [ -d "$LOCAL_TMP_DIR" ]; then
        rm -rf -- "$LOCAL_TMP_DIR"
    fi
}
trap cleanup EXIT

LOCAL_ARTIFACT="${LOCAL_TMP_DIR}/live-decision-${FULL_COMMIT}.tar"

git -C "$LOCAL_REPO" archive \
    --format=tar \
    --prefix=app/ \
    "$FULL_COMMIT" > "$LOCAL_ARTIFACT"

ARCHIVE_COMMIT="$(git get-tar-commit-id < "$LOCAL_ARTIFACT")"
[ "$ARCHIVE_COMMIT" = "$FULL_COMMIT" ] \
    || die "git archive commit header does not match approved commit"

# Required-file checklist, derived from the REAL remote crontab -- see
# this script's own top-of-file comment for the full "why". Fetched
# and computed HERE, locally, before anything is uploaded: a failure
# to reach the remote crontab, or a crontab-referenced script this
# release commit doesn't actually contain, must stop the deploy before
# any artifact transfer, not be discovered partway through the remote
# extraction step.
LOCAL_CRONTAB_FILE="${LOCAL_TMP_DIR}/remote_crontab.txt"
ssh "$DEPLOY_HOST" crontab -l > "$LOCAL_CRONTAB_FILE" \
    || die "Could not fetch the remote crontab from $DEPLOY_HOST -- the required-file list cannot be derived without it."

LOCAL_REQUIRED_FILES="${LOCAL_TMP_DIR}/required_files.txt"
python3 "$(dirname "$0")/derive_required_files_from_crontab.py" \
    --commit "$FULL_COMMIT" --crontab-file "$LOCAL_CRONTAB_FILE" > "$LOCAL_REQUIRED_FILES" \
    || die "Failed to derive the required-file list from the remote crontab -- see the error above."

[ -s "$LOCAL_REQUIRED_FILES" ] \
    || die "Derived required-file list is empty -- refusing to proceed with a checklist that would verify nothing."

REQUIRED_FILES_SHA256="$(shasum -a 256 "$LOCAL_REQUIRED_FILES" | awk '{print $1}')"
REQUIRED_FILES_COUNT="$(wc -l < "$LOCAL_REQUIRED_FILES" | tr -d ' ')"
printf 'Required-file list derived from the real remote crontab: %s file(s), sha256=%s\n' \
    "$REQUIRED_FILES_COUNT" "$REQUIRED_FILES_SHA256"

# FULL-TREE manifest (independent-audit finding, 2026-08-29): the
# required-file list above is a SPOT CHECK (does this specific subset
# exist), never a guarantee every file in the release tree is byte-
# correct -- and a real incident already showed why that distinction
# matters: a file at RUNTIME_ROOT's own top level (a path this script
# has never written to, ever -- see RELEASE_DIR/CURRENT_LINK below,
# both firmly under RUNTIME_ROOT/deploy/) was found stale, and the
# spot-check alone gave no way to prove or disprove whether the ACTUAL
# release tree (RELEASE_DIR, what deploy/current really points at) was
# itself fully, correctly extracted. This manifest covers EVERY
# tracked file in the release, not a curated subset, and is verified
# against RELEASE_DIR specifically (the one tree this script actually
# manages) further down -- never RUNTIME_ROOT's own top level, which
# by design also holds persistent, non-release things (.env, .venv,
# logs, data/live) a git tree was never going to match anyway.
#
# Built by re-extracting the SAME already-verified local artifact (not
# the working tree) into a throwaway directory and hashing every real
# file inside it -- sha256sum's own `<hash>  <path>` format, so the
# remote side can verify it with the standard `sha256sum -c` rather
# than hand-rolled comparison logic.
LOCAL_VERIFY_EXTRACT_DIR="${LOCAL_TMP_DIR}/verify-extract"
mkdir -p "$LOCAL_VERIFY_EXTRACT_DIR"
tar --extract --file="$LOCAL_ARTIFACT" --directory="$LOCAL_VERIFY_EXTRACT_DIR"

LOCAL_TREE_MANIFEST="${LOCAL_TMP_DIR}/full_tree_manifest.txt"
(
    cd "${LOCAL_VERIFY_EXTRACT_DIR}/app" \
        || die "Could not enter the local verification-extract directory"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256
) > "$LOCAL_TREE_MANIFEST"

[ -s "$LOCAL_TREE_MANIFEST" ] \
    || die "Full-tree manifest is empty -- refusing to proceed with a byte-for-byte check that would verify nothing."

TREE_MANIFEST_SHA256="$(shasum -a 256 "$LOCAL_TREE_MANIFEST" | awk '{print $1}')"
TREE_MANIFEST_FILE_COUNT="$(wc -l < "$LOCAL_TREE_MANIFEST" | tr -d ' ')"
printf 'Full-tree manifest built for byte-for-byte verification: %s file(s), sha256=%s\n' \
    "$TREE_MANIFEST_FILE_COUNT" "$TREE_MANIFEST_SHA256"

ARTIFACT_SHA256="$(
    shasum -a 256 "$LOCAL_ARTIFACT" | awk '{print $1}'
)"

printf '\nLive-decision release approval\n'
printf '  Branch:              %s\n' "$SOURCE_BRANCH"
printf '  Commit:              %s\n' "$FULL_COMMIT"
printf '  Tree:                %s\n' "$TREE_SHA"
printf '  Commit time:         %s\n' "$COMMIT_TIME"
printf '  Artifact SHA256:     %s\n' "$ARTIFACT_SHA256"
printf '  Required files:      %s file(s), sha256=%s (derived from the real remote crontab)\n' \
    "$REQUIRED_FILES_COUNT" "$REQUIRED_FILES_SHA256"
printf '  Full-tree manifest:  %s file(s), sha256=%s (byte-for-byte check of RELEASE_DIR)\n' \
    "$TREE_MANIFEST_FILE_COUNT" "$TREE_MANIFEST_SHA256"
printf '  Runtime root:        %s\n' "$RUNTIME_ROOT"
printf '  Worktree clean:      %s\n\n' "$SOURCE_WORKTREE_CLEAN"

printf 'Type the FULL commit hash to approve this deployment: '
IFS= read -r CONFIRMED_COMMIT

[ "$CONFIRMED_COMMIT" = "$FULL_COMMIT" ] \
    || die "Typed approval did not match the full commit hash"

UPLOAD_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REMOTE_PARTIAL="${DEPLOY_ROOT}/incoming/${FULL_COMMIT}.${UPLOAD_TOKEN}.tar.part"
REMOTE_REQUIRED_FILES_PARTIAL="${DEPLOY_ROOT}/incoming/${FULL_COMMIT}.${UPLOAD_TOKEN}.required_files.txt.part"
REMOTE_TREE_MANIFEST_PARTIAL="${DEPLOY_ROOT}/incoming/${FULL_COMMIT}.${UPLOAD_TOKEN}.full_tree_manifest.txt.part"

ssh "$DEPLOY_HOST" "
    set -eu
    test \"\$(id -u)\" = 0
    install -d -m 0700 \
        '${DEPLOY_ROOT}/incoming' \
        '${DEPLOY_ROOT}/artifacts' \
        '${DEPLOY_ROOT}/releases'
"

scp "$LOCAL_ARTIFACT" "${DEPLOY_HOST}:${REMOTE_PARTIAL}"
scp "$LOCAL_REQUIRED_FILES" "${DEPLOY_HOST}:${REMOTE_REQUIRED_FILES_PARTIAL}"
scp "$LOCAL_TREE_MANIFEST" "${DEPLOY_HOST}:${REMOTE_TREE_MANIFEST_PARTIAL}"

ssh "$DEPLOY_HOST" bash -s -- \
    "$FULL_COMMIT" \
    "$TREE_SHA" \
    "$ARTIFACT_SHA256" \
    "$SOURCE_BRANCH" \
    "$COMMIT_TIME" \
    "$SOURCE_WORKTREE_CLEAN" \
    "$DEPLOYED_BY" \
    "$REMOTE_PARTIAL" \
    "$REMOTE_REQUIRED_FILES_PARTIAL" \
    "$REQUIRED_FILES_SHA256" \
    "$REMOTE_TREE_MANIFEST_PARTIAL" \
    "$TREE_MANIFEST_SHA256" <<'REMOTE_BASH'
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
REMOTE_REQUIRED_FILES_PARTIAL="$9"
EXPECTED_REQUIRED_FILES_SHA="${10}"
REMOTE_TREE_MANIFEST_PARTIAL="${11}"
EXPECTED_TREE_MANIFEST_SHA="${12}"

RUNTIME_ROOT="/root/AI-Stock-Radar"
DEPLOY_ROOT="${RUNTIME_ROOT}/deploy"
INCOMING_DIR="${DEPLOY_ROOT}/incoming"
ARTIFACTS_DIR="${DEPLOY_ROOT}/artifacts"
RELEASES_DIR="${DEPLOY_ROOT}/releases"
RELEASE_DIR="${RELEASES_DIR}/${FULL_COMMIT}"
CURRENT_LINK="${DEPLOY_ROOT}/current"
ROOT_MANIFEST="${RUNTIME_ROOT}/DEPLOYED_VERSION.txt"

# LIVE_GUARD_DIRECTORY -- NOT under RUNTIME_ROOT (2026-08-29 correction).
# src/live/position_state.py's own _GUARD_DIRECTORY resolves
# $AI_STOCK_RADAR_GUARD_DIR if set, else Path.home()/".ai_stock_radar_guard".
# The MAIN live system's crontab/.env sets no such variable (confirmed:
# `grep guard .env` on the real host is empty), so the real, running
# code falls back to root's home directory -- /root/.ai_stock_radar_guard
# -- never /root/AI-Stock-Radar/.guard. Direct, decisive confirmation
# this repo's own docs/control_arm_crontab_v3.draft (lines 147-155):
# control-arm's OWN crontab explicitly sets
# AI_STOCK_RADAR_GUARD_DIR=/root/AI-Stock-Radar-Control/.guard
# specifically "without this it would default to
# $HOME/.ai_stock_radar_guard/ and silently collide with the live
# deployment's own guard file, since both run as root on the same
# server" -- i.e. control-arm had to move ITS OWN guard dir aside
# precisely because the main live system already occupies the default,
# untouched, out-of-repo location. Deliberately outside RUNTIME_ROOT
# entirely (position_state.py's own docstring: "A location the deploy
# mechanism never touches is the only way this guard can do its job" --
# an rsync/deploy that could reach this path could also clobber the
# rollback high-water-mark it exists to protect).
LIVE_GUARD_DIRECTORY="/root/.ai_stock_radar_guard"

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

case "$REMOTE_REQUIRED_FILES_PARTIAL" in
    "${INCOMING_DIR}/"*.required_files.txt.part) ;;
    *) die "Unexpected incoming required-files list path: $REMOTE_REQUIRED_FILES_PARTIAL" ;;
esac

case "$EXPECTED_REQUIRED_FILES_SHA" in
    *[!0-9a-f]*|'') die "Invalid required-files SHA-256" ;;
esac
[ "${#EXPECTED_REQUIRED_FILES_SHA}" -eq 64 ] \
    || die "Required-files SHA-256 must contain 64 hexadecimal characters"

case "$REMOTE_TREE_MANIFEST_PARTIAL" in
    "${INCOMING_DIR}/"*.full_tree_manifest.txt.part) ;;
    *) die "Unexpected incoming full-tree manifest path: $REMOTE_TREE_MANIFEST_PARTIAL" ;;
esac

case "$EXPECTED_TREE_MANIFEST_SHA" in
    *[!0-9a-f]*|'') die "Invalid full-tree manifest SHA-256" ;;
esac
[ "${#EXPECTED_TREE_MANIFEST_SHA}" -eq 64 ] \
    || die "Full-tree manifest SHA-256 must contain 64 hexadecimal characters"

test "$(id -u)" = 0 || die "Remote deploy must run as root"
test -f "${RUNTIME_ROOT}/.env" || die "Live .env is missing"
test -x "${RUNTIME_ROOT}/.venv/bin/python3" || die "Live virtualenv is missing"
test -d "${RUNTIME_ROOT}/data/live" || die "Persistent data/live directory is missing"
test -d "${RUNTIME_ROOT}/logs" || die "Persistent logs directory is missing"
test -f "$REMOTE_PARTIAL" || die "Transferred artifact is missing"
test -f "$REMOTE_REQUIRED_FILES_PARTIAL" || die "Transferred required-files list is missing"
test -f "$REMOTE_TREE_MANIFEST_PARTIAL" || die "Transferred full-tree manifest is missing"

REMOTE_REQUIRED_FILES_SHA="$(sha256sum "$REMOTE_REQUIRED_FILES_PARTIAL" | awk '{print $1}')"
[ "$REMOTE_REQUIRED_FILES_SHA" = "$EXPECTED_REQUIRED_FILES_SHA" ] \
    || die "Transferred required-files list SHA-256 mismatch"
[ -s "$REMOTE_REQUIRED_FILES_PARTIAL" ] \
    || die "Transferred required-files list is empty -- refusing to proceed with a checklist that would verify nothing"

REMOTE_TREE_MANIFEST_SHA="$(sha256sum "$REMOTE_TREE_MANIFEST_PARTIAL" | awk '{print $1}')"
[ "$REMOTE_TREE_MANIFEST_SHA" = "$EXPECTED_TREE_MANIFEST_SHA" ] \
    || die "Transferred full-tree manifest SHA-256 mismatch"
[ -s "$REMOTE_TREE_MANIFEST_PARTIAL" ] \
    || die "Transferred full-tree manifest is empty -- refusing to proceed with a byte-for-byte check that would verify nothing"

# LIVE_GUARD_DIRECTORY is deliberately NOT a fail-closed `test -d || die`
# check like the ones above -- position_state.py's own docstring states
# "a fresh server needs no special setup (the file is created on first
# successful save)". Self-heal (create only the empty directory scaffold,
# 0700, matching this script's own umask) if missing; NEVER touch/create
# the high_water_mark.json or lock file inside it either way -- an
# existing guard file's content must survive completely untouched by a
# deploy, per that same module's own explicit design rationale.
if [ ! -d "$LIVE_GUARD_DIRECTORY" ]; then
    install -d -m 0700 "$LIVE_GUARD_DIRECTORY"
fi

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

    # Required-file checklist, read from the crontab-derived list
    # transferred and SHA-256-verified above (REMOTE_REQUIRED_FILES_PARTIAL)
    # -- never a hand-maintained list here. Each line is the exact
    # `src/*`/`scripts/*` relative path
    # scripts/derive_required_files_from_crontab.py discovered by
    # tracing every real crontab entry point's own import chain at this
    # exact release commit. A blank line (a trailing newline in the
    # transferred file) is skipped rather than treated as a bogus
    # zero-length required path.
    while IFS= read -r REQUIRED_FILE || [ -n "$REQUIRED_FILE" ]; do
        [ -n "$REQUIRED_FILE" ] || continue
        test -f "${APP_DIR}/${REQUIRED_FILE}" \
            || die "Module missing from archive (crontab-derived required-file list): ${REQUIRED_FILE}"
    done < "$REMOTE_REQUIRED_FILES_PARTIAL"

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
    # Convenience symlink only -- none of the 5 real crontab entry
    # points read a relative `.guard` path; position_state.py resolves
    # $AI_STOCK_RADAR_GUARD_DIR/$HOME directly via the Python process's
    # own environment, never through this app directory. Points at the
    # REAL location (LIVE_GUARD_DIRECTORY, see its own definition above)
    # so a future relative-path consumer run from here (e.g. a dashboard
    # snapshot builder, per src/dashboard/snapshot_builder.py's own
    # `.guard/order_intents/*.json` read pattern, if ever deployed here)
    # resolves to the SAME directory the live code actually writes to.
    ln -s "$LIVE_GUARD_DIRECTORY" "${APP_DIR}/.guard"
    ln -s "${RUNTIME_ROOT}/logs" "${APP_DIR}/logs"
    ln -s "${RUNTIME_ROOT}/data/live" "${APP_DIR}/data/live"

    DEPLOYED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    MANIFEST_TMP="${APP_DIR}/.DEPLOYED_VERSION.txt.tmp.$$"

    {
        printf 'schema_version=1\n'
        printf 'component=live-daily-decision\n'
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

# BYTE-FOR-BYTE verification of RELEASE_DIR against the full-tree
# manifest (independent-audit finding, 2026-08-29) -- runs for BOTH
# branches above (fresh extraction AND idempotent reuse of an existing
# release directory). This is deliberately NOT folded into either
# branch alone: an earlier, partially-failed deploy attempt could in
# principle leave a RELEASE_DIR whose own DEPLOYED_VERSION.txt fields
# happen to already match (commit/tree/artifact SHA all correct) while
# its actual file CONTENTS do not -- the required-file spot-check a
# few lines above this block would not catch that (it only proves a
# curated subset of paths exist, never that their content, or every
# OTHER file's content, is byte-correct). This is the check that
# closes that gap, for every tracked file, not a subset -- the same
# category of gap already seen once this session (a stale file
# discovered at a path this script has never written to at all,
# RUNTIME_ROOT's own top level -- a different bug from this one, but
# the same root worry: "does what's on disk really match what was
# approved").
TREE_MANIFEST_PATH="${RELEASE_DIR}/FULL_TREE_MANIFEST.sha256"
if [ ! -f "$TREE_MANIFEST_PATH" ]; then
    cp -- "$REMOTE_TREE_MANIFEST_PARTIAL" "$TREE_MANIFEST_PATH"
    chmod 0444 "$TREE_MANIFEST_PATH"
fi

(
    cd "$RELEASE_DIR" || die "Could not enter $RELEASE_DIR for byte-for-byte verification"
    sha256sum --quiet -c "$TREE_MANIFEST_PATH"
) || die "RELEASE_DIR ($RELEASE_DIR) does not byte-for-byte match the approved release's own git tree -- see the sha256sum output above for which file(s) diverged. Refusing to activate this release."

printf 'Byte-for-byte verification passed: every file in %s matches the approved release tree.\n' "$RELEASE_DIR"

# --- Sync verified release CODE into RUNTIME_ROOT's own top level ---
# (2026-08-29, owner-approved -- the fourth incident this session in
# the "release content vs what cron actually executes" category, this
# one root-caused conclusively: the REAL crontab, raw `crontab -l`
# output provided directly by the owner, does `cd /root/AI-Stock-Radar
# && ... scripts/...` -- RUNTIME_ROOT's own top level, NEVER
# deploy/current. This predates every deploy script built this
# session (run_daily_decision.py has been executing successfully from
# there for months, including the real CEG-incident fixes); whatever
# mechanism used to keep it in sync is not part of this repository.
#
# Rather than change the crontab (a standing-configuration change
# needing its own separate, explicit owner sign-off -- not something
# to fold into a deploy script's own unilateral behavior) or abandon
# the immutable-release/rollback/byte-for-byte-checksum properties
# built up over the three prior incidents tonight, this step syncs the
# ALREADY byte-for-byte-verified RELEASE_DIR's own code into
# RUNTIME_ROOT, strictly ON TOP of every guarantee already established
# above -- this is the LAST step before activation, never a substitute
# for any earlier check.
#
# Scope is deliberately narrow: scripts/, src/, config/ ONLY -- the
# exact directories derive_required_files_from_crontab.py's own real
# crontab-derived trace ever reaches (this repo's own docs/, tests/,
# README.md etc. are never referenced by any real cron entry point, so
# they are never synced -- RUNTIME_ROOT's top level does not need to be
# a full mirror of the release, only the parts cron actually imports).
# `.env`/`.env.control`/`.venv`/`data/`/`data/live/`/`logs/`/`.guard/`
# are NEVER in this list and NEVER will be -- the exact same persistent-
# state discipline the archive-exclusion checks above already enforce
# for the release tree itself, now extended to this sync step too.
command -v rsync >/dev/null 2>&1 \
    || die "rsync is required for the RUNTIME_ROOT code sync step but is not installed on this host"

for SYNC_DIR in scripts src config; do
    test -d "${RELEASE_DIR}/${SYNC_DIR}" \
        || die "Verified release is missing its own ${SYNC_DIR}/ directory -- cannot sync what does not exist"
    rsync -a --delete -- "${RELEASE_DIR}/${SYNC_DIR}/" "${RUNTIME_ROOT}/${SYNC_DIR}/" \
        || die "rsync failed while syncing ${SYNC_DIR}/ into RUNTIME_ROOT -- RUNTIME_ROOT may now be in a PARTIAL state; investigate before retrying, do not assume the prior deploy is still intact"
done
sync "$RUNTIME_ROOT"

# Final check: RUNTIME_ROOT's own scripts/, src/, config/ now
# byte-for-byte match the verified release -- reuses the SAME
# full-tree manifest already built and verified against RELEASE_DIR
# above (filtered to just these three prefixes), never a separate,
# independent trust point. This is what directly answers "does
# RUNTIME_ROOT/scripts/send_status_update.py (or any other synced
# file) really match the new release" -- not by spot-checking one
# named file, but by checking every synced file the same way
# RELEASE_DIR itself was checked.
RUNTIME_SYNC_MANIFEST="${RUNTIME_ROOT}/.sync_verify_manifest.$$.sha256"
grep -E '^[^[:space:]]+  \./(scripts|src|config)/' "$TREE_MANIFEST_PATH" > "$RUNTIME_SYNC_MANIFEST" || true
[ -s "$RUNTIME_SYNC_MANIFEST" ] \
    || die "Scoped sync-verification manifest is empty -- no scripts/src/config entries found in the full-tree manifest, which should be impossible given the required-file checklist. Refusing to declare the sync verified."

(
    cd "$RUNTIME_ROOT" || die "Could not enter $RUNTIME_ROOT for sync verification"
    sha256sum --quiet -c "$RUNTIME_SYNC_MANIFEST"
) || die "RUNTIME_ROOT's synced scripts/, src/, config/ do NOT byte-for-byte match the verified release -- see the sha256sum output above for which file(s) diverged. RUNTIME_ROOT may be in a partial/inconsistent state -- investigate before retrying."

RUNTIME_SYNC_FILE_COUNT="$(grep -cE '^[^[:space:]]+  \./(scripts|src|config)/' "$TREE_MANIFEST_PATH")"
rm -f -- "$RUNTIME_SYNC_MANIFEST"
printf 'RUNTIME_ROOT sync verified: %s file(s) in scripts/, src/, config/ byte-for-byte match the release.\n' "$RUNTIME_SYNC_FILE_COUNT"

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
printf 'Byte-for-byte verified against: %s (every tracked file, not a spot check)\n' "$TREE_MANIFEST_PATH"
printf 'RUNTIME_ROOT sync (scripts/, src/, config/): VERIFIED, %s file(s) byte-for-byte match (what the real crontab actually executes)\n' \
    "$RUNTIME_SYNC_FILE_COUNT"
cat "$ROOT_MANIFEST"
REMOTE_BASH
