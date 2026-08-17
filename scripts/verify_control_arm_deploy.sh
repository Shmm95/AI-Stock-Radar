#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: APPROVED_COMMIT=<hash> LOCAL_REPO=/path DEPLOY_HOST=root@host \
#        scripts/verify_control_arm_deploy.sh

: "${APPROVED_COMMIT:?Set APPROVED_COMMIT}"
: "${LOCAL_REPO:?Set LOCAL_REPO}"
: "${DEPLOY_HOST:?Set DEPLOY_HOST}"

FULL_COMMIT="$(
    git -C "$LOCAL_REPO" rev-parse --verify "${APPROVED_COMMIT}^{commit}"
)"
EXPECTED_TREE="$(
    git -C "$LOCAL_REPO" rev-parse "${FULL_COMMIT}^{tree}"
)"

ssh "$DEPLOY_HOST" bash -s -- "$FULL_COMMIT" "$EXPECTED_TREE" <<'VERIFY'
set -Eeuo pipefail

EXPECTED_COMMIT="$1"
EXPECTED_TREE="$2"

RUNTIME_ROOT="/root/AI-Stock-Radar-Control"
DEPLOY_ROOT="${RUNTIME_ROOT}/deploy"
MANIFEST="${RUNTIME_ROOT}/DEPLOYED_VERSION.txt"
EXPECTED_RELEASE="${DEPLOY_ROOT}/releases/${EXPECTED_COMMIT}"

test -L "$MANIFEST"
test "$(readlink -f "${DEPLOY_ROOT}/current")" = "$EXPECTED_RELEASE"
test "$(readlink -f "$MANIFEST")" = "${EXPECTED_RELEASE}/DEPLOYED_VERSION.txt"

grep -Fqx "commit_sha=${EXPECTED_COMMIT}" "$MANIFEST"
grep -Fqx "git_tree_sha=${EXPECTED_TREE}" "$MANIFEST"

ARTIFACT_PATH="$(awk -F= '$1 == "artifact_path" {print $2}' "$MANIFEST")"
EXPECTED_SHA="$(awk -F= '$1 == "artifact_sha256" {print $2}' "$MANIFEST")"

test -f "$ARTIFACT_PATH"
test "$(sha256sum "$ARTIFACT_PATH" | awk '{print $1}')" = "$EXPECTED_SHA"
test "$(git get-tar-commit-id < "$ARTIFACT_PATH")" = "$EXPECTED_COMMIT"

printf 'DEPLOY PROVENANCE VERIFIED\n'
cat "$MANIFEST"
VERIFY
