#!/usr/bin/env bash
set -Eeuo pipefail

# Runs the curated "safety subset" of tests -- see tests/SAFETY_SUBSET.txt
# for what's in it and why (independent-audit finding, 2026-08-24, item 6).
#
# Usage: scripts/run_safety_subset_tests.sh [extra pytest args...]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/tests/SAFETY_SUBSET.txt"

test -f "$MANIFEST"

# Portable read loop, not `mapfile` -- macOS ships bash 3.2, which
# predates `mapfile` (added in bash 4.0).
TEST_PATHS=()
while IFS= read -r line; do
    TEST_PATHS+=("$line")
done < <(grep -v '^\s*#' "$MANIFEST" | grep -v '^\s*$')

for path in "${TEST_PATHS[@]}"; do
    test -f "${REPO_ROOT}/${path}" || {
        echo "safety subset manifest names a file that no longer exists: ${path}" >&2
        exit 1
    }
done

cd "$REPO_ROOT"
exec "${REPO_ROOT}/.venv/bin/python" -m pytest "${TEST_PATHS[@]}" "$@"
