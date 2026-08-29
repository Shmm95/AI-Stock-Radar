#!/usr/bin/env python3
"""Derives the deploy-time required-file checklist automatically from
the REAL, remote crontab -- replaces a hand-maintained file list.

WHY: `deploy_live_decision_release.sh`'s own required-file checklist
was, until this script existed, a manually-built, hand-maintained list
(31 files, itself already verified once via a real `git archive` +
extraction test against the actual crontab -- see that script's own
"REQUIRED-FILE CHECKLIST" comment for the evidence trail). A hand-kept
list can silently drift out of sync the moment the crontab or an
entry point's own import chain changes and nobody remembers to update
it by hand. This script removes that manual-sync requirement
structurally: given the REAL crontab text, it extracts every
`scripts/*.py` path actually invoked, then computes each one's own
FULL transitive import closure by parsing real source (Python `ast`,
never `grep`/regex-on-code) -- from a SPECIFIC commit's own git tree
(`git show <commit>:<path>`), never the working tree, so the computed
list always reflects exactly what THAT release's own code needs, never
whatever happens to be checked out locally at derivation time (a
lesson from an earlier draft of the hand-built list, which was once
traced against the wrong tree and produced a wrong answer).

Usage:
    python3 scripts/derive_required_files_from_crontab.py \\
        --commit <full-40-char-sha> --crontab-file /path/to/crontab.txt

Prints the sorted, deduplicated required-file list, one path per line,
to stdout. Exits non-zero (fail-closed, never a silent empty result)
if the crontab text contains no `scripts/*.py` reference at all, if
any referenced script does not exist at the given commit, or if any
reachable file fails to parse as Python.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys

# Matches a `scripts/<...>.py` path anywhere in the crontab text,
# regardless of exact invocation form (venv-prefixed
# `.venv/bin/python3 scripts/x.py`, plain `python3 scripts/x.py`, or
# just the bare path). Crontab lines that are env-var assignments,
# comments, or non-Python commands (curl, healthchecks pings) never
# contain this pattern, so no separate filtering step is needed.
_SCRIPT_PATH_PATTERN = re.compile(r"scripts/[A-Za-z0-9_./\-]+\.py")


def extract_script_paths_from_crontab(crontab_text: str) -> list[str]:
    """Every distinct `scripts/*.py` path referenced anywhere in the
    real crontab text -- these are the entry points whose own import
    chains get traced below."""
    return sorted(set(_SCRIPT_PATH_PATTERN.findall(crontab_text)))


def _git_show(commit: str, path: str) -> str | None:
    """The exact content of `path` at `commit`'s own git tree, or
    `None` if it does not exist there -- NEVER reads the working tree,
    so the result is correct regardless of what happens to be checked
    out locally."""
    proc = subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def _module_to_path(commit: str, module: str) -> str | None:
    if module.startswith("src.") or module.startswith("scripts."):
        relative = module.replace(".", "/")
    elif "." not in module:
        # Bare, unprefixed name (e.g. `import generate_performance_report`
        # or `from generate_performance_report import X`) -- this
        # codebase's own established convention (documented in multiple
        # scripts/ entry points: running a file directly puts only
        # scripts/ on sys.path[0]) means a bare top-level name can be a
        # real sibling module under scripts/, not just a stdlib/
        # third-party import. Only treated as a project file if
        # `scripts/<name>.py` genuinely exists at this commit -- a bare
        # `import json`/`import argparse` simply won't resolve here and
        # is correctly ignored.
        candidate = f"scripts/{module}.py"
        return candidate if _git_show(commit, candidate) is not None else None
    else:
        return None
    for candidate in (relative + ".py", relative + "/__init__.py"):
        if _git_show(commit, candidate) is not None:
            return candidate
    return None


def _find_project_imports(content: str, *, path: str) -> set[str]:
    """Every module name this file imports that could plausibly be a
    project file -- `src.*`/`scripts.*` prefixed names in either
    `import x.y.z` or `from x.y import z` form (the latter additionally
    records `x.y.z`, the imported name treated as a possible submodule,
    matching `_module_to_path`'s own resolution, so `from src.live
    import position_state` resolves `position_state` correctly as a
    submodule, not just the package `src.live` itself) -- PLUS every
    bare, unprefixed `import x`/`from x import y` name, which
    `_module_to_path` resolves against `scripts/<x>.py` and silently
    discards if it doesn't exist there (stdlib/third-party bare imports
    are never project files, so they resolve to nothing and are
    dropped, never mistaken for one)."""
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError as error:
        raise RuntimeError(f"Failed to parse {path}: {error}") from error
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.") or alias.name.startswith("scripts.") or "." not in alias.name:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # explicit relative import (`from . import x`) -- not used anywhere in this codebase, skipped rather than mis-resolved
            if node.module and (node.module.startswith("src.") or node.module.startswith("scripts.")):
                found.add(node.module)
                for alias in node.names:
                    found.add(f"{node.module}.{alias.name}")
            elif node.module and "." not in node.module:
                found.add(node.module)
    return found


def compute_required_files(commit: str, entry_scripts: list[str]) -> list[str]:
    """The full transitive closure of every entry script's own real
    import chain, at the exact given commit's own tree. Fails closed
    (raises) rather than silently producing a partial list if any
    crontab-referenced entry script does not exist at that commit --
    that would itself be a real, worth-surfacing problem (a crontab
    referencing a script the release doesn't contain), never something
    to quietly skip."""
    visited: set[str] = set()
    queue: list[str] = list(entry_scripts)
    missing_entries: list[str] = []
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        content = _git_show(commit, current)
        if content is None:
            missing_entries.append(current)
            continue
        visited.add(current)
        for module in _find_project_imports(content, path=current):
            resolved = _module_to_path(commit, module)
            if resolved and resolved not in visited:
                queue.append(resolved)
    if missing_entries:
        raise RuntimeError(
            f"Crontab references {len(missing_entries)} script(s) not found at commit "
            f"{commit}: {sorted(missing_entries)}. Refusing to produce a required-file "
            f"list that would silently omit them."
        )
    return sorted(visited)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", required=True, help="Full commit SHA to trace imports against.")
    parser.add_argument("--crontab-file", required=True, type=argparse.FileType("r"), help="Path to the real crontab text (e.g. from `crontab -l`), or '-' for stdin.")
    arguments = parser.parse_args()

    crontab_text = arguments.crontab_file.read()
    entry_scripts = extract_script_paths_from_crontab(crontab_text)
    if not entry_scripts:
        print(
            "No scripts/*.py path found anywhere in the given crontab text -- refusing "
            "to produce an empty required-file list (that would make the deploy's own "
            "sanity check a silent no-op).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        required_files = compute_required_files(arguments.commit, entry_scripts)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    for path in required_files:
        print(path)


if __name__ == "__main__":
    main()
