"""Regression test for the load_dotenv override fix in
src/agents/analyst_agent.py.

Scoped narrowly to this fix, not a full module test suite -- see
tests/test_telegram_notifier.py for the original bug reproduction this
mirrors: python-dotenv's load_dotenv() defaults to override=False, which
skips a key already present in os.environ even if it's an empty string.
Applied consistently here for the same latent risk, even though this
module is unrelated to the live-trading/notification path.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.agents import analyst_agent


def test_module_load_dotenv_call_uses_override_true():
    source = Path(analyst_agent.__file__).read_text(encoding="utf-8")
    assert "load_dotenv(ENV_PATH, override=True)" in source


def test_stale_empty_env_var_is_not_overridden_without_override_true(
    monkeypatch, tmp_path: Path
):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=REAL_KEY_FROM_ENV_FILE\n")
    monkeypatch.setenv("OPENAI_API_KEY", "")  # simulates the stale inherited var

    load_dotenv(env_file)  # the OLD call shape (override defaults to False)

    assert os.environ["OPENAI_API_KEY"] == "", (
        "demonstrates the bug: without override=True, the real .env value "
        "never overwrites the stale empty variable"
    )


def test_override_true_fixes_the_stale_empty_env_var(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=REAL_KEY_FROM_ENV_FILE\n")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    load_dotenv(env_file, override=True)  # the FIXED call shape

    assert os.environ["OPENAI_API_KEY"] == "REAL_KEY_FROM_ENV_FILE"
