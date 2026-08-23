"""Tests for `scripts/run_missing_equity_session_replay.py` -- the CLI
entry point (item 5 of the approved missing-session-replay scope). A
thin wrapper only: these tests confirm the wiring (argument parsing,
STOP-flag short circuit, exit-code-on-failure via re-raise, JSON
summary excludes the non-serializable `raw_bars_by_ticker`), not
`run_missing_session_replay`'s own logic (already covered by
`test_equity_session_orchestrator.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.run_daily_decision as rdd
import scripts.run_missing_equity_session_replay as cli
import src.live.equity_session_detection as detection
from src.live.equity_session_orchestrator import OrchestratorResult


def test_stop_flag_skips_entirely(tmp_path, monkeypatch, capsys):
    stop_flag = tmp_path / "STOP"
    stop_flag.write_text("stop", encoding="utf-8")
    monkeypatch.setattr(rdd, "STOP_FLAG_PATH", stop_flag)
    monkeypatch.setattr(rdd, "_notify_safe", lambda text: True)
    monkeypatch.setattr(
        cli, "run_missing_session_replay",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not be called when STOP is set")),
    )
    monkeypatch.setattr(sys, "argv", ["run_missing_equity_session_replay.py"])
    cli.main()
    assert "STOP flag detected" in capsys.readouterr().out


def test_default_argparse_paths_match_orchestrator_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rdd, "STOP_FLAG_PATH", tmp_path / "no-such-stop")
    captured = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return OrchestratorResult(outcome="CLEAN_NO_OP")

    monkeypatch.setattr(cli, "run_missing_session_replay", _fake)
    monkeypatch.setattr(sys, "argv", ["run_missing_equity_session_replay.py"])
    cli.main()

    assert captured["state_path"] == rdd.DEFAULT_STATE_PATH
    assert captured["decision_log_directory"] == rdd.DEFAULT_DECISION_LOG_DIRECTORY
    assert captured["guard_path"] == rdd.ps.HIGH_WATER_MARK_PATH


def test_result_summary_json_omits_raw_bars_by_ticker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rdd, "STOP_FLAG_PATH", tmp_path / "no-such-stop")

    class _UnserializableFrame:
        pass

    detection_result = detection.DetectionResult(
        expected_sessions=("2026-08-20",), complete_sessions=("2026-08-20",),
        missing_by_session={}, unexpected_future_bars={},
        raw_bars_by_ticker={"AAPL": _UnserializableFrame()},
    )
    monkeypatch.setattr(
        cli, "run_missing_session_replay",
        lambda **k: OrchestratorResult(outcome="CLEAN_NO_OP", detection=detection_result),
    )
    monkeypatch.setattr(sys, "argv", ["run_missing_equity_session_replay.py"])
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["outcome"] == "CLEAN_NO_OP"
    assert payload["detection"]["expected_sessions"] == ["2026-08-20"]
    assert "raw_bars_by_ticker" not in payload["detection"]


def test_fail_closed_error_is_notified_and_reraised(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rdd, "STOP_FLAG_PATH", tmp_path / "no-such-stop")
    notified = []
    monkeypatch.setattr(rdd, "_notify_safe", lambda text: notified.append(text) or True)

    from src.live.equity_session_orchestrator import SessionCursorAheadOfCalendarError

    def _boom(**kwargs):
        raise SessionCursorAheadOfCalendarError("cursor ahead of calendar")

    monkeypatch.setattr(cli, "run_missing_session_replay", _boom)
    monkeypatch.setattr(sys, "argv", ["run_missing_equity_session_replay.py"])

    with pytest.raises(SessionCursorAheadOfCalendarError):
        cli.main()

    assert len(notified) == 1
    assert "FAIL-CLOSED" in notified[0]
