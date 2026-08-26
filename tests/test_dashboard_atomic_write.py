"""Tests for scripts/generate_dashboard_snapshot.py -- the atomic-write
contract and the fail-closed-on-core-failure rule (owner's own spec,
2026-08-25): a core (broker P&L/account) failure must NEVER overwrite
an existing good snapshot with a broken/partial one, and the process
must exit non-zero when that happens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_dashboard_snapshot as gds  # noqa: E402

from src.dashboard import models  # noqa: E402
from src.dashboard import snapshot_builder as sb  # noqa: E402


class _FakeAccount:
    cash = "1000.00"
    equity = "1000.00"
    last_equity = "1000.00"
    buying_power = "2000.00"
    account_number = "PA000TEST"
    status = "ACTIVE"
    trading_blocked = False
    transfers_blocked = False
    account_blocked = False


class _FakeClient:
    def get_account(self):
        return _FakeAccount()

    def get_all_positions(self):
        return []

    def get_orders(self, *, filter=None):
        return []


def _run_main(monkeypatch, *, argv):
    monkeypatch.setattr(sys, "argv", argv)
    gds.main()


def test_successful_run_writes_a_real_atomic_schema_valid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "_build_trading_client", lambda: _FakeClient())
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard"))
    output_path = tmp_path / "output" / "dashboard_snapshot_v1.json"

    _run_main(
        monkeypatch,
        argv=[
            "generate_dashboard_snapshot.py",
            "--output-path", str(output_path),
            "--state-path", str(tmp_path / "position_state.json"),
            "--guard-path", str(tmp_path / "hwm.json"),
            "--decision-log-directory", str(tmp_path / "decisions"),
            "--pass-gate-directory", str(tmp_path / "pass_gate"),
        ],
    )

    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["collection_status"] == models.COLLECTION_STATUS_OK
    assert payload["schema_version"] == 1
    # No leftover temp file from the atomic-write dance.
    leftovers = list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    assert leftovers == []


def test_core_failure_does_not_overwrite_existing_snapshot_and_exits_nonzero(tmp_path, monkeypatch):
    output_path = tmp_path / "output" / "dashboard_snapshot_v1.json"
    output_path.parent.mkdir(parents=True)
    original_content = json.dumps({"schema_version": 1, "collection_status": "OK", "marker": "ORIGINAL_GOOD_SNAPSHOT"})
    output_path.write_text(original_content, encoding="utf-8")

    def _broken_client():
        raise RuntimeError("ALPACA_API_KEY not set")

    monkeypatch.setattr(sb, "_build_trading_client", _broken_client)
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard"))

    with pytest.raises(SystemExit) as exc_info:
        _run_main(
            monkeypatch,
            argv=[
                "generate_dashboard_snapshot.py",
                "--output-path", str(output_path),
                "--state-path", str(tmp_path / "position_state.json"),
                "--guard-path", str(tmp_path / "hwm.json"),
            ],
        )

    assert exc_info.value.code != 0
    # The ORIGINAL file must be completely untouched -- byte for byte.
    assert output_path.read_text(encoding="utf-8") == original_content


def test_core_failure_with_no_prior_snapshot_writes_nothing_at_all(tmp_path, monkeypatch):
    output_path = tmp_path / "output" / "dashboard_snapshot_v1.json"  # never created

    def _broken_client():
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(sb, "_build_trading_client", _broken_client)
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard"))

    with pytest.raises(SystemExit) as exc_info:
        _run_main(
            monkeypatch,
            argv=[
                "generate_dashboard_snapshot.py",
                "--output-path", str(output_path),
                "--state-path", str(tmp_path / "position_state.json"),
                "--guard-path", str(tmp_path / "hwm.json"),
            ],
        )

    assert exc_info.value.code != 0
    assert not output_path.exists()


def test_partial_result_is_still_written(tmp_path, monkeypatch):
    """A PARTIAL result (core succeeded, a secondary section failed) IS
    written -- only a full core failure withholds the write."""
    monkeypatch.setattr(sb, "_build_trading_client", lambda: _FakeClient())
    monkeypatch.setattr(sb, "build_system_view", lambda **kwargs: (None, "SimulatedSystemFailure"))
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard"))
    output_path = tmp_path / "output" / "dashboard_snapshot_v1.json"

    _run_main(
        monkeypatch,
        argv=[
            "generate_dashboard_snapshot.py",
            "--output-path", str(output_path),
            "--state-path", str(tmp_path / "position_state.json"),
            "--guard-path", str(tmp_path / "hwm.json"),
        ],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["collection_status"] == models.COLLECTION_STATUS_PARTIAL
    assert any(e["section"] == "system" for e in payload["errors"])


def test_error_message_never_leaks_raw_exception_text_to_stdout_or_file(tmp_path, monkeypatch, capsys):
    def _broken_client():
        raise RuntimeError("SECRET_CREDENTIAL_VALUE_ABC123")

    monkeypatch.setattr(sb, "_build_trading_client", _broken_client)
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard"))
    output_path = tmp_path / "output" / "dashboard_snapshot_v1.json"

    with pytest.raises(SystemExit):
        _run_main(
            monkeypatch,
            argv=[
                "generate_dashboard_snapshot.py",
                "--output-path", str(output_path),
                "--state-path", str(tmp_path / "position_state.json"),
                "--guard-path", str(tmp_path / "hwm.json"),
            ],
        )

    captured = capsys.readouterr()
    assert "SECRET_CREDENTIAL_VALUE_ABC123" not in captured.err
    assert "SECRET_CREDENTIAL_VALUE_ABC123" not in captured.out
    assert "RuntimeError" in captured.err
