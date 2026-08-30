"""Tests for src/dashboard/http_app.py -- the read-only, GET-only HTTP
contract (owner's own spec, 2026-08-25): exactly three routes, every
other method/path is 404/405, security headers on every response,
503 on a missing/corrupt snapshot, and a staleness flag once the
snapshot is too old.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.dashboard import http_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "dashboard_snapshot_v1.json"
    timeseries_path = tmp_path / "dashboard_timeseries_v2.json"
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_PATH", str(snapshot_path))
    monkeypatch.setenv("DASHBOARD_TIMESERIES_PATH", str(timeseries_path))
    return TestClient(http_app.app), snapshot_path, timeseries_path


def _write_timeseries(path: Path, *, points: list[dict] | None = None, **extra) -> None:
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "collection_status": "OK",
        "portfolio_series": points if points is not None else [
            {"valuation_date": "2026-08-07", "equity_usd": "100000.00"},
            {"valuation_date": "2026-08-08", "equity_usd": "101250.50"},
        ],
    }
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_snapshot(path: Path, *, generated_at: datetime | None = None, **extra) -> None:
    generated_at = generated_at or datetime.now(UTC)
    payload = {"schema_version": 1, "generated_at_utc": generated_at.isoformat(), "collection_status": "OK"}
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_index_serves_html_with_inlined_css_and_js(client):
    test_client, snapshot_path, timeseries_path = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "__DASHBOARD_CSS__" not in response.text
    assert "__DASHBOARD_JS__" not in response.text
    assert "__DASHBOARD_NONCE__" not in response.text
    assert "<style nonce=" in response.text and "</style>" in response.text
    assert "<script nonce=" in response.text and "</script>" in response.text


def test_index_inline_tags_carry_a_nonce_matching_the_csp_header(client):
    """2026-08-30 incident: index.html's <style>/<script> were plain
    inline tags with no nonce and no style-src/script-src CSP override
    -- the browser's own default-src 'self' fallback (correctly, no
    'unsafe-inline' anywhere) refused to run either one, so the page
    never rendered past "Loading...". This is the fix's own contract:
    the CSP header's nonce and the two tags' nonce attribute must be
    the SAME value, and must change on every request (never a fixed,
    guessable value -- that would defeat the point of a nonce)."""
    test_client, snapshot_path, timeseries_path = client
    first = test_client.get("/")
    second = test_client.get("/")
    for response in (first, second):
        csp = response.headers["content-security-policy"]
        assert "'unsafe-inline'" not in csp
        assert "default-src 'self'" in csp
        match = re.search(r"style-src 'self' 'nonce-([^']+)'", csp)
        assert match, csp
        nonce = match.group(1)
        assert f'<style nonce="{nonce}">' in response.text
        assert f'<script nonce="{nonce}">' in response.text
        assert f"'nonce-{nonce}'" in csp.split("script-src")[1]
    assert re.search(r"nonce-([^']+)", first.headers["content-security-policy"]).group(1) != re.search(
        r"nonce-([^']+)", second.headers["content-security-policy"]
    ).group(1)


def test_healthz_returns_ok_without_reading_the_snapshot(client):
    test_client, snapshot_path, timeseries_path = client
    assert not snapshot_path.exists()
    response = test_client.get("/healthz")
    assert response.status_code == 200
    assert response.text == "ok"


def test_snapshot_route_returns_valid_snapshot_with_staleness_flag(client):
    test_client, snapshot_path, timeseries_path = client
    _write_snapshot(snapshot_path)
    response = test_client.get("/api/v1/snapshot")
    assert response.status_code == 200
    body = response.json()
    assert body["_dashboard_stale"] is False
    assert body["schema_version"] == 1


def test_snapshot_route_flags_stale_when_old(client):
    test_client, snapshot_path, timeseries_path = client
    _write_snapshot(snapshot_path, generated_at=datetime.now(UTC) - timedelta(minutes=15))
    response = test_client.get("/api/v1/snapshot")
    assert response.status_code == 200
    assert response.json()["_dashboard_stale"] is True


def test_snapshot_route_returns_503_when_missing(client):
    test_client, snapshot_path, timeseries_path = client
    assert not snapshot_path.exists()
    response = test_client.get("/api/v1/snapshot")
    assert response.status_code == 503


def test_snapshot_route_returns_503_when_corrupt(client):
    test_client, snapshot_path, timeseries_path = client
    snapshot_path.write_text("{ not valid json", encoding="utf-8")
    response = test_client.get("/api/v1/snapshot")
    assert response.status_code == 503


@pytest.mark.parametrize("path", ["/", "/api/v1/snapshot", "/healthz"])
@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_only_get_and_head_are_allowed_on_every_route(client, path, method):
    test_client, snapshot_path, timeseries_path = client
    _write_snapshot(snapshot_path)
    response = getattr(test_client, method)(path)
    assert response.status_code == 405


@pytest.mark.parametrize("path", ["/", "/api/v1/snapshot", "/healthz"])
def test_head_works_on_every_route(client, path):
    test_client, snapshot_path, timeseries_path = client
    _write_snapshot(snapshot_path)
    response = test_client.head(path)
    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/nonexistent", "/api/v1/other", "/static/dashboard.js", "/api/v2/snapshot"])
def test_undefined_paths_are_404(client, path):
    test_client, snapshot_path, timeseries_path = client
    response = test_client.get(path)
    assert response.status_code == 404


@pytest.mark.parametrize("path", ["/", "/api/v1/snapshot", "/healthz"])
def test_security_headers_present_on_every_route(client, path):
    test_client, snapshot_path, timeseries_path = client
    _write_snapshot(snapshot_path)
    response = test_client.get(path)
    headers = response.headers
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert "default-src 'self'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert "camera=()" in headers["permissions-policy"]
    assert "microphone=()" in headers["permissions-policy"]
    assert "geolocation=()" in headers["permissions-policy"]


def test_security_headers_present_even_on_503(client):
    test_client, snapshot_path, timeseries_path = client
    response = test_client.get("/api/v1/snapshot")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_app_only_registers_exactly_three_routes():
    paths = sorted(route.path for route in http_app.app.routes)
    assert paths == ["/", "/api/v1/snapshot", "/healthz"]


class TestEquityCurve:
    """Owner request, 2026-08-30: a minimal equity-curve widget fed from
    Dashboard V2 Phase 1's own public export (`dashboard_timeseries_v2.json`)
    -- a SECOND, independently-produced file, never the private SQLite
    store. This section is deliberately optional/additive: missing or
    corrupt must never turn the whole /api/v1/snapshot route into a 503,
    since the primary V1 data (account/P&L/positions) has nothing to do
    with whether Phase 1's collector has run yet."""

    def test_present_when_timeseries_file_has_real_points(self, client):
        test_client, snapshot_path, timeseries_path = client
        _write_snapshot(snapshot_path)
        _write_timeseries(timeseries_path)
        response = test_client.get("/api/v1/snapshot")
        assert response.status_code == 200
        curve = response.json()["equity_curve"]
        assert curve is not None
        assert curve["points"] == [
            {"date": "2026-08-07", "equity_usd": "100000.00"},
            {"date": "2026-08-08", "equity_usd": "101250.50"},
        ]
        assert curve["collection_status"] == "OK"

    def test_null_when_timeseries_file_missing(self, client):
        test_client, snapshot_path, timeseries_path = client
        _write_snapshot(snapshot_path)
        assert not timeseries_path.exists()
        response = test_client.get("/api/v1/snapshot")
        assert response.status_code == 200
        assert response.json()["equity_curve"] is None

    def test_null_when_timeseries_file_corrupt_never_503s_the_whole_route(self, client):
        test_client, snapshot_path, timeseries_path = client
        _write_snapshot(snapshot_path)
        timeseries_path.write_text("{ not valid json", encoding="utf-8")
        response = test_client.get("/api/v1/snapshot")
        assert response.status_code == 200
        body = response.json()
        assert body["equity_curve"] is None
        assert body["schema_version"] == 1  # primary snapshot data unaffected

    def test_null_when_portfolio_series_is_empty(self, client):
        test_client, snapshot_path, timeseries_path = client
        _write_snapshot(snapshot_path)
        _write_timeseries(timeseries_path, points=[])
        response = test_client.get("/api/v1/snapshot")
        assert response.json()["equity_curve"] is None

    def test_only_valuation_date_and_equity_usd_cross_into_the_payload(self, client):
        """The v2 export's own points also carry daily_return, drawdown,
        quality_status, etc. -- none of that should leak into this
        minimal widget payload, on principle (nothing else is needed to
        draw a line)."""
        test_client, snapshot_path, timeseries_path = client
        _write_snapshot(snapshot_path)
        _write_timeseries(timeseries_path, points=[
            {"valuation_date": "2026-08-07", "equity_usd": "100000.00", "daily_return": "0.01",
             "drawdown": "0", "quality_status": "OK", "cumulative_index": "1.01"},
        ])
        response = test_client.get("/api/v1/snapshot")
        point = response.json()["equity_curve"]["points"][0]
        assert set(point.keys()) == {"date", "equity_usd"}
