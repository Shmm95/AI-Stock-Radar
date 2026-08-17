"""Tests for src/live/issuer_identity_preflight.py -- issuer/asset
identity preflight guarding against ticker recycling (e.g. the real,
SEC-EDGAR-confirmed Q/SNDK cases this module's own docstring cites).

Deterministic and network-free, same discipline as
tests/test_broker_reconciliation.py: a `FakeClient` stands in for
`TradingClient` (never a real one), a small synthetic 3-ticker universe
stands in for the real 44-ticker `CONTROL_UNIVERSE_TICKERS` (monkeypatched
module-global, same LOAD_GLOBAL-rebinding technique documented at length
in `run_control_arm_decision.py`'s own module docstring), and
`_load_sec_company_tickers` is monkeypatched directly rather than hitting
SEC EDGAR. No real Alpaca or SEC call anywhere in this file --
`tests/conftest.py`'s autouse network guard is defense in depth, not the
only thing keeping this file offline.
"""

from __future__ import annotations

import json

import pytest

import src.live.issuer_identity_preflight as iip

_FAKE_TICKERS = ("AAA", "BBB", "CCC")


class FakeAsset:
    def __init__(self, *, id, symbol, name, asset_class="us_equity", exchange="NYSE", status="active", tradable=True, fractionable=True):
        self.id = id
        self.symbol = symbol
        self.name = name
        self.asset_class = asset_class
        self.exchange = exchange
        self.status = status
        self.tradable = tradable
        self.fractionable = fractionable


class FakeClient:
    def __init__(self, assets):
        self._assets = list(assets)

    def get_all_assets(self, request=None):
        return list(self._assets)


def _anchor_record(ticker: str, *, cik: int, name: str = None) -> dict:
    name = name or f"{ticker} Corp"
    return {
        "ticker": ticker,
        "sec_cik": cik,
        "expected_company_name": name,
        "sec_ticker_at_anchor": ticker,
        "alpaca_asset_id": f"asset-id-{ticker}",
        "alpaca_symbol_at_anchor": ticker,
        "alpaca_asset_name_at_anchor": name,
        "asset_class_at_anchor": "us_equity",
        "exchange_at_anchor": "NYSE",
        "status_at_anchor": "active",
        "tradable_at_anchor": True,
        "fractionable_at_anchor": True,
        "anchored_at_utc": "2026-08-17T00:00:00Z",
        "source_record_hashes": {"alpaca_asset_record_sha256": "x", "sec_company_tickers_record_sha256": "y"},
    }


def _write_anchor_artifact(tmp_path, tickers, *, ciks=None):
    ciks = ciks or {t: 1000 + i for i, t in enumerate(tickers)}
    artifact = {
        "schema_version": 1,
        "universe_id": "test_universe",
        "universe_source_commit": "deadbeef",
        "universe_source_snapshot": "test.json",
        "created_at_utc": "2026-08-17T00:00:00Z",
        "approval_status": "PRE_REGISTERED_FROZEN_BEFORE_LIVE",
        "ticker_count": len(tickers),
        "ticker_set_sha256": "irrelevant-for-these-tests",
        "sources": {},
        "anchors": [_anchor_record(t, cik=ciks[t]) for t in tickers],
    }
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path, ciks


def _fake_assets_matching_anchors(tickers):
    return [FakeAsset(id=f"asset-id-{t}", symbol=t, name=f"{t} Corp") for t in tickers]


def _fake_sec_map(tickers, ciks):
    return {t: {"cik_str": ciks[t], "ticker": t, "title": f"{t} Corp"} for t in tickers}


def _write_sec_cache(tmp_path, tickers, ciks):
    # Raw payload shape SEC's own company_tickers.json uses --
    # numerically-indexed, one entry per ticker.
    raw = {str(i): {"cik_str": ciks[t], "ticker": t, "title": f"{t} Corp"} for i, t in enumerate(tickers)}
    path = tmp_path / "company_tickers.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fake_universe(monkeypatch):
    monkeypatch.setattr(iip, "CONTROL_UNIVERSE_TICKERS", _FAKE_TICKERS)


def _run(monkeypatch, tmp_path, *, assets, sec_map, ciks, anchor_tickers=_FAKE_TICKERS, runtime_cache_name="cache.json"):
    anchor_path, _ = _write_anchor_artifact(tmp_path, anchor_tickers, ciks=ciks)
    sec_cache_path = _write_sec_cache(tmp_path, anchor_tickers, ciks)
    monkeypatch.setattr(iip, "_load_sec_company_tickers", lambda force_refresh=False: sec_map)
    client = FakeClient(assets)
    return iip.run_issuer_identity_preflight(
        client,
        anchor_path=anchor_path,
        runtime_cache_path=tmp_path / runtime_cache_name,
        sec_cache_path=sec_cache_path,
    )


def test_all_match_is_a_clean_pass(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    result = _run(
        monkeypatch, tmp_path,
        assets=_fake_assets_matching_anchors(_FAKE_TICKERS),
        sec_map=_fake_sec_map(_FAKE_TICKERS, ciks),
        ciks=ciks,
    )
    assert result.status == iip.STATUS_PASS
    assert result.checked_ticker_count == 3
    assert result.hard_findings == []
    assert result.soft_findings == []


def test_hard_mismatch_asset_id_stops_everything_even_with_only_one_wrong(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    assets = _fake_assets_matching_anchors(_FAKE_TICKERS)
    assets[0].id = "TAMPERED-ASSET-ID"  # AAA's asset_id no longer matches its anchor
    with pytest.raises(iip.IssuerIdentityMismatchError, match="AAA"):
        _run(monkeypatch, tmp_path, assets=assets, sec_map=_fake_sec_map(_FAKE_TICKERS, ciks), ciks=ciks)


def test_hard_mismatch_missing_ticker_stops_everything(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    assets = [a for a in _fake_assets_matching_anchors(_FAKE_TICKERS) if a.symbol != "BBB"]  # BBB vanished
    with pytest.raises(iip.IssuerIdentityMismatchError, match="BBB"):
        _run(monkeypatch, tmp_path, assets=assets, sec_map=_fake_sec_map(_FAKE_TICKERS, ciks), ciks=ciks)


def test_hard_mismatch_sec_cik_changed(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    sec_map = _fake_sec_map(_FAKE_TICKERS, ciks)
    sec_map["CCC"]["cik_str"] = 999999  # CIK silently changed -- different registrant
    with pytest.raises(iip.IssuerIdentityMismatchError, match="CCC"):
        _run(monkeypatch, tmp_path, assets=_fake_assets_matching_anchors(_FAKE_TICKERS), sec_map=sec_map, ciks=ciks)


def test_hard_mismatch_not_tradable(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    assets = _fake_assets_matching_anchors(_FAKE_TICKERS)
    assets[1].tradable = False
    with pytest.raises(iip.IssuerIdentityMismatchError, match="BBB"):
        _run(monkeypatch, tmp_path, assets=assets, sec_map=_fake_sec_map(_FAKE_TICKERS, ciks), ciks=ciks)


def test_hard_mismatch_anchor_set_not_equal_to_universe(monkeypatch, tmp_path):
    """The anchor artifact itself only covers 2 of the 3 (fake) universe
    tickers -- a stale/mismatched anchor file, condition 9."""
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS[:2])}
    with pytest.raises(iip.IssuerIdentityMismatchError, match="anchor_ticker_set"):
        _run(
            monkeypatch, tmp_path,
            assets=_fake_assets_matching_anchors(_FAKE_TICKERS[:2]),
            sec_map=_fake_sec_map(_FAKE_TICKERS[:2], ciks),
            ciks=ciks,
            anchor_tickers=_FAKE_TICKERS[:2],
        )


def test_soft_drift_name_change_does_not_block(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    assets = _fake_assets_matching_anchors(_FAKE_TICKERS)
    assets[0].name = "AAA Corp (Rebranded)"  # CIK/asset_id unchanged
    result = _run(monkeypatch, tmp_path, assets=assets, sec_map=_fake_sec_map(_FAKE_TICKERS, ciks), ciks=ciks)
    assert result.status == iip.STATUS_SOFT_DRIFT
    assert result.hard_findings == []
    assert len(result.soft_findings) == 1
    assert result.soft_findings[0].ticker == "AAA"
    assert result.soft_findings[0].field == "alpaca_asset_name"


def test_soft_drift_exchange_change_does_not_block(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    assets = _fake_assets_matching_anchors(_FAKE_TICKERS)
    assets[2].exchange = "NASDAQ"  # CIK/asset_id unchanged
    result = _run(monkeypatch, tmp_path, assets=assets, sec_map=_fake_sec_map(_FAKE_TICKERS, ciks), ciks=ciks)
    assert result.status == iip.STATUS_SOFT_DRIFT
    assert result.hard_findings == []
    assert len(result.soft_findings) == 1
    assert result.soft_findings[0].field == "exchange"


def test_sec_company_name_drift_is_soft_when_cik_stable(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    sec_map = _fake_sec_map(_FAKE_TICKERS, ciks)
    sec_map["BBB"]["title"] = "BBB Corp (Legally Renamed)"  # CIK unchanged
    result = _run(monkeypatch, tmp_path, assets=_fake_assets_matching_anchors(_FAKE_TICKERS), sec_map=sec_map, ciks=ciks)
    assert result.status == iip.STATUS_SOFT_DRIFT
    assert result.hard_findings == []
    assert any(f.field == "sec_company_name" and f.ticker == "BBB" for f in result.soft_findings)


def test_runtime_cache_is_written_on_every_outcome(monkeypatch, tmp_path):
    ciks = {t: 1000 + i for i, t in enumerate(_FAKE_TICKERS)}
    cache_path = tmp_path / "cache.json"
    _run(
        monkeypatch, tmp_path,
        assets=_fake_assets_matching_anchors(_FAKE_TICKERS),
        sec_map=_fake_sec_map(_FAKE_TICKERS, ciks),
        ciks=ciks,
        runtime_cache_name="cache.json",
    )
    payload = json.loads(cache_path.read_text())
    assert payload["status"] == iip.STATUS_PASS
    assert payload["checked_ticker_count"] == 3
