"""One-time (re-runnable) generator for
`config/control_universe_identity_anchors_v1.json` -- the pre-registered
issuer-identity anchor artifact `src/live/issuer_identity_preflight.py`
checks the control arm's real, live-fetched broker/SEC data against
before every run.

WHY THIS EXISTS: `src/live/control_universe.py` freezes WHICH 44 tickers
the control arm trades. It says nothing about WHO each ticker actually
identifies -- if AA silently started resolving to a different company
tomorrow (a real, documented risk this session's own ticker-recycling
research on Q/SNDK demonstrated is not hypothetical), the frozen ticker
list alone could not detect it. This script records, ONCE, a real,
dated snapshot of each ticker's Alpaca asset identity (asset_id, symbol,
name, class, exchange, status, tradable, fractionable) and SEC EDGAR
issuer identity (CIK, registered title) -- the anchor
`issuer_identity_preflight.py` compares every future run against.

REAL DATA ONLY, NO PARTIAL WRITES: fetches real
`TradingClient(paper=True).get_all_assets()` (unfiltered -- the same
discipline the runtime preflight uses, so a ticker that has gone
inactive/delisted is never silently excluded by a status filter) and
real SEC EDGAR `company_tickers.json` (via
`src.research.pit_universe.membership_query._load_sec_company_tickers`,
reused rather than reimplemented -- that module already solved a real,
documented obstacle: SEC's Akamai bot-management fingerprints Python's
own TLS stack and blocks it, so that function shells out to `curl`
instead; re-solving that here would be pure duplication). If ANY of the
44 tickers is missing from either source, or SEC's ticker->CIK mapping
is ambiguous (the same ticker string resolving to more than one CIK in
the raw source data), generation ABORTS -- zero anchors are written,
not 43 real ones and one placeholder.

Run again to regenerate (e.g. after a deliberate, human-approved
RETIRED_IDENTITY_BREAK event -- that re-anchoring workflow itself is
explicitly out of scope for this task, only the artifact THIS script
produces and the hard-failure behavior that checks against it).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.requests import GetAssetsRequest

import scripts.run_control_arm_decision as carm
from src.live.control_universe import (
    CONTROL_UNIVERSE_SOURCE_COMMIT,
    CONTROL_UNIVERSE_SOURCE_SNAPSHOT,
    CONTROL_UNIVERSE_TICKERS,
)
from src.research.pit_universe.membership_query import _load_sec_company_tickers

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "config" / "control_universe_identity_anchors_v1.json"
UNIVERSE_ID = "control_arm_v1_midcap400_44"


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_hash(record: dict) -> str:
    return _sha256_hex(json.dumps(record, sort_keys=True, default=str))


def generate(*, env_file: str = ".env.control") -> dict:
    carm._load_env_file_fail_closed(env_file)

    client = carm.rdd.order_submission.get_trading_client()
    account_masked = carm.broker_reconciliation.verify_account_identity(client)

    alpaca_fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_assets = client.get_all_assets(GetAssetsRequest())  # unfiltered, deliberately
    assets_by_symbol = {a.symbol: a for a in all_assets}

    sec_fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sec_map = _load_sec_company_tickers()

    missing_alpaca = [t for t in CONTROL_UNIVERSE_TICKERS if t not in assets_by_symbol]
    missing_sec = [t for t in CONTROL_UNIVERSE_TICKERS if t not in sec_map]
    if missing_alpaca or missing_sec:
        raise RuntimeError(
            f"Anchor generation ABORTED -- zero anchors written. Missing from "
            f"Alpaca (unfiltered): {missing_alpaca}. Missing from SEC EDGAR "
            f"company_tickers.json: {missing_sec}. Fix the underlying data gap "
            f"before re-running; this script never writes a partial artifact."
        )

    # Ambiguity check: does the SAME ticker string map to more than one CIK
    # anywhere in the raw SEC source (not just our 44)? A real, mechanical
    # signal that a bare ticker->CIK lookup for that specific symbol cannot
    # be trusted without a human resolving which CIK is meant.
    ambiguous: list[str] = []
    # _load_sec_company_tickers() collapses SEC's own numerically-indexed
    # raw payload into {ticker: entry} -- if the SAME ticker string
    # appeared more than once in the raw payload with DIFFERENT CIKs, only
    # the last one survives that collapse silently. Re-derive from the
    # cached raw file directly to catch that case honestly rather than
    # trusting the already-collapsed view.
    raw_cache_path = Path(__file__).resolve().parent.parent / "src/research/pit_universe/sec_edgar_cache/company_tickers.json"
    raw_payload = json.loads(raw_cache_path.read_text(encoding="utf-8"))
    seen_ciks: dict[str, set[int]] = {}
    for entry in raw_payload.values():
        seen_ciks.setdefault(entry["ticker"], set()).add(entry["cik_str"])
    for ticker in CONTROL_UNIVERSE_TICKERS:
        if len(seen_ciks.get(ticker, set())) > 1:
            ambiguous.append(ticker)
    if ambiguous:
        raise RuntimeError(
            f"Anchor generation ABORTED -- zero anchors written. Ambiguous "
            f"SEC ticker->CIK mapping (same ticker, multiple CIKs) for: "
            f"{ambiguous}."
        )

    anchors = []
    for ticker in CONTROL_UNIVERSE_TICKERS:
        asset = assets_by_symbol[ticker]
        sec_entry = sec_map[ticker]
        asset_record = {
            "id": str(asset.id),
            "symbol": asset.symbol,
            "name": asset.name,
            "asset_class": str(asset.asset_class.value if hasattr(asset.asset_class, "value") else asset.asset_class),
            "exchange": str(asset.exchange.value if hasattr(asset.exchange, "value") else asset.exchange),
            "status": str(asset.status.value if hasattr(asset.status, "value") else asset.status),
            "tradable": asset.tradable,
            "fractionable": asset.fractionable,
        }
        anchors.append(
            {
                "ticker": ticker,
                "sec_cik": sec_entry["cik_str"],
                "expected_company_name": sec_entry["title"],
                "sec_ticker_at_anchor": sec_entry["ticker"],
                "alpaca_asset_id": asset_record["id"],
                "alpaca_symbol_at_anchor": asset_record["symbol"],
                "alpaca_asset_name_at_anchor": asset_record["name"],
                "asset_class_at_anchor": asset_record["asset_class"],
                "exchange_at_anchor": asset_record["exchange"],
                "status_at_anchor": asset_record["status"],
                "tradable_at_anchor": asset_record["tradable"],
                "fractionable_at_anchor": asset_record["fractionable"],
                "anchored_at_utc": alpaca_fetched_at,
                "source_record_hashes": {
                    "alpaca_asset_record_sha256": _record_hash(asset_record),
                    "sec_company_tickers_record_sha256": _record_hash(sec_entry),
                },
            }
        )

    ticker_set_sha256 = _sha256_hex("\n".join(sorted(CONTROL_UNIVERSE_TICKERS)))

    artifact = {
        "schema_version": 1,
        "universe_id": UNIVERSE_ID,
        "universe_source_commit": CONTROL_UNIVERSE_SOURCE_COMMIT,
        "universe_source_snapshot": CONTROL_UNIVERSE_SOURCE_SNAPSHOT,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approval_status": "PRE_REGISTERED_FROZEN_BEFORE_LIVE",
        "ticker_count": len(CONTROL_UNIVERSE_TICKERS),
        "ticker_set_sha256": ticker_set_sha256,
        "sources": {
            "alpaca": {
                "account_identity_masked": account_masked,
                "get_all_assets_call": "unfiltered (GetAssetsRequest with no status/asset_class filter)",
                "total_assets_returned": len(all_assets),
                "fetched_at_utc": alpaca_fetched_at,
            },
            "sec_edgar": {
                "source_url": "https://www.sec.gov/files/company_tickers.json",
                "total_entries": len(sec_map),
                "fetched_at_utc": sec_fetched_at,
            },
        },
        "anchors": anchors,
    }

    if len(artifact["anchors"]) != len(CONTROL_UNIVERSE_TICKERS):
        raise RuntimeError("Internal error: anchor count does not match the control universe ticker count.")

    return artifact


def main() -> int:
    artifact = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(artifact['anchors'])} anchors to {OUTPUT_PATH}")
    print(f"ticker_set_sha256: {artifact['ticker_set_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
