"""Issuer-identity preflight: verifies every one of the control arm's 44
tickers still refers to the SAME real company it referred to when
`config/control_universe_identity_anchors_v1.json` was generated --
before any market-data fetch or decision computation.

WHY THIS EXISTS: `src/live/control_universe.py` freezes WHICH tickers
the control arm trades; nothing in this codebase previously checked WHO
each ticker actually identifies. Ticker recycling is a real, documented
risk this session's own research already demonstrated is not
hypothetical (Q and SNDK both show a real REMOVE-then-later-ADD pattern
where the current company is provably NOT the one that originally used
the ticker -- see `src/research/pit_universe/membership_query.py`'s own
module docstring). This module closes that gap with a real, dated
anchor comparison, every run.

DAILY vs WEEKLY CADENCE: the Alpaca side is checked with a real, fresh
`get_all_assets()` call (unfiltered -- never status/asset_class-filtered,
so a ticker that went inactive is never silently excluded from the
comparison) EVERY run. The SEC EDGAR side (`company_tickers.json`, ~10.4K
entries, "rarely changes") is cached on disk and only re-fetched when
the cache exceeds `SEC_CACHE_MAX_AGE_DAYS` (8) -- roughly weekly in
practice, not a genuine per-run network call. If the cache is already
past 8 days AND a refresh attempt fails, this is itself a HARD,
fail-closed condition (see below) -- an unverifiable CIK is treated the
same as a mismatched one, never silently skipped.

NINE HARD-FAILURE CONDITIONS (any one of these stops the ENTIRE run,
zero progress, NO single-ticker exclusion -- see
`IssuerIdentityMismatchError`):
  1. Ticker not found in Alpaca's unfiltered `get_all_assets()`.
  2. Alpaca `asset_id` differs from the anchor's.
  3. Alpaca `symbol` differs from the anchor's.
  4. SEC CIK differs from the anchor's, OR the ticker is no longer
     present in SEC EDGAR's `company_tickers.json` at all.
  5. `asset_class` is no longer `us_equity`.
  6. `status` is no longer `active`.
  7. `tradable` is no longer `true`.
  8. The ticker now maps to MORE THAN ONE CIK in the raw SEC source data
     (an ambiguous mapping -- the anchor's single CIK can no longer be
     trusted as THE answer for that ticker string).
  9. The anchor artifact's own ticker set does not equal
     `CONTROL_UNIVERSE_TICKERS` -- the anchor file is stale relative to
     the current universe definition (or vice versa).

TWO SOFT ("REVIEW-REQUIRED") DRIFT CONDITIONS -- logged, cached, and
`[CONTROL]`-notifiable by the caller, but NEVER block the run (CIK/asset_id
provably unchanged, so identity itself is not in question):
  - Company/asset display name changed (a real rebrand, for example).
  - Exchange changed.

REUSES, NEVER REIMPLEMENTS: SEC EDGAR fetching goes through
`src.research.pit_universe.membership_query._load_sec_company_tickers`
-- that module already solved a real, documented obstacle (SEC's
Akamai bot-management fingerprints Python's own TLS stack and blocks
it; that function shells out to `curl` instead). Re-solving that here
would be pure duplication of an already-real fix.

RUNTIME CACHE: every check (pass, soft-drift, or fail-closed) writes
`$AI_STOCK_RADAR_GUARD_DIR/issuer_identity_last_check.json` -- the
anchor artifact's own SHA-256, the check timestamp, how many tickers
were verified, the full finding list, and the SEC cache's age at check
time. Written with the same atomic-write discipline as
`position_state.json` (`_atomic_write_text`, reused from
`position_state.py`, not reimplemented).

OUT OF SCOPE FOR THIS TASK, DELIBERATELY: what happens AFTER a human
reviews a hard mismatch and decides the new identity is legitimate (a
`RETIRED_IDENTITY_BREAK`/versioned-anchor-update workflow) is a
separate, later, human-approved process. This module only implements
the correct STOPPING behavior today.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

from src.live.control_universe import CONTROL_UNIVERSE_TICKERS
from src.live.position_state import HIGH_WATER_MARK_PATH, _atomic_write_text
from src.research.pit_universe.membership_query import _load_sec_company_tickers

ANCHOR_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "control_universe_identity_anchors_v1.json"
DEFAULT_RUNTIME_CACHE_PATH = HIGH_WATER_MARK_PATH.parent / "issuer_identity_last_check.json"
SEC_CACHE_MAX_AGE_DAYS = 8

STATUS_PASS = "PASS"
STATUS_SOFT_DRIFT = "SOFT_DRIFT"
STATUS_FAIL_CLOSED = "FAIL_CLOSED"


class IssuerIdentityMismatchError(RuntimeError):
    """Raised (fail-closed) on ANY hard issuer-identity mismatch. The
    entire run must stop -- see module docstring's "NINE HARD-FAILURE
    CONDITIONS". No single-ticker exclusion mechanism exists anywhere in
    this module; that is a deliberate design choice, not an omission."""


@dataclass
class IssuerIdentityFinding:
    ticker: str
    kind: str  # "HARD" | "SOFT"
    field: str
    old_value: str
    new_value: str
    detail: str


@dataclass
class IssuerIdentityCheckResult:
    status: str  # PASS | SOFT_DRIFT | FAIL_CLOSED
    checked_ticker_count: int
    hard_findings: list[IssuerIdentityFinding] = field(default_factory=list)
    soft_findings: list[IssuerIdentityFinding] = field(default_factory=list)
    anchor_sha256: str = ""
    checked_at_utc: str = ""
    sec_data_age_days: float | None = None


def _default_sec_cache_path() -> Path:
    return Path(__file__).resolve().parent.parent / "research" / "pit_universe" / "sec_edgar_cache" / "company_tickers.json"


def _ensure_sec_data_fresh(sec_cache_path: Path | None) -> tuple[dict[str, dict] | None, float | None, str | None]:
    """Returns `(sec_map, age_days, refresh_error)`. `sec_map` is `None`
    ONLY when the cache was stale/missing AND a refresh attempt failed
    -- the caller must treat that as a hard, fail-closed condition (see
    module docstring's "DAILY vs WEEKLY CADENCE")."""
    path = sec_cache_path or _default_sec_cache_path()
    age_days = (time.time() - path.stat().st_mtime) / 86400.0 if path.is_file() else None

    if age_days is not None and age_days <= SEC_CACHE_MAX_AGE_DAYS:
        return _load_sec_company_tickers(force_refresh=False), age_days, None

    try:
        sec_map = _load_sec_company_tickers(force_refresh=True)
        return sec_map, 0.0, None
    except Exception as error:  # noqa: BLE001 - reported, not swallowed; caller fails closed
        return None, age_days, f"{type(error).__name__}: {error}"


def _find_ambiguous_tickers(sec_cache_path: Path | None, tickers: frozenset[str]) -> list[str]:
    """Same raw-payload duplicate-CIK check
    `scripts/generate_control_universe_identity_anchors.py` uses at
    generation time -- re-run at check time since ambiguity could newly
    appear (a ticker reassigned) even if it wasn't ambiguous when
    anchored."""
    path = sec_cache_path or _default_sec_cache_path()
    if not path.is_file():
        return []
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    seen_ciks: dict[str, set[int]] = {}
    for entry in raw_payload.values():
        seen_ciks.setdefault(entry["ticker"], set()).add(entry["cik_str"])
    return sorted(t for t in tickers if len(seen_ciks.get(t, set())) > 1)


def _load_anchor_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IssuerIdentityMismatchError(
            f"Anchor artifact not found: {path} -- fail-closed, cannot verify "
            f"issuer identity without it. Run "
            f"scripts/generate_control_universe_identity_anchors.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_runtime_cache(result: IssuerIdentityCheckResult, path: Path | None) -> None:
    target = path or DEFAULT_RUNTIME_CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor_sha256": result.anchor_sha256,
        "checked_at_utc": result.checked_at_utc,
        "checked_ticker_count": result.checked_ticker_count,
        "sec_data_age_days": result.sec_data_age_days,
        "status": result.status,
        "hard_findings": [asdict(f) for f in result.hard_findings],
        "soft_findings": [asdict(f) for f in result.soft_findings],
    }
    _atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_issuer_identity_preflight(
    client: TradingClient,
    *,
    anchor_path: Path = ANCHOR_PATH,
    runtime_cache_path: Path | None = None,
    sec_cache_path: Path | None = None,
) -> IssuerIdentityCheckResult:
    """Full preflight pass. Raises `IssuerIdentityMismatchError`
    (fail-closed) if ANY hard condition is found -- see module docstring.
    Returns normally (status `PASS` or `SOFT_DRIFT`) otherwise; the
    caller MUST still proceed to `run_daily_decision()` on `SOFT_DRIFT`
    (it is not a failure), but should notify `[CONTROL]` with the drift
    detail. The runtime cache is written on every outcome, including the
    fail-closed one, before the exception is raised."""
    artifact = _load_anchor_artifact(anchor_path)
    anchors_by_ticker: dict[str, dict] = {a["ticker"]: a for a in artifact["anchors"]}
    anchor_ticker_set = frozenset(anchors_by_ticker)
    current_universe = frozenset(CONTROL_UNIVERSE_TICKERS)

    hard_findings: list[IssuerIdentityFinding] = []
    soft_findings: list[IssuerIdentityFinding] = []

    if anchor_ticker_set != current_universe:
        only_in_anchor = sorted(anchor_ticker_set - current_universe)
        only_in_universe = sorted(current_universe - anchor_ticker_set)
        hard_findings.append(
            IssuerIdentityFinding(
                ticker="<universe>", kind="HARD", field="anchor_ticker_set",
                old_value=f"anchor_only={only_in_anchor}", new_value=f"universe_only={only_in_universe}",
                detail="Anchor artifact's ticker set does not equal CONTROL_UNIVERSE_TICKERS -- "
                       "the anchor file is stale relative to the current universe (or vice versa).",
            )
        )

    # Daily: real, unfiltered Alpaca fetch -- never status/asset_class-filtered.
    all_assets = client.get_all_assets(GetAssetsRequest())
    assets_by_symbol = {a.symbol: a for a in all_assets}

    sec_map, sec_age_days, sec_refresh_error = _ensure_sec_data_fresh(sec_cache_path)
    if sec_map is None:
        hard_findings.append(
            IssuerIdentityFinding(
                ticker="<sec-cache>", kind="HARD", field="sec_company_tickers_cache_age",
                old_value=f"age_days={sec_age_days}", new_value=f"refresh_error={sec_refresh_error}",
                detail=f"SEC EDGAR company_tickers.json cache exceeded the "
                       f"{SEC_CACHE_MAX_AGE_DAYS}-day max age and a refresh attempt failed -- "
                       f"cannot verify CIK identity for any ticker this run.",
            )
        )

    for ticker in sorted(current_universe & anchor_ticker_set):
        anchor = anchors_by_ticker[ticker]
        asset = assets_by_symbol.get(ticker)
        if asset is None:
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "alpaca_presence", anchor["alpaca_symbol_at_anchor"], "MISSING",
                                      f"{ticker} not found in unfiltered get_all_assets()")
            )
            continue

        asset_id = str(asset.id)
        if asset_id != anchor["alpaca_asset_id"]:
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "alpaca_asset_id", anchor["alpaca_asset_id"], asset_id,
                                      "Alpaca asset_id changed -- this ticker now refers to a DIFFERENT underlying asset record.")
            )
        if asset.symbol != anchor["alpaca_symbol_at_anchor"]:
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "alpaca_symbol", anchor["alpaca_symbol_at_anchor"], asset.symbol,
                                      "Alpaca symbol differs from anchor.")
            )

        asset_class = str(asset.asset_class.value if hasattr(asset.asset_class, "value") else asset.asset_class)
        if asset_class != "us_equity":
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "asset_class", "us_equity", asset_class, "Asset class is no longer us_equity.")
            )

        status = str(asset.status.value if hasattr(asset.status, "value") else asset.status)
        if status != "active":
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "status", "active", status, "Asset status is no longer active.")
            )

        if not asset.tradable:
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "tradable", "true", "false", "Asset is no longer tradable.")
            )

        exchange = str(asset.exchange.value if hasattr(asset.exchange, "value") else asset.exchange)
        if exchange != anchor["exchange_at_anchor"]:
            soft_findings.append(
                IssuerIdentityFinding(ticker, "SOFT", "exchange", anchor["exchange_at_anchor"], exchange,
                                      "Exchange changed -- CIK/asset_id unaffected, review-required, not blocking.")
            )

        if asset.name != anchor["alpaca_asset_name_at_anchor"]:
            soft_findings.append(
                IssuerIdentityFinding(ticker, "SOFT", "alpaca_asset_name", anchor["alpaca_asset_name_at_anchor"], asset.name,
                                      "Alpaca asset display name changed -- CIK/asset_id unaffected, review-required, not blocking.")
            )

        if sec_map is not None:
            sec_entry = sec_map.get(ticker)
            if sec_entry is None:
                hard_findings.append(
                    IssuerIdentityFinding(ticker, "HARD", "sec_cik_presence", str(anchor["sec_cik"]), "MISSING",
                                          "Ticker no longer present in SEC EDGAR company_tickers.json -- CIK lost.")
                )
            elif sec_entry["cik_str"] != anchor["sec_cik"]:
                hard_findings.append(
                    IssuerIdentityFinding(ticker, "HARD", "sec_cik", str(anchor["sec_cik"]), str(sec_entry["cik_str"]),
                                          "SEC CIK changed -- this ticker now identifies a DIFFERENT registrant.")
                )
            elif sec_entry["title"] != anchor["expected_company_name"]:
                soft_findings.append(
                    IssuerIdentityFinding(ticker, "SOFT", "sec_company_name", anchor["expected_company_name"], sec_entry["title"],
                                          "SEC registered company name changed -- CIK constant, review-required, not blocking.")
                )

    if sec_map is not None:
        for ticker in _find_ambiguous_tickers(sec_cache_path, current_universe & anchor_ticker_set):
            hard_findings.append(
                IssuerIdentityFinding(ticker, "HARD", "sec_cik_ambiguous", "single CIK", "multiple CIKs",
                                      f"{ticker} now maps to more than one CIK in raw SEC source data -- ambiguous, cannot trust the anchor's single CIK.")
            )

    status = STATUS_FAIL_CLOSED if hard_findings else (STATUS_SOFT_DRIFT if soft_findings else STATUS_PASS)
    result = IssuerIdentityCheckResult(
        status=status,
        checked_ticker_count=len(current_universe & anchor_ticker_set),
        hard_findings=hard_findings,
        soft_findings=soft_findings,
        anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        checked_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sec_data_age_days=sec_age_days,
    )
    _write_runtime_cache(result, runtime_cache_path)

    if hard_findings:
        detail_lines = [
            f"  - {f.ticker}: {f.field} changed from {f.old_value!r} to {f.new_value!r} -- {f.detail}"
            for f in hard_findings
        ]
        raise IssuerIdentityMismatchError(
            "Issuer-identity HARD mismatch detected -- ENTIRE RUN STOPPED, zero "
            "progress, no single-ticker exclusion. Affected:\n" + "\n".join(detail_lines)
        )

    return result
