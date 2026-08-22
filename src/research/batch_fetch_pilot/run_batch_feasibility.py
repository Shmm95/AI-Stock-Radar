"""Isolated feasibility test: does alpaca-py's real multi-symbol batch
endpoint (`StockBarsRequest`/`CryptoBarsRequest`) work, is it faster
than this codebase's current sequential per-ticker fetch, and what
happens when a batch request contains an invalid, delisted, or
recycled ticker symbol?

ISOLATED, NOT INTEGRATED: does not modify, and is never imported by,
`src/data/alpaca_market_data.py`, `src/live/data_preparer.py`,
`scripts/run_daily_decision.py`, or `scripts/run_control_arm_decision.py`.
The sequential baselines below call `alpaca_market_data.get_stock_bars`/
`get_crypto_bars` exactly as they exist today (imported, not copied) so
the comparison is against the REAL current code.

Real, already-run results (this session, alpaca-py 0.43.5):

EQUITY (25-ticker subset of `live_universe.LIVE_CONTROLLED_TICKERS`,
400-day daily window):
- Sequential: 10.88s (0.435s/ticker). Batch: 0.92s. Speedup: 11.9x.
  Full 80-ticker universe, batch: 1.73s (vs. ~35.2s sequential, ~20x).
- Invalid symbol ("ZZZINVALID") injected into a real batch: NO
  exception. Batch succeeded, 25/25 valid symbols returned correctly,
  invalid symbol silently ABSENT from the response.

CRYPTO (BTC/USD, ETH/USD, UNI/USD -- the live universe's own 3 crypto
tickers, kept small deliberately per this task's own instruction not to
inflate quota use):
- Sequential: 1.34s (0.446s/ticker). Batch: 0.55s. Speedup: 2.4x (small
  N means fixed per-request overhead dominates more than at equity
  scale -- the win still grows with N, just less dramatically at N=3).
- Invalid symbol ("ZZZINVALID/USD") injected: same behavior as equity
  -- NO exception, silently absent from the response.

RATE-LIMIT / ERROR BEHAVIOR -- researched via alpaca-py's own source
(`alpaca/common/rest.py`, `alpaca/common/constants.py`), deliberately
NOT triggered live (this task's own instruction: don't burn real quota
to force a 429). This is a genuinely different failure mode from the
silent-symbol-drop above, and behaves oppositely:
- A batch call is ONE HTTP request. `RESTClient._request` retries ONLY
  on status codes in `DEFAULT_RETRY_EXCEPTION_CODES = [429, 504]`, up
  to `DEFAULT_RETRY_ATTEMPTS = 3` times, `DEFAULT_RETRY_WAIT_SECONDS = 3`
  apart (confirmed by direct read of `_request`/`_one_request`, not
  assumed).
- If retries are exhausted (429/504 persists) OR the status is any
  OTHER error code (400/401/403/404/500/502/503/...), `_one_request`
  raises `APIError` for the WHOLE request -- there is no partial
  success at the transport level. Unlike the silent per-symbol data
  gap above (which happens INSIDE a 200 OK response body), a real
  429/5xx fails the entire batch together, loudly, as one exception.
  This means a caller does NOT need to worry about "half the batch
  succeeded, half hit a network error" -- that specific combination is
  structurally impossible with one HTTP request/response pair. The
  only partial-failure mode is the data-level one `verify_requested_vs_returned`
  already targets.

DELISTED / RECYCLED TICKER TEST -- real batch call, `["FB", "META",
"COV", "AAPL"]`: NO exception. Returned: AAPL, FB, META. Missing: COV.
- COV (Covidien, a real, confirmed REMOVE event 2015-01-27 in this
  session's own PIT membership data, acquired by Medtronic) has NO
  data via this endpoint -- genuinely delisted tickers are silently
  absent, same as an invalid symbol, already covered by
  `verify_requested_vs_returned`.
- **FB is the serious, previously-undiscovered finding of this task.**
  "FB" was Facebook's own ticker before its 2021 rename to META. It
  DID return real data here -- but comparing FB's and META's actual
  OHLCV values for the same recent dates shows they are NOT equal
  (FB: ~$45, near-zero volume some days; META: ~$600, ~10k+ trades/day)
  -- confirmed via direct `DataFrame.equals()` comparison, not assumed.
  **"FB" has been RECYCLED by an entirely unrelated real security since
  the rename**, and Alpaca's batch endpoint returns THAT security's
  real data under the "FB" symbol with zero warning or distinguishing
  signal. This is WORSE than the silent-drop case: `verify_requested_vs_returned`
  cannot catch it at all, because "FB" is present in both requested and
  returned sets -- the ticker resolves successfully, just to the wrong
  company. Any caller holding an old ticker->identity mapping (exactly
  the kind of mapping a PIT membership reconstruction like
  `src/research/pit_universe/` produces) that queries by ticker symbol
  alone, without cross-checking against a CIK or similar durable
  identifier, is at risk of silently querying the WRONG security's
  price history for a renamed/recycled ticker.

Run directly: `.venv/bin/python -m src.research.batch_fetch_pilot.run_batch_feasibility`
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.data.alpaca_market_data import _require_credentials, get_crypto_bars, get_stock_bars
from src.live.live_universe import LIVE_CONTROLLED_TICKERS

logger = logging.getLogger(__name__)

FETCH_CALENDAR_DAYS = 400  # matches data_preparer.py's own DEFAULT_FETCH_CALENDAR_DAYS
MEASURED_SEQUENTIAL_SECONDS_PER_TICKER = 0.44  # this pilot's own real measurement, reused for cost estimates below


def _equity_subset(n: int | None = None) -> list[str]:
    equity = [t for t in LIVE_CONTROLLED_TICKERS if not t.endswith("-USD")]
    return equity if n is None else equity[:n]


def _crypto_subset() -> list[str]:
    # Alpaca's crypto symbol form is slash, not dash -- same conversion data_preparer.py already does.
    return [t.replace("-", "/") for t in LIVE_CONTROLLED_TICKERS if t.endswith("-USD")]


def _stock_client() -> StockHistoricalDataClient:
    api_key, secret_key = _require_credentials()
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _crypto_client() -> CryptoHistoricalDataClient:
    api_key, secret_key = _require_credentials()
    return CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _window() -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=FETCH_CALENDAR_DAYS)
    return start, end


def fetch_batch(tickers: list[str]) -> "pd.DataFrame":  # noqa: F821 -- pandas, avoided as a hard import here
    start, end = _window()
    request = StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.IEX
    )
    return _stock_client().get_stock_bars(request).df


def fetch_batch_crypto(symbols: list[str]) -> "pd.DataFrame":  # noqa: F821
    start, end = _window()
    request = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end)
    return _crypto_client().get_crypto_bars(request).df


def verify_requested_vs_returned(
    requested: list[str],
    df,
    *,
    on_missing: Callable[[dict], None] | None = None,
) -> dict:
    """The safety check this whole pilot exists to justify: since a
    batch call can silently drop an invalid/unrecognized/delisted
    symbol with NO exception (confirmed empirically, equity and
    crypto, see module docstring), any real adoption of batch fetching
    MUST diff requested vs. returned symbols after every call and treat
    a gap as an error condition -- never silently proceed with fewer
    tickers than requested.

    Hardening added this task, matching the three things the task asked
    to make explicit rather than silently assumed:
    (a) LOGGED clearly -- `logger.warning(...)` fires whenever a gap is
        found, with the exact missing set and counts, independent of
        whatever the caller does next.
    (b) NOTIFICATION HOOK, left unwired on purpose: `on_missing`, an
        optional callback invoked with this function's own result dict
        whenever `missing` is non-empty. This is deliberately NOT
        connected to any real Telegram/[CONTROL] channel here -- that
        would be touching live/control code, out of this task's scope.
        A future live caller could pass e.g.
        `on_missing=lambda r: _notify_control(f"batch fetch gap: {r['missing']}")`
        (matching `run_control_arm_decision.py`'s own existing
        `_notify_control` naming convention) without this function
        itself changing at all.
    (c) Sequential-fallback feasibility is a SEPARATE function,
        `assess_sequential_fallback_feasibility` below -- evaluates,
        does not perform, per this task's explicit instruction not to
        write a working automatic fallback yet.

    NOTE ON WHAT THIS CANNOT CATCH: a symbol that resolves successfully
    but to the WRONG security (the real "FB" ticker-recycling finding
    documented in this module's own docstring) is invisible to a
    requested-vs-returned diff, because the requested symbol IS present
    in the response -- just attached to different real data than a
    caller might expect. This function's own scope is presence/absence,
    not identity verification; see the module docstring's closing note
    on why a durable identifier (CIK or similar), not the ticker symbol
    alone, would be needed to catch that case.
    """
    returned = set(df.index.get_level_values("symbol")) if len(df) else set()
    missing = sorted(set(requested) - returned)
    extra = sorted(returned - set(requested))
    result = {
        "requested_count": len(requested),
        "returned_count": len(returned),
        "missing": missing,
        "extra_unexpected": extra,
        "silent_partial_failure": len(missing) > 0,
    }
    if missing:
        logger.warning(
            "Batch fetch returned fewer symbols than requested: missing=%s "
            "(requested=%d, returned=%d)",
            missing, len(requested), len(returned),
        )
        if on_missing is not None:
            on_missing(result)
    return result


def assess_sequential_fallback_feasibility(missing: list[str], asset_class: str = "EQUITY") -> dict:
    """Evaluates, does NOT perform, whether `missing` tickers could be
    recovered via the existing sequential fetch path -- no network call
    is made here, per this task's explicit instruction to assess
    feasibility only, not implement an automatic fallback.
    """
    existing_function = (
        "src.data.alpaca_market_data.get_stock_bars"
        if asset_class == "EQUITY"
        else "src.data.alpaca_market_data.get_crypto_bars"
    )
    return {
        "feasible": True,
        "mechanism": (
            f"call {existing_function}(ticker, start=..., end=...) once per "
            "missing ticker -- already exists, unmodified, and is exactly "
            "the function this pilot already used as its own sequential "
            "baseline, so it is proven working, not a new code path."
        ),
        "estimated_cost_seconds": round(len(missing) * MEASURED_SEQUENTIAL_SECONDS_PER_TICKER, 2),
        "caveat": (
            "A ticker missing from a batch response is not necessarily "
            "fetchable sequentially either -- if it is missing because the "
            "symbol is genuinely invalid or delisted (e.g. this task's own "
            "COV finding), the sequential call raises the SAME underlying "
            "'no data returned' error (see `_fetch_raw_bars` in "
            "`data_preparer.py`, which already raises ValueError on an "
            "empty response) rather than silently succeeding. Fallback can "
            "only recover tickers dropped for a transient/batch-specific "
            "reason, not ones genuinely absent from the vendor. It also "
            "CANNOT help with the FB-style wrong-identity case, since that "
            "ticker is not missing at all -- sequential fetch would return "
            "the exact same wrong security's data."
        ),
    }


def test_valid_batch(n: int = 25) -> None:
    print(f"=== Test 1: valid batch, {n} real tickers ===")
    tickers = _equity_subset(n)
    t0 = time.time()
    df = fetch_batch(tickers)
    elapsed = time.time() - t0
    check = verify_requested_vs_returned(tickers, df)
    print(f"  elapsed: {elapsed:.3f}s")
    print(f"  {check}")
    print()


def test_invalid_symbol_injection(n: int = 25) -> None:
    print(f"=== Test 2: {n} valid tickers + 1 intentionally invalid ('ZZZINVALID') ===")
    tickers = _equity_subset(n) + ["ZZZINVALID"]
    try:
        t0 = time.time()
        df = fetch_batch(tickers)
        elapsed = time.time() - t0
        check = verify_requested_vs_returned(tickers, df)
        print(f"  NO EXCEPTION -- batch call succeeded in {elapsed:.3f}s")
        print(f"  {check}")
        if check["silent_partial_failure"]:
            print(
                "  CONFIRMED: invalid symbol silently dropped, no error raised. "
                "Real adoption MUST diff requested-vs-returned after every call."
            )
    except Exception as error:
        print(f"  EXCEPTION RAISED: {type(error).__name__}: {error}")
    print()


def test_timing_comparison(n: int = 25) -> None:
    print(f"=== Test 3: sequential (existing get_stock_bars) vs batch, {n} tickers ===")
    tickers = _equity_subset(n)
    start, end = _window()

    t0 = time.time()
    for ticker in tickers:
        get_stock_bars(ticker, start=start, end=end)  # existing, unmodified function
    sequential_elapsed = time.time() - t0

    t0 = time.time()
    fetch_batch(tickers)
    batch_elapsed = time.time() - t0

    print(f"  sequential: {sequential_elapsed:.3f}s ({sequential_elapsed/n:.3f}s/ticker)")
    print(f"  batch:      {batch_elapsed:.3f}s")
    print(f"  speedup:    {sequential_elapsed/batch_elapsed:.1f}x")
    print()


def test_full_universe_batch() -> None:
    tickers = _equity_subset()
    print(f"=== Test 4: full equity universe batch, {len(tickers)} tickers ===")
    t0 = time.time()
    df = fetch_batch(tickers)
    elapsed = time.time() - t0
    check = verify_requested_vs_returned(tickers, df)
    print(f"  elapsed: {elapsed:.3f}s")
    print(f"  {check}")
    print()


def test_valid_batch_crypto() -> None:
    symbols = _crypto_subset()
    print(f"=== Crypto Test 1: valid batch, {len(symbols)} real crypto tickers ({symbols}) ===")
    t0 = time.time()
    df = fetch_batch_crypto(symbols)
    elapsed = time.time() - t0
    check = verify_requested_vs_returned(symbols, df)
    print(f"  elapsed: {elapsed:.3f}s")
    print(f"  {check}")
    print()


def test_invalid_symbol_injection_crypto() -> None:
    symbols = _crypto_subset() + ["ZZZINVALID/USD"]
    print(f"=== Crypto Test 2: {len(symbols)-1} valid + 1 intentionally invalid ===")
    try:
        t0 = time.time()
        df = fetch_batch_crypto(symbols)
        elapsed = time.time() - t0
        check = verify_requested_vs_returned(symbols, df)
        print(f"  NO EXCEPTION -- batch call succeeded in {elapsed:.3f}s")
        print(f"  {check}")
        if check["silent_partial_failure"]:
            print("  CONFIRMED: same silent-drop behavior as equity.")
    except Exception as error:
        print(f"  EXCEPTION RAISED: {type(error).__name__}: {error}")
    print()


def test_timing_comparison_crypto() -> None:
    symbols = _crypto_subset()
    print(f"=== Crypto Test 3: sequential (existing get_crypto_bars) vs batch, {len(symbols)} tickers ===")
    start, end = _window()

    t0 = time.time()
    for sym in symbols:
        get_crypto_bars(sym, start=start, end=end)  # existing, unmodified function
    sequential_elapsed = time.time() - t0

    t0 = time.time()
    fetch_batch_crypto(symbols)
    batch_elapsed = time.time() - t0

    print(f"  sequential: {sequential_elapsed:.3f}s ({sequential_elapsed/len(symbols):.3f}s/ticker)")
    print(f"  batch:      {batch_elapsed:.3f}s")
    print(f"  speedup:    {sequential_elapsed/batch_elapsed:.1f}x "
          f"(small N -- fixed per-request overhead dominates more than at equity scale)")
    print()


def test_delisted_and_recycled_ticker() -> None:
    """Real batch call mixing a known-good control (AAPL), a genuinely
    delisted ticker with a real confirmed removal event in this
    session's own PIT data (COV -- Covidien, removed from the S&P 500
    2015-01-27, acquired by Medtronic), and a renamed ticker pair
    (FB -> META, Facebook's 2021 rename). See module docstring for the
    full finding -- FB is NOT missing and NOT an alias of META, it is a
    real, different, RECYCLED security today, which
    `verify_requested_vs_returned` cannot detect (the symbol IS
    present, just attached to the wrong company)."""
    tickers = ["FB", "META", "COV", "AAPL"]
    print(f"=== Test 5: delisted (COV) + renamed/recycled (FB) ticker test, {tickers} ===")
    try:
        t0 = time.time()
        df = fetch_batch(tickers)
        elapsed = time.time() - t0
        check = verify_requested_vs_returned(tickers, df)
        print(f"  NO EXCEPTION -- elapsed {elapsed:.3f}s")
        print(f"  {check}")
        if "FB" in check.get("missing", []) or "FB" not in set(df.index.get_level_values("symbol")):
            print("  FB was absent this run (consistent with a fully-retired symbol).")
        else:
            fb = df.xs("FB", level="symbol")
            meta = df.xs("META", level="symbol")
            identical = fb.reset_index(drop=True).equals(meta.reset_index(drop=True))
            print(f"  FB present. FB data identical to META's? {identical}")
            if not identical:
                print(
                    "  CONFIRMED: FB is a RECYCLED ticker for an unrelated real security today "
                    "-- NOT the old Facebook, NOT an alias of META. verify_requested_vs_returned "
                    "cannot catch this (FB is 'present', just wrong)."
                )
        if "COV" in check["missing"]:
            print("  COV (genuinely delisted, real 2015-01-27 PIT removal event) correctly silently absent.")
    except Exception as error:
        print(f"  EXCEPTION RAISED: {type(error).__name__}: {error}")
    print()


if __name__ == "__main__":
    test_valid_batch()
    test_invalid_symbol_injection()
    test_timing_comparison()
    test_full_universe_batch()
    test_valid_batch_crypto()
    test_invalid_symbol_injection_crypto()
    test_timing_comparison_crypto()
    test_delisted_and_recycled_ticker()
