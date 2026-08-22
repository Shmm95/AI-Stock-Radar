"""Equity cross-sectional 12-1 momentum, native-universe (S&P 500 PIT)
proxy -- strategy_id EQUITY_CROSS_SECTIONAL_MOMENTUM_12_1_K1_DECILE_SP500_PIT_ALPACA_IEX_V1.

See the frozen pre-registration artifact
(data/research/equity_cross_sectional_momentum_native_proxy_preregistration_*.json,
status=FROZEN_BEFORE_MARKET_DATA_FETCH_AND_BACKTEST) for the full,
locked-in parameter set this module implements. Every constant below is
copied verbatim from that artifact, not re-decided here.

STUDY_CLASSIFICATION: NATIVE_REPRODUCTION_PROXY -- NOT a full
Jegadeesh-Titman replication (that used every NYSE/AMEX stock with
sufficient history; this uses a S&P 500 point-in-time-reconstructed
large-cap proxy). RESEARCH_ONLY, not a deployment-readiness test. Never
apply this module's output to the live 83-ticker universe, the existing
cap=6/6% position-sizing engine, or any order/execution path.

ISOLATION GUARANTEES (load-bearing, not incidental):
- Never imports src/live/live_universe.py or src/live/control_universe.py
  -- the live/current ticker universe must never leak into a historical,
  point-in-time study.
- Never imports or calls anything from src/backtest/portfolio_backtest_engine.py
  (the frozen EMA/RSI/cap=6 engine) -- this module's portfolio
  construction (decile ranking, equal weight, K=1 non-overlapping
  holding) is entirely separate and reimplemented here on purpose; the
  two strategies share no code path.
- src/data/alpaca_market_data.py is READ from (`_require_credentials`,
  a pure env-var reader, no fetch/adjustment logic) but never modified
  and never used for the actual bar fetch -- that existing helper's
  `get_stock_bars` does not specify `adjustment` (returns RAW bars) and
  is not batched; this module's own `fetch_point_in_time_bars` pins
  Adjustment.ALL/DataFeed.IEX/asof explicitly and batches by symbol,
  deliberately not reusing that helper for the actual fetch.
- `src.live.order_submission.get_trading_client` (paper=True, read-only
  calendar/account access, an existing generic factory with no
  engine coupling) is reused for the one calendar call this module
  needs -- not reinvented.
- `classify_regime` is imported, unmodified, from
  src/backtest/run_portfolio_benchmark_gap_attribution.py -- a pure
  function (`float -> str`) with no engine coupling, "the same
  thresholds" the spec asks for by construction, not a re-derivation.
- Sharpe/Sortino annualize with sqrt(252) (equity-only), NOT
  src/research/performance_report_v1/metrics.py's TRADING_DAYS_PER_YEAR
  = 365.25 (that constant is deliberately calendar-day-based because
  that report's portfolio holds crypto 24/7 alongside equities -- wrong
  for this equity-only study, so a separate, small metrics
  implementation lives in this module instead of importing that one).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from src.backtest.run_portfolio_benchmark_gap_attribution import classify_regime
from src.data.alpaca_market_data import _require_credentials
from src.research.pit_universe.membership_query import get_membership_as_of

# --- Frozen identity (pre-registration artifact, verbatim) -----------------

STRATEGY_ID = "EQUITY_CROSS_SECTIONAL_MOMENTUM_12_1_K1_DECILE_SP500_PIT_ALPACA_IEX_V1"
STUDY_CLASSIFICATION = "NATIVE_REPRODUCTION_PROXY"
DEPLOYMENT_STATUS = "RESEARCH_ONLY"

FORMATION_MONTH_RANGE = ("2018-12", "2025-10")
PERFORMANCE_MONTH_RANGE = ("2019-01", "2025-11")
HOLDING_PERIODS_COUNT = 83

# --- PIT source artifacts (verified against these exact hashes before any
# market data fetch -- see verify_pit_artifact_hashes) --------------------

PIT_SUMMARY_ARTIFACT_PATH = Path("data/research/sp500_pit_membership_2015_2025_v1_20260815_215706.json")
PIT_SUMMARY_ARTIFACT_SHA256 = "889cb5adebae1bba90f807c61875683bebf6ff69d8fca966eb3798cf59dde6ef"
PIT_RECONSTRUCTION_INPUTS_PATH = Path(
    "data/research/sp500_pit_membership_2015_2025_v1_reconstruction_inputs_20260815_215706.json"
)
PIT_RECONSTRUCTION_INPUTS_SHA256 = "cc7b7a8b3abb535f6e9cae6ef624448bbf7c95979c59b61bcbd96db1130c9bb7"

# --- Frozen parameters -------------------------------------------------

MINIMUM_ELIGIBLE_N = 300
BAR_COVERAGE_MIN_PERCENT = 95.0
VOLUME_LOOKBACK_SESSIONS = 63
VOLUME_MIN_POSITIVE_SESSIONS = 60
DECILE_COUNT = 10
HOLDING_MONTHS = 1

COMMISSION_RATE = 0.0005  # 0.05% -- portfolio_backtest_models.py PortfolioBacktestConfig default, reused as a VALUE only
MINIMUM_FEE_USD = 1.0
SLIPPAGE_BPS = 5.0

TRADING_DAYS_PER_YEAR = 252  # equity-only; see module docstring's isolation note on metrics.py's 365.25

SELECTION_SEED = None
BOOTSTRAP_SEED = 20260820

STRICT_MODE = "strict"
PERMISSIVE_MODE = "permissive"

SPY_SYMBOL = "SPY"


class PitArtifactShaMismatchError(RuntimeError):
    """Raised by verify_pit_artifact_hashes -- fail-closed abort rule
    #3 from the pre-registration artifact: a recomputed SHA-256 that
    does not match the frozen value must halt the run BEFORE any market
    data fetch."""


class MinimumEligibleUniverseError(RuntimeError):
    """Raised when a formation month's eligible ticker count falls below
    MINIMUM_ELIGIBLE_N -- fail-closed abort rule #1."""


class IncompletePricePanelError(RuntimeError):
    """Raised when a held ticker's exit-date price bar is missing from
    the fetched panel -- fail-closed abort rule #2. The official run
    must halt, never silently drop the trade -- see
    `evaluate_exit_prices`'s own docstring for the explicitly-labeled
    sensitivity-variant escape hatch."""


def verify_pit_artifact_hashes(
    summary_path: Path = PIT_SUMMARY_ARTIFACT_PATH,
    summary_expected_sha256: str = PIT_SUMMARY_ARTIFACT_SHA256,
    reconstruction_inputs_path: Path = PIT_RECONSTRUCTION_INPUTS_PATH,
    reconstruction_inputs_expected_sha256: str = PIT_RECONSTRUCTION_INPUTS_SHA256,
) -> None:
    """MUST be called before any market data fetch (see
    run_native_proxy_backtest.py's own ordering). Recomputes each PIT
    source artifact's real SHA-256 and compares against the value frozen
    in the pre-registration artifact -- raises PitArtifactShaMismatchError
    on any mismatch or missing file, never silently proceeds."""
    for path, expected in (
        (summary_path, summary_expected_sha256),
        (reconstruction_inputs_path, reconstruction_inputs_expected_sha256),
    ):
        if not path.is_file():
            raise PitArtifactShaMismatchError(
                f"{path} not found -- cannot verify PIT source artifact before any market data fetch."
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise PitArtifactShaMismatchError(
                f"{path}: expected SHA-256 {expected!r}, recomputed {actual!r} -- "
                f"ABORTING before any market data fetch (pre-registration artifact's own abort rule)."
            )


SOCKET_TIMEOUT_SECONDS = 60.0
_socket_timeout_applied = False


def _apply_socket_timeout() -> None:
    """Real, confirmed root-cause fix, not a defensive guess: `grep
    -n timeout` over alpaca-py's own `alpaca/common/rest.py` (the
    module every SDK call in this file funnels through) found ZERO
    matches -- the library sets no HTTP timeout anywhere, so a stalled
    connection blocks the calling thread forever with nothing to break
    out of it. This was reproduced for real during this study's own
    first full-run attempt: the process sat at 0:01.52 CPU time across
    an 8+ minute wall-clock stall with the exact same single TCP
    connection never changing (confirmed via `lsof`/`ps`, not assumed).
    `socket.setdefaulttimeout` is process-wide (affects every socket
    this process opens, not just Alpaca's), applied once, idempotently,
    here -- the one real choke point every fetch call in this module
    goes through."""
    global _socket_timeout_applied
    if not _socket_timeout_applied:
        import socket

        socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
        _socket_timeout_applied = True


def _call_with_hard_timeout(fn, *args, timeout_seconds: float, **kwargs):
    """`socket.setdefaulttimeout` alone was NOT sufficient -- reproduced
    for real a second time (same CLOSE_WAIT-stuck-connection signature,
    confirmed via `lsof`, this time with the process's OWN CPU time
    frozen even across an actively-polled window) after the first fix
    was already applied. requests/urllib3's connection pooling does not
    reliably consult the process-wide socket default in every code
    path `alpaca-py` exercises. This is the actual hard backstop: runs
    `fn(*args, **kwargs)` in a one-shot worker thread and raises
    `TimeoutError` if it has not returned within `timeout_seconds`,
    regardless of what the stuck call is blocked on internally. The
    orphaned worker thread (Python cannot forcibly kill a thread) is an
    accepted cost for a short-lived batch script -- the caller's retry
    loop treats the TimeoutError exactly like any other transient
    failure and moves on; it does not wait for the orphan."""
    import concurrent.futures

    # Deliberately NOT a `with` block: ThreadPoolExecutor.__exit__ calls
    # shutdown(wait=True), which would block on the very orphaned thread
    # this function exists to stop waiting on, defeating the timeout
    # entirely. shutdown(wait=False) here lets the orphan finish (or
    # never finish) in the background without blocking this call's
    # return.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False)


def get_isolated_stock_historical_client() -> StockHistoricalDataClient:
    """Isolated client builder for this module's own real batched fetch
    -- reuses `_require_credentials` (a pure env-var reader) from
    src/data/alpaca_market_data.py, never that module's `get_stock_bars`
    (see module docstring's isolation note). Applies
    `_apply_socket_timeout` before returning -- see that function's own
    docstring for why this is load-bearing, not optional."""
    _apply_socket_timeout()
    api_key, secret_key = _require_credentials()
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


# --- Real calendar (no weekday assumption anywhere) -------------------


@dataclass(frozen=True)
class RealCalendar:
    """Every real NYSE session in a fetched range, sorted ascending.
    Every date-arithmetic helper below this point resolves against this
    real set -- never `datetime.timedelta`/weekday guessing."""

    sessions: tuple[date_type, ...]

    def __post_init__(self) -> None:
        if list(self.sessions) != sorted(self.sessions):
            raise ValueError("RealCalendar.sessions must be sorted ascending.")

    def last_session_of_month(self, year: int, month: int) -> date_type:
        matches = [d for d in self.sessions if d.year == year and d.month == month]
        if not matches:
            raise ValueError(f"No real NYSE session found for {year}-{month:02d} in this calendar's range.")
        return max(matches)

    def first_session_after(self, d: date_type) -> date_type:
        for session in self.sessions:
            if session > d:
                return session
        raise ValueError(f"No real NYSE session found after {d} in this calendar's range.")

    def trailing_sessions(self, as_of: date_type, count: int) -> tuple[date_type, ...]:
        """The `count` most recent real sessions on or before `as_of`."""
        eligible = [d for d in self.sessions if d <= as_of]
        return tuple(eligible[-count:])

    def sessions_between(self, start: date_type, end: date_type) -> tuple[date_type, ...]:
        """Real sessions in (start, end] -- start exclusive, end inclusive."""
        return tuple(d for d in self.sessions if start < d <= end)


def fetch_real_calendar(trading_client: TradingClient, start: date_type, end: date_type) -> RealCalendar:
    """The ONE real Alpaca calendar call this whole study needs -- see
    module docstring. `trading_client` should come from
    `src.live.order_submission.get_trading_client()` (reused, not
    reinvented)."""
    request = GetCalendarRequest(start=start, end=end)
    entries = trading_client.get_calendar(request)
    sessions = tuple(sorted(entry.date for entry in entries))
    return RealCalendar(sessions=sessions)


def month_range(start_month: str, end_month: str) -> list[tuple[int, int]]:
    """Inclusive (year, month) tuples from 'YYYY-MM' start_month through end_month."""
    start_y, start_m = (int(x) for x in start_month.split("-"))
    end_y, end_m = (int(x) for x in end_month.split("-"))
    months: list[tuple[int, int]] = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + delta
    return total // 12, total % 12 + 1


@dataclass(frozen=True)
class FormationEvent:
    """One monthly rebalance event. `formation_date` is the real,
    Alpaca-calendar-confirmed last NYSE session of the formation month.
    `execution_date` is the real first NYSE session after formation
    (entry Open). `exit_date` is the NEXT formation event's own
    execution_date -- K=1 month, non-overlapping, Open-to-Open."""

    formation_month: str  # 'YYYY-MM'
    formation_date: date_type
    execution_date: date_type
    exit_date: date_type
    m1_month_end: date_type  # last session of formation_month - 1 (momentum end)
    m12_month_end: date_type  # last session of formation_month - 12 (momentum start)


def build_formation_schedule(
    calendar: RealCalendar,
    formation_month_range: tuple[str, str] = FORMATION_MONTH_RANGE,
) -> list[FormationEvent]:
    """Real, Alpaca-calendar-derived formation/execution/exit schedule --
    no weekday assumption. `exit_date` for the LAST formation event is
    the execution_date of the month immediately following the range end
    (one extra month is required from the calendar fetch for this)."""
    months = month_range(*formation_month_range)
    events: list[FormationEvent] = []
    for year, month in months:
        formation_date = calendar.last_session_of_month(year, month)
        execution_date = calendar.first_session_after(formation_date)
        next_year, next_month = _add_months(year, month, 1)
        next_formation_date = calendar.last_session_of_month(next_year, next_month)
        exit_date = calendar.first_session_after(next_formation_date)
        m1_year, m1_month = _add_months(year, month, -1)
        m12_year, m12_month = _add_months(year, month, -12)
        events.append(
            FormationEvent(
                formation_month=f"{year:04d}-{month:02d}",
                formation_date=formation_date,
                execution_date=execution_date,
                exit_date=exit_date,
                m1_month_end=calendar.last_session_of_month(m1_year, m1_month),
                m12_month_end=calendar.last_session_of_month(m12_year, m12_month),
            )
        )
    return events


# --- PIT integration ----------------------------------------------------


@dataclass(frozen=True)
class PitEligibleUniverse:
    formation_date: str
    mode: str
    raw_ticker_count: int
    identity_warnings: frozenset[str]
    eligible_tickers: frozenset[str]
    uncertain: bool
    unresolved_events_in_window: int
    unresolved_tickers_in_window: frozenset[str]


def query_pit_eligible_universe(formation_date_iso: str, mode: str) -> PitEligibleUniverse:
    """Step 1-3 of the pre-registration artifact's PIT integration
    order: real get_membership_as_of query, then identity_warnings
    tickers excluded from the ranking universe. History/liquidity gates
    (step 4) are applied separately, downstream, by
    `evaluate_history_liquidity_eligibility` -- never live_universe.py/
    control_universe.py, see module docstring."""
    result = get_membership_as_of(formation_date_iso, mode=mode)
    eligible = result.tickers - result.identity_warnings
    return PitEligibleUniverse(
        formation_date=formation_date_iso,
        mode=mode,
        raw_ticker_count=result.ticker_count,
        identity_warnings=result.identity_warnings,
        eligible_tickers=eligible,
        uncertain=result.uncertain,
        unresolved_events_in_window=result.unresolved_events_in_window,
        unresolved_tickers_in_window=result.unresolved_tickers_in_window,
    )


# --- Real, batched, point-in-time-adjusted market data fetch -----------


def chunk_symbols(symbols: list[str], chunk_size: int = 150) -> list[list[str]]:
    ordered = sorted(symbols)
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]


def fetch_point_in_time_bars(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: date_type,
    end: date_type,
    asof: date_type,
    chunk_size: int = 150,
) -> dict[str, pd.DataFrame]:
    """Real, batched StockBarsRequest -- Adjustment.ALL, DataFeed.IEX,
    asof=asof (point-in-time corporate-action adjustment, avoiding
    look-ahead from adjustments not yet knowable as of `asof`). Returns
    {symbol: DataFrame} with a DatetimeIndex, one entry per symbol that
    had at least one bar in range -- a symbol absent from the result had
    zero bars in [start, end] on this feed.

    Each batch is retried up to `max_retries` times on a socket-level
    timeout/connection error (see `_apply_socket_timeout`'s own
    docstring for why this library has no built-in timeout at all) --
    a single transient stall must not silently abort an entire
    multi-hour study run; a batch that still fails after retries raises
    the real underlying error, never silently returns partial/empty
    data for it."""
    import socket
    import time

    _apply_socket_timeout()
    out: dict[str, pd.DataFrame] = {}
    for batch_index, batch in enumerate(chunk_symbols(symbols, chunk_size=chunk_size)):
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
            asof=asof.isoformat(),
        )
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                barset = _call_with_hard_timeout(
                    client.get_stock_bars, request, timeout_seconds=SOCKET_TIMEOUT_SECONDS
                )
                break
            except (socket.timeout, OSError, ConnectionError, TimeoutError) as error:
                if attempt == max_retries:
                    raise
                print(
                    f"[RETRY] batch {batch_index} attempt {attempt}/{max_retries} failed "
                    f"({type(error).__name__}: {error}) -- retrying.",
                    flush=True,
                )
                time.sleep(2 * attempt)
        df = barset.df
        if df is None or df.empty:
            continue
        for symbol in df.index.get_level_values("symbol").unique():
            symbol_frame = df.xs(symbol, level="symbol").sort_index()
            symbol_frame.index = pd.to_datetime(symbol_frame.index).tz_localize(None).normalize()
            out[symbol] = symbol_frame
    return out


# --- History/liquidity eligibility gates --------------------------------


@dataclass(frozen=True)
class EligibilityResult:
    ticker: str
    eligible: bool
    reason: str
    bar_coverage_percent: float | None = None
    positive_volume_sessions: int | None = None
    momentum: float | None = None


def compute_momentum(panel: pd.DataFrame, m12_month_end: date_type, m1_month_end: date_type) -> float:
    """MOM_12_1 = AdjustedClose(m1_month_end) / AdjustedClose(m12_month_end) - 1.

    Reads EXACTLY these two rows from `panel` and nothing else -- no
    bar dated the formation date itself, no bar after m1_month_end, is
    ever read here. This is the entire signal formula; kept as its own
    function (rather than inlined inside the eligibility-gate bundle
    below) so its no-leakage property is directly testable in isolation
    from the unrelated bar-coverage/volume gates."""
    p12 = float(panel.loc[panel.index.date == m12_month_end, "close"].iloc[0])
    p1 = float(panel.loc[panel.index.date == m1_month_end, "close"].iloc[0])
    return p1 / p12 - 1.0


def evaluate_history_liquidity_eligibility(
    ticker: str,
    panel: pd.DataFrame | None,
    event: FormationEvent,
    expected_window_sessions: tuple[date_type, ...],
) -> EligibilityResult:
    """Step 4 of the PIT integration order -- applied only to tickers
    that already survived PIT-membership + identity_warnings exclusion.
    ALL of the following must hold, per the pre-registration artifact's
    eligibility_gates section:
    - m12/m1 month-end closes and the entry Open all present and > 0
    - >= 95% real-session bar coverage across the M-12..M-1 window
    - positive volume in >= 60 of the last 63 real sessions on/before
      the formation date
    """
    if panel is None or panel.empty:
        return EligibilityResult(ticker, False, "no price data returned by the feed")

    index_dates = {ts.date() for ts in panel.index}

    if event.m12_month_end not in index_dates or event.m1_month_end not in index_dates:
        return EligibilityResult(ticker, False, "missing month-end close for M-12 or M-1")

    p12 = float(panel.loc[panel.index.date == event.m12_month_end, "close"].iloc[0])
    p1 = float(panel.loc[panel.index.date == event.m1_month_end, "close"].iloc[0])
    if p12 <= 0 or p1 <= 0:
        return EligibilityResult(ticker, False, "non-positive momentum price")

    if event.execution_date not in index_dates:
        return EligibilityResult(ticker, False, "missing entry Open bar")
    entry_open = float(panel.loc[panel.index.date == event.execution_date, "open"].iloc[0])
    if entry_open <= 0:
        return EligibilityResult(ticker, False, "non-positive entry Open")

    covered = sum(1 for d in expected_window_sessions if d in index_dates)
    coverage_percent = 100.0 * covered / len(expected_window_sessions) if expected_window_sessions else 0.0
    if coverage_percent < BAR_COVERAGE_MIN_PERCENT:
        return EligibilityResult(ticker, False, "insufficient bar coverage", bar_coverage_percent=coverage_percent)

    trailing_63_dates = [d for d in expected_window_sessions if d <= event.formation_date][-VOLUME_LOOKBACK_SESSIONS:]
    if not trailing_63_dates:
        trailing_63_dates = [d for d in index_dates if d <= event.formation_date]
        trailing_63_dates = sorted(trailing_63_dates)[-VOLUME_LOOKBACK_SESSIONS:]
    positive_volume_sessions = sum(
        1
        for d in trailing_63_dates
        if d in index_dates and float(panel.loc[panel.index.date == d, "volume"].iloc[0]) > 0
    )
    if positive_volume_sessions < VOLUME_MIN_POSITIVE_SESSIONS:
        return EligibilityResult(
            ticker, False, "insufficient positive-volume sessions",
            bar_coverage_percent=coverage_percent, positive_volume_sessions=positive_volume_sessions,
        )

    momentum = compute_momentum(panel, event.m12_month_end, event.m1_month_end)
    return EligibilityResult(
        ticker, True, "eligible",
        bar_coverage_percent=coverage_percent,
        positive_volume_sessions=positive_volume_sessions,
        momentum=momentum,
    )


# --- Deterministic decile ranking / portfolio construction -------------


def rank_into_deciles(momentum_by_ticker: dict[str, float]) -> dict[str, int]:
    """Decile 0 = highest momentum (Winner), decile 9 = lowest (Loser).
    Descending momentum sort, ALPHABETICAL ticker order as the
    deterministic tie-break on equal momentum -- see pre-registration
    artifact's signal.tie_break field."""
    ordered = sorted(momentum_by_ticker.items(), key=lambda item: (-item[1], item[0]))
    n = len(ordered)
    deciles: dict[str, int] = {}
    for index, (ticker, _momentum) in enumerate(ordered):
        decile = min(DECILE_COUNT - 1, (index * DECILE_COUNT) // n)
        deciles[ticker] = decile
    return deciles


@dataclass(frozen=True)
class TickerLeg:
    ticker: str
    entry_open: float
    exit_open: float | None
    gross_return: float | None
    net_return: float | None
    exit_status: str  # "closed" | "INCOMPLETE_PRICE_PANEL"


def _apply_cost(gross_return: float) -> float:
    """Round-trip commission (entry + exit) + slippage, per the
    pre-registration artifact's cost_assumptions. Applied as a return
    deduction, not a per-share dollar amount, since this is a
    proportional-position-size study (no fixed share counts modeled)."""
    round_trip_commission = 2 * COMMISSION_RATE
    round_trip_slippage = 2 * (SLIPPAGE_BPS / 10_000.0)
    return gross_return - round_trip_commission - round_trip_slippage


def build_ticker_leg(
    ticker: str,
    entry_panel: pd.DataFrame,
    exit_panel: pd.DataFrame | None,
    event: FormationEvent,
    *,
    allow_last_valid_price_fallback: bool = False,
) -> TickerLeg:
    """One ticker's one-month leg. Raises IncompletePricePanelError if
    the exit-date Open bar is missing and
    `allow_last_valid_price_fallback` is False (the OFFICIAL run's
    behavior -- see abort rule #2). Setting the flag True is the
    explicitly-labeled, SEPARATE sensitivity-variant escape hatch the
    pre-registration artifact's missing_data_handling section allows --
    never used by the official run."""
    entry_dates = {ts.date() for ts in entry_panel.index}
    if event.execution_date not in entry_dates:
        raise IncompletePricePanelError(f"{ticker}: entry Open bar missing for {event.execution_date}")
    entry_open = float(entry_panel.loc[entry_panel.index.date == event.execution_date, "open"].iloc[0])

    exit_dates = {ts.date() for ts in exit_panel.index} if exit_panel is not None else set()
    if event.exit_date not in exit_dates:
        if not allow_last_valid_price_fallback:
            raise IncompletePricePanelError(
                f"{ticker}: exit Open bar missing for {event.exit_date} -- official run halts "
                f"(INCOMPLETE_PRICE_PANEL), trade is never silently dropped."
            )
        if exit_panel is None or exit_panel.empty:
            return TickerLeg(ticker, entry_open, None, None, None, "INCOMPLETE_PRICE_PANEL")
        last_valid = exit_panel[exit_panel.index.date <= event.exit_date]
        if last_valid.empty:
            return TickerLeg(ticker, entry_open, None, None, None, "INCOMPLETE_PRICE_PANEL")
        exit_open = float(last_valid["close"].iloc[-1])
    else:
        exit_open = float(exit_panel.loc[exit_panel.index.date == event.exit_date, "open"].iloc[0])

    gross_return = exit_open / entry_open - 1.0
    net_return = _apply_cost(gross_return)
    return TickerLeg(ticker, entry_open, exit_open, gross_return, net_return, "closed")


@dataclass(frozen=True)
class MonthlyPortfolioReturn:
    formation_month: str
    winner_tickers: tuple[str, ...]
    loser_tickers: tuple[str, ...]
    winner_gross_return: float
    winner_net_return: float
    loser_gross_return: float
    loser_net_return: float
    wml_gross_return: float
    wml_net_return: float


def build_monthly_portfolio_return(
    event: FormationEvent,
    deciles: dict[str, int],
    legs_by_ticker: dict[str, TickerLeg],
) -> MonthlyPortfolioReturn:
    """`legs_by_ticker` may be a strict SUBSET of the tickers `deciles`
    selected into the Winner/Loser groups -- a ticker excluded upstream
    for a real, disclosed reason (missing post-entry exit data, e.g. a
    corporate action/delisting -- see `run_one_formation_month`'s own
    docstring for the exact, user-approved handling) simply has no leg
    here and is excluded from the equal-weight average, never causing a
    KeyError or a silently-wrong return."""
    winner_tickers = tuple(sorted(t for t, d in deciles.items() if d == 0 and t in legs_by_ticker))
    loser_tickers = tuple(sorted(t for t, d in deciles.items() if d == DECILE_COUNT - 1 and t in legs_by_ticker))

    def _equal_weight_mean(tickers: tuple[str, ...], field_name: str) -> float:
        values = [getattr(legs_by_ticker[t], field_name) for t in tickers]
        return sum(values) / len(values) if values else 0.0

    winner_gross = _equal_weight_mean(winner_tickers, "gross_return")
    winner_net = _equal_weight_mean(winner_tickers, "net_return")
    loser_gross = _equal_weight_mean(loser_tickers, "gross_return")
    loser_net = _equal_weight_mean(loser_tickers, "net_return")

    return MonthlyPortfolioReturn(
        formation_month=event.formation_month,
        winner_tickers=winner_tickers,
        loser_tickers=loser_tickers,
        winner_gross_return=winner_gross,
        winner_net_return=winner_net,
        loser_gross_return=loser_gross,
        loser_net_return=loser_net,
        wml_gross_return=winner_gross - loser_gross,
        wml_net_return=winner_net - loser_net,
    )


# --- Metrics (equity-only, sqrt(252) -- see module docstring) ----------


def cagr(total_return_fraction: float, years: float) -> float:
    if years <= 0:
        return 0.0
    return (1.0 + total_return_fraction) ** (1.0 / years) - 1.0


def max_drawdown_percent(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min() * 100.0)


def sharpe_sortino(monthly_returns: pd.Series) -> tuple[float | None, float | None]:
    """Annualized with sqrt(12) applied to MONTHLY returns scaled to an
    equivalent of sqrt(TRADING_DAYS_PER_YEAR)/sqrt(12) -- this study's
    return series is monthly (K=1 non-overlapping), not daily, so the
    correct annualization factor for a monthly series is sqrt(12), and
    TRADING_DAYS_PER_YEAR=252 is used only for the equity-curve/CAGR
    "years" conversion elsewhere, never applied directly to a monthly
    return series (that would over-annualize by construction)."""
    if monthly_returns.std(ddof=1) == 0 or monthly_returns.empty:
        return None, None
    mean = monthly_returns.mean()
    std = monthly_returns.std(ddof=1)
    sharpe = (mean / std) * (12 ** 0.5)
    downside = monthly_returns[monthly_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else None
    sortino = (mean / downside_std) * (12 ** 0.5) if downside_std not in (None, 0) else None
    return float(sharpe), (float(sortino) if sortino is not None else None)


def profit_factor(monthly_returns: pd.Series) -> float | None:
    gains = monthly_returns[monthly_returns > 0].sum()
    losses = -monthly_returns[monthly_returns < 0].sum()
    if losses == 0:
        return None
    return float(gains / losses)


def build_equity_curve(monthly_returns: pd.Series, starting_capital: float = 100_000.0) -> pd.Series:
    return starting_capital * (1.0 + monthly_returns).cumprod()
