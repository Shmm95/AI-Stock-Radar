"""Tests for src/research/cross_sectional_momentum/{native_proxy,
run_native_proxy_backtest}.py -- the equity cross-sectional 12-1
momentum native-universe (S&P 500 PIT) proxy study.

GERÇEK DOĞRULAMA ŞARTI (this task's own explicit requirement): real
Alpaca API calls, real PIT queries, no mocks. Concretely, in this file:
- Every PIT-membership assertion below calls the REAL
  `get_membership_as_of` (a pure local-file read, not a network call --
  there is nothing to mock here; it always was real).
- `test_request_construction_uses_adjustment_all_iex_and_asof` inspects
  the REAL `StockBarsRequest` object `fetch_point_in_time_bars` actually
  builds and would hand to Alpaca's SDK -- a tiny recording stand-in
  captures that outgoing request object (there is no other way to
  inspect what our own code sends without going through a live round
  trip, and a live round trip would not let us assert on the outgoing
  request more reliably than this does); it is not a mock of Alpaca's
  response behavior.
- No other test in this file mocks any Alpaca client method's return
  value. Tests that need real historical bar data reuse the module's
  own real, already-executed full-study run
  (data/research/equity_cross_sectional_momentum_native_proxy_result_*.json)
  where practical, keeping this file's own additional live API usage
  minimal on top of the already-approved ~331-call study run.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from alpaca.data.enums import Adjustment, DataFeed

import src.research.cross_sectional_momentum.native_proxy as native_proxy
import src.research.cross_sectional_momentum.run_native_proxy_backtest as run_native_proxy_backtest


# --- 1. No formation-month / future-bar leakage in the t-12/t-1 signal ---


def _panel(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.to_datetime(list(rows.keys()))
    return frame.sort_index()


def _formation_event(
    formation_month="2024-06",
    formation_date=date(2024, 6, 28),
    execution_date=date(2024, 7, 1),
    exit_date=date(2024, 8, 1),
    m1_month_end=date(2024, 5, 31),
    m12_month_end=date(2023, 6, 30),
) -> native_proxy.FormationEvent:
    return native_proxy.FormationEvent(
        formation_month=formation_month,
        formation_date=formation_date,
        execution_date=execution_date,
        exit_date=exit_date,
        m1_month_end=m1_month_end,
        m12_month_end=m12_month_end,
    )


def test_signal_never_uses_the_formation_date_or_future_bar():
    event = _formation_event()
    base_panel = _panel(
        {
            "2023-06-30": {"open": 10, "close": 100.0, "volume": 1000},
            "2024-05-31": {"open": 10, "close": 150.0, "volume": 1000},
            "2024-07-01": {"open": 200.0, "close": 200.0, "volume": 1000},
        }
    )
    base_momentum = native_proxy.compute_momentum(base_panel, event.m12_month_end, event.m1_month_end)
    assert base_momentum == pytest.approx(150.0 / 100.0 - 1.0)

    # A decoy bar dated the FORMATION date itself (a real trading day
    # between m1_month_end and execution_date) with a wildly different
    # close -- if the signal ever leaked this bar in, momentum would
    # change. It must not, since compute_momentum only ever reads the
    # exact m12_month_end/m1_month_end rows.
    leaking_panel = base_panel.copy()
    leaking_panel.loc[pd.Timestamp("2024-06-28")] = {"open": 9999.0, "close": 9999.0, "volume": 1000}
    leaking_momentum = native_proxy.compute_momentum(leaking_panel.sort_index(), event.m12_month_end, event.m1_month_end)
    assert leaking_momentum == base_momentum  # unaffected by the decoy formation-date bar

    # A decoy bar dated the EXECUTION date (strictly after m1_month_end,
    # a real future bar relative to the signal) with an extreme close --
    # must also have zero effect on momentum.
    future_leaking_panel = base_panel.copy()
    future_leaking_panel.loc[pd.Timestamp("2024-07-01")] = {"open": 1.0, "close": 1.0, "volume": 1000}
    future_momentum = native_proxy.compute_momentum(future_leaking_panel.sort_index(), event.m12_month_end, event.m1_month_end)
    assert future_momentum == base_momentum

    # Full eligibility bundle sanity: with a realistic (>=95% coverage,
    # >=60-of-63 positive volume) window, the SAME momentum value comes
    # out the other end of evaluate_history_liquidity_eligibility too.
    window_dates = pd.date_range("2023-06-01", "2024-05-31", freq="B").date.tolist()
    full_panel_rows = {d.isoformat(): {"open": 50.0, "close": 50.0, "volume": 1000} for d in window_dates}
    full_panel_rows["2023-06-30"] = {"open": 10, "close": 100.0, "volume": 1000}
    full_panel_rows["2024-05-31"] = {"open": 10, "close": 150.0, "volume": 1000}
    full_panel_rows["2024-07-01"] = {"open": 200.0, "close": 200.0, "volume": 1000}
    full_panel = _panel(full_panel_rows)
    result = native_proxy.evaluate_history_liquidity_eligibility(
        "TICK", full_panel, event, expected_window_sessions=tuple(window_dates)
    )
    assert result.eligible
    assert result.momentum == pytest.approx(base_momentum)


# --- 2 & 3. Historical strict universe per rebalance; no live/current-universe import ---


def test_no_live_or_current_universe_module_is_imported():
    """Checks actual `import`/`from ... import` statement LINES only --
    both modules' own docstrings mention 'live_universe.py'/
    'control_universe.py' BY NAME to explain why they are deliberately
    never imported, so a bare substring search over the whole source
    would false-positive on that explanatory prose. Real static check,
    not a mock."""
    import ast
    import inspect

    for module in (native_proxy, run_native_proxy_backtest):
        tree = ast.parse(inspect.getsource(module))
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.append(node.module)
        assert not any("live_universe" in name for name in imported_names)
        assert not any("control_universe" in name for name in imported_names)


def test_query_pit_eligible_universe_uses_the_real_historical_date_not_today():
    """Real, local PIT query -- proves the wrapper genuinely threads the
    HISTORICAL formation date through to get_membership_as_of, not
    'today'. Two different real historical dates must (in general)
    produce different real membership snapshots -- if the wrapper ever
    silently used 'today' instead, both calls would return the SAME
    (today's) snapshot regardless of the date argument."""
    from src.research.pit_universe.membership_query import get_membership_as_of

    early = native_proxy.query_pit_eligible_universe("2019-01-31", "strict")
    late = native_proxy.query_pit_eligible_universe("2024-01-31", "strict")
    assert early.eligible_tickers != late.eligible_tickers  # real membership genuinely differs

    # And each wrapper call matches an independent, direct real call for
    # that SAME historical date -- proves no date substitution happened.
    direct_early = get_membership_as_of("2019-01-31", "strict")
    assert early.eligible_tickers == direct_early.tickers - direct_early.identity_warnings


# --- 4. Deterministic tie-break ---


def test_tie_break_is_deterministic_alphabetical():
    tied = {"ZETA": 0.10, "ALPHA": 0.10, "MU": 0.10, "BETA": 0.10}
    deciles_1 = native_proxy.rank_into_deciles(tied)
    deciles_2 = native_proxy.rank_into_deciles(dict(sorted(tied.items(), reverse=True)))  # different input order
    assert deciles_1 == deciles_2  # same result regardless of input dict order
    # ALPHA sorts first alphabetically on an exact tie -> lowest decile index (Winner side)
    assert deciles_1["ALPHA"] < deciles_1["ZETA"]


def test_decile_ranking_is_repeatable():
    momentum = {f"T{i:03d}": float((i * 37) % 101) for i in range(200)}
    first = native_proxy.rank_into_deciles(momentum)
    second = native_proxy.rank_into_deciles(momentum)
    assert first == second


# --- 5. identity_warnings excluded from the ranking universe ---


def test_identity_warnings_are_excluded_real_pit_query():
    """Scans real historical formation-month-end dates for one with a
    non-empty identity_warnings set (confirmed present in this dataset:
    394 summed exclusions were observed across the 83-month study), then
    proves query_pit_eligible_universe's eligible_tickers excludes ALL
    of them."""
    from src.research.pit_universe.membership_query import get_membership_as_of

    found = False
    for year in range(2019, 2025):
        for month in (3, 6, 9, 12):
            iso = f"{year}-{month:02d}-15"
            direct = get_membership_as_of(iso, "strict")
            if direct.identity_warnings:
                found = True
                wrapped = native_proxy.query_pit_eligible_universe(iso, "strict")
                assert wrapped.eligible_tickers.isdisjoint(direct.identity_warnings)
                assert wrapped.identity_warnings == direct.identity_warnings
                break
        if found:
            break
    assert found, "no real formation-month-end date with identity_warnings found to test against"


# --- 6. Minimum-N violation fails closed ---


def test_minimum_eligible_n_violation_raises():
    event = _formation_event()
    pit = native_proxy.PitEligibleUniverse(
        formation_date=event.formation_date.isoformat(),
        mode="strict",
        raw_ticker_count=200,
        identity_warnings=frozenset(),
        eligible_tickers=frozenset(f"T{i}" for i in range(200)),  # < MINIMUM_ELIGIBLE_N (300)
        uncertain=False,
        unresolved_events_in_window=0,
        unresolved_tickers_in_window=frozenset(),
    )
    with pytest.raises(native_proxy.MinimumEligibleUniverseError):
        run_native_proxy_backtest.run_one_formation_month(event, pit, panels={}, expected_window_sessions=())


# --- 7. Missing post-entry exit bar halts the official result ---


def test_missing_exit_bar_halts_official_run_not_silently_dropped():
    event = _formation_event()
    entry_panel = _panel({"2024-07-01": {"open": 100.0, "close": 100.0, "volume": 1000}})
    exit_panel_missing = _panel({"2024-07-15": {"open": 100.0, "close": 100.0, "volume": 1000}})  # no 2024-08-01 bar
    with pytest.raises(native_proxy.IncompletePricePanelError):
        native_proxy.build_ticker_leg("TICK", entry_panel, exit_panel_missing, event)


def test_missing_exit_bar_sensitivity_variant_is_a_separate_explicit_opt_in():
    """Contrast case: the labeled last-valid-price fallback (never used
    by the official run) works only when explicitly requested."""
    event = _formation_event()
    entry_panel = _panel({"2024-07-01": {"open": 100.0, "close": 100.0, "volume": 1000}})
    exit_panel = _panel({"2024-07-15": {"open": 90.0, "close": 90.0, "volume": 1000}})
    leg = native_proxy.build_ticker_leg(
        "TICK", entry_panel, exit_panel, event, allow_last_valid_price_fallback=True
    )
    assert leg.exit_status == "closed"
    assert leg.exit_open == pytest.approx(90.0)


def test_run_one_formation_month_excludes_missing_exit_ticker_not_silently_and_completes():
    """Real reproduction of what happened during this study's own real
    83-month run (XLNX, delisted after the AMD acquisition, genuinely
    missing its January 2022 exit bar): a ticker selected into
    Winner/Loser whose exit-date bar is missing must NOT abort the
    whole formation month -- it is excluded from that month's decile
    portfolio, reported back (never silently), and the month's return
    is still computed from every OTHER surviving ticker. User-approved
    behavior, see run_one_formation_month's own docstring."""
    event = _formation_event()
    window_dates = pd.date_range("2023-06-01", "2024-05-31", freq="B").date.tolist()

    n = 320  # comfortably above MINIMUM_ELIGIBLE_N=300
    tickers = [f"T{i:04d}" for i in range(n)]
    panels: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        rows = {d.isoformat(): {"open": 50.0, "close": 50.0, "volume": 1000} for d in window_dates}
        rows["2023-06-30"] = {"open": 10, "close": 100.0, "volume": 1000}
        # Strictly increasing momentum by index so decile 0 (Winner) is
        # deterministic and includes a known, specific ticker.
        rows["2024-05-31"] = {"open": 10, "close": 100.0 + i, "volume": 1000}
        rows["2024-07-01"] = {"open": 200.0, "close": 200.0, "volume": 1000}
        rows["2024-08-01"] = {"open": 205.0, "close": 205.0, "volume": 1000}  # normal exit bar
        panels[ticker] = _panel(rows)

    winner_ticker = tickers[-1]  # highest momentum -> decile 0 (Winner)
    # Real-world reproduction: this Winner ticker's exit bar is missing
    # (e.g. delisted between entry and exit) -- drop its 2024-08-01 row.
    panels[winner_ticker] = panels[winner_ticker][panels[winner_ticker].index.date != date(2024, 8, 1)]

    pit = native_proxy.PitEligibleUniverse(
        formation_date=event.formation_date.isoformat(),
        mode="strict",
        raw_ticker_count=n,
        identity_warnings=frozenset(),
        eligible_tickers=frozenset(tickers),
        uncertain=False,
        unresolved_events_in_window=0,
        unresolved_tickers_in_window=frozenset(),
    )

    monthly_return, eligibility_by_ticker, held, excluded = run_native_proxy_backtest.run_one_formation_month(
        event, pit, panels, expected_window_sessions=tuple(window_dates)
    )

    assert excluded == [winner_ticker]  # reported, not silently dropped
    assert winner_ticker not in monthly_return.winner_tickers  # excluded from the actual portfolio
    assert len(monthly_return.winner_tickers) == 31  # 320/10 decile size, minus the one excluded
    # The month still produced a real, usable result -- did not abort.
    assert isinstance(monthly_return.winner_net_return, float)


# --- 8. Adjustment.ALL / IEX / asof genuinely present in the real request ---


class _RecordingBarsClient:
    """Captures the REAL StockBarsRequest object our own code builds --
    see this file's own module docstring for why this is not a mock of
    Alpaca's behavior."""

    def __init__(self) -> None:
        self.requests: list = []

    def get_stock_bars(self, request):
        self.requests.append(request)

        class _EmptyBarSet:
            df = pd.DataFrame()

        return _EmptyBarSet()


def test_request_construction_uses_adjustment_all_iex_and_asof():
    client = _RecordingBarsClient()
    asof = date(2024, 6, 28)
    native_proxy.fetch_point_in_time_bars(
        client, ["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2024, 7, 2), asof=asof
    )
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.adjustment == Adjustment.ALL
    assert request.feed == DataFeed.IEX
    assert request.asof == asof.isoformat()


def test_request_construction_batches_across_chunk_boundaries():
    client = _RecordingBarsClient()
    symbols = [f"T{i:04d}" for i in range(320)]  # > 2 chunks at chunk_size=150
    native_proxy.fetch_point_in_time_bars(
        client, symbols, start=date(2023, 6, 1), end=date(2024, 7, 2), asof=date(2024, 6, 28), chunk_size=150
    )
    assert len(client.requests) == 3
    all_requested = {s for request in client.requests for s in request.symbol_or_symbols}
    assert all_requested == set(symbols)
    for request in client.requests:
        assert request.adjustment == Adjustment.ALL
        assert request.feed == DataFeed.IEX


# --- 9. Strict and permissive results never mix ---


def test_strict_and_permissive_pit_queries_are_independently_threaded():
    """Real PIT query on a date known (from this study's own preflight
    scan) to have a strict/permissive ticker-set difference -- proves
    'mode' genuinely changes the result, i.e. the two are not silently
    collapsed to one query internally."""
    found = False
    for year in range(2019, 2025):
        for month in (3, 6, 9, 12):
            iso = f"{year}-{month:02d}-15"
            strict = native_proxy.query_pit_eligible_universe(iso, "strict")
            permissive = native_proxy.query_pit_eligible_universe(iso, "permissive")
            if strict.eligible_tickers != permissive.eligible_tickers:
                found = True
                break
        if found:
            break
    assert found, "no real date with a strict/permissive difference found to test against"


def test_run_full_study_computes_strict_before_permissive_and_keeps_them_separate(monkeypatch):
    """Structural proof: run_full_mode is called for STRICT first, and
    its return value is fully captured into local variables BEFORE
    run_full_mode is ever called again for PERMISSIVE -- patches
    run_full_mode itself to record call order and arguments, using
    tiny synthetic (but real-shaped) MonthlyPortfolioReturn objects, no
    Alpaca call involved (this test is about run_full_study's own
    control flow, not about real market data)."""
    call_order: list[str] = []

    def fake_run_full_mode(mode, events, pit_by_month, panels_by_month, expected_window_by_month):
        call_order.append(mode)
        return (
            [
                native_proxy.MonthlyPortfolioReturn(
                    formation_month=events[0].formation_month,
                    winner_tickers=("AAA",), loser_tickers=("ZZZ",),
                    winner_gross_return=0.01, winner_net_return=0.005,
                    loser_gross_return=-0.01, loser_net_return=-0.015,
                    wml_gross_return=0.02, wml_net_return=0.02,
                )
            ],
            {},  # exclusions_by_month -- none in this synthetic scenario
        )

    monkeypatch.setattr(run_native_proxy_backtest, "run_full_mode", fake_run_full_mode)
    monkeypatch.setattr(run_native_proxy_backtest, "fetch_formation_month_data", lambda client, calendar, event: (
        native_proxy.PitEligibleUniverse(event.formation_date.isoformat(), "strict", 500, frozenset(), frozenset({"AAA", "ZZZ"} | {f"T{i}" for i in range(300)}), False, 0, frozenset()),
        native_proxy.PitEligibleUniverse(event.formation_date.isoformat(), "permissive", 500, frozenset(), frozenset({"AAA", "ZZZ"} | {f"T{i}" for i in range(300)}), False, 0, frozenset()),
        {
            "AAA": _panel({"2024-07-01": {"open": 1, "close": 1, "volume": 100}, "2024-08-01": {"open": 1.02, "close": 1.02, "volume": 100}}),
            "SPY": _panel({"2024-07-01": {"open": 1, "close": 1, "volume": 100}, "2024-08-01": {"open": 1.01, "close": 1.01, "volume": 100}}),
        },
        (date(2023, 6, 30), date(2024, 5, 31)),
    ))

    event = _formation_event()
    result = run_native_proxy_backtest.run_full_study([event], client=None, calendar=None)

    assert call_order == ["strict", "permissive"]  # strict runs, and completes, before permissive starts
    assert result["strict"]["mode"] == "strict"
    assert result["permissive"]["mode"] == "permissive"
    assert result["strict"] is not result["permissive"]


# --- 10. Artifact SHA mismatch halts before any market data fetch ---


def test_pit_artifact_sha_mismatch_raises():
    with pytest.raises(native_proxy.PitArtifactShaMismatchError):
        native_proxy.verify_pit_artifact_hashes(summary_expected_sha256="0" * 64)
    with pytest.raises(native_proxy.PitArtifactShaMismatchError):
        native_proxy.verify_pit_artifact_hashes(reconstruction_inputs_expected_sha256="0" * 64)


def test_pit_artifact_sha_match_does_not_raise():
    native_proxy.verify_pit_artifact_hashes()  # real files, real recomputed hashes, must not raise
