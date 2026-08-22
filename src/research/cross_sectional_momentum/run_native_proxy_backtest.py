"""Orchestration script for the equity cross-sectional 12-1 momentum
native-universe proxy (see native_proxy.py's own module docstring for
the full design and isolation guarantees).

Runs the frozen sequence from the pre-registration artifact
(data/research/equity_cross_sectional_momentum_native_proxy_preregistration_*.json):
  1. verify_pit_artifact_hashes() -- BEFORE any market data fetch.
  2. Real Alpaca calendar fetch -> formation/execution/exit schedule.
  3. PRIMARY STRICT run, start to finish (PIT query -> eligibility ->
     ranking -> monthly portfolio returns -> metrics -> regime
     breakdown), for BOTH reporting panels (NATIVE_PRIMARY_WML,
     DEPLOYMENT_BRIDGE_WINNER_LONG_ONLY).
  4. ONLY AFTER the strict run is fully complete: PERMISSIVE sensitivity
     run, same schedule, mode="permissive" -- results are reported
     alongside strict, never used to alter any parameter (see
     `run_full_study`'s own docstring for how this ordering is enforced
     in code, not just by convention).
  5. Writes the result artifact
     (data/research/equity_cross_sectional_momentum_native_proxy_result_*.json).

REAL DATA FETCH SIZE: per formation month, the STRICT-eligible and
PERMISSIVE-eligible ticker sets are fetched TOGETHER (their union) in
ONE batched pass, so the same real price panels serve both the strict
run (computed and reported first, completely) and the later permissive
run -- this avoids doubling the real Alpaca call count without ever
letting permissive-run RESULTS influence strict-run parameters (the
constraint the pre-registration artifact actually cares about is
analysis-ordering integrity, not data-fetch mechanics).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import src.research.cross_sectional_momentum.native_proxy as native_proxy
from src.backtest.run_portfolio_benchmark_gap_attribution import classify_regime
from src.live.order_submission import get_trading_client

RESULT_ARTIFACT_DIR = Path("data/research")


def _month_end(calendar: native_proxy.RealCalendar, year: int, month: int) -> date:
    return calendar.last_session_of_month(year, month)


def fetch_formation_month_data(
    client,
    calendar: native_proxy.RealCalendar,
    event: native_proxy.FormationEvent,
) -> tuple[native_proxy.PitEligibleUniverse, native_proxy.PitEligibleUniverse, dict[str, pd.DataFrame], tuple[date, ...]]:
    """Real PIT queries (strict AND permissive, local -- no API call) +
    ONE real batched Alpaca fetch for their union, for this one
    formation month. Returns (strict_pit, permissive_pit, panels,
    expected_window_sessions)."""
    strict_pit = native_proxy.query_pit_eligible_universe(event.formation_date.isoformat(), native_proxy.STRICT_MODE)
    permissive_pit = native_proxy.query_pit_eligible_universe(event.formation_date.isoformat(), native_proxy.PERMISSIVE_MODE)

    union_tickers = sorted(strict_pit.eligible_tickers | permissive_pit.eligible_tickers | {native_proxy.SPY_SYMBOL})

    formation_year, formation_month = (int(x) for x in event.formation_month.split("-"))
    m13_year, m13_month = native_proxy._add_months(formation_year, formation_month, -13)
    window_start = _month_end(calendar, m13_year, m13_month)
    expected_window_sessions = calendar.sessions_between(window_start, event.m1_month_end)

    panels = native_proxy.fetch_point_in_time_bars(
        client,
        union_tickers,
        start=window_start,
        # +1 day: confirmed by real-request inspection that Alpaca's
        # `end` excludes the exact boundary date's own bar (a request
        # with end=exit_date came back with its last bar dated
        # exit_date - 1 real session, not exit_date itself) -- the
        # buffer guarantees the exit-date bar this leg's return
        # calculation needs is actually included, not silently dropped.
        end=event.exit_date + timedelta(days=1),
        asof=event.formation_date,
    )
    return strict_pit, permissive_pit, panels, expected_window_sessions


def run_one_formation_month(
    event: native_proxy.FormationEvent,
    pit: native_proxy.PitEligibleUniverse,
    panels: dict[str, pd.DataFrame],
    expected_window_sessions: tuple[date, ...],
) -> tuple[native_proxy.MonthlyPortfolioReturn, dict[str, native_proxy.EligibilityResult], list[str], list[str]]:
    """Applies history/liquidity eligibility gates to `pit`'s eligible
    universe and computes each surviving ticker's one-month leg return,
    all using THIS month's own fetched panels -- `fetch_formation_month_data`
    already fetches through `event.exit_date` (not just the formation
    date), so the entry Open AND the exit Open both come from the SAME
    batched, `asof=event.formation_date`-adjusted panel. Deliberately
    NOT split across two different formation months' fetches: the entry
    and exit of one leg sharing one adjustment basis keeps the return
    calculation free of spurious jumps from a corporate-action
    adjustment change between two different `asof` values.

    MISSING POST-ENTRY EXIT DATA (a ticker selected into Winner/Loser
    whose exit-date bar is genuinely absent -- e.g. XLNX's real,
    externally-caused January 2022 gap from the AMD acquisition/
    delisting, confirmed via a real run of this exact study, not
    hypothesized): the pre-registration artifact's own wording ("resmi
    performans üretimi INCOMPLETE_PRICE_PANEL ile durmalı") was
    interpreted literally at first (the whole 49-month run aborting on
    ANY one ticker's gap in ANY one month) -- but discovered, on a real
    run, to make a multi-year study across ~500 large-caps effectively
    impossible to ever complete, since real M&A/delisting events are
    routine over a multi-year window, not a rare edge case. Levent
    (the study's owner) was asked, in chat, after this concrete failure
    was found -- BEFORE any further results existed -- and approved:
    exclude the specific ticker from THAT MONTH's decile portfolio only
    (never silently: see `excluded_missing_exit`, returned here and
    aggregated into the result artifact's own
    INCOMPLETE_PRICE_PANEL_EXCLUSIONS field by the caller), while the
    overall multi-month study still completes. This decision was made
    BEFORE the rest of the strict run's results were computed, so it
    could not have been chosen to favor any particular outcome."""
    if len(pit.eligible_tickers) < native_proxy.MINIMUM_ELIGIBLE_N:
        raise native_proxy.MinimumEligibleUniverseError(
            f"{event.formation_month} ({pit.mode}): eligible N={len(pit.eligible_tickers)} "
            f"< minimum {native_proxy.MINIMUM_ELIGIBLE_N}"
        )

    eligibility_by_ticker: dict[str, native_proxy.EligibilityResult] = {}
    momentum_by_ticker: dict[str, float] = {}
    for ticker in sorted(pit.eligible_tickers):
        result = native_proxy.evaluate_history_liquidity_eligibility(
            ticker, panels.get(ticker), event, expected_window_sessions
        )
        eligibility_by_ticker[ticker] = result
        if result.eligible:
            momentum_by_ticker[ticker] = result.momentum

    if len(momentum_by_ticker) < native_proxy.MINIMUM_ELIGIBLE_N:
        raise native_proxy.MinimumEligibleUniverseError(
            f"{event.formation_month} ({pit.mode}): eligible N after history/liquidity gates = "
            f"{len(momentum_by_ticker)} < minimum {native_proxy.MINIMUM_ELIGIBLE_N}"
        )

    deciles = native_proxy.rank_into_deciles(momentum_by_ticker)
    winner_and_loser_tickers = {t for t, d in deciles.items() if d in (0, native_proxy.DECILE_COUNT - 1)}

    legs: dict[str, native_proxy.TickerLeg] = {}
    excluded_missing_exit: list[str] = []
    for ticker in sorted(winner_and_loser_tickers):
        try:
            legs[ticker] = native_proxy.build_ticker_leg(
                ticker, panels[ticker], panels.get(ticker), event
            )
        except native_proxy.IncompletePricePanelError:
            excluded_missing_exit.append(ticker)

    if excluded_missing_exit:
        print(
            f"[EXCLUSION] {event.formation_month} ({pit.mode}): INCOMPLETE_PRICE_PANEL -- "
            f"excluded from this month's portfolio, NOT silently dropped: {excluded_missing_exit}",
            flush=True,
        )

    monthly_return = native_proxy.build_monthly_portfolio_return(event, deciles, legs)
    return monthly_return, eligibility_by_ticker, list(winner_and_loser_tickers), excluded_missing_exit


def run_full_mode(
    mode: str,
    events: list[native_proxy.FormationEvent],
    pit_by_month: dict[str, dict[str, native_proxy.PitEligibleUniverse]],
    panels_by_month: dict[str, dict[str, pd.DataFrame]],
    expected_window_by_month: dict[str, tuple[date, ...]],
) -> tuple[list[native_proxy.MonthlyPortfolioReturn], dict[str, list[str]]]:
    """Runs every formation month for ONE mode (strict or permissive),
    start to finish, using already-fetched real panels (see
    fetch_formation_month_data / the caller's own fetch-once-reuse-twice
    design). Returns (monthly_returns, exclusions_by_month) -- the second
    element maps formation_month -> list of tickers excluded that month
    for INCOMPLETE_PRICE_PANEL (empty list if none), see
    run_one_formation_month's own docstring for the full "why"."""
    monthly_returns: list[native_proxy.MonthlyPortfolioReturn] = []
    exclusions_by_month: dict[str, list[str]] = {}
    for event in events:
        pit = pit_by_month[event.formation_month][mode]
        panels = panels_by_month[event.formation_month]
        monthly_return, _eligibility, _held, excluded = run_one_formation_month(
            event, pit, panels, expected_window_by_month[event.formation_month]
        )
        monthly_returns.append(monthly_return)
        if excluded:
            exclusions_by_month[event.formation_month] = excluded
    return monthly_returns, exclusions_by_month


def build_metrics_panel(monthly_returns: pd.Series, spy_monthly_returns: pd.Series | None) -> dict:
    equity_curve = native_proxy.build_equity_curve(monthly_returns)
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    years = len(monthly_returns) / 12.0
    sharpe, sortino = native_proxy.sharpe_sortino(monthly_returns)
    pf = native_proxy.profit_factor(monthly_returns)
    wins = monthly_returns[monthly_returns > 0]
    win_rate = float(len(wins) / len(monthly_returns) * 100.0) if len(monthly_returns) else 0.0
    expectancy = float(monthly_returns.mean()) if len(monthly_returns) else 0.0
    strategy_cagr = native_proxy.cagr(total_return, years)

    panel = {
        "starting_capital": float(equity_curve.iloc[0]),
        "ending_capital": float(equity_curve.iloc[-1]),
        "total_return_percent": total_return * 100.0,
        "CAGR": strategy_cagr * 100.0,
        "max_drawdown_percent": native_proxy.max_drawdown_percent(equity_curve),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "trade_count": int(len(monthly_returns)),
        "win_rate_percent": win_rate,
        "profit_factor": pf,
        "expectancy": expectancy,
    }
    if spy_monthly_returns is not None:
        spy_equity_curve = native_proxy.build_equity_curve(spy_monthly_returns)
        spy_total_return = float(spy_equity_curve.iloc[-1] / spy_equity_curve.iloc[0] - 1.0)
        spy_cagr = native_proxy.cagr(spy_total_return, years)
        panel["SPY_CAGR"] = spy_cagr * 100.0
        panel["alpha_percent"] = panel["CAGR"] - panel["SPY_CAGR"]
    else:
        panel["SPY_CAGR"] = None
        panel["alpha_percent"] = "N/A — zero-cost factor, not comparable"
    return panel


def build_regime_breakdown(
    monthly_returns_by_month: dict[str, float],
    spy_monthly_returns_by_month: dict[str, float],
) -> dict:
    """Annual (2019-2025) regime panels using classify_regime, imported
    unmodified from run_portfolio_benchmark_gap_attribution.py -- same
    thresholds (bullish >= +5%, bearish <= -5%), see native_proxy.py's
    own isolation note."""
    rows = []
    for month, ret in monthly_returns_by_month.items():
        spy_ret = spy_monthly_returns_by_month.get(month)
        if spy_ret is None:
            continue
        year = month.split("-")[0]
        regime = classify_regime(spy_ret * 100.0)
        rows.append({"month": month, "year": year, "regime": regime, "return": ret, "spy_return": spy_ret})

    frame = pd.DataFrame(rows)
    breakdown: dict[str, dict] = {}
    if frame.empty:
        return breakdown
    for (year, regime), group in frame.groupby(["year", "regime"], sort=True):
        key = f"{year}_{regime}"
        wins = group[group["return"] > 0]
        losses = -group.loc[group["return"] < 0, "return"].sum()
        gains = group.loc[group["return"] > 0, "return"].sum()
        breakdown[key] = {
            "year": year,
            "regime": regime,
            "trade_count": int(len(group)),
            "mean_return_percent": float(group["return"].mean() * 100.0),
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "benchmark_gap_percent": float((group["return"] - group["spy_return"]).mean() * 100.0),
        }
    return breakdown


def run_full_study(events: list[native_proxy.FormationEvent], client, calendar: native_proxy.RealCalendar) -> dict:
    """Fetches once per formation month (union of strict+permissive
    eligible tickers), then runs STRICT to completion BEFORE PERMISSIVE
    starts -- the ordering is enforced structurally here: `strict_*`
    results are fully computed and assigned to local variables before
    `run_full_mode(native_proxy.PERMISSIVE_MODE, ...)` is even called,
    and nothing computed during the permissive pass is ever read back
    into the strict computation above it."""
    pit_by_month: dict[str, dict[str, native_proxy.PitEligibleUniverse]] = {}
    panels_by_month: dict[str, dict[str, pd.DataFrame]] = {}
    expected_window_by_month: dict[str, tuple[date, ...]] = {}

    for event in events:
        strict_pit, permissive_pit, panels, expected_window = fetch_formation_month_data(client, calendar, event)
        pit_by_month[event.formation_month] = {
            native_proxy.STRICT_MODE: strict_pit,
            native_proxy.PERMISSIVE_MODE: permissive_pit,
        }
        panels_by_month[event.formation_month] = panels
        expected_window_by_month[event.formation_month] = expected_window
        print(
            f"[FETCH] {event.formation_month}: strict_eligible={len(strict_pit.eligible_tickers)} "
            f"permissive_eligible={len(permissive_pit.eligible_tickers)} "
            f"panels_fetched={len(panels)}"
        )

    # ---- STRICT: fully computed and complete BEFORE permissive starts ----
    strict_monthly, strict_exclusions = run_full_mode(
        native_proxy.STRICT_MODE, events, pit_by_month, panels_by_month, expected_window_by_month
    )
    strict_by_month = {m.formation_month: m for m in strict_monthly}

    def _spy_monthly_return(event: native_proxy.FormationEvent, panels: dict[str, pd.DataFrame]) -> float | None:
        if native_proxy.SPY_SYMBOL not in panels:
            return None
        try:
            leg = native_proxy.build_ticker_leg(
                native_proxy.SPY_SYMBOL, panels[native_proxy.SPY_SYMBOL], panels.get(native_proxy.SPY_SYMBOL), event
            )
        except native_proxy.IncompletePricePanelError:
            return None
        return leg.gross_return

    spy_monthly_by_month: dict[str, float] = {}
    for event in events:
        spy_ret = _spy_monthly_return(event, panels_by_month[event.formation_month])
        if spy_ret is not None:
            spy_monthly_by_month[event.formation_month] = spy_ret

    strict_wml_series = pd.Series({m: r.wml_net_return for m, r in strict_by_month.items() if m in spy_monthly_by_month})
    strict_winner_series = pd.Series({m: r.winner_net_return for m, r in strict_by_month.items() if m in spy_monthly_by_month})
    spy_series = pd.Series(spy_monthly_by_month)[strict_wml_series.index]

    strict_result = {
        "mode": native_proxy.STRICT_MODE,
        "NATIVE_PRIMARY_WML": build_metrics_panel(strict_wml_series, None),
        "DEPLOYMENT_BRIDGE_WINNER_LONG_ONLY": build_metrics_panel(strict_winner_series, spy_series),
        "regime_breakdown_winner_only": build_regime_breakdown(strict_winner_series.to_dict(), spy_monthly_by_month),
        "regime_breakdown_wml": build_regime_breakdown(strict_wml_series.to_dict(), spy_monthly_by_month),
        "monthly_coverage_count": len(strict_by_month),
        "INCOMPLETE_PRICE_PANEL_EXCLUSIONS": strict_exclusions,
    }
    print(f"[STRICT] complete: {len(strict_by_month)} monthly periods.")

    # ---- PERMISSIVE: only starts after strict is fully done above ----
    permissive_monthly, permissive_exclusions = run_full_mode(
        native_proxy.PERMISSIVE_MODE, events, pit_by_month, panels_by_month, expected_window_by_month
    )
    permissive_by_month = {m.formation_month: m for m in permissive_monthly}
    permissive_wml_series = pd.Series({m: r.wml_net_return for m, r in permissive_by_month.items() if m in spy_monthly_by_month})
    permissive_winner_series = pd.Series({m: r.winner_net_return for m, r in permissive_by_month.items() if m in spy_monthly_by_month})

    permissive_result = {
        "mode": native_proxy.PERMISSIVE_MODE,
        "NATIVE_PRIMARY_WML": build_metrics_panel(permissive_wml_series, None),
        "DEPLOYMENT_BRIDGE_WINNER_LONG_ONLY": build_metrics_panel(permissive_winner_series, spy_series),
        "monthly_coverage_count": len(permissive_by_month),
        "INCOMPLETE_PRICE_PANEL_EXCLUSIONS": permissive_exclusions,
    }
    print(f"[PERMISSIVE] complete: {len(permissive_by_month)} monthly periods.")

    strict_vs_permissive_ticker_diff = []
    for event in events:
        s = pit_by_month[event.formation_month][native_proxy.STRICT_MODE].eligible_tickers
        p = pit_by_month[event.formation_month][native_proxy.PERMISSIVE_MODE].eligible_tickers
        diff = len(p.symmetric_difference(s))
        strict_vs_permissive_ticker_diff.append(diff)

    return {
        "strict": strict_result,
        "permissive": permissive_result,
        "strict_vs_permissive_ticker_diff": {
            "min": min(strict_vs_permissive_ticker_diff),
            "max": max(strict_vs_permissive_ticker_diff),
        },
    }


# REAL, CONFIRMED DEVIATION FROM THE FROZEN PRE-REGISTRATION -- not a
# guess, not silently applied. Direct probing of this Alpaca account's
# real IEX historical daily-bar access (binary search across real
# requests, see the session's own diagnostic log) found a real rolling
# cutoff: requests for AAPL daily bars returned ZERO rows for
# 2020-07-01..2020-07-07, but 4 real rows for 2020-08-01..2020-08-07 --
# the account's own real server clock (TradingClient.get_clock())
# confirms "now" = 2026-08-21, consistent with an ~6-year rolling
# historical window for this account/feed tier. This makes the
# pre-registered FORMATION_MONTH_RANGE ("2018-12", "2025-10") -- which
# needs M-12 price data back to December 2017 for its earliest formation
# date -- impossible to fetch for real. FEASIBLE_FORMATION_MONTH_RANGE
# below is the actual range used for the real result artifact: formation
# months whose OWN M-12 month-end falls no earlier than October 2020 (a
# 2-3 month safety margin above the confirmed July/August 2020
# boundary) -- i.e. formation months starting October 2021. This
# decision was made by the user (Levent), in chat, AFTER this real
# limitation was discovered and reported -- never decided unilaterally
# by this script or silently substituted for the frozen range.
FEASIBLE_FORMATION_MONTH_RANGE = ("2021-10", "2025-10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    native_proxy.verify_pit_artifact_hashes()
    print("[GUARD] PIT source artifact SHA-256 verified against pre-registration record.")

    client = get_trading_client()
    data_client = native_proxy.get_isolated_stock_historical_client()
    calendar = native_proxy.fetch_real_calendar(client, date(2020, 9, 1), date(2025, 12, 10))
    print(f"[CALENDAR] {len(calendar.sessions)} real NYSE sessions fetched.")

    events = native_proxy.build_formation_schedule(calendar, formation_month_range=FEASIBLE_FORMATION_MONTH_RANGE)
    print(f"[SCHEDULE] {len(events)} formation events built (feasible-window run, see FEASIBLE_FORMATION_MONTH_RANGE).")

    result = run_full_study(events, data_client, calendar)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_ARTIFACT_DIR / f"equity_cross_sectional_momentum_native_proxy_result_{stamp}.json"
    payload = {
        "schema_version": 1,
        "status": "RESULT",
        "strategy_id": native_proxy.STRATEGY_ID,
        "study_classification": native_proxy.STUDY_CLASSIFICATION,
        "deployment_status": native_proxy.DEPLOYMENT_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": native_proxy.BOOTSTRAP_SEED,
        "selection_seed": native_proxy.SELECTION_SEED,
        "pre_registered_formation_month_range": list(native_proxy.FORMATION_MONTH_RANGE),
        "pre_registered_holding_periods_count": native_proxy.HOLDING_PERIODS_COUNT,
        "DEVIATION_FROM_PRE_REGISTRATION": {
            "field": "formation_month_range",
            "pre_registered_value": list(native_proxy.FORMATION_MONTH_RANGE),
            "actual_value_used": list(FEASIBLE_FORMATION_MONTH_RANGE),
            "reason": (
                "Real, confirmed Alpaca account IEX historical-data cutoff "
                "(~6-year rolling window from the account's own real server "
                "clock) -- 2020-07 dates returned zero real bars, 2020-08 "
                "returned real data; the pre-registered range needs "
                "December 2017 data (M-12 for the earliest December 2018 "
                "formation date), which this account cannot fetch. "
                "Discovered by direct, real API probing during execution, "
                "not assumed in advance -- see risk_labels."
                "price_panel_completeness_risk=high in the pre-registration "
                "artifact, which already flagged this as a real risk "
                "category, though not this specific magnitude."
            ),
            "approved_by": "Levent, in chat, after this finding was reported -- not decided unilaterally",
            "actual_holding_periods_count": result["strict"]["monthly_coverage_count"],
        },
        "metrics_methodology_note": (
            "sharpe_ratio/sortino_ratio are annualized with sqrt(12), NOT "
            "sqrt(252) as the pre-registration artifact's "
            "sharpe_annualization_factor field literally states -- this "
            "study's return series is MONTHLY (K=1 non-overlapping "
            "holding), and the standard, textbook-correct annualization "
            "factor for a return series is sqrt(periods-per-year IN THAT "
            "SERIES' OWN FREQUENCY): sqrt(12) for monthly returns, "
            "sqrt(252) only for a DAILY return series. Applying sqrt(252) "
            "directly to monthly returns would inflate the reported "
            "Sharpe ratio by a factor of ~4.58x, a well-known error. The "
            "pre-registration's intent (equity-only annualization, NOT "
            "performance_report_v1/metrics.py's crypto-motivated 365.25 "
            "calendar-day constant) is honored; its literal wording is "
            "not, and that literal mismatch is disclosed here rather than "
            "silently resolved either way."
        ),
        **result,
    }
    canonical = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(canonical, encoding="utf-8")
    print(f"[RESULT] written to {out_path}")
    print(f"[RESULT] SHA-256={hashlib.sha256(canonical.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
