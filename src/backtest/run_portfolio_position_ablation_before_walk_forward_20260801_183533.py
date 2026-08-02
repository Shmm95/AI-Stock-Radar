"""Maximum-open-position ablation for the shared-cash portfolio engine.

The module downloads and prepares the tested universe once, then runs the same
strategy and risk configuration with several maximum-open-position limits.
Each variant is evaluated against an equal-weight buy-and-hold benchmark scaled
to that variant's own average market exposure.

This is a historical research tool only. It cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import (
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
)
DEFAULT_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BTC-USD",
    "ETH-USD",
)
DEFAULT_PERIOD = "5y"
DEFAULT_REGIME_PERIOD = "10y"
DEFAULT_POSITION_LIMITS = (3, 4, 5)
DEFAULT_OUTPUT_DIRECTORY = (
    Path("data") / "backtests" / "portfolio" / "position_ablation"
)


def _money(value: float) -> float:
    return round(float(value), 2)


def _price(value: float) -> float:
    return round(float(value), 8)


def _is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith(("-USD", "-EUR", "-GBP"))


def calculate_cagr(
    initial: float,
    ending: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Calculate annualized return for a dated equity interval."""

    years = max((end - start).days, 0) / 365.25
    if years <= 0 or initial <= 0 or ending <= 0:
        return 0.0
    return round(((ending / initial) ** (1 / years) - 1) * 100, 4)


def return_drawdown_ratio(total_return: float, drawdown: float) -> float:
    """Return total-return divided by maximum drawdown."""

    if drawdown <= 0:
        return float("inf") if total_return > 0 else 0.0
    return round(total_return / drawdown, 4)


def run_matched_benchmark(
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    initial_cash: float,
    exposure_percent: float,
    config: PortfolioBacktestConfig,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run an equal-weight buy-and-hold benchmark at fixed exposure."""

    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero.")
    if not 0 < exposure_percent <= 100:
        raise ValueError(
            "exposure_percent must be greater than 0 and at most 100."
        )
    if not data_by_ticker:
        raise ValueError("data_by_ticker cannot be empty.")

    normalized: dict[str, pd.DataFrame] = {}
    for raw_ticker, raw_data in data_by_ticker.items():
        ticker = raw_ticker.strip().upper()
        data = raw_data.copy()
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_convert(None)
        data = data.sort_index().loc[
            ~data.index.duplicated(keep="last")
        ]
        missing = [
            column for column in ("Open", "Close")
            if column not in data.columns
        ]
        if missing:
            raise ValueError(
                f"{ticker} is missing: {', '.join(missing)}"
            )
        for column in ("Open", "Close"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["Open", "Close"])
        if data.empty:
            raise ValueError(f"Market data is empty for {ticker}.")
        normalized[ticker] = data[["Open", "Close"]]

    symbols = sorted(normalized)
    timeline = pd.DatetimeIndex(
        sorted(
            set().union(*(set(data.index) for data in normalized.values()))
        )
    )
    if len(timeline) < 2:
        raise ValueError("Benchmark timeline requires at least two dates.")

    commission = config.commission_rate
    minimum_fee = config.minimum_fee
    slippage = config.slippage_bps / 10_000

    def fee(notional: float) -> float:
        if notional <= 0:
            return 0.0
        return _money(max(notional * commission, minimum_fee))

    cash = float(initial_cash)
    budget = initial_cash * exposure_percent / 100
    allocation = budget / len(symbols)
    holdings: dict[str, tuple[float, float]] = {}
    total_fees = 0.0

    for ticker in symbols:
        data = normalized[ticker]
        first_date = data.index[data.index >= timeline[0]][0]
        entry = _price(
            float(data.loc[first_date, "Open"]) * (1 + slippage)
        )
        fractional = (
            config.allow_fractional_crypto
            if _is_crypto(ticker)
            else config.allow_fractional_stocks
        )
        quantity = (
            round(max(allocation / entry, 0.0), 6)
            if fractional
            else float(max(floor(allocation / entry), 0))
        )
        value = quantity * entry
        entry_fee = fee(value)

        while quantity > 0 and value + entry_fee > cash:
            quantity = (
                round(max(cash - minimum_fee, 0.0) / entry, 6)
                if fractional
                else max(quantity - 1, 0)
            )
            value = quantity * entry
            entry_fee = fee(value)

        if quantity > 0:
            cash -= value + entry_fee
            total_fees += entry_fee
            holdings[ticker] = (quantity, entry)

    if not holdings:
        raise ValueError("Benchmark could not open any holdings.")

    cash_after_entries = cash
    peak = initial_cash
    max_dd_amount = 0.0
    max_dd_percent = 0.0
    last_prices: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for date in timeline:
        for ticker, data in normalized.items():
            if date in data.index:
                last_prices[ticker] = float(data.loc[date, "Close"])
        positions = sum(
            quantity * last_prices.get(ticker, entry)
            for ticker, (quantity, entry) in holdings.items()
        )
        equity = cash + positions
        peak = max(peak, equity)
        dd_amount = max(peak - equity, 0.0)
        dd_percent = dd_amount / peak * 100 if peak else 0.0
        max_dd_amount = max(max_dd_amount, dd_amount)
        max_dd_percent = max(max_dd_percent, dd_percent)
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "cash": _money(cash),
                "positions_market_value": _money(positions),
                "total_equity": _money(equity),
                "drawdown_percent": round(dd_percent, 4),
            }
        )

    ending_equity = cash
    for ticker, (quantity, _entry) in holdings.items():
        data = normalized[ticker]
        final_date = data.index[data.index <= timeline[-1]][-1]
        exit_price = _price(
            float(data.loc[final_date, "Close"]) * (1 - slippage)
        )
        value = quantity * exit_price
        exit_fee = fee(value)
        ending_equity += value - exit_fee
        total_fees += exit_fee

    ending_equity = _money(ending_equity)
    return_amount = _money(ending_equity - initial_cash)
    return_percent = round(return_amount / initial_cash * 100, 4)
    curve = pd.DataFrame(rows)
    curve.loc[
        curve.index[-1],
        ["cash", "positions_market_value", "total_equity"],
    ] = [ending_equity, 0.0, ending_equity]

    summary = {
        "label": label,
        "target_exposure_percent": round(exposure_percent, 4),
        "initial_cash": _money(initial_cash),
        "invested_budget": _money(budget),
        "ending_equity": ending_equity,
        "total_return_amount": return_amount,
        "total_return_percent": return_percent,
        "cagr_percent": calculate_cagr(
            initial_cash, ending_equity, timeline[0], timeline[-1]
        ),
        "maximum_drawdown_amount": _money(max_dd_amount),
        "maximum_drawdown_percent": round(max_dd_percent, 4),
        "return_drawdown_ratio": return_drawdown_ratio(
            return_percent, max_dd_percent
        ),
        "total_fees": _money(total_fees),
        "invested_tickers": len(holdings),
        "cash_after_entries": _money(cash_after_entries),
        "start_date": timeline[0].isoformat(),
        "end_date": timeline[-1].isoformat(),
    }
    return summary, curve


def normalize_position_limits(values: Iterable[int]) -> tuple[int, ...]:
    """Return unique, ascending, positive position limits."""

    normalized = sorted({int(value) for value in values})
    if not normalized:
        raise ValueError("At least one position limit is required.")
    if any(value <= 0 for value in normalized):
        raise ValueError("Position limits must be positive integers.")
    return tuple(normalized)


def build_variant_config(
    base_config: PortfolioBacktestConfig,
    maximum_open_positions: int,
) -> PortfolioBacktestConfig:
    """Clone a base config while changing only the open-position limit."""

    if maximum_open_positions <= 0:
        raise ValueError("maximum_open_positions must be positive.")
    variant = replace(
        base_config,
        maximum_open_positions=int(maximum_open_positions),
    )
    variant.validate()
    return variant


def execution_config(config: PortfolioBacktestConfig) -> dict[str, Any]:
    """Convert execution assumptions for the matched benchmark helper."""

    return {
        "commission_rate": config.commission_rate,
        "minimum_fee": config.minimum_fee,
        "slippage_bps": config.slippage_bps,
        "allow_fractional_stocks": config.allow_fractional_stocks,
        "allow_fractional_crypto": config.allow_fractional_crypto,
    }


def annual_returns_from_equity(
    equity_curve: Sequence[Any],
    *,
    initial_cash: float,
) -> pd.DataFrame:
    """Calculate chained calendar-year returns from portfolio equity points."""

    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero.")

    rows = [
        {
            "timestamp": pd.Timestamp(point.timestamp),
            "total_equity": float(point.total_equity),
        }
        for point in equity_curve
    ]
    if not rows:
        return pd.DataFrame(columns=["year", "return_percent"])

    frame = pd.DataFrame(rows)
    if frame["timestamp"].dt.tz is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(None)
    frame = (
        frame.dropna(subset=["timestamp", "total_equity"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )
    frame["year"] = frame["timestamp"].dt.year
    year_ends = frame.groupby("year")["total_equity"].last()

    previous = float(initial_cash)
    output: list[dict[str, Any]] = []
    for year, ending in year_ends.items():
        ending_value = float(ending)
        annual_return = (
            (ending_value / previous - 1) * 100 if previous > 0 else 0.0
        )
        output.append(
            {
                "year": int(year),
                "return_percent": round(annual_return, 4),
            }
        )
        previous = ending_value

    return pd.DataFrame(output)


def ticker_contributions(result: PortfolioBacktestResult) -> pd.DataFrame:
    """Aggregate trade-level PnL and quality metrics by ticker."""

    grouped: dict[str, list[Any]] = defaultdict(list)
    for trade in result.trades:
        grouped[trade.ticker].append(trade)

    rows: list[dict[str, Any]] = []
    for ticker in sorted(result.tickers):
        trades = grouped.get(ticker, [])
        winners = [trade for trade in trades if trade.net_pnl > 0]
        losers = [trade for trade in trades if trade.net_pnl < 0]
        gross_profit = sum(trade.net_pnl for trade in winners)
        gross_loss = abs(sum(trade.net_pnl for trade in losers))
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        rows.append(
            {
                "ticker": ticker,
                "asset_class": (
                    trades[0].asset_class
                    if trades
                    else (
                        "CRYPTO"
                        if ticker.endswith(("-USD", "-EUR", "-GBP"))
                        else "EQUITY"
                    )
                ),
                "completed_trades": len(trades),
                "winning_trades": len(winners),
                "losing_trades": len(losers),
                "win_rate_percent": round(
                    len(winners) / len(trades) * 100 if trades else 0.0,
                    4,
                ),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "net_pnl": round(sum(trade.net_pnl for trade in trades), 2),
                "profit_factor": (
                    round(profit_factor, 4)
                    if isfinite(profit_factor)
                    else float("inf")
                ),
                "total_fees": round(
                    sum(trade.total_fees for trade in trades),
                    2,
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    positive_total = float(frame.loc[frame["net_pnl"] > 0, "net_pnl"].sum())
    frame["positive_pnl_share_percent"] = frame["net_pnl"].apply(
        lambda value: round(value / positive_total * 100, 4)
        if value > 0 and positive_total > 0
        else 0.0
    )
    return frame


def rejection_counts(result: PortfolioBacktestResult) -> dict[str, int]:
    """Count rejected signals by deterministic reason code."""

    return dict(Counter(item.reason_code for item in result.rejections))


def concentration_metrics(ticker_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize how concentrated positive PnL is across tickers."""

    if ticker_frame.empty:
        return {
            "profitable_ticker_count": 0,
            "losing_ticker_count": 0,
            "largest_positive_ticker": None,
            "largest_positive_ticker_pnl": 0.0,
            "largest_positive_pnl_share_percent": 0.0,
            "top_3_positive_pnl_share_percent": 0.0,
        }

    positive = ticker_frame.loc[ticker_frame["net_pnl"] > 0].sort_values(
        "net_pnl", ascending=False
    )
    losing = ticker_frame.loc[ticker_frame["net_pnl"] < 0]
    positive_total = float(positive["net_pnl"].sum())

    if positive.empty or positive_total <= 0:
        largest_ticker = None
        largest_pnl = 0.0
        largest_share = 0.0
        top_three_share = 0.0
    else:
        largest_ticker = str(positive.iloc[0]["ticker"])
        largest_pnl = float(positive.iloc[0]["net_pnl"])
        largest_share = largest_pnl / positive_total * 100
        top_three_share = float(positive.head(3)["net_pnl"].sum()) / positive_total * 100

    return {
        "profitable_ticker_count": int(len(positive)),
        "losing_ticker_count": int(len(losing)),
        "largest_positive_ticker": largest_ticker,
        "largest_positive_ticker_pnl": round(largest_pnl, 2),
        "largest_positive_pnl_share_percent": round(largest_share, 4),
        "top_3_positive_pnl_share_percent": round(top_three_share, 4),
    }


def build_variant_summary(
    result: PortfolioBacktestResult,
    matched_benchmark: dict[str, Any],
    annual: pd.DataFrame,
    ticker_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build one compact comparison row for an ablation variant."""

    start = pd.Timestamp(result.start_date)
    end = pd.Timestamp(result.end_date)
    counts = rejection_counts(result)
    concentration = concentration_metrics(ticker_frame)

    annual_values = (
        annual["return_percent"].astype(float).tolist()
        if not annual.empty
        else []
    )

    return {
        "maximum_open_positions": result.config.maximum_open_positions,
        "maximum_open_positions_observed": (
            result.maximum_open_positions_observed
        ),
        "initial_cash": result.initial_cash,
        "ending_equity": result.ending_equity,
        "total_return_amount": result.total_return_amount,
        "total_return_percent": result.total_return_percent,
        "cagr_percent": calculate_cagr(
            result.initial_cash,
            result.ending_equity,
            start,
            end,
        ),
        "maximum_drawdown_percent": result.maximum_drawdown_percent,
        "return_drawdown_ratio": return_drawdown_ratio(
            result.total_return_percent,
            result.maximum_drawdown_percent,
        ),
        "completed_trades": result.completed_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate_percent": result.win_rate_percent,
        "profit_factor": result.profit_factor,
        "total_fees": result.total_fees,
        "average_exposure_percent": result.average_exposure_percent,
        "rejected_signals": result.rejected_signals,
        "max_open_position_rejections": counts.get(
            "MAX_OPEN_POSITIONS", 0
        ),
        "max_total_risk_rejections": counts.get("MAX_TOTAL_OPEN_RISK", 0),
        "max_crypto_allocation_rejections": counts.get(
            "MAX_CRYPTO_ALLOCATION", 0
        ),
        "insufficient_cash_rejections": counts.get(
            "INSUFFICIENT_CASH", 0
        ),
        "matched_benchmark_return_percent": matched_benchmark[
            "total_return_percent"
        ],
        "matched_benchmark_drawdown_percent": matched_benchmark[
            "maximum_drawdown_percent"
        ],
        "excess_return_vs_matched_percent": round(
            result.total_return_percent
            - matched_benchmark["total_return_percent"],
            4,
        ),
        "drawdown_advantage_vs_matched_percent": round(
            matched_benchmark["maximum_drawdown_percent"]
            - result.maximum_drawdown_percent,
            4,
        ),
        "positive_year_count": sum(value > 0 for value in annual_values),
        "negative_year_count": sum(value < 0 for value in annual_values),
        "worst_year_return_percent": (
            round(min(annual_values), 4) if annual_values else 0.0
        ),
        "best_year_return_percent": (
            round(max(annual_values), 4) if annual_values else 0.0
        ),
        **concentration,
    }


def run_position_ablation(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    position_limits: Iterable[int] = DEFAULT_POSITION_LIMITS,
) -> dict[str, Any]:
    """Run every position-limit variant against the same prepared data."""

    limits = normalize_position_limits(position_limits)
    summaries: list[dict[str, Any]] = []
    ticker_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    rejection_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    raw_results: dict[str, Any] = {}

    for sequence, limit in enumerate(limits, start=1):
        print()
        print("=" * 100)
        print(
            f"POSITION LIMIT ABLATION [{sequence}/{len(limits)}] — "
            f"MAX OPEN POSITIONS = {limit}"
        )
        print("=" * 100)

        config = build_variant_config(base_config, limit)
        result = run_portfolio_backtest(
            data_by_ticker=data_by_ticker,
            config=config,
            include_benchmark=False,
        )

        matched, _matched_curve = run_matched_benchmark(
            data_by_ticker,
            initial_cash=config.initial_cash,
            exposure_percent=max(result.average_exposure_percent, 0.0001),
            config=config,
            label=f"MATCHED_{limit}_POSITIONS",
        )
        annual = annual_returns_from_equity(
            result.equity_curve,
            initial_cash=result.initial_cash,
        )
        tickers = ticker_contributions(result)
        summary = build_variant_summary(result, matched, annual, tickers)

        summaries.append(summary)
        raw_results[str(limit)] = result.to_dict()

        if not tickers.empty:
            tickers.insert(0, "maximum_open_positions", limit)
            ticker_frames.append(tickers)

        if not annual.empty:
            annual.insert(0, "maximum_open_positions", limit)
            annual_frames.append(annual)

        for reason_code, count in sorted(rejection_counts(result).items()):
            rejection_rows.append(
                {
                    "maximum_open_positions": limit,
                    "reason_code": reason_code,
                    "count": count,
                }
            )

        equity = pd.DataFrame(
            [point.to_dict() for point in result.equity_curve]
        )
        if not equity.empty:
            equity.insert(0, "maximum_open_positions", limit)
            equity_frames.append(equity)

        print(
            f"Return={result.total_return_percent:+.4f}%  "
            f"MaxDD={result.maximum_drawdown_percent:.4f}%  "
            f"PF={result.profit_factor:.4f}  "
            f"Exposure={result.average_exposure_percent:.4f}%  "
            f"Trades={result.completed_trades}  "
            f"Rejected={result.rejected_signals}"
        )

    summary_frame = pd.DataFrame(summaries).sort_values(
        "maximum_open_positions"
    )
    return {
        "summary": summary_frame.reset_index(drop=True),
        "tickers": (
            pd.concat(ticker_frames, ignore_index=True)
            if ticker_frames
            else pd.DataFrame()
        ),
        "annual": (
            pd.concat(annual_frames, ignore_index=True)
            if annual_frames
            else pd.DataFrame()
        ),
        "rejections": pd.DataFrame(rejection_rows),
        "equity": (
            pd.concat(equity_frames, ignore_index=True)
            if equity_frames
            else pd.DataFrame()
        ),
        "raw_results": raw_results,
        "base_config": base_config.to_dict(),
        "position_limits": list(limits),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_position_ablation(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    """Save JSON and comparison CSV artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    paths = {
        "json": output_directory / f"position_ablation_{stamp}.json",
        "summary": output_directory
        / f"position_ablation_summary_{stamp}.csv",
        "tickers": output_directory
        / f"position_ablation_tickers_{stamp}.csv",
        "annual": output_directory
        / f"position_ablation_annual_{stamp}.csv",
        "rejections": output_directory
        / f"position_ablation_rejections_{stamp}.csv",
        "equity": output_directory
        / f"position_ablation_equity_{stamp}.csv",
    }

    json_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_config": bundle["base_config"],
        "position_limits": bundle["position_limits"],
        "summary": bundle["summary"].to_dict(orient="records"),
        "ticker_contributions": bundle["tickers"].to_dict(
            orient="records"
        ),
        "annual_returns": bundle["annual"].to_dict(orient="records"),
        "rejection_counts": bundle["rejections"].to_dict(
            orient="records"
        ),
        "runs": bundle["raw_results"],
    }

    with paths["json"].open("w", encoding="utf-8") as file:
        json.dump(_json_safe(json_payload), file, ensure_ascii=False, indent=2)
        file.write("\n")

    bundle["summary"].to_csv(paths["summary"], index=False)
    bundle["tickers"].to_csv(paths["tickers"], index=False)
    bundle["annual"].to_csv(paths["annual"], index=False)
    bundle["rejections"].to_csv(paths["rejections"], index=False)
    bundle["equity"].to_csv(paths["equity"], index=False)

    return paths


def print_position_ablation(bundle: dict[str, Any]) -> None:
    """Print the compact comparison table."""

    summary = bundle["summary"]
    columns = [
        "maximum_open_positions",
        "total_return_percent",
        "cagr_percent",
        "maximum_drawdown_percent",
        "return_drawdown_ratio",
        "profit_factor",
        "average_exposure_percent",
        "completed_trades",
        "rejected_signals",
        "excess_return_vs_matched_percent",
        "worst_year_return_percent",
    ]

    print()
    print("=" * 132)
    print("AI STOCK RADAR — MAXIMUM OPEN POSITION ABLATION")
    print("=" * 132)
    print(
        summary[columns].to_string(
            index=False,
            formatters={
                "total_return_percent": lambda value: f"{value:+.4f}%",
                "cagr_percent": lambda value: f"{value:+.4f}%",
                "maximum_drawdown_percent": lambda value: f"{value:.4f}%",
                "return_drawdown_ratio": lambda value: f"{value:.4f}",
                "profit_factor": lambda value: f"{value:.4f}",
                "average_exposure_percent": lambda value: f"{value:.4f}%",
                "excess_return_vs_matched_percent": (
                    lambda value: f"{value:+.4f}%"
                ),
                "worst_year_return_percent": lambda value: f"{value:+.4f}%",
            },
        )
    )
    print("=" * 132)
    print(
        "Interpretation rule: prefer stability across limits, controlled "
        "drawdown, durable profit factor, and reduced concentration—not "
        "the single highest return alone."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare shared-cash portfolio results under different maximum "
            "open-position limits."
        )
    )

    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=list(DEFAULT_POSITION_LIMITS),
    )
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)

    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--max-position", type=float, default=25.0)
    parser.add_argument("--max-total-risk", type=float, default=4.0)
    parser.add_argument("--max-crypto", type=float, default=25.0)

    parser.add_argument("--stock-stop", type=float, default=5.0)
    parser.add_argument("--crypto-stop", type=float, default=5.0)
    parser.add_argument("--stock-trailing", type=float, default=7.5)
    parser.add_argument("--crypto-trailing", type=float, default=7.5)

    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--minimum-fee", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)

    parser.add_argument("--fractional-stocks", action="store_true")
    parser.add_argument("--no-fractional-crypto", action="store_true")
    parser.add_argument("--no-crypto-regime", action="store_true")
    parser.add_argument("--no-force-close", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    tickers = arguments.tickers or list(DEFAULT_TICKERS)
    limits = normalize_position_limits(arguments.positions)

    base_config = PortfolioBacktestConfig(
        initial_cash=arguments.initial_cash,
        risk_per_trade_percent=arguments.risk,
        maximum_position_percent=arguments.max_position,
        maximum_total_open_risk_percent=arguments.max_total_risk,
        maximum_crypto_allocation_percent=arguments.max_crypto,
        maximum_open_positions=max(limits),
        stock_stop_loss_percent=arguments.stock_stop,
        crypto_stop_loss_percent=arguments.crypto_stop,
        stock_trailing_close_percent=arguments.stock_trailing,
        crypto_trailing_close_percent=arguments.crypto_trailing,
        commission_rate=arguments.commission_rate,
        minimum_fee=arguments.minimum_fee,
        slippage_bps=arguments.slippage_bps,
        allow_fractional_stocks=arguments.fractional_stocks,
        allow_fractional_crypto=(not arguments.no_fractional_crypto),
        force_close_at_end=(not arguments.no_force_close),
    )
    base_config.validate()

    from src.backtest.run_portfolio_backtest import prepare_portfolio_data

    prepared_data = prepare_portfolio_data(
        tickers=tickers,
        period=arguments.period,
        regime_period=arguments.regime_period,
        use_crypto_regime=(not arguments.no_crypto_regime),
    )

    bundle = run_position_ablation(
        data_by_ticker=prepared_data,
        base_config=base_config,
        position_limits=limits,
    )
    print_position_ablation(bundle)

    if not arguments.no_save:
        paths = save_position_ablation(
            bundle,
            output_directory=arguments.output_directory,
        )
        print()
        print("=" * 100)
        print("POSITION ABLATION FILES")
        print("=" * 100)
        for label, path in paths.items():
            print(f"{label.upper():<12} {path.resolve()}")
        print("=" * 100)

    print()
    print("Position-limit ablation completed successfully.")


if __name__ == "__main__":
    main()
