"""Exposure-matched benchmark analysis for portfolio backtest JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.download_stock import download_stock_data


DEFAULT_PERIOD = "10y"
DEFAULT_OUTPUT_DIR = (
    Path("data") / "backtests" / "portfolio" / "benchmark_analysis"
)


def _money(value: float) -> float:
    return round(float(value), 2)


def _price(value: float) -> float:
    return round(float(value), 8)


def _is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith(("-USD", "-EUR", "-GBP"))


def _tickers(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        ticker = str(raw).strip().upper()
        if ticker and ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
    if not result:
        raise ValueError("At least one ticker is required.")
    return result


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{name} is not a valid timestamp.")
    if result.tzinfo is not None:
        result = result.tz_convert(None)
    return result


def load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Portfolio result JSON not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    required = {
        "config",
        "tickers",
        "start_date",
        "end_date",
        "initial_cash",
        "ending_equity",
        "total_return_percent",
        "maximum_drawdown_percent",
        "average_exposure_percent",
        "equity_curve",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("Missing JSON fields: " + ", ".join(missing))
    return payload


def _execution_config(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload["config"]
    config = {
        "commission_rate": float(raw.get("commission_rate", 0.0005)),
        "minimum_fee": float(raw.get("minimum_fee", 1.0)),
        "slippage_bps": float(raw.get("slippage_bps", 5.0)),
        "allow_fractional_stocks": bool(
            raw.get("allow_fractional_stocks", False)
        ),
        "allow_fractional_crypto": bool(
            raw.get("allow_fractional_crypto", True)
        ),
    }
    for name in ("commission_rate", "minimum_fee", "slippage_bps"):
        if config[name] < 0:
            raise ValueError(f"{name} cannot be negative.")
    return config


def normalize_data(
    data: pd.DataFrame,
    *,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if data.empty:
        raise ValueError(f"Market data is empty for {ticker}.")
    result = data.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in result.columns
        ]
    missing = [c for c in ("Open", "Close") if c not in result.columns]
    if missing:
        raise ValueError(f"{ticker} is missing: {', '.join(missing)}")

    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    result = result.sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    for column in ("Open", "Close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Open", "Close"])
    result = result.loc[
        (result.index >= start) & (result.index <= end),
        ["Open", "Close"],
    ]
    if result.empty:
        raise ValueError(f"{ticker} has no data in the requested period.")
    if (result <= 0).any().any():
        raise ValueError(f"{ticker} contains non-positive prices.")
    return result


def prepare_data(
    tickers: Iterable[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period: str,
) -> dict[str, pd.DataFrame]:
    symbols = _tickers(tickers)
    prepared: dict[str, pd.DataFrame] = {}
    print("\n" + "=" * 92)
    print("BENCHMARK MARKET-DATA PREPARATION")
    print("=" * 92)
    for number, ticker in enumerate(symbols, start=1):
        print(f"[{number}/{len(symbols)}] {ticker}")
        prepared[ticker] = normalize_data(
            download_stock_data(ticker, period=period),
            ticker=ticker,
            start=start,
            end=end,
        )
        data = prepared[ticker]
        print(
            f"  Bars={len(data)} Start={data.index.min().date()} "
            f"End={data.index.max().date()}"
        )
    print("=" * 92)
    return prepared


def calculate_cagr(
    initial: float,
    ending: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    years = max((end - start).days, 0) / 365.25
    if years <= 0 or initial <= 0 or ending <= 0:
        return 0.0
    return round(((ending / initial) ** (1 / years) - 1) * 100, 4)


def return_drawdown_ratio(total_return: float, drawdown: float) -> float:
    if drawdown <= 0:
        return float("inf") if total_return > 0 else 0.0
    return round(total_return / drawdown, 4)


def run_benchmark(
    data_by_ticker: dict[str, pd.DataFrame],
    *,
    initial_cash: float,
    exposure_percent: float,
    config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero.")
    if not 0 < exposure_percent <= 100:
        raise ValueError("exposure_percent must be greater than 0 and at most 100.")
    if not data_by_ticker:
        raise ValueError("data_by_ticker cannot be empty.")

    symbols = sorted(_tickers(data_by_ticker))
    timeline = pd.DatetimeIndex(
        sorted(set().union(*(set(data.index) for data in data_by_ticker.values())))
    )
    if len(timeline) < 2:
        raise ValueError("Benchmark timeline requires at least two dates.")

    commission = config["commission_rate"]
    minimum_fee = config["minimum_fee"]
    slippage = config["slippage_bps"] / 10_000

    def fee(notional: float) -> float:
        return _money(max(notional * commission, minimum_fee)) if notional > 0 else 0.0

    cash = float(initial_cash)
    budget = initial_cash * exposure_percent / 100
    allocation = budget / len(symbols)
    holdings: dict[str, tuple[float, float]] = {}
    total_fees = 0.0

    for ticker in symbols:
        data = data_by_ticker[ticker]
        first_date = data.index[data.index >= timeline[0]][0]
        entry = _price(float(data.loc[first_date, "Open"]) * (1 + slippage))
        fractional = (
            config["allow_fractional_crypto"]
            if _is_crypto(ticker)
            else config["allow_fractional_stocks"]
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
        for ticker, data in data_by_ticker.items():
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
    for ticker, (quantity, _) in holdings.items():
        data = data_by_ticker[ticker]
        final_date = data.index[data.index <= timeline[-1]][-1]
        exit_price = _price(float(data.loc[final_date, "Close"]) * (1 - slippage))
        value = quantity * exit_price
        exit_fee = fee(value)
        ending_equity += value - exit_fee
        total_fees += exit_fee

    ending_equity = _money(ending_equity)
    return_amount = _money(ending_equity - initial_cash)
    return_percent = round(return_amount / initial_cash * 100, 4)
    curve = pd.DataFrame(rows)
    curve.loc[curve.index[-1], ["cash", "positions_market_value", "total_equity"]] = [
        ending_equity,
        0.0,
        ending_equity,
    ]

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


def strategy_curve(payload: dict[str, Any]) -> pd.DataFrame:
    curve = pd.DataFrame(payload["equity_curve"])
    missing = sorted({"timestamp", "total_equity"} - set(curve.columns))
    if missing:
        raise ValueError("equity_curve is missing: " + ", ".join(missing))
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])
    if curve["timestamp"].dt.tz is not None:
        curve["timestamp"] = curve["timestamp"].dt.tz_convert(None)
    curve["total_equity"] = pd.to_numeric(curve["total_equity"], errors="coerce")
    curve = curve.dropna(subset=["timestamp", "total_equity"])
    return (
        curve.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        [["timestamp", "total_equity"]]
        .reset_index(drop=True)
    )


def annual_returns(
    strategy: pd.DataFrame,
    full: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    initial_cash: float,
) -> pd.DataFrame:
    frames = {
        "strategy_return_percent": strategy,
        "full_benchmark_return_percent": full,
        "matched_benchmark_return_percent": matched,
    }
    year_ends: dict[str, dict[int, float]] = {}
    years: set[int] = set()

    for name, frame in frames.items():
        work = frame.copy()
        work["year"] = pd.to_datetime(work["timestamp"]).dt.year
        values = work.groupby("year")["total_equity"].last().to_dict()
        year_ends[name] = {int(year): float(value) for year, value in values.items()}
        years.update(year_ends[name])

    previous = {name: initial_cash for name in frames}
    rows: list[dict[str, Any]] = []
    for year in sorted(years):
        row: dict[str, Any] = {"year": year}
        for name in frames:
            ending = year_ends[name].get(year)
            row[name] = (
                round((ending / previous[name] - 1) * 100, 4)
                if ending is not None
                else None
            )
            if ending is not None:
                previous[name] = ending
        s = row["strategy_return_percent"]
        m = row["matched_benchmark_return_percent"]
        row["strategy_excess_vs_matched_percent"] = (
            round(s - m, 4) if s is not None and m is not None else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_analysis(
    payload: dict[str, Any],
    *,
    source_json: str,
    data_by_ticker: dict[str, pd.DataFrame],
    target_exposure: float | None = None,
) -> dict[str, Any]:
    symbols = sorted(_tickers(payload["tickers"]))
    missing = sorted(set(symbols) - set(data_by_ticker))
    if missing:
        raise ValueError("Missing market data for: " + ", ".join(missing))

    initial = float(payload["initial_cash"])
    ending = float(payload["ending_equity"])
    strategy_return = float(payload["total_return_percent"])
    strategy_dd = float(payload["maximum_drawdown_percent"])
    strategy_exposure = float(payload["average_exposure_percent"])
    matched_exposure = strategy_exposure if target_exposure is None else target_exposure
    if not 0 < matched_exposure <= 100:
        raise ValueError("Matched exposure must be between 0 and 100.")

    start = _timestamp(payload["start_date"], "start_date")
    end = _timestamp(payload["end_date"], "end_date")
    config = _execution_config(payload)

    full_summary, full_curve = run_benchmark(
        data_by_ticker,
        initial_cash=initial,
        exposure_percent=100.0,
        config=config,
        label="FULL_BENCHMARK",
    )
    matched_summary, matched_curve = run_benchmark(
        data_by_ticker,
        initial_cash=initial,
        exposure_percent=float(matched_exposure),
        config=config,
        label="MATCHED_BENCHMARK",
    )

    strategy_summary = {
        "average_exposure_percent": round(strategy_exposure, 4),
        "ending_equity": _money(ending),
        "total_return_percent": round(strategy_return, 4),
        "cagr_percent": calculate_cagr(initial, ending, start, end),
        "maximum_drawdown_percent": round(strategy_dd, 4),
        "return_drawdown_ratio": return_drawdown_ratio(strategy_return, strategy_dd),
    }

    source_benchmark = payload.get("benchmark") or {}
    source_return = source_benchmark.get("total_return_percent")
    source_dd = source_benchmark.get("maximum_drawdown_percent")

    strategy_frame = strategy_curve(payload)
    full_frame = full_curve[["timestamp", "total_equity"]]
    matched_frame = matched_curve[["timestamp", "total_equity"]]
    annual = annual_returns(
        strategy_frame,
        full_frame,
        matched_frame,
        initial_cash=initial,
    )

    equity = (
        strategy_frame.rename(columns={"total_equity": "strategy_equity"})
        .set_index("timestamp")
        .join(
            full_frame.rename(columns={"total_equity": "full_benchmark_equity"})
            .set_index("timestamp"),
            how="outer",
        )
        .join(
            matched_frame.rename(
                columns={"total_equity": "matched_benchmark_equity"}
            ).set_index("timestamp"),
            how="outer",
        )
        .sort_index()
        .ffill()
        .reset_index()
    )

    comparison = pd.DataFrame(
        [
            {"series": "STRATEGY", **strategy_summary},
            {"series": "FULL_BENCHMARK", **full_summary},
            {"series": "MATCHED_BENCHMARK", **matched_summary},
        ]
    )

    summary = {
        "source_json": source_json,
        "created_at": datetime.now(UTC).isoformat(),
        "tickers": symbols,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "initial_cash": _money(initial),
        "strategy": strategy_summary,
        "source_benchmark": {
            "total_return_percent": source_return,
            "maximum_drawdown_percent": source_dd,
        },
        "full_benchmark": full_summary,
        "matched_benchmark": matched_summary,
        "reconciliation": {
            "return_difference_vs_source": (
                round(full_summary["total_return_percent"] - float(source_return), 4)
                if source_return is not None else None
            ),
            "drawdown_difference_vs_source": (
                round(full_summary["maximum_drawdown_percent"] - float(source_dd), 4)
                if source_dd is not None else None
            ),
        },
        "comparison": {
            "strategy_excess_return_vs_full_percent": round(
                strategy_return - full_summary["total_return_percent"], 4
            ),
            "strategy_excess_return_vs_matched_percent": round(
                strategy_return - matched_summary["total_return_percent"], 4
            ),
            "strategy_cagr_advantage_vs_matched_percent": round(
                strategy_summary["cagr_percent"] - matched_summary["cagr_percent"], 4
            ),
            "strategy_drawdown_advantage_vs_matched_percent": round(
                matched_summary["maximum_drawdown_percent"] - strategy_dd, 4
            ),
            "strategy_return_drawdown_advantage_vs_matched": round(
                strategy_summary["return_drawdown_ratio"]
                - matched_summary["return_drawdown_ratio"],
                4,
            ),
        },
        "annual_returns": annual.to_dict(orient="records"),
    }
    return {
        "summary": summary,
        "comparison": comparison,
        "annual": annual,
        "equity": equity,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_analysis(bundle: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths = {
        "json": output_dir / f"benchmark_analysis_{stamp}.json",
        "comparison": output_dir / f"benchmark_comparison_{stamp}.csv",
        "annual": output_dir / f"benchmark_annual_{stamp}.csv",
        "equity": output_dir / f"benchmark_equity_{stamp}.csv",
    }
    with paths["json"].open("w", encoding="utf-8") as file:
        json.dump(_json_safe(bundle["summary"]), file, indent=2)
        file.write("\n")
    bundle["comparison"].to_csv(paths["comparison"], index=False)
    bundle["annual"].to_csv(paths["annual"], index=False)
    bundle["equity"].to_csv(paths["equity"], index=False)
    return paths


def print_analysis(bundle: dict[str, Any], tolerance: float) -> None:
    summary = bundle["summary"]
    strategy = summary["strategy"]
    full = summary["full_benchmark"]
    matched = summary["matched_benchmark"]
    comparison = summary["comparison"]
    recon = summary["reconciliation"]

    print("\n" + "=" * 92)
    print("AI STOCK RADAR — EXPOSURE-MATCHED BENCHMARK ANALYSIS")
    print("=" * 92)
    print(f"Strategy return:              {strategy['total_return_percent']:+.4f}%")
    print(f"Strategy CAGR:                {strategy['cagr_percent']:+.4f}%")
    print(f"Strategy max drawdown:        {strategy['maximum_drawdown_percent']:.4f}%")
    print(f"Strategy avg exposure:        {strategy['average_exposure_percent']:.4f}%")
    print(f"Full benchmark return:        {full['total_return_percent']:+.4f}%")
    print(f"Matched benchmark return:     {matched['total_return_percent']:+.4f}%")
    print(f"Matched benchmark drawdown:   {matched['maximum_drawdown_percent']:.4f}%")
    print(
        f"Strategy excess vs matched:  "
        f"{comparison['strategy_excess_return_vs_matched_percent']:+.4f}%"
    )
    values = [value for value in recon.values() if value is not None]
    if values:
        status = "PASS" if max(abs(value) for value in values) <= tolerance else "WARNING"
        print(f"Source reconciliation:        {status}")
        print(f"Return difference:            {recon['return_difference_vs_source']:+.4f}%")
        print(f"Drawdown difference:          {recon['drawdown_difference_vs_source']:+.4f}%")
    print("=" * 92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exposure-matched portfolio benchmark analysis."
    )
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--target-exposure", type=float, default=None)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reconciliation-tolerance", type=float, default=0.25)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.result_json.expanduser().resolve()
    payload = load_result(source)
    start = _timestamp(payload["start_date"], "start_date")
    end = _timestamp(payload["end_date"], "end_date")
    data = prepare_data(
        payload["tickers"], start=start, end=end, period=args.period
    )
    bundle = build_analysis(
        payload,
        source_json=str(source),
        data_by_ticker=data,
        target_exposure=args.target_exposure,
    )
    print_analysis(bundle, args.reconciliation_tolerance)

    if not args.no_save:
        paths = save_analysis(bundle, args.output_directory)
        print("\nBENCHMARK ANALYSIS FILES")
        for name, path in paths.items():
            print(f"{name.upper():<12} {path.resolve()}")
    print("\nExposure-matched benchmark analysis completed successfully.")


if __name__ == "__main__":
    main()
