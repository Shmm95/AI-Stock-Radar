"""Portfolio risk-parameter ablation for the shared-cash backtest engine.

The maximum-open-position limit is held fixed while risk per trade and maximum
aggregate open risk are tested on the same prepared market data. Every variant
uses the same entry/exit logic, fees, slippage, universe, and crypto regime.

This module is historical research only. It cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    DEFAULT_REGIME_PERIOD,
    DEFAULT_TICKERS,
    annual_returns_from_equity,
    build_variant_summary,
    rejection_counts,
    run_matched_benchmark,
    ticker_contributions,
)

DEFAULT_PERIOD = "5y"
DEFAULT_MAXIMUM_OPEN_POSITIONS = 4
DEFAULT_RISK_PER_TRADE_VALUES = (0.5, 0.75, 1.0)
DEFAULT_MAXIMUM_TOTAL_RISK_VALUES = (3.0, 4.0, 5.0)
DEFAULT_BASELINE_RISK_PER_TRADE = 1.0
DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK = 4.0
DEFAULT_OUTPUT_DIRECTORY = (
    Path("data") / "backtests" / "portfolio" / "risk_ablation"
)


def _normalize_percent_values(
    values: Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    normalized = sorted({round(float(value), 8) for value in values})
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{name} values must be positive.")
    return tuple(normalized)


def normalize_risk_per_trade_values(
    values: Iterable[float],
) -> tuple[float, ...]:
    return _normalize_percent_values(values, name="risk_per_trade")


def normalize_maximum_total_risk_values(
    values: Iterable[float],
) -> tuple[float, ...]:
    return _normalize_percent_values(values, name="maximum_total_open_risk")


def _format_percent_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def risk_variant_id(
    risk_per_trade_percent: float,
    maximum_total_open_risk_percent: float,
) -> str:
    return (
        f"R{_format_percent_token(risk_per_trade_percent)}_"
        f"T{_format_percent_token(maximum_total_open_risk_percent)}"
    )


def build_risk_grid(
    *,
    risk_per_trade_values: Iterable[float],
    maximum_total_risk_values: Iterable[float],
) -> tuple[dict[str, float | str], ...]:
    risks = normalize_risk_per_trade_values(risk_per_trade_values)
    totals = normalize_maximum_total_risk_values(maximum_total_risk_values)

    rows: list[dict[str, float | str]] = []
    for risk_per_trade in risks:
        for maximum_total_risk in totals:
            rows.append(
                {
                    "variant_id": risk_variant_id(
                        risk_per_trade,
                        maximum_total_risk,
                    ),
                    "risk_per_trade_percent": risk_per_trade,
                    "maximum_total_open_risk_percent": maximum_total_risk,
                }
            )
    return tuple(rows)


def build_risk_variant_config(
    base_config: PortfolioBacktestConfig,
    *,
    risk_per_trade_percent: float,
    maximum_total_open_risk_percent: float,
    maximum_open_positions: int,
) -> PortfolioBacktestConfig:
    if maximum_open_positions <= 0:
        raise ValueError("maximum_open_positions must be positive.")

    config = replace(
        base_config,
        risk_per_trade_percent=float(risk_per_trade_percent),
        maximum_total_open_risk_percent=float(
            maximum_total_open_risk_percent
        ),
        maximum_open_positions=int(maximum_open_positions),
    )
    config.validate()
    return config


def rank_risk_candidates(
    summary: pd.DataFrame,
    *,
    baseline_risk_per_trade_percent: float = (
        DEFAULT_BASELINE_RISK_PER_TRADE
    ),
    baseline_maximum_total_open_risk_percent: float = (
        DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK
    ),
) -> pd.DataFrame:
    """Rank variants with deterministic rank aggregation.

    Ranking prioritizes risk-adjusted return, matched-benchmark excess return,
    profit factor, total return, and lower drawdown. A complete tie is resolved
    toward the existing V1 baseline.
    """

    required = {
        "variant_id",
        "risk_per_trade_percent",
        "maximum_total_open_risk_percent",
        "return_drawdown_ratio",
        "excess_return_vs_matched_percent",
        "profit_factor",
        "total_return_percent",
        "maximum_drawdown_percent",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Risk summary is missing: {', '.join(missing)}")
    if summary.empty:
        raise ValueError("Risk summary cannot be empty.")

    ranked = summary.copy().reset_index(drop=True)
    rules = (
        ("return_drawdown_ratio", False),
        ("excess_return_vs_matched_percent", False),
        ("profit_factor", False),
        ("total_return_percent", False),
        ("maximum_drawdown_percent", True),
    )
    rank_columns: list[str] = []
    for column, ascending in rules:
        values = pd.to_numeric(ranked[column], errors="coerce")
        values = values.fillna(float("inf") if ascending else float("-inf"))
        rank_column = f"rank_{column}"
        ranked[rank_column] = values.rank(method="min", ascending=ascending)
        rank_columns.append(rank_column)

    ranked["selection_rank_sum"] = ranked[rank_columns].sum(axis=1)
    ranked["distance_to_baseline_risk"] = (
        ranked["risk_per_trade_percent"].astype(float)
        - float(baseline_risk_per_trade_percent)
    ).abs()
    ranked["distance_to_baseline_total_risk"] = (
        ranked["maximum_total_open_risk_percent"].astype(float)
        - float(baseline_maximum_total_open_risk_percent)
    ).abs()

    ranked = ranked.sort_values(
        by=[
            "selection_rank_sum",
            "return_drawdown_ratio",
            "excess_return_vs_matched_percent",
            "profit_factor",
            "maximum_drawdown_percent",
            "total_return_percent",
            "distance_to_baseline_risk",
            "distance_to_baseline_total_risk",
            "risk_per_trade_percent",
            "maximum_total_open_risk_percent",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["selected_candidate"] = ranked["rank"].eq(1)
    return ranked


def run_portfolio_risk_ablation(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    risk_per_trade_values: Iterable[float] = (
        DEFAULT_RISK_PER_TRADE_VALUES
    ),
    maximum_total_risk_values: Iterable[float] = (
        DEFAULT_MAXIMUM_TOTAL_RISK_VALUES
    ),
    maximum_open_positions: int = DEFAULT_MAXIMUM_OPEN_POSITIONS,
    baseline_risk_per_trade_percent: float = (
        DEFAULT_BASELINE_RISK_PER_TRADE
    ),
    baseline_maximum_total_open_risk_percent: float = (
        DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK
    ),
) -> dict[str, Any]:
    if not data_by_ticker:
        raise ValueError("data_by_ticker cannot be empty.")

    grid = build_risk_grid(
        risk_per_trade_values=risk_per_trade_values,
        maximum_total_risk_values=maximum_total_risk_values,
    )

    summaries: list[dict[str, Any]] = []
    ticker_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    rejection_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    raw_results: dict[str, Any] = {}

    for sequence, variant in enumerate(grid, start=1):
        variant_id = str(variant["variant_id"])
        risk_per_trade = float(variant["risk_per_trade_percent"])
        maximum_total_risk = float(
            variant["maximum_total_open_risk_percent"]
        )

        print()
        print("=" * 104)
        print(
            f"PORTFOLIO RISK ABLATION [{sequence}/{len(grid)}] — "
            f"{variant_id} | RISK/TRADE={risk_per_trade:.2f}% | "
            f"MAX TOTAL RISK={maximum_total_risk:.2f}%"
        )
        print("=" * 104)

        config = build_risk_variant_config(
            base_config,
            risk_per_trade_percent=risk_per_trade,
            maximum_total_open_risk_percent=maximum_total_risk,
            maximum_open_positions=maximum_open_positions,
        )
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
            label=f"MATCHED_{variant_id}",
        )
        annual = annual_returns_from_equity(
            result.equity_curve,
            initial_cash=result.initial_cash,
        )
        tickers = ticker_contributions(result)
        summary = build_variant_summary(result, matched, annual, tickers)
        summary.update(
            {
                "variant_id": variant_id,
                "risk_per_trade_percent": risk_per_trade,
                "maximum_total_open_risk_percent": maximum_total_risk,
                "configured_maximum_open_positions": (
                    maximum_open_positions
                ),
                "nominal_four_position_risk_percent": round(
                    risk_per_trade * maximum_open_positions,
                    4,
                ),
                "effective_nominal_open_risk_cap_percent": round(
                    min(
                        risk_per_trade * maximum_open_positions,
                        maximum_total_risk,
                    ),
                    4,
                ),
            }
        )
        summaries.append(summary)
        raw_results[variant_id] = result.to_dict()

        if not tickers.empty:
            tickers.insert(0, "variant_id", variant_id)
            tickers.insert(1, "risk_per_trade_percent", risk_per_trade)
            tickers.insert(
                2,
                "maximum_total_open_risk_percent",
                maximum_total_risk,
            )
            ticker_frames.append(tickers)

        if not annual.empty:
            annual.insert(0, "variant_id", variant_id)
            annual.insert(1, "risk_per_trade_percent", risk_per_trade)
            annual.insert(
                2,
                "maximum_total_open_risk_percent",
                maximum_total_risk,
            )
            annual_frames.append(annual)

        for reason_code, count in sorted(rejection_counts(result).items()):
            rejection_rows.append(
                {
                    "variant_id": variant_id,
                    "risk_per_trade_percent": risk_per_trade,
                    "maximum_total_open_risk_percent": maximum_total_risk,
                    "reason_code": reason_code,
                    "count": count,
                }
            )

        equity = pd.DataFrame(
            [point.to_dict() for point in result.equity_curve]
        )
        if not equity.empty:
            equity.insert(0, "variant_id", variant_id)
            equity.insert(1, "risk_per_trade_percent", risk_per_trade)
            equity.insert(
                2,
                "maximum_total_open_risk_percent",
                maximum_total_risk,
            )
            equity_frames.append(equity)

        print(
            f"Return={result.total_return_percent:+.4f}%  "
            f"MaxDD={result.maximum_drawdown_percent:.4f}%  "
            f"PF={result.profit_factor:.4f}  "
            f"Exposure={result.average_exposure_percent:.4f}%  "
            f"Trades={result.completed_trades}  "
            f"Rejected={result.rejected_signals}"
        )

    summary_frame = pd.DataFrame(summaries)
    ranked_summary = rank_risk_candidates(
        summary_frame,
        baseline_risk_per_trade_percent=(
            baseline_risk_per_trade_percent
        ),
        baseline_maximum_total_open_risk_percent=(
            baseline_maximum_total_open_risk_percent
        ),
    )

    return {
        "summary": ranked_summary,
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
        "risk_grid": list(grid),
        "maximum_open_positions": maximum_open_positions,
        "baseline": {
            "risk_per_trade_percent": (
                baseline_risk_per_trade_percent
            ),
            "maximum_total_open_risk_percent": (
                baseline_maximum_total_open_risk_percent
            ),
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_portfolio_risk_ablation(
    bundle: dict[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    paths = {
        "json": output_directory / f"portfolio_risk_ablation_{stamp}.json",
        "summary": output_directory
        / f"portfolio_risk_ablation_summary_{stamp}.csv",
        "tickers": output_directory
        / f"portfolio_risk_ablation_tickers_{stamp}.csv",
        "annual": output_directory
        / f"portfolio_risk_ablation_annual_{stamp}.csv",
        "rejections": output_directory
        / f"portfolio_risk_ablation_rejections_{stamp}.csv",
        "equity": output_directory
        / f"portfolio_risk_ablation_equity_{stamp}.csv",
    }

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Full-period shared-cash portfolio risk ablation. Maximum open "
            "positions remain fixed while risk per trade and maximum total "
            "open risk are varied on identical prepared data."
        ),
        "base_config": bundle["base_config"],
        "maximum_open_positions": bundle["maximum_open_positions"],
        "baseline": bundle["baseline"],
        "risk_grid": bundle["risk_grid"],
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
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")

    bundle["summary"].to_csv(paths["summary"], index=False)
    bundle["tickers"].to_csv(paths["tickers"], index=False)
    bundle["annual"].to_csv(paths["annual"], index=False)
    bundle["rejections"].to_csv(paths["rejections"], index=False)
    bundle["equity"].to_csv(paths["equity"], index=False)
    return paths


def print_portfolio_risk_ablation(bundle: dict[str, Any]) -> None:
    summary = bundle["summary"]
    columns = [
        "rank",
        "variant_id",
        "risk_per_trade_percent",
        "maximum_total_open_risk_percent",
        "effective_nominal_open_risk_cap_percent",
        "total_return_percent",
        "cagr_percent",
        "maximum_drawdown_percent",
        "return_drawdown_ratio",
        "profit_factor",
        "average_exposure_percent",
        "completed_trades",
        "excess_return_vs_matched_percent",
    ]

    print()
    print("=" * 170)
    print("AI STOCK RADAR — PORTFOLIO RISK ABLATION")
    print("=" * 170)
    print(
        summary[columns].to_string(
            index=False,
            formatters={
                "risk_per_trade_percent": lambda value: f"{value:.2f}%",
                "maximum_total_open_risk_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "effective_nominal_open_risk_cap_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "total_return_percent": lambda value: f"{value:+.4f}%",
                "cagr_percent": lambda value: f"{value:+.4f}%",
                "maximum_drawdown_percent": lambda value: f"{value:.4f}%",
                "return_drawdown_ratio": lambda value: f"{value:.4f}",
                "profit_factor": lambda value: f"{value:.4f}",
                "average_exposure_percent": lambda value: f"{value:.4f}%",
                "excess_return_vs_matched_percent": (
                    lambda value: f"{value:+.4f}%"
                ),
            },
        )
    )
    print("=" * 170)
    winner = summary.iloc[0]
    print(
        "Top full-period candidate: "
        f"{winner['variant_id']} — this is exploratory, not a live-setting "
        "decision. The winner must pass a risk walk-forward next."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare risk-per-trade and maximum-total-open-risk settings "
            "with a fixed shared-cash portfolio position limit."
        )
    )
    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--risks",
        nargs="+",
        type=float,
        default=list(DEFAULT_RISK_PER_TRADE_VALUES),
    )
    parser.add_argument(
        "--total-risks",
        nargs="+",
        type=float,
        default=list(DEFAULT_MAXIMUM_TOTAL_RISK_VALUES),
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAXIMUM_OPEN_POSITIONS,
    )
    parser.add_argument(
        "--baseline-risk",
        type=float,
        default=DEFAULT_BASELINE_RISK_PER_TRADE,
    )
    parser.add_argument(
        "--baseline-total-risk",
        type=float,
        default=DEFAULT_BASELINE_MAXIMUM_TOTAL_RISK,
    )
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)

    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--max-position", type=float, default=25.0)
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
    risks = normalize_risk_per_trade_values(arguments.risks)
    total_risks = normalize_maximum_total_risk_values(
        arguments.total_risks
    )

    base_config = PortfolioBacktestConfig(
        initial_cash=arguments.initial_cash,
        risk_per_trade_percent=arguments.baseline_risk,
        maximum_position_percent=arguments.max_position,
        maximum_total_open_risk_percent=arguments.baseline_total_risk,
        maximum_crypto_allocation_percent=arguments.max_crypto,
        maximum_open_positions=arguments.max_positions,
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

    bundle = run_portfolio_risk_ablation(
        data_by_ticker=prepared_data,
        base_config=base_config,
        risk_per_trade_values=risks,
        maximum_total_risk_values=total_risks,
        maximum_open_positions=arguments.max_positions,
        baseline_risk_per_trade_percent=arguments.baseline_risk,
        baseline_maximum_total_open_risk_percent=(
            arguments.baseline_total_risk
        ),
    )
    print_portfolio_risk_ablation(bundle)

    if not arguments.no_save:
        paths = save_portfolio_risk_ablation(
            bundle,
            output_directory=arguments.output_directory,
        )
        print()
        print("=" * 104)
        print("PORTFOLIO RISK ABLATION FILES")
        print("=" * 104)
        for label, path in paths.items():
            print(f"{label.upper():<12} {path.resolve()}")
        print("=" * 104)

    print()
    print("Portfolio risk ablation completed successfully.")


if __name__ == "__main__":
    main()
