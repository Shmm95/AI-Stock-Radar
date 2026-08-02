"""Rolling out-of-sample validation of portfolio initial-stop settings.

Each training window ranks the complete stock/crypto stop grid using only data
inside that window. The selected profile is then tested on the immediately
following unseen window and compared with four fixed controls. Historical
research only; this module cannot place orders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import DEFAULT_REGIME_PERIOD, DEFAULT_TICKERS
from src.backtest.run_portfolio_risk_walk_forward import _aggregate_model, _comparison_counts
from src.backtest.run_portfolio_stop_ablation import (
    DEFAULT_STOPS, normalize_stops, run_portfolio_stop_ablation, stop_variant_id,
)
from src.backtest.run_portfolio_walk_forward import (
    WalkForwardWindow, build_walk_forward_windows, slice_prepared_data,
)

DEFAULT_TRAIN_MONTHS = 24
DEFAULT_TEST_MONTHS = 6
DEFAULT_STEP_MONTHS = 6
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/stop_walk_forward")

MODEL_DYNAMIC = "DYNAMIC_SELECTED"
MODEL_BASELINE = "FIXED_BASELINE"
MODEL_RANK_WINNER = "FIXED_RANK_WINNER"
MODEL_MAX_RETURN = "FIXED_MAX_RETURN"
MODEL_LOW_DRAWDOWN = "FIXED_LOW_DRAWDOWN"


@dataclass(frozen=True, slots=True, order=True)
class StopProfile:
    stock_stop_loss_percent: float
    crypto_stop_loss_percent: float

    def __post_init__(self):
        stock = round(float(self.stock_stop_loss_percent), 8)
        crypto = round(float(self.crypto_stop_loss_percent), 8)
        if not 0 < stock <= 100 or not 0 < crypto <= 100:
            raise ValueError("Stop percentages must be in (0, 100].")
        object.__setattr__(self, "stock_stop_loss_percent", stock)
        object.__setattr__(self, "crypto_stop_loss_percent", crypto)

    @property
    def variant_id(self) -> str:
        return stop_variant_id(self.stock_stop_loss_percent, self.crypto_stop_loss_percent)

    def to_dict(self) -> dict[str, Any]:
        return {"variant_id": self.variant_id,
                "stock_stop_loss_percent": self.stock_stop_loss_percent,
                "crypto_stop_loss_percent": self.crypto_stop_loss_percent}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model: str
    profile: StopProfile


def _profile_from_row(row: pd.Series | dict[str, Any]) -> StopProfile:
    return StopProfile(float(row["stock_stop_loss_percent"]),
                       float(row["crypto_stop_loss_percent"]))


def _summary_row(frame: pd.DataFrame, profile: StopProfile) -> dict[str, Any]:
    rows = frame.loc[frame["variant_id"] == profile.variant_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {profile.variant_id}.")
    return rows.iloc[0].to_dict()


def _variant_records(frame: pd.DataFrame, profile: StopProfile) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.loc[frame["variant_id"] == profile.variant_id].to_dict("records")


def _run_profile(data: dict[str, pd.DataFrame], config: PortfolioBacktestConfig,
                 profile: StopProfile) -> dict[str, Any]:
    return run_portfolio_stop_ablation(
        data_by_ticker=data, base_config=config,
        stock_stops=(profile.stock_stop_loss_percent,),
        crypto_stops=(profile.crypto_stop_loss_percent,), baseline_stop=5.0)


def _append_scaled_equity(output: list[dict[str, Any]], *, equity: pd.DataFrame,
                          window: WalkForwardWindow, model: str,
                          profile: StopProfile, opening_capital: float,
                          initial_cash: float) -> float:
    if equity.empty:
        return opening_capital
    curve = equity.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])
    curve = curve.sort_values("timestamp")
    scale = opening_capital / initial_cash
    for row in curve.itertuples(index=False):
        output.append({"window_id": window.window_id, "model": model,
                       **profile.to_dict(), "timestamp": pd.Timestamp(row.timestamp),
                       "total_equity": round(float(row.total_equity) * scale, 6)})
    return round(float(curve.iloc[-1]["total_equity"]) * scale, 6)


def _validate_profile(label: str, profile: StopProfile,
                      stocks: Sequence[float], cryptos: Sequence[float]) -> None:
    if profile.stock_stop_loss_percent not in stocks:
        raise ValueError(f"{label} stock stop must be included in candidates.")
    if profile.crypto_stop_loss_percent not in cryptos:
        raise ValueError(f"{label} crypto stop must be included in candidates.")


def run_portfolio_stop_walk_forward(*, data_by_ticker: dict[str, pd.DataFrame],
        base_config: PortfolioBacktestConfig,
        stock_stops: Iterable[float] = DEFAULT_STOPS,
        crypto_stops: Iterable[float] = DEFAULT_STOPS,
        baseline_profile: StopProfile = StopProfile(5.0, 5.0),
        rank_winner_profile: StopProfile = StopProfile(5.0, 3.5),
        max_return_profile: StopProfile = StopProfile(3.5, 3.5),
        low_drawdown_profile: StopProfile = StopProfile(7.5, 3.5),
        train_months: int = DEFAULT_TRAIN_MONTHS,
        test_months: int = DEFAULT_TEST_MONTHS,
        step_months: int = DEFAULT_STEP_MONTHS) -> dict[str, Any]:
    stocks, cryptos = normalize_stops(stock_stops), normalize_stops(crypto_stops)
    controls = (("Baseline", baseline_profile), ("Rank winner", rank_winner_profile),
                ("Max return", max_return_profile), ("Low drawdown", low_drawdown_profile))
    for label, profile in controls:
        _validate_profile(label, profile, stocks, cryptos)
    base_config.validate()
    windows = build_walk_forward_windows(data_by_ticker, train_months=train_months,
        test_months=test_months, step_months=step_months)
    model_profiles_fixed = (
        ModelProfile(MODEL_BASELINE, baseline_profile),
        ModelProfile(MODEL_RANK_WINNER, rank_winner_profile),
        ModelProfile(MODEL_MAX_RETURN, max_return_profile),
        ModelProfile(MODEL_LOW_DRAWDOWN, low_drawdown_profile),
    )
    training_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    stitched_rows: list[dict[str, Any]] = []
    model_names = (MODEL_DYNAMIC, MODEL_BASELINE, MODEL_RANK_WINNER,
                   MODEL_MAX_RETURN, MODEL_LOW_DRAWDOWN)
    capitals = {model: float(base_config.initial_cash) for model in model_names}

    for sequence, window in enumerate(windows, 1):
        print(f"\nSTOP WALK-FORWARD [{sequence}/{len(windows)}] {window.window_id}")
        train_data = slice_prepared_data(data_by_ticker, start=window.train_start,
                                         end_exclusive=window.train_end_exclusive)
        train_bundle = run_portfolio_stop_ablation(
            data_by_ticker=train_data, base_config=base_config,
            stock_stops=stocks, crypto_stops=cryptos, baseline_stop=5.0)
        ranked = train_bundle["summary"].copy().reset_index(drop=True)
        ranked = ranked.rename(columns={"rank": "training_rank",
                                        "selected_candidate": "selected_for_test"})
        selected = _profile_from_row(ranked.iloc[0])
        for record in ranked.to_dict("records"):
            training_rows.append({**window.to_dict(), **record})

        test_data = slice_prepared_data(data_by_ticker, start=window.test_start,
                                        end_exclusive=window.test_end_exclusive)
        unique_profiles = {profile.variant_id: profile for profile in
                           (selected, baseline_profile, rank_winner_profile,
                            max_return_profile, low_drawdown_profile)}
        bundles = {key: _run_profile(test_data, base_config, profile)
                   for key, profile in unique_profiles.items()}
        all_models = (ModelProfile(MODEL_DYNAMIC, selected), *model_profiles_fixed)
        summaries: dict[str, dict[str, Any]] = {}
        for item in all_models:
            bundle = bundles[item.profile.variant_id]
            summary = _summary_row(bundle["summary"], item.profile)
            summaries[item.model] = summary
            test_rows.append({"window_id": window.window_id, "model": item.model,
                "selected_variant_id": selected.variant_id,
                "selected_stock_stop_loss_percent": selected.stock_stop_loss_percent,
                "selected_crypto_stop_loss_percent": selected.crypto_stop_loss_percent,
                "test_start": window.test_start.isoformat(),
                "test_end_exclusive": window.test_end_exclusive.isoformat(),
                **item.profile.to_dict(), **summary})
            for record in _variant_records(bundle["tickers"], item.profile):
                ticker_rows.append({"window_id": window.window_id, "model": item.model, **record})
            for record in _variant_records(bundle["rejections"], item.profile):
                rejection_rows.append({"window_id": window.window_id, "model": item.model, **record})
            equity = bundle["equity"].loc[bundle["equity"]["variant_id"] == item.profile.variant_id]
            capitals[item.model] = _append_scaled_equity(
                stitched_rows, equity=equity, window=window, model=item.model,
                profile=item.profile, opening_capital=capitals[item.model],
                initial_cash=base_config.initial_cash)

        baseline = summaries[MODEL_BASELINE]
        row: dict[str, Any] = {**window.to_dict(), **{f"selected_{k}": v for k, v in selected.to_dict().items()}}
        for item in all_models:
            short = {MODEL_DYNAMIC:"dynamic", MODEL_BASELINE:"baseline",
                     MODEL_RANK_WINNER:"rank_winner", MODEL_MAX_RETURN:"max_return",
                     MODEL_LOW_DRAWDOWN:"low_drawdown"}[item.model]
            summary = summaries[item.model]
            row[f"{short}_variant_id"] = item.profile.variant_id
            row[f"{short}_return_percent"] = summary["total_return_percent"]
            row[f"{short}_return_advantage_vs_baseline_percent"] = round(
                summary["total_return_percent"] - baseline["total_return_percent"], 4)
            row[f"{short}_max_drawdown_percent"] = summary["maximum_drawdown_percent"]
            row[f"{short}_profit_factor"] = summary["profit_factor"]
            row[f"{short}_excess_vs_matched_percent"] = summary["excess_return_vs_matched_percent"]
        window_rows.append(row)
        print(f"SELECTED={selected.variant_id} dynamic={summaries[MODEL_DYNAMIC]['total_return_percent']:+.4f}% "
              f"baseline={baseline['total_return_percent']:+.4f}%")

    windows_frame, training_frame = pd.DataFrame(window_rows), pd.DataFrame(training_rows)
    test_frame, tickers_frame = pd.DataFrame(test_rows), pd.DataFrame(ticker_rows)
    rejections_frame, stitched_frame = pd.DataFrame(rejection_rows), pd.DataFrame(stitched_rows)
    aggregate = pd.DataFrame([_aggregate_model(test_frame, stitched_frame, model=model,
        initial_cash=base_config.initial_cash, first_test_start=windows[0].test_start,
        last_test_end_exclusive=windows[-1].test_end_exclusive) for model in model_names])
    by_model = aggregate.set_index("model"); base = by_model.loc[MODEL_BASELINE]
    comparison: dict[str, Any] = {}
    for prefix, model in (("dynamic",MODEL_DYNAMIC),("rank_winner",MODEL_RANK_WINNER),
                          ("max_return",MODEL_MAX_RETURN),("low_drawdown",MODEL_LOW_DRAWDOWN)):
        candidate = by_model.loc[model]
        column = f"{prefix}_return_advantage_vs_baseline_percent"
        counts = _comparison_counts(windows_frame, column)
        comparison.update({
            f"{prefix}_minus_baseline_compounded_return_percent": round(
                candidate.compounded_return_percent - base.compounded_return_percent, 4),
            f"{prefix}_drawdown_advantage_vs_baseline_percent": round(
                base.maximum_drawdown_percent - candidate.maximum_drawdown_percent, 4),
            f"{prefix}_minus_baseline_return_drawdown_ratio": round(
                candidate.return_drawdown_ratio - base.return_drawdown_ratio, 4),
            f"{prefix}_window_win_count": counts["win_count"],
            f"{prefix}_window_tie_count": counts["tie_count"],
            f"{prefix}_window_loss_count": counts["loss_count"]})
    selected_ids = windows_frame["selected_variant_id"].astype(str)
    counts = Counter(selected_ids)
    selection_summary = {
        "variant_selection_counts": {stop_variant_id(s,c): int(counts.get(stop_variant_id(s,c),0))
                                     for s in stocks for c in cryptos},
        "baseline_selected_count": int(counts.get(baseline_profile.variant_id,0)),
        "selection_change_count": int(selected_ids.ne(selected_ids.shift()).iloc[1:].sum())}
    return {"windows": windows_frame, "training_candidates": training_frame,
        "test_runs": test_frame, "aggregate": aggregate,
        "ticker_contributions": tickers_frame, "rejections": rejections_frame,
        "stitched_equity": stitched_frame, "comparison": comparison,
        "selection_summary": selection_summary,
        "window_definitions": [window.to_dict() for window in windows],
        "base_config": base_config.to_dict(), "stock_stops": list(stocks),
        "crypto_stops": list(cryptos), "profiles": {item.model:item.profile.to_dict()
        for item in model_profiles_fixed}, "train_months": train_months,
        "test_months": test_months, "step_months": step_months}


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, float) and not isfinite(value): return None
    if isinstance(value, dict): return {str(k): _json_safe(v) for k,v in value.items()}
    if isinstance(value, list): return [_json_safe(v) for v in value]
    return value


def save_portfolio_stop_walk_forward(bundle: dict[str, Any], *,
        output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    names = {"windows":"windows", "training":"training_candidates",
             "test_runs":"test_runs", "aggregate":"aggregate",
             "tickers":"ticker_contributions", "rejections":"rejections",
             "equity":"stitched_equity"}
    paths = {name: output_directory / f"portfolio_stop_walk_forward_{name}_{stamp}.csv"
             for name in names}
    paths["json"] = output_directory / f"portfolio_stop_walk_forward_{stamp}.json"
    payload = {"created_at":datetime.now(UTC).isoformat(),
        "method":"24m train / 6m unseen test rolling portfolio stop validation",
        "base_config":bundle["base_config"], "stock_stops":bundle["stock_stops"],
        "crypto_stops":bundle["crypto_stops"], "profiles":bundle["profiles"],
        "window_definitions":bundle["window_definitions"],
        "aggregate":bundle["aggregate"].to_dict("records"),
        "comparison":bundle["comparison"], "selection_summary":bundle["selection_summary"]}
    paths["json"].write_text(json.dumps(_json_safe(payload), indent=2)+"\n", encoding="utf-8")
    for name, key in names.items(): bundle[key].to_csv(paths[name], index=False)
    return paths


def _profile_arg(values: Sequence[float]) -> StopProfile:
    return StopProfile(values[0], values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio initial-stop walk-forward")
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--stock-stops", nargs="+", type=float, default=list(DEFAULT_STOPS))
    parser.add_argument("--crypto-stops", nargs="+", type=float, default=list(DEFAULT_STOPS))
    parser.add_argument("--baseline", nargs=2, type=float, default=(5.0,5.0), metavar=("STOCK","CRYPTO"))
    parser.add_argument("--rank-winner", nargs=2, type=float, default=(5.0,3.5), metavar=("STOCK","CRYPTO"))
    parser.add_argument("--max-return", nargs=2, type=float, default=(3.5,3.5), metavar=("STOCK","CRYPTO"))
    parser.add_argument("--low-drawdown", nargs=2, type=float, default=(7.5,3.5), metavar=("STOCK","CRYPTO"))
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--period", default="5y"); parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--max-total-risk", type=float, default=4.0)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--max-position", type=float, default=25.0)
    parser.add_argument("--max-crypto", type=float, default=25.0)
    parser.add_argument("--stock-trailing", type=float, default=7.5)
    parser.add_argument("--crypto-trailing", type=float, default=7.5)
    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--minimum-fee", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fractional-stocks", action="store_true")
    parser.add_argument("--no-fractional-crypto", action="store_true")
    parser.add_argument("--no-crypto-regime", action="store_true")
    parser.add_argument("--no-force-close", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = PortfolioBacktestConfig(initial_cash=args.initial_cash,
        risk_per_trade_percent=args.risk,
        maximum_total_open_risk_percent=args.max_total_risk,
        maximum_open_positions=args.max_positions,
        maximum_position_percent=args.max_position,
        maximum_crypto_allocation_percent=args.max_crypto,
        stock_stop_loss_percent=5.0, crypto_stop_loss_percent=5.0,
        stock_trailing_close_percent=args.stock_trailing,
        crypto_trailing_close_percent=args.crypto_trailing,
        commission_rate=args.commission_rate, minimum_fee=args.minimum_fee,
        slippage_bps=args.slippage_bps,
        allow_fractional_stocks=args.fractional_stocks,
        allow_fractional_crypto=not args.no_fractional_crypto,
        force_close_at_end=not args.no_force_close)
    from src.backtest.run_portfolio_backtest import prepare_portfolio_data
    data = prepare_portfolio_data(tickers=args.tickers or list(DEFAULT_TICKERS),
        period=args.period, regime_period=args.regime_period,
        use_crypto_regime=not args.no_crypto_regime)
    bundle = run_portfolio_stop_walk_forward(data_by_ticker=data, base_config=config,
        stock_stops=args.stock_stops, crypto_stops=args.crypto_stops,
        baseline_profile=_profile_arg(args.baseline),
        rank_winner_profile=_profile_arg(args.rank_winner),
        max_return_profile=_profile_arg(args.max_return),
        low_drawdown_profile=_profile_arg(args.low_drawdown),
        train_months=args.train_months, test_months=args.test_months,
        step_months=args.step_months)
    print(bundle["aggregate"].to_string(index=False))
    if not args.no_save:
        for name,path in save_portfolio_stop_walk_forward(
                bundle, output_directory=args.output_directory).items():
            print(name, path.resolve())


if __name__ == "__main__": main()
