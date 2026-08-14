"""The three signal components, computed independently so the research
script can measure each in isolation as well as combined. Every
parameter choice is explained inline where it is defined -- see the
report for the empirical comparison that motivated (or didn't move)
each default.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalParameters:
    # --- Core signal: RSI(2) + SMA(200) trend filter (Connors-style) ---
    # Period=2 is the defining feature of the Connors "RSI2" mean-reversion
    # method: a lookback this short is deliberately noisy/reactive so it
    # reacts to a single sharp pullback rather than a multi-bar trend
    # change (that job belongs to the trend filter below, not RSI itself).
    rsi_period: int = 2
    # Connors' own equity research uses <5/<10 as "deeply oversold" on
    # DAILY bars. Hourly crypto bars are more volatile per-bar than daily
    # equity bars, so a raw <10 fires far more often here than it would
    # on daily data -- see the report's frequency-vs-quality tradeoff
    # for why this default was kept rather than tightened to <5.
    rsi_entry_threshold: float = 10.0
    # Exit once the snap-back is far enough along that the "deeply
    # oversold" edge this trade was taken for is gone.
    rsi_exit_threshold: float = 70.0
    # SMA(200) on HOURLY bars is ~8.3 days of trend context, NOT the
    # ~10-month window Connors' daily-bar SMA(200) represents. This is a
    # genuine scale change from the daily methodology, not an oversight
    # -- flagged explicitly in the report rather than assumed equivalent.
    # Kept at 200 (as specified) rather than rescaled, since rescaling
    # would be a second uncontrolled variable this phase-1 prototype
    # deliberately avoids introducing.
    trend_sma_period: int = 200
    # Backstop exit if RSI never reclaims 70 -- a mean-reversion trade
    # that hasn't reverted within a trading day (24 hourly bars) is
    # failing to do what it was taken for; holding longer just accumulates
    # trend risk the SMA(200) filter was supposed to keep this trade out of.
    max_holding_bars: int = 24

    # --- Parallel/OR signal: VWAP deviation ---
    # Rolling window for the VWAP itself: 24 bars (~1 day) is long enough
    # to smooth out single-bar noise but short enough to track a real,
    # recent "fair value" rather than a multi-day average that would lag
    # a genuine intraday dip.
    vwap_window: int = 24
    # Rolling window for the deviation's own standard deviation (of price
    # around the rolling VWAP) -- same 24-bar window as the VWAP itself
    # for consistency, not a separately-tuned value.
    vwap_std_window: int = 24
    # Entry: price at least this many standard deviations BELOW the
    # rolling VWAP. 2.0 is the standard Bollinger-style "statistically
    # unusual" threshold -- a deliberately conventional starting point,
    # not tuned to this data, since tuning it would confound the
    # component-isolation comparison the report is built around.
    vwap_deviation_entry: float = 2.0
    # Exit once price recovers back to (or above) the rolling VWAP --
    # the deviation this trade was taken for has closed.
    vwap_exit_at_or_above_mean: bool = True

    # --- Volume quality weighting (never a gate) ---
    volume_ma_window: int = 20
    # Below the volume MA is "weak" -- still taken, at reduced size. The
    # exact ratio (0.5x) is a simple, symmetric halving rather than a
    # separately-optimized value, matching this phase's "measure, don't
    # optimize" mandate (see the report for why this stays untuned).
    weak_volume_size_multiplier: float = 0.5


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI (the standard formula; period=2 is what makes
    this Connors' RSI2, not a different indicator)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Wilder's formula is undefined (0/0) when avg_loss is also 0 --
    # every recent bar was a gain, which is unambiguously NOT oversold.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def compute_rolling_vwap(frame: pd.DataFrame, window: int) -> pd.Series:
    """Volume-weighted rolling price, using Alpaca's own per-bar `vwap`
    field (the real intra-bar trade-weighted price) as the price input,
    not a cruder (H+L+C)/3 typical-price proxy."""
    price_volume = frame["vwap"] * frame["volume"]
    rolling_pv = price_volume.rolling(window, min_periods=window).sum()
    rolling_volume = frame["volume"].rolling(window, min_periods=window).sum()
    return rolling_pv / rolling_volume.replace(0.0, float("nan"))


def compute_signals(frame: pd.DataFrame, params: SignalParameters) -> pd.DataFrame:
    """Adds every intermediate + final signal column. Zero-volume bars
    are real (see the data-quality report) and must not corrupt the
    volume-MA comparison or the VWAP calculation -- both already handle
    them via `.replace(0.0, nan)` / pandas' own NaN propagation, so a
    zero-volume bar simply never qualifies as "strong" and never
    contributes a real price to the rolling VWAP average."""
    out = frame.copy()
    out["rsi2"] = compute_rsi(out["close"], params.rsi_period)
    out["sma_trend"] = out["close"].rolling(params.trend_sma_period, min_periods=params.trend_sma_period).mean()
    out["trend_up"] = out["close"] > out["sma_trend"]

    out["rolling_vwap"] = compute_rolling_vwap(out, params.vwap_window)
    price_dev = out["close"] - out["rolling_vwap"]
    out["vwap_dev_std"] = price_dev.rolling(params.vwap_std_window, min_periods=params.vwap_std_window).std()
    out["vwap_z"] = price_dev / out["vwap_dev_std"].replace(0.0, float("nan"))

    out["volume_ma"] = out["volume"].rolling(params.volume_ma_window, min_periods=params.volume_ma_window).mean()
    out["volume_strong"] = out["volume"] >= out["volume_ma"]

    out["rsi_signal"] = out["trend_up"] & (out["rsi2"] < params.rsi_entry_threshold)
    out["vwap_signal"] = out["trend_up"] & (out["vwap_z"] <= -params.vwap_deviation_entry)
    out["combined_signal"] = out["rsi_signal"] | out["vwap_signal"]

    return out
