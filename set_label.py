# -*- coding: utf-8 -*-
"""
Oracle Labeler - future price simulation for trading labels.

Supported labeling modes:
  - oracle_best: search TP x SL ATR grids and keep the best TP-first trade.
  - fixed_rule: use one fixed TP/SL ATR rule, useful for realistic analysis.

Labels:
  0 = LONG
  1 = NEUTRAL
  2 = SHORT
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

try:
    import numba
except ModuleNotFoundError:
    class _NumbaFallback:  # type: ignore[no-redef]
        @staticmethod
        def njit(func=None, **_kwargs):
            if func is None:
                return lambda wrapped: wrapped
            return func

    numba = _NumbaFallback()  # type: ignore[assignment]

from config import cfg

LabelingMode = Literal["oracle_best", "fixed_rule"]
EntryMode = Literal["current_close", "next_open"]

LONG = 0
NEUTRAL = 1
SHORT = 2

EXIT_TP = 0
EXIT_SL = 1
EXIT_TIMEOUT = 2
EXIT_NONE = 3

LABEL_NAMES = {LONG: "LONG", NEUTRAL: "NEUTRAL", SHORT: "SHORT"}
EXIT_REASON_NAMES = {
    EXIT_TP: "TP",
    EXIT_SL: "SL",
    EXIT_TIMEOUT: "TIMEOUT",
    EXIT_NONE: "NONE",
}
MA_SPECS = (
    ("sma20", "sma", 20),
    ("sma50", "sma", 50),
    ("sma200", "sma", 200),
    ("ema5", "ema", 5),
    ("ema9", "ema", 9),
    ("ema20", "ema", 20),
    ("ema50", "ema", 50),
    ("ema200", "ema", 200),
    ("hma9", "hma", 9),
    ("hma21", "hma", 21),
    ("hma50", "hma", 50),
)
MA_STATUS_COLUMNS = tuple(f"price_above_{name}" for name, _, _ in MA_SPECS)
TECHNICAL_CONDITION_COLUMNS = (
    "condition_rsi14_oversold",
    "condition_rsi14_neutral",
    "condition_rsi14_overbought",
    "condition_macd_bullish",
    "condition_macd_bearish",
    "condition_macd_hist_positive",
    "condition_macd_hist_negative",
    "condition_adx_strong",
    "condition_adx_weak",
    "condition_dmp_above_dmn",
    "condition_dmn_above_dmp",
    "condition_bb_lower_zone",
    "condition_bb_middle_zone",
    "condition_bb_upper_zone",
    "condition_bb_squeeze",
    "condition_stochrsi_oversold",
    "condition_stochrsi_overbought",
    "condition_stochrsi_k_above_d",
    "condition_volume_surprise_high",
    "condition_volume_surprise_low",
    "condition_roc3_positive",
    "condition_roc3_negative",
    "condition_ema9_above_ema21",
    "condition_ema5_above_ema20",
    "condition_ema5_cross_above_ema20",
    "condition_ema5_cross_below_ema20",
    "condition_ema21_above_ema50",
    "condition_ema50_above_ema200",
    "condition_sma50_above_sma200",
    "condition_price_above_hma9",
    "condition_price_below_hma9",
    "condition_hma9_above_hma21",
    "condition_hma9_cross_above_hma21",
    "condition_hma9_cross_below_hma21",
    "condition_trend_regime_up",
    "condition_trend_regime_down",
    "condition_rsi_rising",
    "condition_rsi_falling",
    "condition_momentum_up",
    "condition_momentum_down",
    "condition_roc12_positive",
    "condition_roc12_negative",
    "condition_above_vwap",
    "condition_below_vwap",
    "condition_obv_rising",
    "condition_obv_falling",
    "condition_mtf_4h_trend_up",
    "condition_mtf_4h_trend_down",
    "condition_mtf_4h_rsi_bullish",
    "condition_mtf_4h_rsi_bearish",
    "condition_mtf_1d_trend_up",
    "condition_mtf_1d_trend_down",
    "condition_mtf_1d_rsi_bullish",
    "condition_mtf_1d_rsi_bearish",
)

# ── Column schema ─────────────────────────────────────────────────────────────
# All columns that are derived from FUTURE price data.
# NEVER use these as ML model features — they cause data leakage.
LABEL_COLUMNS: tuple[str, ...] = (
    "direction_label",
    "take_profit_pct",
    "stop_loss_pct",
    "expected_return_pct",
    "time_to_target",
    "magnitude_label",
    "entry_price",
    "exit_price",
    "exit_reason",
    "bars_to_exit",
    "trade_return_pct",
    "gross_return_pct",
    "net_return_pct",
    "direction_name",
    "sma_filter_used",
    "labeling_mode",
    "entry_mode",
    "rr_ratio",
    "label_valid",
    "oracle_direction_label",
    "oracle_direction_name",
    "confirmation_score",
    "confirmation_bull_score",
    "confirmation_bear_score",
    "confirmation_edge",
    "confirmation_filter_used",
    "label_filter_reason",
)

_RAW_OHLCV: frozenset[str] = frozenset({"Open", "High", "Low", "Close", "Volume"})


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the columns that are safe to use as ML model input features.

    Excludes:
      - All future-derived label columns (LABEL_COLUMNS)
      - Raw OHLCV price columns
    What remains are causal technical indicators computed from past/current
    bars only — safe for supervised learning without look-ahead bias.

    Usage:
        features = df[get_feature_columns(df)]
        labels   = df["direction_label"]
    """
    exclude = frozenset(LABEL_COLUMNS) | _RAW_OHLCV
    return [c for c in df.columns if c not in exclude]


def _causal_atr_pct(df: pd.DataFrame, length: int) -> np.ndarray:
    """Compute ATR% using only current/past OHLC values."""
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    atr_pct = (atr / df["Close"]) * 100.0
    return atr_pct.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)


def _causal_sma(df: pd.DataFrame, length: int = 20) -> np.ndarray:
    """Compute SMA using current/past closes only."""
    if "sma_20" in df.columns:
        return df["sma_20"].ffill().fillna(df["Close"]).to_numpy(dtype=np.float64)
    return df["Close"].rolling(length).mean().fillna(df["Close"]).to_numpy(dtype=np.float64)


def _weighted_moving_average(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=np.float64)
    return series.rolling(length, min_periods=length).apply(
        lambda values: float(np.dot(values, weights) / weights.sum()),
        raw=True,
    )


def _hma(series: pd.Series, length: int) -> pd.Series:
    half_length = max(int(length / 2), 1)
    sqrt_length = max(int(np.sqrt(length)), 1)
    raw_hma = 2.0 * _weighted_moving_average(series, half_length) - _weighted_moving_average(series, length)
    return _weighted_moving_average(raw_hma, sqrt_length)


def _atr_ratio(df: pd.DataFrame, atr_pct: np.ndarray) -> np.ndarray:
    """Volatility ratio used by the existing oracle filter."""
    if "atr" in df.columns:
        atr = df["atr"]
    else:
        atr = pd.Series(atr_pct, index=df.index)
    return (atr / atr.rolling(100).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=np.float64)


def _add_moving_average_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal SMA/EMA values and price-above status columns.

    These columns are computed from Close values up to the signal candle only.
    They are safe for signal-candle analytics and feature inspection.
    """
    out = df.copy()
    close = out["Close"]

    for name, ma_type, period in MA_SPECS:
        if name not in out.columns:
            if ma_type == "sma":
                out[name] = close.rolling(period, min_periods=period).mean()
            elif ma_type == "ema":
                out[name] = close.ewm(span=period, adjust=False, min_periods=period).mean()
            else:
                out[name] = _hma(close, period)

        out[f"price_above_{name}"] = (close > out[name]).fillna(False).astype(bool)

    return out


def _safe_div(numerator, denominator):
    return numerator / (denominator + 1e-9)


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = _atr_series(high, low, close, length)
    dmp = 100.0 * _safe_div(plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean(), atr)
    dmn = 100.0 * _safe_div(minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean(), atr)
    dx = 100.0 * _safe_div((dmp - dmn).abs(), dmp + dmn)
    adx = dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return adx, dmp, dmn


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    return (
        df[["Open", "High", "Low", "Close", "Volume"]]
        .resample(rule, label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def _basic_context_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out[f"{prefix}log_return_1"] = np.log(close / close.shift(1))
    out[f"{prefix}natr"] = _safe_div(_atr_series(high, low, close), close) * 100.0

    ema9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out[f"{prefix}dist_ema_9"] = _safe_div(close - ema9, ema9)
    out[f"{prefix}dist_ema_21"] = _safe_div(close - ema21, ema21)
    out[f"{prefix}dist_ema_50"] = _safe_div(close - ema50, ema50)
    out[f"{prefix}ema_9_21_spread"] = _safe_div(ema9 - ema21, ema21)
    out[f"{prefix}ema_21_50_spread"] = _safe_div(ema21 - ema50, ema50)
    out[f"{prefix}trend_regime"] = np.sign(close - ema21)

    out[f"{prefix}rsi_14"] = _rsi(close, 14)
    out[f"{prefix}rsi_slope_7_21"] = _safe_div(_rsi(close, 7) - _rsi(close, 21), 100.0)
    adx, _, _ = _adx_dmi(high, low, close)
    out[f"{prefix}adx"] = adx

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    out[f"{prefix}bb_position"] = _safe_div(close - bb_lower, bb_upper - bb_lower)
    out[f"{prefix}volume_surprise"] = _safe_div(volume - volume.rolling(20).mean(), volume.rolling(20).std())
    return out


def _higher_timeframe_features(df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    htf = _resample_ohlcv(df, rule)
    if htf.empty:
        return pd.DataFrame(index=df.index)
    features = _basic_context_features(htf, prefix=prefix).shift(1)
    keep = [col for col in features.columns if col in {
        f"{prefix}log_return_1",
        f"{prefix}natr",
        f"{prefix}dist_ema_9",
        f"{prefix}dist_ema_21",
        f"{prefix}dist_ema_50",
        f"{prefix}ema_9_21_spread",
        f"{prefix}ema_21_50_spread",
        f"{prefix}trend_regime",
        f"{prefix}rsi_14",
        f"{prefix}rsi_slope_7_21",
        f"{prefix}adx",
        f"{prefix}bb_position",
        f"{prefix}volume_surprise",
    }]
    return features[keep].reindex(df.index, method="ffill")


def _add_technical_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal technical features and signal-condition flags."""
    out = df.copy().sort_index()
    out = out.drop(
        columns=[
            col for col in out.columns
            if col.startswith(("mtf_4h_", "mtf_1d_", "condition_"))
        ],
        errors="ignore",
    )
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    open_ = out["Open"]
    volume = out["Volume"]
    candle_range = (high - low).replace(0, np.nan)

    for period in (1, 3, 6, 12, 24):
        out[f"log_return_{period}"] = np.log(close / close.shift(period))

    out["atr"] = _atr_series(high, low, close)
    out["natr"] = _safe_div(out["atr"], close) * 100.0
    out["realized_vol_12"] = out["log_return_1"].rolling(12).std()
    out["realized_vol_24"] = out["log_return_1"].rolling(24).std()
    out["volatility_regime_72"] = _safe_div(out["realized_vol_24"], out["realized_vol_24"].rolling(72).mean())
    out["atr_ratio"] = _safe_div(out["atr"], out["atr"].rolling(100).mean())

    out["rsi_7"] = _rsi(close, 7)
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_21"] = _rsi(close, 21)
    out["rsi_slope_7_21"] = _safe_div(out["rsi_7"] - out["rsi_21"], 100.0)
    rsi_min = out["rsi_14"].rolling(14).min()
    rsi_max = out["rsi_14"].rolling(14).max()
    out["stochrsi_k"] = _safe_div(out["rsi_14"] - rsi_min, rsi_max - rsi_min) * 100.0
    out["stochrsi_d"] = out["stochrsi_k"].rolling(3).mean()

    ema5 = close.ewm(span=5, adjust=False, min_periods=5).mean()
    ema9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    hma9 = _hma(close, 9)
    hma21 = _hma(close, 21)
    hma50 = _hma(close, 50)
    out["ema5"] = ema5
    out["ema9"] = ema9
    out["ema20"] = ema20
    out["ema21"] = ema21
    out["ema50"] = ema50
    out["ema200"] = ema200
    out["sma50"] = sma50
    out["sma200"] = sma200
    out["hma9"] = hma9
    out["hma21"] = hma21
    out["hma50"] = hma50
    out["dist_ema_5"] = _safe_div(close - ema5, ema5)
    out["dist_ema_9"] = _safe_div(close - ema9, ema9)
    out["dist_ema_20"] = _safe_div(close - ema20, ema20)
    out["dist_ema_21"] = _safe_div(close - ema21, ema21)
    out["dist_ema_50"] = _safe_div(close - ema50, ema50)
    out["dist_hma_9"] = _safe_div(close - hma9, hma9)
    out["dist_hma_21"] = _safe_div(close - hma21, hma21)
    out["ema_5_20_spread"] = _safe_div(ema5 - ema20, ema20)
    out["ema_9_21_spread"] = _safe_div(ema9 - ema21, ema21)
    out["ema_21_50_spread"] = _safe_div(ema21 - ema50, ema50)
    out["hma_9_21_spread"] = _safe_div(hma9 - hma21, hma21)
    out["trend_regime"] = np.sign(close - ema21)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    adx, dmp, dmn = _adx_dmi(high, low, close)
    out["adx"] = adx
    out["dmp"] = dmp
    out["dmn"] = dmn

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    out["bb_width"] = _safe_div(bb_upper - bb_lower, close)
    out["bb_position"] = _safe_div(close - bb_lower, bb_upper - bb_lower)
    out["momentum_slope"] = out["log_return_1"].rolling(5).mean() - out["log_return_1"].rolling(20).mean()

    for period in (3, 6, 12, 24):
        out[f"roc_{period}"] = close.pct_change(period) * 100.0
    out["volume_ratio_5"] = _safe_div(volume, volume.rolling(5).mean())
    out["volume_ratio_10"] = _safe_div(volume, volume.rolling(10).mean())
    out["candle_directional_score"] = _safe_div((close - open_).abs(), candle_range)
    out["micro_position_6"] = _safe_div(close - low.rolling(6).min(), high.rolling(6).max() - low.rolling(6).min())

    out["body_pct_range"] = _safe_div((close - open_).abs(), candle_range)
    out["buy_pressure"] = _safe_div(close - low, candle_range)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    out["wick_imbalance"] = _safe_div(upper_wick - lower_wick, candle_range)
    out["candle_vs_atr"] = _safe_div(high - low, out["atr"])
    out["dist_high_24"] = _safe_div(close - high.rolling(24).max(), high.rolling(24).max())
    out["dist_low_24"] = _safe_div(close - low.rolling(24).min(), low.rolling(24).min())
    out["dist_high_72"] = _safe_div(close - high.rolling(72).max(), high.rolling(72).max())
    out["dist_low_72"] = _safe_div(close - low.rolling(72).min(), low.rolling(72).min())
    out["intraday_position"] = _safe_div(close - low.rolling(24).min(), high.rolling(24).max() - low.rolling(24).min())

    typical_price = (high + low + close) / 3.0
    vwap = _safe_div((typical_price * volume).rolling(20).sum(), volume.rolling(20).sum())
    out["vwap"] = vwap
    out["price_to_vwap"] = _safe_div(close - vwap, vwap)
    out["vol_ratio_20"] = _safe_div(volume, volume.rolling(20).mean())
    out["volume_surprise_50"] = _safe_div(volume - volume.rolling(50).mean(), volume.rolling(50).std())
    out["vol_trend_5_20"] = _safe_div(volume.rolling(5).mean() - volume.rolling(20).mean(), volume.rolling(20).mean())
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    out["obv_slope_20"] = _safe_div(obv - obv.shift(20), close * volume.rolling(20).mean())
    out["efficiency_ratio_10"] = _safe_div((close - close.shift(10)).abs(), candle_range.rolling(10).sum())
    out["range_compression_10_50"] = _safe_div(candle_range.rolling(10).mean(), candle_range.rolling(50).mean())
    simple_return_1 = close.pct_change()
    out["surprise_20"] = _safe_div(simple_return_1 - simple_return_1.rolling(20).mean(), simple_return_1.rolling(20).std())

    volume_momentum = (close.pct_change() * volume).rolling(10).sum()
    out["volume_momentum_norm"] = _safe_div(volume_momentum, volume_momentum.rolling(50).std())
    out["avg_close_pos_10"] = _safe_div(close - low, high - low).rolling(10).mean()
    bb_width_norm = _safe_div(out["bb_width"], out["bb_width"].rolling(50).mean())
    out["bb_squeeze"] = (bb_width_norm < 0.5).astype(float)
    out["vol_percentile_100"] = out["realized_vol_24"].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) else 0.5,
        raw=False,
    )

    if isinstance(out.index, pd.DatetimeIndex):
        hour = out.index.hour + out.index.minute / 60.0
        dow = out.index.dayofweek
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 0.0
        out["dow_sin"] = 0.0
        out["dow_cos"] = 0.0

    out = pd.concat(
        [
            out,
            _higher_timeframe_features(out, "4h", "mtf_4h_"),
            _higher_timeframe_features(out, "1D", "mtf_1d_"),
        ],
        axis=1,
    )

    condition_data = {
        "condition_rsi14_oversold": out["rsi_14"] < 30.0,
        "condition_rsi14_neutral": out["rsi_14"].between(30.0, 70.0, inclusive="both"),
        "condition_rsi14_overbought": out["rsi_14"] > 70.0,
        "condition_macd_bullish": out["macd"] > out["macd_signal"],
        "condition_macd_bearish": out["macd"] < out["macd_signal"],
        "condition_macd_hist_positive": out["macd_hist"] > 0.0,
        "condition_macd_hist_negative": out["macd_hist"] < 0.0,
        "condition_adx_strong": out["adx"] >= 25.0,
        "condition_adx_weak": out["adx"] < 20.0,
        "condition_dmp_above_dmn": out["dmp"] > out["dmn"],
        "condition_dmn_above_dmp": out["dmn"] > out["dmp"],
        "condition_bb_lower_zone": out["bb_position"] < 0.2,
        "condition_bb_middle_zone": out["bb_position"].between(0.2, 0.8, inclusive="both"),
        "condition_bb_upper_zone": out["bb_position"] > 0.8,
        "condition_bb_squeeze": out["bb_squeeze"] > 0.0,
        "condition_stochrsi_oversold": out["stochrsi_k"] < 20.0,
        "condition_stochrsi_overbought": out["stochrsi_k"] > 80.0,
        "condition_stochrsi_k_above_d": out["stochrsi_k"] > out["stochrsi_d"],
        "condition_volume_surprise_high": out["volume_surprise_50"] > 1.0,
        "condition_volume_surprise_low": out["volume_surprise_50"] < -1.0,
        "condition_roc3_positive": out["roc_3"] > 0.0,
        "condition_roc3_negative": out["roc_3"] < 0.0,
        # ── EMA / SMA trend stack (directional — like price_above_ema*) ──────
        "condition_ema9_above_ema21": out["ema_9_21_spread"] > 0.0,
        "condition_ema5_above_ema20": out["ema_5_20_spread"] > 0.0,
        "condition_ema5_cross_above_ema20": (ema5 > ema20) & (ema5.shift(1) <= ema20.shift(1)),
        "condition_ema5_cross_below_ema20": (ema5 < ema20) & (ema5.shift(1) >= ema20.shift(1)),
        "condition_ema21_above_ema50": out["ema_21_50_spread"] > 0.0,
        "condition_ema50_above_ema200": ema50 > ema200,
        "condition_sma50_above_sma200": sma50 > sma200,
        "condition_price_above_hma9": close > hma9,
        "condition_price_below_hma9": close < hma9,
        "condition_hma9_above_hma21": out["hma_9_21_spread"] > 0.0,
        "condition_hma9_cross_above_hma21": (hma9 > hma21) & (hma9.shift(1) <= hma21.shift(1)),
        "condition_hma9_cross_below_hma21": (hma9 < hma21) & (hma9.shift(1) >= hma21.shift(1)),
        # ── Trend regime ────────────────────────────────────────────────────
        "condition_trend_regime_up": out["trend_regime"] > 0.0,
        "condition_trend_regime_down": out["trend_regime"] < 0.0,
        # ── Momentum direction ──────────────────────────────────────────────
        "condition_rsi_rising": out["rsi_slope_7_21"] > 0.0,
        "condition_rsi_falling": out["rsi_slope_7_21"] < 0.0,
        "condition_momentum_up": out["momentum_slope"] > 0.0,
        "condition_momentum_down": out["momentum_slope"] < 0.0,
        "condition_roc12_positive": out["roc_12"] > 0.0,
        "condition_roc12_negative": out["roc_12"] < 0.0,
        # ── VWAP location ───────────────────────────────────────────────────
        "condition_above_vwap": out["Close"] > out["vwap"],
        "condition_below_vwap": out["Close"] < out["vwap"],
        # ── On-balance-volume direction ─────────────────────────────────────
        "condition_obv_rising": out["obv_slope_20"] > 0.0,
        "condition_obv_falling": out["obv_slope_20"] < 0.0,
        # ── Higher-timeframe trend alignment (4h / 1d) ──────────────────────
        "condition_mtf_4h_trend_up": out["mtf_4h_trend_regime"] > 0.0,
        "condition_mtf_4h_trend_down": out["mtf_4h_trend_regime"] < 0.0,
        "condition_mtf_4h_rsi_bullish": out["mtf_4h_rsi_14"] > 50.0,
        "condition_mtf_4h_rsi_bearish": out["mtf_4h_rsi_14"] < 50.0,
        "condition_mtf_1d_trend_up": out["mtf_1d_trend_regime"] > 0.0,
        "condition_mtf_1d_trend_down": out["mtf_1d_trend_regime"] < 0.0,
        "condition_mtf_1d_rsi_bullish": out["mtf_1d_rsi_14"] > 50.0,
        "condition_mtf_1d_rsi_bearish": out["mtf_1d_rsi_14"] < 50.0,
    }
    out = pd.concat([out, pd.DataFrame(condition_data, index=out.index)], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    bool_cols = list(MA_STATUS_COLUMNS) + list(TECHNICAL_CONDITION_COLUMNS)
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    return out.ffill().fillna(0.0)


@numba.njit
def _clip_pct(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


@numba.njit
def _simulate_trade_path(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    row_index: int,
    lookahead: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    direction_is_long: bool,
    entry_price: float,
) -> tuple[int, float, float, float, int]:
    """
    Walk forward after signal generation and return the first outcome.

    Conservative behavior is preserved: if TP and SL are both touched in the
    same candle, SL wins because candle intrabar order is unknown.
    """
    if direction_is_long:
        tp_price = entry_price * (1.0 + take_profit_pct / 100.0)
        sl_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    else:
        tp_price = entry_price * (1.0 - take_profit_pct / 100.0)
        sl_price = entry_price * (1.0 + stop_loss_pct / 100.0)

    max_exit_index = min(row_index + lookahead, len(close_prices) - 1)

    for step in range(1, lookahead + 1):
        bar_index = row_index + step
        if bar_index >= len(close_prices):
            break

        bar_high = high_prices[bar_index]
        bar_low = low_prices[bar_index]

        if direction_is_long:
            hit_sl = bar_low <= sl_price
            hit_tp = bar_high >= tp_price
        else:
            hit_sl = bar_high >= sl_price
            hit_tp = bar_low <= tp_price

        if hit_sl:
            return EXIT_SL, sl_price, -stop_loss_pct, -stop_loss_pct, step
        if hit_tp:
            return EXIT_TP, tp_price, take_profit_pct, take_profit_pct, step

    exit_price = close_prices[max_exit_index]
    if direction_is_long:
        gross_return_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:
        gross_return_pct = ((entry_price - exit_price) / entry_price) * 100.0
    return EXIT_TIMEOUT, exit_price, gross_return_pct, gross_return_pct, lookahead


@numba.njit
def _compute_labels_jit(
    row_count: int,
    lookahead: int,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    atr_pct: np.ndarray,
    tp_multipliers: np.ndarray,
    sl_multipliers: np.ndarray,
    fixed_tp_atr_multiplier: float,
    fixed_sl_atr_multiplier: float,
    trade_cost_pct: float,
    slippage_fraction: float,
    min_rr: float,
    min_target_pct: float,
    max_target_pct: float,
    min_stop_pct: float,
    max_stop_pct: float,
    sma20: np.ndarray,
    atr_ratio: np.ndarray,
    labeling_mode_code: int,
    entry_mode_code: int,
    use_sma_filter: bool,
):
    labels = np.full(row_count, NEUTRAL)
    tp_out = np.zeros(row_count)
    sl_out = np.zeros(row_count)
    expected_return = np.zeros(row_count)
    time_to_target = np.full(row_count, float(lookahead))
    magnitude = np.zeros(row_count)
    entry_prices = np.full(row_count, np.nan)
    exit_prices = np.full(row_count, np.nan)
    exit_reasons = np.full(row_count, EXIT_NONE)
    bars_to_exit = np.zeros(row_count)
    gross_returns = np.zeros(row_count)
    net_returns = np.zeros(row_count)
    rr_ratios = np.zeros(row_count)

    for row_index in range(row_count - lookahead):
        current_atr = atr_pct[row_index]
        if current_atr <= 0.0:
            continue

        if entry_mode_code == 1:
            entry_bar_index = row_index + 1
            if entry_bar_index >= row_count:
                continue
            base_entry = open_prices[entry_bar_index]
        else:
            base_entry = close_prices[row_index]

        best_label = NEUTRAL
        best_tp = 0.0
        best_sl = 0.0
        best_entry = np.nan
        best_exit = np.nan
        best_exit_reason = EXIT_NONE
        best_bars = 0
        best_gross = 0.0
        best_net = 0.0
        best_rr = 0.0

        tp_loop_count = len(tp_multipliers)
        sl_loop_count = len(sl_multipliers)
        if labeling_mode_code == 1:
            tp_loop_count = 1
            sl_loop_count = 1

        for tp_i in range(tp_loop_count):
            if labeling_mode_code == 1:
                tp_pct = current_atr * fixed_tp_atr_multiplier
            else:
                tp_pct = current_atr * tp_multipliers[tp_i]
            tp_pct = _clip_pct(tp_pct, min_target_pct, max_target_pct)

            for sl_i in range(sl_loop_count):
                if labeling_mode_code == 1:
                    sl_pct = current_atr * fixed_sl_atr_multiplier
                else:
                    sl_pct = current_atr * sl_multipliers[sl_i]
                sl_pct = _clip_pct(sl_pct, min_stop_pct, max_stop_pct)

                rr = tp_pct / max(sl_pct, 1e-9)
                if rr < min_rr:
                    continue

                # Keep the old volatility guard for oracle mode only. Fixed
                # rule mode should test the rule as-is for fair comparison.
                if labeling_mode_code == 0 and atr_ratio[row_index] > 2.0:
                    continue

                allow_long = (not use_sma_filter) or close_prices[row_index] > sma20[row_index]
                allow_short = (not use_sma_filter) or close_prices[row_index] < sma20[row_index]

                if allow_long:
                    entry_long = base_entry * (1.0 + slippage_fraction)
                    reason, exit_price, gross, _, bars = _simulate_trade_path(
                        close_prices, high_prices, low_prices, row_index, lookahead,
                        tp_pct, sl_pct, True, entry_long,
                    )
                    net = gross - trade_cost_pct
                    if reason == EXIT_TP and net > best_net:
                        best_label = LONG
                        best_tp = tp_pct
                        best_sl = sl_pct
                        best_entry = entry_long
                        best_exit = exit_price
                        best_exit_reason = reason
                        best_bars = bars
                        best_gross = gross
                        best_net = net
                        best_rr = rr

                if allow_short:
                    entry_short = base_entry * (1.0 - slippage_fraction)
                    reason, exit_price, gross, _, bars = _simulate_trade_path(
                        close_prices, high_prices, low_prices, row_index, lookahead,
                        tp_pct, sl_pct, False, entry_short,
                    )
                    net = gross - trade_cost_pct
                    if reason == EXIT_TP and net > best_net:
                        best_label = SHORT
                        best_tp = tp_pct
                        best_sl = sl_pct
                        best_entry = entry_short
                        best_exit = exit_price
                        best_exit_reason = reason
                        best_bars = bars
                        best_gross = gross
                        best_net = net
                        best_rr = rr

        if best_label != NEUTRAL:
            labels[row_index] = best_label
            tp_out[row_index] = best_tp
            sl_out[row_index] = best_sl
            expected_return[row_index] = best_net
            time_to_target[row_index] = float(best_bars)
            magnitude[row_index] = best_tp if best_label == LONG else -best_tp
            entry_prices[row_index] = best_entry
            exit_prices[row_index] = best_exit
            exit_reasons[row_index] = best_exit_reason
            bars_to_exit[row_index] = best_bars
            gross_returns[row_index] = best_gross
            net_returns[row_index] = best_net
            rr_ratios[row_index] = best_rr

    return (
        labels,
        tp_out,
        sl_out,
        expected_return,
        time_to_target,
        magnitude,
        entry_prices,
        exit_prices,
        exit_reasons,
        bars_to_exit,
        gross_returns,
        net_returns,
        rr_ratios,
    )


def _max_drawdown_pct(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0) / 100.0).cumprod()
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    return float(drawdown.min())


def _profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


BULLISH_CONFIRMATION_COLUMNS: tuple[str, ...] = (
    "price_above_ema5",
    "price_above_ema9",
    "price_above_ema20",
    "price_above_hma9",
    "condition_ema5_above_ema20",
    "condition_ema5_cross_above_ema20",
    "condition_ema9_above_ema21",
    "condition_hma9_above_hma21",
    "condition_hma9_cross_above_hma21",
    "condition_trend_regime_up",
    "condition_macd_bullish",
    "condition_macd_hist_positive",
    "condition_dmp_above_dmn",
    "condition_rsi_rising",
    "condition_momentum_up",
    "condition_roc3_positive",
    "condition_roc12_positive",
    "condition_above_vwap",
    "condition_obv_rising",
    "condition_mtf_4h_trend_up",
    "condition_mtf_4h_rsi_bullish",
)
BULLISH_REQUIRED_COLUMNS: tuple[str, ...] = ()
BEARISH_CONFIRMATION_COLUMNS: tuple[str, ...] = (
    "price_above_ema5",
    "price_above_ema9",
    "price_above_ema20",
    "price_above_hma9",
    "condition_ema5_above_ema20",
    "condition_ema5_cross_below_ema20",
    "condition_hma9_cross_below_hma21",
    "condition_trend_regime_down",
    "condition_macd_bearish",
    "condition_macd_hist_negative",
    "condition_dmn_above_dmp",
    "condition_rsi_falling",
    "condition_momentum_down",
    "condition_roc3_negative",
    "condition_roc12_negative",
    "condition_below_vwap",
    "condition_obv_falling",
    "condition_mtf_4h_trend_down",
    "condition_mtf_4h_rsi_bearish",
)
BEARISH_REQUIRED_COLUMNS: tuple[str, ...] = ()


def _sum_true_columns(df: pd.DataFrame, columns: tuple[str, ...], invert: frozenset[str] = frozenset()) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].fillna(False).astype(bool)
        if column in invert:
            values = ~values
        score += values.astype(float)
    return score


def _apply_confirmation_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    training = cfg.training
    use_filter = bool(training.USE_TECHNICAL_CONFIRMATION_FILTER)

    bearish_invert = frozenset({"price_above_ema5", "price_above_ema9", "price_above_ema20", "price_above_hma9", "condition_ema5_above_ema20"})
    bull_score = _sum_true_columns(out, BULLISH_CONFIRMATION_COLUMNS)
    bear_score = _sum_true_columns(out, BEARISH_CONFIRMATION_COLUMNS, invert=bearish_invert)
    edge = (bull_score - bear_score).abs()

    out["oracle_direction_label"] = out["direction_label"].astype(int)
    out["oracle_direction_name"] = out["direction_name"]
    out["confirmation_bull_score"] = bull_score
    out["confirmation_bear_score"] = bear_score
    out["confirmation_score"] = np.where(out["direction_label"] == LONG, bull_score, np.where(out["direction_label"] == SHORT, bear_score, 0.0))
    out["confirmation_edge"] = edge
    out["confirmation_filter_used"] = use_filter
    out["label_filter_reason"] = np.where(out["direction_label"] == NEUTRAL, "oracle_neutral", "confirmed")

    if not use_filter:
        out.loc[out["direction_label"] != NEUTRAL, "label_filter_reason"] = "filter_disabled"
        return out

    long_required = pd.Series(True, index=out.index)
    for column in BULLISH_REQUIRED_COLUMNS:
        if column in out.columns:
            long_required &= out[column].fillna(False).astype(bool)

    short_required = pd.Series(True, index=out.index)
    for column in BEARISH_REQUIRED_COLUMNS:
        if column not in out.columns:
            continue
        values = out[column].fillna(False).astype(bool)
        if column in bearish_invert:
            values = ~values
        short_required &= values

    is_trade = out["direction_label"].isin([LONG, SHORT])
    weak = is_trade & (out["confirmation_score"] < float(training.MIN_CONFIRMATION_SCORE))
    conflicted = is_trade & (out["confirmation_edge"] < float(training.MIN_CONFIRMATION_EDGE))
    missing_required = (
        ((out["direction_label"] == LONG) & ~long_required)
        | ((out["direction_label"] == SHORT) & ~short_required)
    )
    non_positive_return = is_trade & (out["net_return_pct"] <= float(training.MIN_CONFIRMED_RETURN_PCT))
    invalid = is_trade & ~out["label_valid"].astype(bool)

    filters = [
        (invalid, "invalid_lookahead_window"),
        (non_positive_return, "non_positive_return"),
        (missing_required, "missing_required_trend_confirmation"),
        (weak, "weak_confirmation"),
        (conflicted, "conflicting_confirmation"),
    ]
    reject = pd.Series(False, index=out.index)
    for mask, reason in filters:
        active = mask & ~reject
        out.loc[active, "label_filter_reason"] = reason
        reject |= mask

    out.loc[reject, ["direction_label", "take_profit_pct", "stop_loss_pct", "expected_return_pct", "magnitude_label", "bars_to_exit", "trade_return_pct", "gross_return_pct", "net_return_pct", "rr_ratio"]] = 0
    out.loc[reject, "direction_label"] = NEUTRAL
    out.loc[reject, "direction_name"] = LABEL_NAMES[NEUTRAL]
    out.loc[reject, "exit_reason"] = EXIT_REASON_NAMES[EXIT_NONE]
    out.loc[reject, ["entry_price", "exit_price"]] = np.nan
    return out


class OracleLabeler:
    """Generate trading labels and compare label quality without ML training."""

    def generate_labels(
        self,
        df: pd.DataFrame,
        labeling_mode: LabelingMode = "oracle_best",
        use_sma_filter: bool = True,
        entry_mode: EntryMode = "next_open",
        fixed_tp_atr_multiplier: float = 1.0,
        fixed_sl_atr_multiplier: float = 0.5,
    ) -> pd.DataFrame:
        """
        Annotate OHLCV data with labels and trade diagnostics.

        Existing API still works: calling generate_labels(df) returns the same
        core label columns, now with extra columns for analysis.
        """
        if labeling_mode not in ("oracle_best", "fixed_rule"):
            raise ValueError("labeling_mode must be 'oracle_best' or 'fixed_rule'")
        if entry_mode not in ("current_close", "next_open"):
            raise ValueError("entry_mode must be 'current_close' or 'next_open'")

        required = {"Open", "High", "Low", "Close"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

        training = cfg.training
        lookahead = training.LOOKAHEAD_BARS
        atr_pct = _causal_atr_pct(df, training.ATR_LENGTH)
        sma20 = _causal_sma(df)
        atr_ratio = _atr_ratio(df, atr_pct)

        trade_cost_pct = cfg.testing.ROUND_TRIP_FEE_PCT + 2.0 * cfg.testing.SLIPPAGE_PCT
        slippage_fraction = cfg.testing.SLIPPAGE_PCT / 100.0

        results = _compute_labels_jit(
            len(df),
            lookahead,
            df["Open"].to_numpy(dtype=np.float64),
            df["Close"].to_numpy(dtype=np.float64),
            df["High"].to_numpy(dtype=np.float64),
            df["Low"].to_numpy(dtype=np.float64),
            atr_pct,
            np.array(training.TP_ATR_MULTIPLIERS, dtype=np.float64),
            np.array(training.SL_ATR_MULTIPLIERS, dtype=np.float64),
            float(fixed_tp_atr_multiplier),
            float(fixed_sl_atr_multiplier),
            float(trade_cost_pct),
            float(slippage_fraction),
            float(training.ORACLE_MIN_RR),
            float(training.MIN_ATR_TARGET_PCT),
            float(training.MAX_ATR_TARGET_PCT),
            float(training.MIN_ATR_STOP_PCT),
            float(training.MAX_ATR_STOP_PCT),
            sma20,
            atr_ratio,
            0 if labeling_mode == "oracle_best" else 1,
            0 if entry_mode == "current_close" else 1,
            bool(use_sma_filter),
        )

        (
            labels,
            tp_pct,
            sl_pct,
            exp_ret,
            time_to_target,
            magnitude,
            entry_prices,
            exit_prices,
            exit_reason_codes,
            bars_to_exit,
            gross_returns,
            net_returns,
            rr_ratios,
        ) = results

        df = _add_moving_average_status_columns(df)
        df = _add_technical_signal_features(df)
        df["direction_label"] = labels.astype(int)
        df["take_profit_pct"] = tp_pct
        df["stop_loss_pct"] = sl_pct
        df["expected_return_pct"] = exp_ret
        df["time_to_target"] = time_to_target
        df["magnitude_label"] = magnitude

        df["entry_price"] = entry_prices
        df["exit_price"] = exit_prices
        df["exit_reason"] = pd.Series(exit_reason_codes, index=df.index).map(EXIT_REASON_NAMES)
        df["bars_to_exit"] = bars_to_exit
        df["trade_return_pct"] = net_returns
        df["gross_return_pct"] = gross_returns
        df["net_return_pct"] = net_returns
        df["direction_name"] = pd.Series(labels.astype(int), index=df.index).map(LABEL_NAMES)
        df["sma_filter_used"] = bool(use_sma_filter)
        df["labeling_mode"] = labeling_mode
        df["entry_mode"] = entry_mode
        df["rr_ratio"] = rr_ratios

        # Mark the last `lookahead` rows as invalid: the label loop stops at
        # row_count-lookahead, so these rows are NEUTRAL due to data truncation
        # (not market conditions). Exclude them from model training.
        label_valid = np.ones(len(df), dtype=bool)
        label_valid[-lookahead:] = False
        df["label_valid"] = label_valid
        df = _apply_confirmation_filter(df)
        return df

    def compare_sma_filter_stats(
        self,
        df: pd.DataFrame,
        labeling_mode: LabelingMode = "fixed_rule",
        entry_mode: EntryMode = "next_open",
        fixed_tp_atr_multiplier: float = 1.0,
        fixed_sl_atr_multiplier: float = 0.5,
    ) -> pd.DataFrame:
        """Run labels with and without SMA20 filter and return trading metrics."""
        reports = []
        for use_sma_filter, name in [(False, "without_sma20"), (True, "with_sma20")]:
            labeled = self.generate_labels(
                df.copy(),
                labeling_mode=labeling_mode,
                use_sma_filter=use_sma_filter,
                entry_mode=entry_mode,
                fixed_tp_atr_multiplier=fixed_tp_atr_multiplier,
                fixed_sl_atr_multiplier=fixed_sl_atr_multiplier,
            )
            row = self._stats_row(labeled)
            row["scenario"] = name
            reports.append(row)

        columns = ["scenario"] + [col for col in reports[0].keys() if col != "scenario"]
        return pd.DataFrame(reports)[columns]

    def label_quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a compact label quality report for an already-labeled DataFrame."""
        counts = df["direction_label"].value_counts()
        total = len(df)
        trades = df[df["direction_label"] != NEUTRAL]
        reasons = df["exit_reason"].value_counts() if "exit_reason" in df.columns else pd.Series(dtype=int)

        report = {
            "total_rows": total,
            "long_count": int(counts.get(LONG, 0)),
            "neutral_count": int(counts.get(NEUTRAL, 0)),
            "short_count": int(counts.get(SHORT, 0)),
            "long_pct": self._pct(counts.get(LONG, 0), total),
            "neutral_pct": self._pct(counts.get(NEUTRAL, 0), total),
            "short_pct": self._pct(counts.get(SHORT, 0), total),
            "avg_tp_pct": float(trades["take_profit_pct"].mean()) if not trades.empty else 0.0,
            "avg_sl_pct": float(trades["stop_loss_pct"].mean()) if not trades.empty else 0.0,
            "avg_return_pct": float(trades["net_return_pct"].mean()) if not trades.empty else 0.0,
            "avg_time_to_target": float(trades["bars_to_exit"].mean()) if not trades.empty else 0.0,
            "timeout_trades": int(reasons.get("TIMEOUT", 0)),
            "sl_hits": int(reasons.get("SL", 0)),
            "tp_hits": int(reasons.get("TP", 0)),
        }
        return pd.DataFrame([report])

    def moving_average_relationship_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare BUY/SELL signal quality by price relationship to SMA/EMA values.

        Input can be raw OHLCV plus labels or an already-labeled DataFrame. The
        report only evaluates rows where direction_label is LONG or SHORT.
        """
        required = {"direction_label", "net_return_pct", "gross_return_pct"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame must be labeled first. Missing: {sorted(missing)}")

        analyzed = _add_moving_average_status_columns(df)
        signals = analyzed[analyzed["direction_label"].isin([LONG, SHORT])].copy()

        rows: list[dict[str, float | int | str]] = []
        for direction_value, direction_name, expected_above_label in (
            (LONG, "BUY", "above"),
            (SHORT, "SELL", "below"),
        ):
            direction_trades = signals[signals["direction_label"] == direction_value]
            profitable_direction = direction_trades[direction_trades["net_return_pct"] > 0]

            for ma_name, _, _ in MA_SPECS:
                status_col = f"price_above_{ma_name}"
                for status_value, relationship in ((True, "above"), (False, "below")):
                    subset = direction_trades[direction_trades[status_col] == status_value]
                    rows.append(
                        self._ma_stats_row(
                            subset=subset,
                            direction_trades=direction_trades,
                            profitable_direction=profitable_direction,
                            direction_name=direction_name,
                            ma_name=ma_name,
                            relationship=relationship,
                            is_expected_trend_side=relationship == expected_above_label,
                        )
                    )

        report = pd.DataFrame(rows)
        if report.empty:
            return report

        return report.sort_values(
            ["direction", "moving_average", "is_expected_trend_side"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    def best_moving_average_conditions(self, df: pd.DataFrame, min_trades: int = 10) -> pd.DataFrame:
        """Return MA conditions ranked by profit factor, net return, and win rate."""
        report = self.moving_average_relationship_report(df)
        if report.empty:
            return report
        filtered = report[report["total_trades"] >= min_trades].copy()
        return filtered.sort_values(
            ["profit_factor", "net_return_pct", "win_rate_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def technical_condition_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare BUY/SELL signal quality across RSI, MACD, ADX, BB, volume, and ROC conditions.
        """
        required = {"direction_label", "net_return_pct", "gross_return_pct"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame must be labeled first. Missing: {sorted(missing)}")

        analyzed = _add_technical_signal_features(df)
        signals = analyzed[analyzed["direction_label"].isin([LONG, SHORT])].copy()

        rows: list[dict[str, float | int | str]] = []
        for condition in TECHNICAL_CONDITION_COLUMNS:
            for direction_value, direction_name in ((LONG, "BUY"), (SHORT, "SELL")):
                direction_trades = signals[signals["direction_label"] == direction_value]
                profitable_direction = direction_trades[direction_trades["net_return_pct"] > 0]
                subset = direction_trades[direction_trades[condition]]
                rows.append(
                    self._condition_stats_row(
                        subset=subset,
                        direction_trades=direction_trades,
                        profitable_direction=profitable_direction,
                        direction_name=direction_name,
                        condition=condition,
                    )
                )

        return pd.DataFrame(rows)

    def best_technical_conditions(self, df: pd.DataFrame, min_trades: int = 10) -> pd.DataFrame:
        """Return technical conditions ranked by profit factor, net return, and win rate."""
        report = self.technical_condition_report(df)
        if report.empty:
            return report
        filtered = report[report["total_trades"] >= min_trades].copy()
        return filtered.sort_values(
            ["profit_factor", "net_return_pct", "win_rate_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def _stats_row(self, df: pd.DataFrame) -> dict[str, float | int]:
        total = len(df)
        trades = df[df["direction_label"] != NEUTRAL].copy()
        longs = trades[trades["direction_label"] == LONG]
        shorts = trades[trades["direction_label"] == SHORT]
        returns = trades["net_return_pct"] if not trades.empty else pd.Series(dtype=float)
        gross_returns = trades["gross_return_pct"] if not trades.empty else pd.Series(dtype=float)
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        return {
            "total_rows": int(total),
            "total_trades": int(len(trades)),
            "long_trades": int(len(longs)),
            "short_trades": int(len(shorts)),
            "neutral_count": int((df["direction_label"] == NEUTRAL).sum()),
            "trade_rate_pct": self._pct(len(trades), total),
            "gross_return_pct": float(gross_returns.sum()) if not gross_returns.empty else 0.0,
            "net_return_pct": float(returns.sum()) if not returns.empty else 0.0,
            "avg_return_per_trade": float(returns.mean()) if not returns.empty else 0.0,
            "win_rate_pct": self._pct(len(wins), len(trades)),
            "loss_rate_pct": self._pct(len(losses), len(trades)),
            "profit_factor": _profit_factor(returns),
            "average_win_pct": float(wins.mean()) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean()) if not losses.empty else 0.0,
            "max_drawdown_pct": _max_drawdown_pct(returns),
            "avg_bars_to_exit": float(trades["bars_to_exit"].mean()) if not trades.empty else 0.0,
            "median_bars_to_exit": float(trades["bars_to_exit"].median()) if not trades.empty else 0.0,
            "avg_tp_pct": float(trades["take_profit_pct"].mean()) if not trades.empty else 0.0,
            "avg_sl_pct": float(trades["stop_loss_pct"].mean()) if not trades.empty else 0.0,
            "long_win_rate_pct": self._pct((longs["net_return_pct"] > 0).sum(), len(longs)),
            "short_win_rate_pct": self._pct((shorts["net_return_pct"] > 0).sum(), len(shorts)),
        }

    def _ma_stats_row(
        self,
        subset: pd.DataFrame,
        direction_trades: pd.DataFrame,
        profitable_direction: pd.DataFrame,
        direction_name: str,
        ma_name: str,
        relationship: str,
        is_expected_trend_side: bool,
    ) -> dict[str, float | int | str]:
        returns = subset["net_return_pct"] if not subset.empty else pd.Series(dtype=float)
        gross_returns = subset["gross_return_pct"] if not subset.empty else pd.Series(dtype=float)
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        profitable_subset_count = int((subset["net_return_pct"] > 0).sum()) if not subset.empty else 0
        return {
            "direction": direction_name,
            "moving_average": ma_name,
            "relationship": relationship,
            "condition": f"price_{relationship}_{ma_name}",
            "is_expected_trend_side": bool(is_expected_trend_side),
            "total_trades": int(len(subset)),
            "trade_pct_of_direction": self._pct(len(subset), len(direction_trades)),
            "profitable_trades": profitable_subset_count,
            "profitable_trade_pct_of_direction": self._pct(profitable_subset_count, len(profitable_direction)),
            "win_rate_pct": self._pct(len(wins), len(subset)),
            "loss_rate_pct": self._pct(len(losses), len(subset)),
            "gross_return_pct": float(gross_returns.sum()) if not gross_returns.empty else 0.0,
            "net_return_pct": float(returns.sum()) if not returns.empty else 0.0,
            "avg_return_per_trade": float(returns.mean()) if not returns.empty else 0.0,
            "profit_factor": _profit_factor(returns),
            "average_win_pct": float(wins.mean()) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean()) if not losses.empty else 0.0,
            "max_drawdown_pct": _max_drawdown_pct(returns),
            "avg_bars_to_exit": float(subset["bars_to_exit"].mean()) if not subset.empty else 0.0,
        }

    def _condition_stats_row(
        self,
        subset: pd.DataFrame,
        direction_trades: pd.DataFrame,
        profitable_direction: pd.DataFrame,
        direction_name: str,
        condition: str,
    ) -> dict[str, float | int | str]:
        returns = subset["net_return_pct"] if not subset.empty else pd.Series(dtype=float)
        gross_returns = subset["gross_return_pct"] if not subset.empty else pd.Series(dtype=float)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        profitable_subset_count = int((subset["net_return_pct"] > 0).sum()) if not subset.empty else 0
        return {
            "direction": direction_name,
            "condition": condition,
            "total_trades": int(len(subset)),
            "trade_pct_of_direction": self._pct(len(subset), len(direction_trades)),
            "profitable_trades": profitable_subset_count,
            "profitable_trade_pct_of_direction": self._pct(profitable_subset_count, len(profitable_direction)),
            "win_rate_pct": self._pct(len(wins), len(subset)),
            "loss_rate_pct": self._pct(len(losses), len(subset)),
            "gross_return_pct": float(gross_returns.sum()) if not gross_returns.empty else 0.0,
            "net_return_pct": float(returns.sum()) if not returns.empty else 0.0,
            "avg_return_per_trade": float(returns.mean()) if not returns.empty else 0.0,
            "profit_factor": _profit_factor(returns),
            "average_win_pct": float(wins.mean()) if not wins.empty else 0.0,
            "average_loss_pct": float(losses.mean()) if not losses.empty else 0.0,
            "max_drawdown_pct": _max_drawdown_pct(returns),
            "avg_bars_to_exit": float(subset["bars_to_exit"].mean()) if not subset.empty else 0.0,
        }

    @staticmethod
    def _pct(numerator: float, denominator: float) -> float:
        return float(numerator / denominator * 100.0) if denominator else 0.0
