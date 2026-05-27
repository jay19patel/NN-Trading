import numpy as np
import pandas as pd
import pandas_ta as ta


EPS = 1e-9


def _safe_div(numerator, denominator):
    """Vectorized division that stays finite around zero denominators."""
    return numerator / (denominator + EPS)


def _series_or_zero(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return series.reindex(index)


def _add_bbands(df: pd.DataFrame, close: pd.Series, prefix: str = "") -> None:
    bbands = ta.bbands(close, length=20)
    if bbands is not None and len(bbands.columns) >= 3:
        lower = bbands.iloc[:, 0]
        upper = bbands.iloc[:, 2]
        df[f"{prefix}bb_width"] = _safe_div(upper - lower, close)
        df[f"{prefix}bb_position"] = _safe_div(close - lower, upper - lower)
    else:
        df[f"{prefix}bb_width"] = 0.0
        df[f"{prefix}bb_position"] = 0.0


def _add_directional_indicators(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series, prefix: str = "") -> None:
    macd = ta.macd(close)
    if macd is not None and len(macd.columns) >= 3:
        df[f"{prefix}macd"] = macd.iloc[:, 0]
        df[f"{prefix}macd_hist"] = macd.iloc[:, 1]
        df[f"{prefix}macd_signal"] = macd.iloc[:, 2]
    else:
        df[f"{prefix}macd"] = 0.0
        df[f"{prefix}macd_hist"] = 0.0
        df[f"{prefix}macd_signal"] = 0.0

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and len(adx.columns) >= 3:
        df[f"{prefix}adx"] = adx.iloc[:, 0]
        df[f"{prefix}dmp"] = adx.iloc[:, 1]
        df[f"{prefix}dmn"] = adx.iloc[:, 2]
    else:
        df[f"{prefix}adx"] = 0.0
        df[f"{prefix}dmp"] = 0.0
        df[f"{prefix}dmn"] = 0.0


def _add_ema_stack(df: pd.DataFrame, close: pd.Series, prefix: str = "", lengths: tuple[int, ...] = (9, 21, 50)) -> None:
    emas: dict[int, pd.Series] = {}
    for length in lengths:
        ema = ta.ema(close, length=length)
        emas[length] = _series_or_zero(ema, close.index)
        df[f"{prefix}dist_ema_{length}"] = _safe_div(close - emas[length], emas[length])

    if 9 in emas and 21 in emas:
        df[f"{prefix}ema_9_21_spread"] = _safe_div(emas[9] - emas[21], emas[21])
    if 21 in emas and 50 in emas:
        df[f"{prefix}ema_21_50_spread"] = _safe_div(emas[21] - emas[50], emas[50])

    if 21 in emas:
        df[f"{prefix}trend_regime"] = np.sign(close - emas[21])


def _add_rsi_suite(df: pd.DataFrame, close: pd.Series, prefix: str = "") -> None:
    df[f"{prefix}rsi_7"] = ta.rsi(close, length=7)
    df[f"{prefix}rsi_14"] = ta.rsi(close, length=14)
    df[f"{prefix}rsi_21"] = ta.rsi(close, length=21)
    df[f"{prefix}rsi_slope_7_21"] = _safe_div(df[f"{prefix}rsi_7"] - df[f"{prefix}rsi_21"], 100.0)

    stoch_rsi = ta.stochrsi(close, length=14)
    if stoch_rsi is not None and len(stoch_rsi.columns) >= 2:
        df[f"{prefix}stochrsi_k"] = stoch_rsi.iloc[:, 0]
        df[f"{prefix}stochrsi_d"] = stoch_rsi.iloc[:, 1]
    else:
        df[f"{prefix}stochrsi_k"] = 0.0
        df[f"{prefix}stochrsi_d"] = 0.0


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["Open", "High", "Low", "Close", "Volume"]]
        .resample(rule, label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def _higher_timeframe_features(df: pd.DataFrame, rule: str, prefix: str, ema_lengths: tuple[int, ...]) -> pd.DataFrame:
    """
    Build completed higher-timeframe regime features and align them to base rows.

    The shifted frame means each 1h row only sees the previous completed 4h/1d
    candle, avoiding accidental lookahead from an in-progress higher-timeframe bar.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(index=df.index)

    htf = _resample_ohlcv(df, rule)
    if htf.empty:
        return pd.DataFrame(index=df.index)

    features = pd.DataFrame(index=htf.index)
    close = htf["Close"]
    high = htf["High"]
    low = htf["Low"]
    volume = htf["Volume"]

    features[f"{prefix}log_return_1"] = np.log(close / close.shift(1))
    features[f"{prefix}log_return_3"] = np.log(close / close.shift(3))
    features[f"{prefix}natr"] = _safe_div(ta.atr(high, low, close, length=14), close) * 100.0
    _add_ema_stack(features, close, prefix=prefix, lengths=ema_lengths)
    _add_rsi_suite(features, close, prefix=prefix)

    adx = ta.adx(high, low, close, length=14)
    features[f"{prefix}adx"] = adx.iloc[:, 0] if adx is not None and len(adx.columns) else 0.0
    _add_bbands(features, close, prefix=prefix)
    features[f"{prefix}volume_surprise"] = _safe_div(volume - volume.rolling(20).mean(), volume.rolling(20).std())

    return features.shift(1).reindex(df.index, method="ffill")


def _add_scalping_microstructure(df: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series, open_: pd.Series, volume: pd.Series) -> None:
    """
    Microstructure features specifically useful for 5m scalping.
    These capture short-term momentum and candle quality signals that
    the model uses to identify high-probability small moves.
    """
    candle_range = (high - low).replace(0, np.nan)

    # Short-term momentum (key for scalping)
    df["roc_3"] = close.pct_change(3)    # 15-min momentum
    df["roc_6"] = close.pct_change(6)    # 30-min momentum
    df["roc_12"] = close.pct_change(12)  # 1-hour momentum

    # Volume confirmation for entry quality
    df["volume_ratio_5"] = _safe_div(volume, volume.rolling(5).mean())
    df["volume_ratio_10"] = _safe_div(volume, volume.rolling(10).mean())

    # Candle quality score (clean directional bars = better entries)
    df["candle_directional_score"] = _safe_div(
        (close - open_).abs(), candle_range
    )

    # Price position within recent micro-range (5m context)
    high_6 = high.rolling(6).max()
    low_6 = low.rolling(6).min()
    df["micro_position_6"] = _safe_div(close - low_6, high_6 - low_6)

    # SMA20 for oracle-consistent trend filter
    df["sma_20"] = close.rolling(20).mean().fillna(close)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compact, trader-oriented features for the neural network.

    Feature philosophy:
      - keep log returns, not duplicate raw returns
      - describe trend with EMA stacks and RSI state
      - include volatility, volume, candle quality, and market structure
      - add previous completed 4h/1d regime context without lookahead
    """
    df = df.copy().sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]
    simple_return_1 = close.pct_change()

    # Returns and volatility
    for period in (1, 3, 6, 12, 24):
        df[f"log_return_{period}"] = np.log(close / close.shift(period))

    df["atr"] = ta.atr(high, low, close, length=14)
    df["natr"] = _safe_div(df["atr"], close) * 100.0
    df["realized_vol_12"] = df["log_return_1"].rolling(12).std()
    df["realized_vol_24"] = df["log_return_1"].rolling(24).std()
    df["volatility_regime_72"] = _safe_div(df["realized_vol_24"], df["realized_vol_24"].rolling(72).mean())
    df["atr_ratio"] = _safe_div(df["atr"], df["atr"].rolling(100).mean())

    # Momentum, trend, and mean reversion
    _add_rsi_suite(df, close)
    _add_ema_stack(df, close)
    _add_directional_indicators(df, high, low, close)
    _add_bbands(df, close)

    supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
    if supertrend is not None and "SUPERT_10_3" in supertrend.columns:
        df["supertrend"] = supertrend["SUPERT_10_3"]
    else:
        df["supertrend"] = close.rolling(10).mean()
    # REMOVED: trend_strength (redundant with dist_ema features)
    # df["trend_strength"] = _safe_div((close - df["supertrend"]).abs(), close)
    df["momentum_slope"] = df["log_return_1"].rolling(5).mean() - df["log_return_1"].rolling(20).mean()

    # Scalping microstructure features
    _add_scalping_microstructure(df, close, high, low, open_, volume)

    # Candle shape and local structure
    candle_range = (high - low).replace(0, np.nan)
    df["body_pct_range"] = _safe_div((close - open_).abs(), candle_range)
    df["upper_wick_pct_range"] = _safe_div(high - np.maximum(open_, close), candle_range)
    df["lower_wick_pct_range"] = _safe_div(np.minimum(open_, close) - low, candle_range)
    df["buy_pressure"] = _safe_div(close - low, candle_range)
    df["wick_imbalance"] = df["upper_wick_pct_range"] - df["lower_wick_pct_range"]
    df["candle_vs_atr"] = _safe_div(high - low, df["atr"])

    rolling_high_24 = high.rolling(24).max()
    rolling_low_24 = low.rolling(24).min()
    rolling_high_72 = high.rolling(72).max()
    rolling_low_72 = low.rolling(72).min()
    df["dist_high_24"] = _safe_div(close - rolling_high_24, rolling_high_24)
    df["dist_low_24"] = _safe_div(close - rolling_low_24, rolling_low_24)
    df["dist_high_72"] = _safe_div(close - rolling_high_72, rolling_high_72)
    df["dist_low_72"] = _safe_div(close - rolling_low_72, rolling_low_72)
    df["intraday_position"] = _safe_div(close - low.rolling(24).min(), high.rolling(24).max() - low.rolling(24).min())

    # Volume and participation
    vwap = ta.vwap(high, low, close, volume)
    df["vwap"] = vwap if vwap is not None else close.rolling(20).mean()
    df["price_to_vwap"] = _safe_div(close - df["vwap"], df["vwap"])
    df["vol_ratio_20"] = _safe_div(volume, volume.rolling(20).mean())
    df["volume_surprise_50"] = _safe_div(volume - volume.rolling(50).mean(), volume.rolling(50).std())
    df["vol_trend_5_20"] = _safe_div(volume.rolling(5).mean() - volume.rolling(20).mean(), volume.rolling(20).mean())
    obv = ta.obv(close, volume)
    df["obv_slope_20"] = _safe_div(obv - obv.shift(20), close * volume.rolling(20).mean())

    # Regime and information features
    df["efficiency_ratio_10"] = _safe_div((close - close.shift(10)).abs(), candle_range.rolling(10).sum())
    df["range_compression_10_50"] = _safe_div(candle_range.rolling(10).mean(), candle_range.rolling(50).mean())
    df["surprise_20"] = _safe_div(simple_return_1 - simple_return_1.rolling(20).mean(), simple_return_1.rolling(20).std())
    df["shock_elasticity_12"] = _safe_div(simple_return_1.abs(), df["realized_vol_12"])

    # ── NEW HIGH-IMPACT FEATURES ──────────────────────────────────────────
    # 1. Price Rate of Change (ROC) - direct % change
    df["roc_12"] = ((close - close.shift(12)) / close.shift(12)) * 100.0
    df["roc_24"] = ((close - close.shift(24)) / close.shift(24)) * 100.0

    # 2. Volume-Weighted Momentum (institutional flow detection)
    df["volume_momentum"] = (close.pct_change() * volume).rolling(10).sum()
    df["volume_momentum_norm"] = _safe_div(df["volume_momentum"], df["volume_momentum"].rolling(50).std())

    # 3. Close position within bar (order flow proxy)
    df["close_position_in_bar"] = _safe_div(close - low, high - low)
    df["avg_close_pos_10"] = df["close_position_in_bar"].rolling(10).mean()

    # 4. Bollinger Band squeeze (volatility breakout predictor)
    bb_width_norm = _safe_div(df["bb_width"], df["bb_width"].rolling(50).mean())
    df["bb_squeeze"] = (bb_width_norm < 0.5).astype(float)

    # 5. Volatility regime percentile
    df["vol_percentile_100"] = df["realized_vol_24"].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    # ──────────────────────────────────────────────────────────────────────

    # Time/session features from the candle timestamp if available.
    idx = pd.to_datetime(df.index, errors="coerce")
    if not idx.isna().all():
        hour = idx.hour + idx.minute / 60.0
        day_of_week = idx.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 0.0

    # Previous completed higher-timeframe context.
    htf_4h = _higher_timeframe_features(df, "4h", "mtf_4h_", ema_lengths=(9, 21, 50))
    htf_1d = _higher_timeframe_features(df, "1D", "mtf_1d_", ema_lengths=(9, 21))
    df = pd.concat([df, htf_4h, htf_1d], axis=1)

    # Cleanup: forward-fill true indicators, then zero-fill remaining warmup gaps.
    df = df.replace([np.inf, -np.inf], np.nan)
    indicator_cols = [
        col for col in df.columns
        if any(token in col for token in ("rsi", "macd", "adx", "dmp", "dmn", "ema", "atr", "bb_", "supertrend", "vwap"))
    ]
    df[indicator_cols] = df[indicator_cols].ffill()
    return df.fillna(0)


def get_feature_columns():
    return [
        # 1h price and volatility
        "log_return_1", "log_return_3", "log_return_6", "log_return_12", "log_return_24",
        "natr", "realized_vol_12", "realized_vol_24", "volatility_regime_72", "atr_ratio",

        # 1h trend and momentum
        "rsi_7", "rsi_14", "rsi_21", "rsi_slope_7_21", "stochrsi_k",
        "dist_ema_9", "dist_ema_21", "dist_ema_50",
        "ema_9_21_spread", "ema_21_50_spread", "trend_regime",
        "macd", "macd_hist", "adx",
        "bb_width", "bb_position", "momentum_slope",

        # Scalping microstructure
        "roc_3", "roc_6", "roc_12",
        "volume_ratio_5", "volume_ratio_10",
        "candle_directional_score",
        "micro_position_6",

        # 1h candle, structure, volume, and time
        "body_pct_range", "buy_pressure", "wick_imbalance", "candle_vs_atr",
        "dist_high_24", "dist_low_24", "dist_high_72", "dist_low_72", "intraday_position",
        "price_to_vwap", "vol_ratio_20", "volume_surprise_50", "vol_trend_5_20", "obv_slope_20",
        "efficiency_ratio_10", "range_compression_10_50", "surprise_20",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",

        # NEW: High-impact features
        "roc_24",
        "volume_momentum_norm",
        "avg_close_pos_10",
        "bb_squeeze",
        "vol_percentile_100",

        # 4h regime context
        "mtf_4h_log_return_1", "mtf_4h_natr",
        "mtf_4h_dist_ema_21", "mtf_4h_dist_ema_50",
        "mtf_4h_ema_9_21_spread", "mtf_4h_ema_21_50_spread", "mtf_4h_trend_regime",
        "mtf_4h_rsi_14", "mtf_4h_rsi_slope_7_21", "mtf_4h_adx",
        "mtf_4h_bb_position", "mtf_4h_volume_surprise",

        # 1d regime context
        "mtf_1d_log_return_1", "mtf_1d_natr",
        "mtf_1d_dist_ema_9", "mtf_1d_dist_ema_21",
        "mtf_1d_ema_9_21_spread", "mtf_1d_trend_regime",
        "mtf_1d_rsi_14", "mtf_1d_rsi_slope_7_21",
        "mtf_1d_bb_position", "mtf_1d_volume_surprise",
    ]
