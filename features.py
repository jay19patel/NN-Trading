# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats
from config import config
from ui_utils import console, get_progress


FEATURE_CACHE_VERSION = config.data.FEATURE_CACHE_VERSION


def _ffill_only(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Forward fill only so we do not leak future values into earlier rows."""
    return series.ffill()


def _daily_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    session_key = pd.to_datetime(df.index).normalize()
    cumulative_pv = (typical_price * df["Volume"]).groupby(session_key).cumsum()
    cumulative_volume = df["Volume"].groupby(session_key).cumsum()
    return cumulative_pv / cumulative_volume.replace(0, np.nan)


def _simulate_trade_path(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    row_index: int,
    lookahead: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    direction: str,
) -> tuple[str, float]:
    """Return first-hit path outcome and signed pnl percent for a candidate trade."""
    entry_price = float(close_prices[row_index])
    final_close = float(close_prices[row_index + lookahead])

    if direction == "long":
        take_profit_price = entry_price * (1 + take_profit_pct / 100.0)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)
    else:
        take_profit_price = entry_price * (1 - take_profit_pct / 100.0)
        stop_loss_price = entry_price * (1 + stop_loss_pct / 100.0)

    for step in range(1, lookahead + 1):
        bar_high = float(high_prices[row_index + step])
        bar_low = float(low_prices[row_index + step])
        if direction == "long":
            hit_stop = bar_low <= stop_loss_price
            hit_target = bar_high >= take_profit_price
        else:
            hit_stop = bar_high >= stop_loss_price
            hit_target = bar_low <= take_profit_price

        if hit_stop and hit_target:
            return "FAILED", -stop_loss_pct
        if hit_stop:
            return "FAILED", -stop_loss_pct
        if hit_target:
            return "SUCCESS", take_profit_pct

    if direction == "long":
        timeout_pnl = ((final_close - entry_price) / entry_price) * 100.0
    else:
        timeout_pnl = ((entry_price - final_close) / entry_price) * 100.0
    return "TIMEOUT", timeout_pnl


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fundamental price and volume based features"""
    df['close_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_1'] = df['Close'].pct_change()
    df['return_5'] = df['Close'].pct_change(5)
    df['return_10'] = df['Close'].pct_change(10)
    df['return_20'] = df['Close'].pct_change(20)

    df['open_close_return'] = np.log(df['Close'] / df['Open'])
    df['high_open_return'] = np.log(df['High'] / df['Open'])
    df['low_open_return'] = np.log(df['Low'] / df['Open'])

    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['volume_std'] = df['Volume'].rolling(20).std()

    df['daily_range'] = (df['High'] - df['Low']) / df['Close']
    df['range_pct'] = ((df['High'] - df['Low']) / df['Open']) * 100

    return df

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add various moving averages to identify trends"""
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'EMA_{period}'] = _ffill_only(ta.ema(df['Close'], length=period))

    for period in [20, 50, 100, 200]:
        df[f'SMA_{period}'] = _ffill_only(ta.sma(df['Close'], length=period))

    df['price_to_ema_5'] = (df['Close'] - df['EMA_5']) / df['EMA_5']
    df['price_to_ema_20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['price_to_ema_50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    df['price_to_ema_100'] = (df['Close'] - df['EMA_100']) / df['EMA_100']
    df['price_to_ema_200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
    df['price_to_sma_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['price_to_sma_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

    df['ema_5_10_cross'] = df['EMA_5'] - df['EMA_10']
    df['ema_10_20_cross'] = df['EMA_10'] - df['EMA_20']
    df['ema_5_10_cross_pct'] = (df['EMA_5'] - df['EMA_10']) / df['Close']
    df['ema_10_20_cross_pct'] = (df['EMA_10'] - df['EMA_20']) / df['Close']

    df['VWAP'] = _daily_vwap(df)
    df['price_to_vwap'] = (df['Close'] - df['VWAP']) / df['VWAP']

    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum oscillators to identify overbought/oversold conditions"""
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = _ffill_only(ta.rsi(df['Close'], length=period))

    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = _ffill_only(macd['MACD_12_26_9'])
    df['MACD_signal'] = _ffill_only(macd['MACDs_12_26_9'])
    df['MACD_hist'] = _ffill_only(macd['MACDh_12_26_9'])
    df['MACD_hist_pct'] = df['MACD_hist'] / df['Close']

    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Stoch_K'] = _ffill_only(stoch['STOCHk_14_3_3'])
    df['Stoch_D'] = _ffill_only(stoch['STOCHd_14_3_3'])

    df['CCI_20'] = _ffill_only(ta.cci(df['High'], df['Low'], df['Close'], length=20))
    df['WilliamsR_14'] = _ffill_only(ta.willr(df['High'], df['Low'], df['Close'], length=14))
    df['ROC_10'] = _ffill_only(ta.roc(df['Close'], length=10))
    df['ROC_20'] = _ffill_only(ta.roc(df['Close'], length=20))

    return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility measures to gauge market risk and movement potential"""
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = _ffill_only(ta.atr(df['High'], df['Low'], df['Close'], length=period))

    df['ATR_pct'] = (df['ATR_14'] / df['Close']) * 100

    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
         df['BB_upper'] = _ffill_only(bbands.iloc[:, 2])
         df['BB_middle'] = _ffill_only(bbands.iloc[:, 1])
         df['BB_lower'] = _ffill_only(bbands.iloc[:, 0])

    df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle']) * 100
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
    if kc is not None:
        df['KC_upper'] = _ffill_only(kc.iloc[:, 2])
        df['KC_lower'] = _ffill_only(kc.iloc[:, 0])

    df['volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['volatility_20'] = df['Close'].pct_change().rolling(20).std()
    df['volatility_50'] = df['Close'].pct_change().rolling(50).std()

    return df

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators that measure trend strength and direction"""
    for period in [14, 20]:
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=period)
        df[f'ADX_{period}'] = _ffill_only(adx_df[f'ADX_{period}'])
        df[f'DMP_{period}'] = _ffill_only(adx_df[f'DMP_{period}'])
        df[f'DMN_{period}'] = _ffill_only(adx_df[f'DMN_{period}'])

    df['directional_bias'] = df['DMP_14'] - df['DMN_14']

    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    df['supertrend'] = _ffill_only(supertrend["SUPERT_10_3"])
    df['supertrend_direction'] = _ffill_only(supertrend["SUPERTd_10_3"])
    df['supertrend_distance'] = (df['Close'] - df['supertrend']) / df['Close']

    df['st_flip'] = df['supertrend_direction'].diff().abs()
    df['bars_since_flip'] = df.groupby((df['st_flip'] == 2).cumsum()).cumcount()

    aroon = ta.aroon(df['High'], df['Low'], length=25)
    df['aroon_up'] = _ffill_only(aroon['AROONU_25'])
    df['aroon_down'] = _ffill_only(aroon['AROOND_25'])
    df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']

    return df

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Choppiness Index to distinguish between trending and ranging markets"""
    # Chop Index: n=14
    n = 14
    tr = ta.true_range(df['High'], df['Low'], df['Close'])
    atr_sum = tr.rolling(window=n).sum()
    max_high = df['High'].rolling(window=n).max()
    min_low = df['Low'].rolling(window=n).min()
    
    df['chop_index'] = 100 * np.log10(atr_sum / (max_high - min_low + 1e-8)) / np.log10(n)
    df['is_choppy'] = (df['chop_index'] > 61.8).astype(int)
    df['is_trending'] = (df['chop_index'] < 38.2).astype(int)
    
    return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators to confirm price movements"""
    df['OBV'] = _ffill_only(ta.obv(df['Close'], df['Volume']))
    df['OBV_ema'] = _ffill_only(ta.ema(df['OBV'], length=20))
    df['OBV_change_20'] = df['OBV'].pct_change(20).replace([np.inf, -np.inf], np.nan)
    df['AD'] = _ffill_only(ta.ad(df['High'], df['Low'], df['Close'], df['Volume']))
    df['CMF'] = _ffill_only(ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20))
    df['MFI'] = _ffill_only(ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14))
    df['VPT'] = _ffill_only(ta.pvt(df['Close'], df['Volume']))
    df['volume_change_5'] = df['Volume'].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df['volume_zscore_20'] = (
        (df['Volume'] - df['Volume'].rolling(20).mean())
        / (df['Volume'].rolling(20).std() + 1e-6)
    )
    return df

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern recognition features"""
    df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_imbalance'] = (df['upper_wick'] - df['lower_wick']) / df['Close']
    df['wick_to_body'] = (df['total_wick'] / (abs(df['Close'] - df['Open']) + 0.0001))
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
    df['is_bullish'] = (df['Close'] > df['Open']).astype(int)
    df['candle_strength'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.0001)
    df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
    df['gap_size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    return df

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add statistical measures for distribution and anomaly detection"""
    for period in [10, 20, 50]:
        df[f'zscore_{period}'] = ((df['Close'] - df['Close'].rolling(period).mean()) /
                                   (df['Close'].rolling(period).std() + 0.0001))
    df['skew_20'] = df['return_1'].rolling(20).skew()
    df['kurt_20'] = df['return_1'].rolling(20).kurt()
    df['percentile_rank_20'] = df['Close'].rolling(20).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5
    )
    return df

def add_advanced_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sophisticated volatility and market microstructure features"""
    new_features = pd.DataFrame(index=df.index)
    new_features['realized_var_20'] = (df['return_1'] ** 2).rolling(20).sum()
    new_features['bipower_var'] = (abs(df['return_1']) * abs(df['return_1'].shift())).rolling(20).sum()
    new_features['jump_strength'] = new_features['realized_var_20'] - new_features['bipower_var']
    new_features['vol_cluster'] = df['volatility_10'].rolling(20).std()
    new_features['vol_regime'] = (df['volatility_10'] > df['volatility_10'].rolling(50).mean()).astype(int)
    new_features['range_compression'] = ((df['High'] - df['Low']).rolling(10).mean() /
                                (df['High'] - df['Low']).rolling(50).mean())
    new_features['range_velocity'] = (df['High'] - df['Low']).pct_change()
    new_features['fractal_proxy'] = df['ATR_pct'] / (df['volatility_10'] + 0.0001)
    new_features['vol_reversion_speed'] = (df['volatility_10'] - df['volatility_10'].shift(10)) / 10
    return pd.concat([df, new_features], axis=1)

def add_advanced_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sophisticated trend quality and momentum features"""
    new_features = pd.DataFrame(index=df.index)
    new_features['efficiency_ratio'] = (abs(df['Close'] - df['Close'].shift(10)) /
                               (df['High'].rolling(10).max() - df['Low'].rolling(10).min() + 0.0001))
    new_features['trend_persistence'] = np.sign(df['return_1']).rolling(10).sum()
    new_features['trend_smoothness'] = (abs(df['Close'] - df['Close'].shift(20)) /
                               (df['return_1'].rolling(20).std() + 0.0001))
    new_features['path_curvature'] = df['return_1'].diff().abs().rolling(10).mean()
    new_features['trend_strength'] = abs(df['Close'] - df['supertrend']) / df['Close']
    new_features['trend_acceleration'] = new_features['trend_strength'].diff()
    new_features['dir_entropy'] = df['return_1'].rolling(20).apply(
        lambda x: -np.mean(np.sign(x) * np.log(np.abs(np.sign(x)) + 1e-6))
    )
    return pd.concat([df, new_features], axis=1)

def add_information_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add entropy and information-based features for pattern recognition"""
    new_features = pd.DataFrame(index=df.index)
    new_features['price_entropy'] = df['return_1'].rolling(20).apply(
        lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1) if len(x) > 0 else 0,
        raw=False
    )
    new_features['surprise'] = ((df['return_1'] - df['return_1'].rolling(20).mean()) /
                      (df['return_1'].rolling(20).std() + 1e-6))
    new_features['shock_elasticity'] = df['return_1'].abs() / (df['volatility_10'] + 1e-6)
    return pd.concat([df, new_features], axis=1)

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features related to market structure and liquidity"""
    new_features = pd.DataFrame(index=df.index)
    new_features['buy_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
    new_features['slippage_proxy'] = (df['High'] - df['Low']) / df['Close'].rolling(10).mean()
    new_features['stop_hunt_proxy'] = (df['High'] - df['Low']) / (df['ATR_14'] + 0.0001)
    new_features['amihud_illiquidity'] = abs(df['return_1']) / (df['Volume'] + 1)
    return pd.concat([df, new_features], axis=1)

def add_risk_reward_features(df: pd.DataFrame, lookahead: int = 192) -> pd.DataFrame:
    """
    Add forward-looking risk/reward metrics for target creation.
    FIXED: BUG #3 - Removed shift(-1) to prevent data leakage.
    Now uses high-performance numpy-based lookahead slices.
    """
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    future_max_high = np.full(n, np.nan)
    future_min_low = np.full(n, np.nan)
    
    # Accurate lookahead window without data leakage
    for i in range(n):
        # We look from i+1 up to i+1+lookahead
        # This ensures current bar 'i' doesn't know about bars it shouldn't
        end_idx = min(i + 1 + lookahead, n)
        if i + 1 < end_idx:
            future_max_high[i] = np.max(highs[i+1 : end_idx])
            future_min_low[i] = np.min(lows[i+1 : end_idx])
            
    new_features = pd.DataFrame(index=df.index)
    new_features['upside_pct'] = ((future_max_high - closes) / closes) * 100
    new_features['downside_pct'] = ((future_min_low - closes) / closes) * 100
    new_features['future_drawdown_pct'] = ((future_min_low - future_max_high) / (future_max_high + 1e-6)) * 100
    
    # Calculate ratios (used for report, but will be excluded from training inputs)
    new_features['reward_risk_ratio'] = new_features['upside_pct'] / (abs(new_features['downside_pct']) + 1e-6)
    new_features['edge_ratio'] = new_features['upside_pct'] / (abs(new_features['downside_pct']) + 1e-6)
    new_features['pain_ratio'] = df['return_10'] / (abs(new_features['future_drawdown_pct']) + 1e-6)
    
    return pd.concat([df, new_features], axis=1)

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features combining multiple indicators"""
    new_features = pd.DataFrame(index=df.index)
    new_features['rsi_vol'] = df['RSI_14'] * df['volatility_10']
    new_features['rsi_atr'] = df['RSI_14'] * df['ATR_pct']
    new_features['trend_volume'] = df['trend_strength'] * (df['volume_ratio'] if 'volume_ratio' in df else 1.0)
    new_features['adx_volume'] = df['ADX_14'] * (df['volume_ratio'] if 'volume_ratio' in df else 1.0)
    new_features['bb_rsi'] = df['BB_position'] * df['RSI_14']
    new_features['vol_atr_ratio'] = (df['volume_ratio'] if 'volume_ratio' in df else 1.0) / (df['ATR_pct'] + 0.0001)
    return pd.concat([df, new_features], axis=1)


def add_pattern_setup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Past-only setup features for scalp pattern learning.

    These columns describe *where a trader might look*, while oracle labels later
    decide whether that setup actually paid after costs. This keeps the model from
    trying to predict every noisy candle as a trade.
    """
    new_features = pd.DataFrame(index=df.index)

    volume_ratio = df.get("volume_ratio", pd.Series(1.0, index=df.index)).fillna(1.0)
    atr_pct = df.get("ATR_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
    bb_width = df.get("BB_width", pd.Series(0.0, index=df.index)).fillna(0.0)
    bb_width_ma = bb_width.rolling(50).mean()
    close_position = df.get("close_position", pd.Series(0.5, index=df.index)).fillna(0.5)
    bb_position = df.get("BB_position", pd.Series(0.5, index=df.index)).fillna(0.5)
    rsi = df.get("RSI_14", pd.Series(50.0, index=df.index)).fillna(50.0)
    macd_hist_pct = df.get("MACD_hist_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
    trend_strength = df.get("trend_strength", pd.Series(0.0, index=df.index)).fillna(0.0)
    adx = df.get("ADX_14", pd.Series(0.0, index=df.index)).fillna(0.0)
    directional_bias = df.get("directional_bias", pd.Series(0.0, index=df.index)).fillna(0.0)
    return_5 = df.get("return_5", pd.Series(0.0, index=df.index)).fillna(0.0)
    return_10 = df.get("return_10", pd.Series(0.0, index=df.index)).fillna(0.0)
    zscore_20 = df.get("zscore_20", pd.Series(0.0, index=df.index)).fillna(0.0)
    wick_imbalance = df.get("wick_imbalance", pd.Series(0.0, index=df.index)).fillna(0.0)
    candle_strength = df.get("candle_strength", pd.Series(0.0, index=df.index)).fillna(0.0)
    chop_index = df.get("chop_index", pd.Series(50.0, index=df.index)).fillna(50.0)
    supertrend_direction = df.get("supertrend_direction", pd.Series(0.0, index=df.index)).fillna(0.0)
    price_to_ema_20 = df.get("price_to_ema_20", pd.Series(0.0, index=df.index)).fillna(0.0)

    trend_long = (
        (supertrend_direction > 0)
        & (price_to_ema_20 > -0.002)
        & (directional_bias > 0)
        & (adx > 14)
    )
    trend_short = (
        (supertrend_direction < 0)
        & (price_to_ema_20 < 0.002)
        & (directional_bias < 0)
        & (adx > 14)
    )

    pullback_long = trend_long & (rsi.between(38, 58)) & (return_5 < 0.002) & (close_position > 0.35)
    pullback_short = trend_short & (rsi.between(42, 62)) & (return_5 > -0.002) & (close_position < 0.65)

    compression = (bb_width < bb_width_ma) & (chop_index < 65) & (atr_pct > 0.08)
    breakout_long = compression & (close_position > 0.68) & (return_5 > 0) & (volume_ratio > 1.05)
    breakout_short = compression & (close_position < 0.32) & (return_5 < 0) & (volume_ratio > 1.05)

    reversal_long = (
        (zscore_20 < -1.0)
        & (rsi < 42)
        & (wick_imbalance < 0)
        & (close_position > 0.45)
        & (candle_strength > 0.25)
    )
    reversal_short = (
        (zscore_20 > 1.0)
        & (rsi > 58)
        & (wick_imbalance > 0)
        & (close_position < 0.55)
        & (candle_strength > 0.25)
    )

    trend_quality = np.clip((adx - 12.0) / 25.0, 0.0, 1.0)
    volume_quality = np.clip((volume_ratio - 0.8) / 1.2, 0.0, 1.0)
    volatility_quality = np.clip(atr_pct / 0.8, 0.0, 1.0)
    momentum_long = np.clip((macd_hist_pct * 1000.0) + (return_10 * 8.0), -1.0, 1.0)
    momentum_short = np.clip((-macd_hist_pct * 1000.0) + (-return_10 * 8.0), -1.0, 1.0)

    long_setup_score = (
        pullback_long.astype(float) * 0.35
        + breakout_long.astype(float) * 0.35
        + reversal_long.astype(float) * 0.30
        + trend_quality * 0.10
        + volume_quality * 0.08
        + volatility_quality * 0.07
        + np.clip(momentum_long, 0.0, 1.0) * 0.10
    )
    short_setup_score = (
        pullback_short.astype(float) * 0.35
        + breakout_short.astype(float) * 0.35
        + reversal_short.astype(float) * 0.30
        + trend_quality * 0.10
        + volume_quality * 0.08
        + volatility_quality * 0.07
        + np.clip(momentum_short, 0.0, 1.0) * 0.10
    )

    new_features["trend_pullback_long_setup"] = pullback_long.astype(int)
    new_features["trend_pullback_short_setup"] = pullback_short.astype(int)
    new_features["breakout_long_setup"] = breakout_long.astype(int)
    new_features["breakout_short_setup"] = breakout_short.astype(int)
    new_features["reversal_long_setup"] = reversal_long.astype(int)
    new_features["reversal_short_setup"] = reversal_short.astype(int)
    new_features["long_setup_score"] = np.clip(long_setup_score, 0.0, 1.0)
    new_features["short_setup_score"] = np.clip(short_setup_score, 0.0, 1.0)
    new_features["setup_score_spread"] = new_features["long_setup_score"] - new_features["short_setup_score"]
    new_features["compression_breakout_score"] = compression.astype(float) * volume_quality

    return pd.concat([df, new_features], axis=1)

def add_oracle_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build path-consistent oracle labels from future excursion estimates.

    The forward-looking columns remain excluded from model inputs; they are only used
    to create labels that match the same first-hit logic used in the backtest.
    """
    if "upside_pct" not in df.columns or "downside_pct" not in df.columns:
        raise ValueError(
            "Oracle labeling requires 'upside_pct' and 'downside_pct'. "
            "Run add_risk_reward_features() first."
        )

    strategy = config.strategy
    row_count = len(df)
    lookahead = config.features.LOOKAHEAD_BARS

    upside_pct = df["upside_pct"].values.astype(np.float64)
    downside_pct = df["downside_pct"].values.astype(np.float64)
    abs_down_pct = np.abs(downside_pct)
    close_prices = df["Close"].values.astype(np.float64)
    high_prices = df["High"].values.astype(np.float64)
    low_prices = df["Low"].values.astype(np.float64)
    long_setup_score = df.get("long_setup_score", pd.Series(1.0, index=df.index)).values.astype(np.float64)
    short_setup_score = df.get("short_setup_score", pd.Series(1.0, index=df.index)).values.astype(np.float64)
    estimated_trade_cost_pct = (
        strategy.ROUND_TRIP_FEE_PCT
        + (2.0 * strategy.SLIPPAGE_PCT)
        + strategy.COST_BUFFER_PCT
    )

    # ------------------------------------------------------------------
    # LONG oracle levels
    # ------------------------------------------------------------------
    oracle_tp_long = np.clip(
        upside_pct * strategy.ORACLE_TP_CAPTURE_RATIO,
        strategy.ORACLE_MIN_TP_PCT,
        strategy.ORACLE_MAX_TP_PCT,
    )
    oracle_sl_long = np.clip(
        abs_down_pct * strategy.ORACLE_SL_CAPTURE_RATIO,
        strategy.ORACLE_MIN_SL_PCT,
        strategy.ORACLE_MAX_SL_PCT,
    )
    oracle_rr_long = oracle_tp_long / (oracle_sl_long + 1e-6)

    # ------------------------------------------------------------------
    # SHORT oracle levels (symmetric: profit from falling, risk from rising)
    # ------------------------------------------------------------------
    oracle_tp_short = np.clip(
        abs_down_pct * strategy.ORACLE_TP_CAPTURE_RATIO,
        strategy.ORACLE_MIN_TP_PCT,
        strategy.ORACLE_MAX_TP_PCT,
    )
    oracle_sl_short = np.clip(
        upside_pct * strategy.ORACLE_SL_CAPTURE_RATIO,
        strategy.ORACLE_MIN_SL_PCT,
        strategy.ORACLE_MAX_SL_PCT,
    )
    oracle_rr_short = oracle_tp_short / (oracle_sl_short + 1e-6)

    labels = np.ones(row_count, dtype=np.float64)
    label_tp_pct = np.full(row_count, strategy.ORACLE_MIN_TP_PCT, dtype=np.float64)
    label_sl_pct = np.full(row_count, strategy.ORACLE_MIN_SL_PCT, dtype=np.float64)
    actual_pnl_pct_arr = np.zeros(row_count, dtype=np.float64)
    winning_rr = np.ones(row_count, dtype=np.float64)
    best_capacity = np.zeros(row_count, dtype=np.float64)

    for row_index in range(row_count - lookahead):
        long_outcome, long_pnl = _simulate_trade_path(
            close_prices,
            high_prices,
            low_prices,
            row_index,
            lookahead,
            float(oracle_tp_long[row_index]),
            float(oracle_sl_long[row_index]),
            "long",
        )
        short_outcome, short_pnl = _simulate_trade_path(
            close_prices,
            high_prices,
            low_prices,
            row_index,
            lookahead,
            float(oracle_tp_short[row_index]),
            float(oracle_sl_short[row_index]),
            "short",
        )

        long_eligible = (
            upside_pct[row_index] >= strategy.ORACLE_MIN_UPSIDE_PCT
            and oracle_rr_long[row_index] >= strategy.ORACLE_MIN_RR
            and oracle_tp_long[row_index] >= estimated_trade_cost_pct
            and long_outcome == "SUCCESS"
            and (
                not strategy.REQUIRE_PATTERN_SETUP_FOR_ORACLE
                or long_setup_score[row_index] >= strategy.MIN_ORACLE_SETUP_SCORE
            )
        )
        short_eligible = (
            abs_down_pct[row_index] >= strategy.ORACLE_MIN_DOWNSIDE_PCT
            and oracle_rr_short[row_index] >= strategy.ORACLE_MIN_RR
            and oracle_tp_short[row_index] >= estimated_trade_cost_pct
            and short_outcome == "SUCCESS"
            and (
                not strategy.REQUIRE_PATTERN_SETUP_FOR_ORACLE
                or short_setup_score[row_index] >= strategy.MIN_ORACLE_SETUP_SCORE
            )
        )

        if long_eligible and (
            not short_eligible
            or long_pnl > short_pnl
            or (np.isclose(long_pnl, short_pnl) and oracle_rr_long[row_index] >= oracle_rr_short[row_index])
        ):
            labels[row_index] = 0.0
            label_tp_pct[row_index] = oracle_tp_long[row_index]
            label_sl_pct[row_index] = oracle_sl_long[row_index]
            actual_pnl_pct_arr[row_index] = long_pnl - estimated_trade_cost_pct
            winning_rr[row_index] = oracle_rr_long[row_index]
            best_capacity[row_index] = upside_pct[row_index]
        elif short_eligible:
            labels[row_index] = 2.0
            label_tp_pct[row_index] = oracle_tp_short[row_index]
            label_sl_pct[row_index] = oracle_sl_short[row_index]
            actual_pnl_pct_arr[row_index] = short_pnl - estimated_trade_cost_pct
            winning_rr[row_index] = oracle_rr_short[row_index]
            best_capacity[row_index] = abs_down_pct[row_index]

    # ------------------------------------------------------------------
    # Capacity score — how much "room" exists in the winning direction
    # Normalized 0-1; model uses this to learn position sizing
    # ------------------------------------------------------------------
    oracle_capacity_score = np.clip(best_capacity / strategy.ORACLE_MAX_TP_PCT, 0.0, 1.0)

    # ------------------------------------------------------------------
    # qty_ratio — composite sizing signal for the sizing_head target
    # Higher R:R + bigger capacity = larger sizing confidence
    # ------------------------------------------------------------------
    rr_bonus       = np.clip((winning_rr - strategy.ORACLE_MIN_RR) * 0.10, 0.0, 0.30)
    capacity_bonus = np.clip(oracle_capacity_score * 0.20, 0.0, 0.20)
    qty_ratio_arr  = np.clip(0.50 + rr_bonus + capacity_bonus, 0.10, 1.0)
    qty_ratio_arr[labels == 1] = 0.30   # Neutral signals: minimal sizing

    # ------------------------------------------------------------------
    # Write outputs
    # oracle_* columns are label-only; excluded from model inputs (get_feature_cols)
    # label_* columns feed build_target_arrays() for training targets
    # ------------------------------------------------------------------
    df["oracle_tp_pct"]        = label_tp_pct
    df["oracle_sl_pct"]        = label_sl_pct
    df["oracle_rr_ratio"]      = winning_rr
    df["oracle_capacity_score"] = oracle_capacity_score
    df["label_take_profit_pct"] = label_tp_pct
    df["label_stop_loss_pct"]   = label_sl_pct
    df["direction_label"]       = labels
    df["actual_pnl_pct"]        = actual_pnl_pct_arr
    df["label_qty_ratio"]       = qty_ratio_arr
    return df


def add_target_labels(df: pd.DataFrame, lookahead: int = 192) -> pd.DataFrame:
    """
    First-hit direction labels plus per-bar TP/SL percentages used for those paths.
    Also calculates 'actual_pnl_pct' and a synthetic 'label_qty_ratio' for the FRD RL architecture.
    """
    strategy = config.strategy
    row_count = len(df)
    close_prices = df["Close"].values
    high_prices = df["High"].values
    low_prices = df["Low"].values

    if strategy.USE_DYNAMIC_TP_SL_LABELS and "ATR_pct" in df.columns:
        atr_pct = df["ATR_pct"].values.astype(np.float64)
        take_profit_pct_per_row = np.clip(
            strategy.TP_ATR_MULTIPLIER * atr_pct,
            strategy.LABEL_TP_PCT_MIN,
            strategy.LABEL_TP_PCT_MAX,
        )
        stop_loss_pct_per_row = np.clip(
            strategy.SL_ATR_MULTIPLIER * atr_pct,
            strategy.LABEL_SL_PCT_MIN,
            strategy.LABEL_SL_PCT_MAX,
        )
    else:
        take_profit_pct_per_row = np.full(row_count, strategy.TARGET_PROFIT_PCT, dtype=np.float64)
        stop_loss_pct_per_row = np.full(row_count, strategy.STOP_LOSS_PCT, dtype=np.float64)

    labels = np.ones(row_count)
    actual_pnl_pct_arr = np.zeros(row_count, dtype=np.float64)

    for row_index in range(row_count - lookahead):
        take_profit_pct = float(take_profit_pct_per_row[row_index])
        stop_loss_pct = float(stop_loss_pct_per_row[row_index])

        long_outcome, long_pnl = _simulate_trade_path(
            close_prices,
            high_prices,
            low_prices,
            row_index,
            lookahead,
            take_profit_pct,
            stop_loss_pct,
            "long",
        )
        short_outcome, short_pnl = _simulate_trade_path(
            close_prices,
            high_prices,
            low_prices,
            row_index,
            lookahead,
            take_profit_pct,
            stop_loss_pct,
            "short",
        )

        if long_outcome == "SUCCESS" and (
            short_outcome != "SUCCESS" or long_pnl >= short_pnl
        ):
            labels[row_index] = 0
            actual_pnl_pct_arr[row_index] = long_pnl
        elif short_outcome == "SUCCESS":
            labels[row_index] = 2
            actual_pnl_pct_arr[row_index] = short_pnl
        else:
            actual_pnl_pct_arr[row_index] = 0.0

    df["label_take_profit_pct"] = take_profit_pct_per_row
    df["label_stop_loss_pct"] = stop_loss_pct_per_row
    df["direction_label"] = labels
    df["actual_pnl_pct"] = actual_pnl_pct_arr
    
    # Synthetic qty ratio: 1.0 for confident targets, 0.5 for others
    qty_ratio_arr = np.full(row_count, 0.5)
    qty_ratio_arr[labels != 1] = 0.8 # Higher base quantity for clear setups
    if 'trend_strength' in df.columns:
        trend_proxy = df['trend_strength'].fillna(0).values
        norm_trend = np.clip(trend_proxy * 5, 0, 0.2)
        qty_ratio_arr = np.clip(qty_ratio_arr + norm_trend, 0.1, 1.0)
    df["label_qty_ratio"] = qty_ratio_arr
    
    return df

def create_full_feature_set(df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
    """Complete feature engineering pipeline with progress tracking"""
    
    steps = [
        ("Basic Attributes", add_basic_features),
        ("Moving Averages", add_moving_averages),
        ("Momentum Indicators", add_momentum_indicators),
        ("Volatility Measures", add_volatility_indicators),
        ("Trend Strength", add_trend_indicators),
        ("Volume Analysis", add_volume_indicators),
        ("Candle Patterns", add_candle_features),
        ("Statistical Features", add_statistical_features),
        ("Adv. Volatility", add_advanced_volatility_features),
        ("Adv. Trend", add_advanced_trend_features),
        ("Entropy/Info Theory", add_information_theory_features),
        ("Microstructure", add_microstructure_features),
        ("Market Regime (Chop)", add_regime_features),
        ("Risk-Reward Profile", lambda d: add_risk_reward_features(d, lookahead)),
        ("Interactions", add_interaction_features),
        ("Pattern Setups", add_pattern_setup_features),
        # Route to Oracle labeler if enabled; falls back to ATR-based labeler
        ("Oracle Target Labels" if config.strategy.USE_ORACLE_LABELS else "ATR Target Labels",
         add_oracle_target_labels if config.strategy.USE_ORACLE_LABELS else add_target_labels),
    ]
    
    with get_progress() as progress:
        task = progress.add_task("[highlight]Feature Engineering Pipeline...[/highlight]", total=len(steps))
        
        for name, func in steps:
            progress.update(task, description=f"Adding {name}...")
            df = func(df)
            progress.update(task, advance=1)

    # Clean up without backward-filling future values into earlier rows.
    df["feature_cache_version"] = FEATURE_CACHE_VERSION
    df = df.ffill().fillna(0)
    console.print(f"[success]✅ Feature engineering complete![/success] Total features: [bold]{len(df.columns)}[/bold]")
    return df
