# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats
from config import config
from ui_utils import console, get_progress

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
    for period in [5, 10, 20, 50, 100]:
        df[f'EMA_{period}'] = ta.ema(df['Close'], length=period).bfill()

    for period in [20, 50, 100, 200]:
        df[f'SMA_{period}'] = ta.sma(df['Close'], length=period).bfill()

    df['price_to_ema_5'] = (df['Close'] - df['EMA_5']) / df['EMA_5']
    df['price_to_ema_20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['price_to_sma_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    df['ema_5_10_cross'] = df['EMA_5'] - df['EMA_10']
    df['ema_10_20_cross'] = df['EMA_10'] - df['EMA_20']

    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume']).bfill()
    df['price_to_vwap'] = (df['Close'] - df['VWAP']) / df['VWAP']

    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum oscillators to identify overbought/oversold conditions"""
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period).bfill()

    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9'].bfill()
    df['MACD_signal'] = macd['MACDs_12_26_9'].bfill()
    df['MACD_hist'] = macd['MACDh_12_26_9'].bfill()

    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3'].bfill()
    df['Stoch_D'] = stoch['STOCHd_14_3_3'].bfill()

    df['CCI_20'] = ta.cci(df['High'], df['Low'], df['Close'], length=20).bfill()
    df['WilliamsR_14'] = ta.willr(df['High'], df['Low'], df['Close'], length=14).bfill()
    df['ROC_10'] = ta.roc(df['Close'], length=10).bfill()
    df['ROC_20'] = ta.roc(df['Close'], length=20).bfill()

    return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility measures to gauge market risk and movement potential"""
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period).bfill()

    df['ATR_pct'] = (df['ATR_14'] / df['Close']) * 100

    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
         df['BB_upper'] = bbands.iloc[:, 2].bfill()
         df['BB_middle'] = bbands.iloc[:, 1].bfill()
         df['BB_lower'] = bbands.iloc[:, 0].bfill()

    df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle']) * 100
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
    if kc is not None:
        df['KC_upper'] = kc.iloc[:, 2].bfill()
        df['KC_lower'] = kc.iloc[:, 0].bfill()

    df['volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['volatility_20'] = df['Close'].pct_change().rolling(20).std()
    df['volatility_50'] = df['Close'].pct_change().rolling(50).std()

    return df

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators that measure trend strength and direction"""
    for period in [14, 20]:
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=period)
        df[f'ADX_{period}'] = adx_df[f'ADX_{period}'].bfill()
        df[f'DMP_{period}'] = adx_df[f'DMP_{period}'].bfill()
        df[f'DMN_{period}'] = adx_df[f'DMN_{period}'].bfill()

    df['directional_bias'] = df['DMP_14'] - df['DMN_14']

    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    df['supertrend'] = supertrend["SUPERT_10_3"].bfill()
    df['supertrend_direction'] = supertrend["SUPERTd_10_3"].bfill()

    df['st_flip'] = df['supertrend_direction'].diff().abs()
    df['bars_since_flip'] = df.groupby((df['st_flip'] == 2).cumsum()).cumcount()

    aroon = ta.aroon(df['High'], df['Low'], length=25)
    df['aroon_up'] = aroon['AROONU_25'].bfill()
    df['aroon_down'] = aroon['AROOND_25'].bfill()
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
    df['OBV'] = ta.obv(df['Close'], df['Volume']).bfill()
    df['OBV_ema'] = ta.ema(df['OBV'], length=20).bfill()
    df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume']).bfill()
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20).bfill()
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).bfill()
    df['VPT'] = ta.pvt(df['Close'], df['Volume']).bfill()
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

def add_oracle_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oracle Labeling Engine — uses ACTUAL future price data to derive per-bar TP and SL.

    Strategy (as approved):
      - Look N bars ahead (done already in add_risk_reward_features)
      - upside_pct  = (future_max_high - close) / close * 100
      - downside_pct = (future_min_low  - close) / close * 100  [negative number]

      - LONG  TP = upside_pct    * ORACLE_TP_CAPTURE_RATIO  (e.g. 70% of max high)
      - LONG  SL = abs_down_pct  * ORACLE_SL_CAPTURE_RATIO  (e.g. 50% of min low)
      - SHORT TP = abs_down_pct  * ORACLE_TP_CAPTURE_RATIO
      - SHORT SL = upside_pct    * ORACLE_SL_CAPTURE_RATIO

      Direction:
        LONG  if upside > abs_downside AND long_rr  >= ORACLE_MIN_RR AND upside >= MIN_UPSIDE
        SHORT if abs_down > upside     AND short_rr >= ORACLE_MIN_RR AND abs_down >= MIN_DOWN
        NEUTRAL otherwise

    Requires: upside_pct, downside_pct columns (from add_risk_reward_features).
    NOTE: These columns are excluded from model inputs to prevent data leakage.
    """
    if "upside_pct" not in df.columns or "downside_pct" not in df.columns:
        raise ValueError(
            "Oracle labeling requires 'upside_pct' and 'downside_pct'. "
            "Run add_risk_reward_features() first."
        )

    strategy = config.strategy
    row_count = len(df)

    upside_pct   = df["upside_pct"].values.astype(np.float64)   # positive: max future high %
    downside_pct = df["downside_pct"].values.astype(np.float64)  # negative: min future low %
    abs_down_pct = np.abs(downside_pct)

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

    # ------------------------------------------------------------------
    # Direction labeling
    # LONG  wins when upside capacity is greater
    # SHORT wins when downside capacity is greater
    # Both need their R:R to be >= ORACLE_MIN_RR
    # ------------------------------------------------------------------
    long_mask = (
        (upside_pct >= strategy.ORACLE_MIN_UPSIDE_PCT)
        & (oracle_rr_long >= strategy.ORACLE_MIN_RR)
        & (upside_pct > abs_down_pct)
    )
    short_mask = (
        (abs_down_pct >= strategy.ORACLE_MIN_DOWNSIDE_PCT)
        & (oracle_rr_short >= strategy.ORACLE_MIN_RR)
        & (abs_down_pct > upside_pct)
        & ~long_mask  # Prevent double-labeling
    )

    labels = np.ones(row_count, dtype=np.float64)       # 1 = NEUTRAL (default)
    labels[long_mask]  = 0.0                            # 0 = LONG
    labels[short_mask] = 2.0                            # 2 = SHORT

    # Set per-row TP/SL from the winning direction
    label_tp_pct = np.where(labels == 0, oracle_tp_long,
                   np.where(labels == 2, oracle_tp_short,
                            strategy.ORACLE_MIN_TP_PCT))
    label_sl_pct = np.where(labels == 0, oracle_sl_long,
                   np.where(labels == 2, oracle_sl_short,
                            strategy.ORACLE_MIN_SL_PCT))

    # Actual PnL stored for the RL loss (positive = profitable outcome)
    actual_pnl_pct_arr = np.zeros(row_count, dtype=np.float64)
    actual_pnl_pct_arr[labels == 0] =  oracle_tp_long[labels == 0]
    actual_pnl_pct_arr[labels == 2] =  oracle_tp_short[labels == 2]

    # ------------------------------------------------------------------
    # Capacity score — how much "room" exists in the winning direction
    # Normalized 0-1; model uses this to learn position sizing
    # ------------------------------------------------------------------
    best_capacity = np.where(labels == 0, upside_pct,
                    np.where(labels == 2, abs_down_pct, 0.0))
    oracle_capacity_score = np.clip(best_capacity / strategy.ORACLE_MAX_TP_PCT, 0.0, 1.0)

    # ------------------------------------------------------------------
    # qty_ratio — composite sizing signal for the sizing_head target
    # Higher R:R + bigger capacity = larger sizing confidence
    # ------------------------------------------------------------------
    winning_rr = np.where(labels == 0, oracle_rr_long,
                 np.where(labels == 2, oracle_rr_short, 1.0))
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
        entry_price = close_prices[row_index]
        take_profit_pct = float(take_profit_pct_per_row[row_index])
        stop_loss_pct = float(stop_loss_pct_per_row[row_index])

        take_profit_price_long = entry_price * (1 + take_profit_pct / 100)
        stop_loss_price_long = entry_price * (1 - stop_loss_pct / 100)

        # Baseline: timeout PnL
        final_close = close_prices[row_index + lookahead]

        # Try long
        long_failed = False
        for step_index in range(1, lookahead + 1):
            bar_low = low_prices[row_index + step_index]
            bar_high = high_prices[row_index + step_index]
            hit_stop_first = bar_low <= stop_loss_price_long
            hit_target_first = bar_high >= take_profit_price_long
            if hit_stop_first and hit_target_first:
                long_failed = True
                break
            if hit_stop_first:
                long_failed = True
                break
            if hit_target_first:
                labels[row_index] = 0
                actual_pnl_pct_arr[row_index] = take_profit_pct
                break

        if labels[row_index] == 1:
            # Try short
            take_profit_price_short = entry_price * (1 - take_profit_pct / 100)
            stop_loss_price_short = entry_price * (1 + stop_loss_pct / 100)
            for step_index in range(1, lookahead + 1):
                bar_low = low_prices[row_index + step_index]
                bar_high = high_prices[row_index + step_index]
                hit_stop_first = bar_high >= stop_loss_price_short
                hit_target_first = bar_low <= take_profit_price_short
                if hit_stop_first and hit_target_first:
                    actual_pnl_pct_arr[row_index] = -stop_loss_pct
                    break
                if hit_stop_first:
                    actual_pnl_pct_arr[row_index] = -stop_loss_pct
                    break
                if hit_target_first:
                    labels[row_index] = 2
                    actual_pnl_pct_arr[row_index] = take_profit_pct
                    break
                    
            if labels[row_index] == 1:
                # Still Hold
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

    # Clean up NaNs
    df = df.ffill().bfill().fillna(0)
    console.print(f"[success]✅ Feature engineering complete![/success] Total features: [bold]{len(df.columns)}[/bold]")
    return df
