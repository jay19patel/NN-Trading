# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats

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

def add_risk_reward_features(df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
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

def add_target_labels(df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
    """
    Add classification labels using 'First-Hit' logic.
    Label 0 (BUY): Hits Target BEFORE Stop Loss
    Label 2 (SELL): Hits -Target BEFORE +Stop Loss
    Label 1 (NEUTRAL): No target hit or Stop Loss hit first
    """
    target = 1.0  # 1.0% Profit Target
    stop = 1.0    # 1.0% Stop Loss
    
    df['direction_label'] = 1  # Default to Neutral
    
    # We need to look ahead for each row
    close_prices = df['Close'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    
    labels = np.ones(len(df))
    
    for i in range(len(df) - lookahead):
        entry_price = close_prices[i]
        tp_price = entry_price * (1 + target / 100)
        sl_price = entry_price * (1 - stop / 100)
        
        # Bullish Check
        for j in range(1, lookahead + 1):
            if low_prices[i + j] <= sl_price:
                break # Hit SL first, neutral
            if high_prices[i + j] >= tp_price:
                labels[i] = 0 # Hit TP first
                break
                
        # Bearish Check (only if not already labelled as Buy)
        if labels[i] == 1:
            tp_sell = entry_price * (1 - target / 100)
            sl_sell = entry_price * (1 + stop / 100)
            for j in range(1, lookahead + 1):
                if high_prices[i + j] >= sl_sell:
                    break # Hit SL first
                if low_prices[i + j] <= tp_sell:
                    labels[i] = 2 # Hit TP first
                    break
                    
    df['direction_label'] = labels
    return df

def create_full_feature_set(df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
    """Complete feature engineering pipeline"""
    print("Adding basic features...")
    df = add_basic_features(df)
    print("Adding moving averages...")
    df = add_moving_averages(df)
    print("Adding momentum indicators...")
    df = add_momentum_indicators(df)
    print("Adding volatility indicators...")
    df = add_volatility_indicators(df)
    print("Adding trend indicators...")
    df = add_trend_indicators(df)
    print("Adding volume indicators...")
    df = add_volume_indicators(df)
    print("Adding candle features...")
    df = add_candle_features(df)
    print("Adding statistical features...")
    df = add_statistical_features(df)
    print("⭐ Adding advanced volatility features...")
    df = add_advanced_volatility_features(df)
    print("⭐ Adding advanced trend features...")
    df = add_advanced_trend_features(df)
    print("⭐ Adding information theory features...")
    df = add_information_theory_features(df)
    print("⭐ Adding microstructure features...")
    df = add_microstructure_features(df)
    print("⭐ Adding risk-reward features...")
    df = add_risk_reward_features(df, lookahead)
    print("Adding interaction features...")
    df = add_interaction_features(df)
    print("Adding target labels...")
    df = add_target_labels(df)
    
    # FIXED: BUG #7 - Reordered bfill/ffill. ffill MUST come first.
    df = df.ffill().bfill().fillna(0)
    print(f"\n✅ Feature engineering complete!")
    print(f"Total features: {len(df.columns)}")
    return df
