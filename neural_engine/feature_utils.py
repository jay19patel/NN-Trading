import pandas as pd
import numpy as np
import pandas_ta as ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to be used as features for the neural network.
    """
    df = df.copy()
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_ = df['Open']
    volume = df['Volume']

    # Returns
    df['return_1'] = df['Close'].pct_change()
    df['return_3'] = df['Close'].pct_change(3)
    df['return_5'] = df['Close'].pct_change(5)
    df['return_10'] = df['Close'].pct_change(10)
    df['return_20'] = df['Close'].pct_change(20)
    df['log_return_1'] = np.log(close / close.shift(1))
    df['log_return_3'] = np.log(close / close.shift(3))
    df['log_return_5'] = np.log(close / close.shift(5))
    df['log_return_10'] = np.log(close / close.shift(10))
    df['log_return_20'] = np.log(close / close.shift(20))
    
    # Volatility
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['natr'] = (df['atr'] / df['Close']) * 100
    df['realized_vol_10'] = df['log_return_1'].rolling(10).std()
    df['realized_vol_20'] = df['log_return_1'].rolling(20).std()
    df['volatility_regime_50'] = df['realized_vol_20'] / (df['realized_vol_20'].rolling(50).mean() + 1e-9)
    
    # RSI
    df['rsi'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_hist'] = macd.iloc[:, 1]
        df['macd_signal'] = macd.iloc[:, 2]
    else:
        df['macd'] = 0.0
        df['macd_hist'] = 0.0
        df['macd_signal'] = 0.0
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df['adx'] = adx.iloc[:, 0]
        df['dmp'] = adx.iloc[:, 1]
        df['dmn'] = adx.iloc[:, 2]
    else:
        df['adx'] = 0.0
        df['dmp'] = 0.0
        df['dmn'] = 0.0
    
    # Moving Averages
    df['sma_20'] = ta.sma(df['Close'], length=20)
    df['sma_50'] = ta.sma(df['Close'], length=50)
    df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
    df['dist_sma_50'] = (df['Close'] - df['sma_50']) / df['sma_50']
    df['trend_sma_20_50'] = (df['sma_20'] - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    # BBands
    bbands = ta.bbands(df['Close'], length=20)
    if bbands is not None:
        # Use first and last columns for upper and lower as naming can vary
        df['bb_upper'] = bbands.iloc[:, 2]
        df['bb_lower'] = bbands.iloc[:, 0]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['Close'] + 1e-9)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    else:
        df['bb_width'] = 0.0
        df['bb_position'] = 0.0

    # Candle shape and rolling structure
    candle_range = (high - low).replace(0, np.nan)
    df['body_pct_range'] = (close - open_).abs() / candle_range
    df['upper_wick_pct_range'] = (high - np.maximum(open_, close)) / candle_range
    df['lower_wick_pct_range'] = (np.minimum(open_, close) - low) / candle_range
    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    rolling_high_50 = high.rolling(50).max()
    rolling_low_50 = low.rolling(50).min()
    df['dist_high_20'] = (close - rolling_high_20) / (rolling_high_20 + 1e-9)
    df['dist_low_20'] = (close - rolling_low_20) / (rolling_low_20 + 1e-9)
    df['dist_high_50'] = (close - rolling_high_50) / (rolling_high_50 + 1e-9)
    df['dist_low_50'] = (close - rolling_low_50) / (rolling_low_50 + 1e-9)
    
    # Volume
    df['vol_sma_20'] = ta.sma(df['Volume'], length=20)
    df['vol_ratio'] = df['Volume'] / (df['vol_sma_20'] + 1e-9)
    df['vol_zscore_20'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    df['vol_trend_5_20'] = (volume.rolling(5).mean() - volume.rolling(20).mean()) / (volume.rolling(20).mean() + 1e-9)

    # Time/session features from the candle timestamp if available.
    idx = pd.to_datetime(df.index, errors='coerce')
    if not idx.isna().all():
        hour = idx.hour + idx.minute / 60.0
        day_of_week = idx.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    else:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 0.0
        df['dow_sin'] = 0.0
        df['dow_cos'] = 0.0
        
    # === ADVANCED SCALPING FEATURES ===
    
    # VWAP & Microstructure
    vwap = ta.vwap(high, low, close, volume)
    df['vwap'] = vwap if vwap is not None else close.rolling(20).mean()
    df['price_to_vwap'] = (close - df['vwap']) / (df['vwap'] + 1e-9)
    df['buy_pressure'] = (close - low) / (candle_range + 1e-9)
    df['wick_imbalance'] = df['upper_wick_pct_range'] - df['lower_wick_pct_range']
    
    # Advanced Volatility & Efficiency
    df['atr_ratio'] = df['atr'] / (df['atr'].rolling(100).mean() + 1e-9)  # High = Over-extended volatility
    df['efficiency_ratio'] = (close - close.shift(10)).abs() / (candle_range.rolling(10).sum() + 1e-9) # Trending vs Choppy
    df['range_compression'] = candle_range.rolling(10).mean() / (candle_range.rolling(50).mean() + 1e-9)
    df['fractal_proxy'] = df['natr'] / (df['realized_vol_10'] + 1e-9)
    
    # Advanced Trend & Momentum
    supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
    if supertrend is not None and "SUPERT_10_3" in supertrend.columns:
        df['supertrend'] = supertrend["SUPERT_10_3"]
    else:
        df['supertrend'] = close.rolling(10).mean()
    df['trend_strength'] = (close - df['supertrend']).abs() / (close + 1e-9)
    df['rsi_7'] = ta.rsi(close, length=7)
    
    # Information Theory
    df['surprise'] = (df['return_1'] - df['return_1'].rolling(20).mean()) / (df['return_1'].rolling(20).std() + 1e-9)
    df['shock_elasticity'] = df['return_1'].abs() / (df['realized_vol_10'] + 1e-9)
    
    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def get_feature_columns():
    return [
        'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
        'log_return_1', 'log_return_3', 'log_return_5', 'log_return_10', 'log_return_20',
        'natr', 'realized_vol_10', 'realized_vol_20', 'volatility_regime_50',
        'rsi', 'macd', 'macd_hist', 'macd_signal', 'adx', 'dmp', 'dmn',
        'dist_sma_20', 'dist_sma_50', 'trend_sma_20_50',
        'bb_width', 'bb_position',
        'body_pct_range', 'upper_wick_pct_range', 'lower_wick_pct_range',
        'dist_high_20', 'dist_low_20', 'dist_high_50', 'dist_low_50',
        'vol_ratio', 'vol_zscore_20', 'vol_trend_5_20',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'price_to_vwap', 'buy_pressure', 'wick_imbalance',
        'atr_ratio', 'efficiency_ratio', 'range_compression', 'fractal_proxy',
        'trend_strength', 'rsi_7',
        'surprise', 'shock_elasticity'
    ]
