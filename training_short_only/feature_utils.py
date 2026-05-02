import pandas as pd
import numpy as np
import pandas_ta as ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to be used as features for the neural network.
    """
    df = df.copy()
    
    # Returns
    df['return_1'] = df['Close'].pct_change()
    df['return_5'] = df['Close'].pct_change(5)
    
    # Volatility
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['natr'] = (df['atr'] / df['Close']) * 100
    
    # RSI
    df['rsi'] = ta.rsi(df['Close'], length=14)
    
    # Moving Averages
    df['sma_20'] = ta.sma(df['Close'], length=20)
    df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
    
    # BBands
    bbands = ta.bbands(df['Close'], length=20)
    if bbands is not None:
        # Use first and last columns for upper and lower as naming can vary
        df['bb_upper'] = bbands.iloc[:, 2]
        df['bb_lower'] = bbands.iloc[:, 0]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['Close'] + 1e-9)
    
    # Volume
    df['vol_sma_20'] = ta.sma(df['Volume'], length=20)
    df['vol_ratio'] = df['Volume'] / (df['vol_sma_20'] + 1e-9)
    
    # Cleanup
    df = df.fillna(0)
    return df

def get_feature_columns():
    return [
        'return_1', 'return_5', 'natr', 'rsi', 
        'dist_sma_20', 'bb_width', 'vol_ratio'
    ]
