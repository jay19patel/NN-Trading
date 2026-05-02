# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy

class RandomStrategy(BaseStrategy):
    """A baseline strategy that generates random signals."""
    
    @property
    def name(self) -> str:
        return "Random"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        row_count = len(df)
        
        # 0: Long, 1: Neutral, 2: Short
        # Randomly choose between these
        df['ai_verdict'] = np.random.choice([0, 1, 2], size=row_count, p=[0.05, 0.90, 0.05])
        
        # Set some fixed/random TP and SL for these random trades
        df['ai_take_profit_pct'] = 0.5 + np.random.random(row_count) * 0.5 # 0.5% to 1.0%
        df['ai_stop_loss_pct'] = df['ai_take_profit_pct'] / 2.0 # Fixed 1:2 RR
        
        df['ai_qty_ratio'] = 1.0
        df['ai_confidence'] = 0.5
        df['ai_directional_edge'] = 0.0
        
        return df
