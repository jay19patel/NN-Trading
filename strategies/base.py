# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame with OHLC data and returns the same DataFrame 
        with signal columns: ai_verdict, ai_take_profit_pct, ai_stop_loss_pct, etc.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the strategy."""
        pass
