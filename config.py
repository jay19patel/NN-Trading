# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import os

@dataclass
class DataConfig:
    SYMBOLS: list[str] = ("BTCUSD", "ETHUSD")
    INTERVAL: str = "1m"
    TOTAL_DAYS: int = 30
    CACHE_VALID_MINS: int = 0  # Rebuild every time for 100% accuracy

@dataclass
class FeatureConfig:
    LOOKAHEAD_BARS: int = 60
    FEATURE_CACHE_VERSION: int = 100

@dataclass
class StrategyConfig:
    INITIAL_CAPITAL_USD: float = 1000.0
    RISK_PER_TRADE_PCT_OF_EQUITY: float = 2.0
    MAX_POSITION_NOTIONAL_PCT_OF_EQUITY: float = 0.95
    ROUND_TRIP_FEE_PCT: float = 0.05   # Realistic crypto fees
    SLIPPAGE_PCT: float = 0.02         # Realistic slippage
    
    # Oracle Filter Settings
    ORACLE_MIN_TP_PCT: float = 0.80   # Strong moves
    ORACLE_MIN_SL_PCT: float = 0.20
    ORACLE_MAX_TP_PCT: float = 5.0
    ORACLE_MAX_SL_PCT: float = 2.0
    ORACLE_TP_CAPTURE_RATIO: float = 0.80 # Buffer for slippage
    ORACLE_SL_CAPTURE_RATIO: float = 0.01 
    ORACLE_MIN_RR: float = 3.0
    ORACLE_MIN_UPSIDE_PCT: float = 1.0    # Strict upside
    ORACLE_MIN_DOWNSIDE_PCT: float = 1.0  # Strict downside
    
    # Execution Logic
    PARALLEL_SLOTS: int = 100
    MAX_DAILY_TRADES: int = 1000
    COOLDOWN_BARS: int = 0
    MAX_CONSECUTIVE_LOSSES: int = 10
    DAILY_STOP_LOSS_PCT: float = 10.0
    BREAK_EVEN_AFTER_R: float = 1000.0   # Set very high to disable
    TRAIL_STOP_AFTER_R: float = 1000.0   # Set very high to disable
    TRAIL_STOP_R_MULTIPLE: float = 2.0
    
    # Required for the script to run
    USE_ORACLE_LABELS: bool = True
    TARGET_PROFIT_PCT: float = 1.0
    STOP_LOSS_PCT: float = 0.5
    MIN_REWARD_RISK_RATIO: float = 1.0
    MIN_DIRECTIONAL_EDGE: float = 0.0
    AI_CONFIDENCE_THRESHOLD: float = 0.0

@dataclass
class TradingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    
    @property
    def DEVICE(self):
        return "cpu"

config = TradingConfig()

def bars_per_day(interval: str) -> int:
    inv = {"15m": 96, "1h": 24, "1d": 1}
    return inv.get(interval, 96)
