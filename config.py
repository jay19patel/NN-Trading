# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import os

@dataclass
class DataConfig:
    SYMBOLS: list[str] = ("ETHUSD",)
    INTERVAL: str = "1h"
    TOTAL_DAYS: int = 30
    CACHE_VALID_MINS: int = 0  # Rebuild every time for 100% accuracy

@dataclass
class FeatureConfig:
    LOOKAHEAD_BARS: int = 24 # 1 full day for 1h
    PURGE_BARS: int = 24
    FEATURE_CACHE_VERSION: int = 100

@dataclass
class StrategyConfig:
    INITIAL_CAPITAL_USD: float = 50.0
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
    ORACLE_MIN_RR: float = 2.5 # Higher quality signals
    ORACLE_MIN_UPSIDE_PCT: float = 1.2    # Strict upside
    ORACLE_MIN_DOWNSIDE_PCT: float = 1.0  # Strict downside
    
    # Execution Logic
    PARALLEL_SLOTS: int = 1
    MAX_DAILY_TRADES: int = 25
    COOLDOWN_BARS: int = 12
    MAX_CONSECUTIVE_LOSSES: int = 2
    DAILY_STOP_LOSS_PCT: float = 4.0
    BREAK_EVEN_AFTER_R: float = 1000.0   # Set very high to disable
    TRAIL_STOP_AFTER_R: float = 1000.0   # Set very high to disable
    TRAIL_STOP_R_MULTIPLE: float = 2.0
    
    # Required for the script to run
    USE_ORACLE_LABELS: bool = True
    TARGET_PROFIT_PCT: float = 1.0
    STOP_LOSS_PCT: float = 0.5
    MIN_REWARD_RISK_RATIO: float = 1.5   # Loosened from 2.0
    MIN_DIRECTIONAL_EDGE: float = 0.02   # Loosened from 0.05
    AI_CONFIDENCE_THRESHOLD: float = 0.35 # Loosened from 0.40
    LONG_CONFIDENCE_THRESHOLD: float = 0.35
    SHORT_CONFIDENCE_THRESHOLD: float = 0.35
    MIN_EXPECTED_RETURN_PCT: float = 0.01 # Loosened from 0.03
    USE_REGIME_FILTER: bool = False      # Disabled to let NN learn
    REGIME_MIN_ADX: float = 15.0
    LONG_RSI_MIN: float = 40.0           # Loosened
    LONG_RSI_MAX: float = 80.0           # Loosened
    SHORT_RSI_MIN: float = 20.0          # Loosened
    SHORT_RSI_MAX: float = 60.0          # Loosened
    LONG_MIN_TREND_SMA_20_50: float = -0.01
    SHORT_MAX_TREND_SMA_20_50: float = 0.01
    TP_GRID_PCT: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
    SL_GRID_PCT: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    ATR_LENGTH: int = 14
    TP_ATR_MULTIPLIERS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    SL_ATR_MULTIPLIERS: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)
    MIN_ATR_STOP_PCT: float = 0.25
    MAX_ATR_STOP_PCT: float = 1.50
    MIN_ATR_TARGET_PCT: float = 0.80 # Increased from 0.50 to cover fees better
    MAX_ATR_TARGET_PCT: float = 5.00

@dataclass
class ModelConfig:
    HIDDEN_DIM: int = 128
    NUM_HEADS: int = 4
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.2
    MAX_SEQ_LEN: int = 128
    WINDOW_SIZE: int = 48  # Number of past bars to look at

@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LR: float = 0.0005
    FOCAL_NEUTRAL_VIOLATION_SCALE: float = 2.0
    TRAINING_DATA_DAYS: int = 300
    VALIDATION_DATA_DAYS: int = 60
    TEST_DATA_DAYS: int = 30
    EARLY_STOP_PATIENCE: int = 7
    SHORT_OBJECTIVE: str = "three_class_long_neutral_short"
    WALK_FORWARD_FOLDS: int = 4
    WALK_FORWARD_VAL_DAYS: int = 30
    THRESHOLD_MIN_TRADES: int = 50
    
@dataclass
class TradingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @property
    def DEVICE(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

config = TradingConfig()

def bars_per_day(interval: str) -> int:
    inv = {"15m": 96, "1h": 24, "1d": 1}
    return inv.get(interval, 96)
