# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch
import os

@dataclass
class DataConfig:
    """Settings for data fetching and caching"""
    SYMBOLS: list[str] = ("BTCUSD", "ETHUSD")  # List of symbols to train on
    INTERVAL: str = "1h"                        # Timeframe bars ki length
    TOTAL_DAYS: int = 365                        # 1 full year of history for robust training
    CACHE_VALID_MINS: int = 120                  # Feature cache validity (2 hours)

@dataclass
class FeatureConfig:
    """Settings for feature engineering and pruning"""
    LOOKAHEAD_BARS: int = 24                     # 6 hours (24 bars * 15m). Sweet spot between 16 and 96.
    PRUNING_THRESHOLD: float = 0.03              # Spearman correlation pruning floor
    # Auto-pruning thresholds (applied after scaling in training_utils)
    VARIANCE_FLOOR: float = 0.01                 # Drop features with var < 0.01 after StandardScaler
    CORRELATION_CEILING: float = 0.95            # Drop one of each pair with |corr| > 0.95

@dataclass
class ModelConfig:
    """Transformer Architecture Hyperparameters"""
    HIDDEN_DIM: int = 48                         # Sweet spot: 32 was underfitting, 64 was overfitting
    NUM_HEADS: int = 4                           # Attention heads (48 / 4 = 12 dim per head)
    NUM_LAYERS: int = 2                          # 2 layers for pattern depth
    DROPOUT: float = 0.3                         # Aggressive regularization (was 0.2)
    SEQ_LEN: int = 48                            # 12 hours lookback (48 bars * 15m)

@dataclass
class TrainingConfig:
    """Settings for the AI training process"""
    # Chronological holdouts: train -> val -> test (oldest -> newest)
    VAL_DAYS: int = 15                           # Increased from 4 — more reliable validation signal
    TEST_DAYS: int = 30                          # 1 month test window
    EPOCHS: int = 30                             # Increased epochs for better training depth
    RL_FINE_TUNE_EPOCHS: int = 15                # ENABLED: Phase 2 RL with frozen backbone
    EARLY_STOP_PATIENCE: int = 7                 # REDUCED: Stop faster when overfitting starts
    BATCH_SIZE: int = 512                        # M4 GPU efficient batch size
    LEARNING_RATE: float = 0.0001                # Phase 1 learning rate
    RL_LEARNING_RATE: float = 0.00002            # Phase 2 RL: slower to preserve Phase 1 knowledge
    WEIGHT_DECAY: float = 0.001                  # REDUCED from 0.01: was killing gradients
    USE_WEIGHTED_SAMPLER: bool = False            # Replaced by Focal Loss (no class weights needed)

    # Loss weights
    LOSS_ALPHA: float = 1.0                      # Signal (direction) loss weight — primary objective
    LOSS_BETA: float = 0.3                       # Sizing loss weight — secondary
    LOSS_GAMMA: float = 0.05                     # PnL penalty weight (Phase 2 only)

    # Focal Loss hyperparameters
    FOCAL_GAMMA: float = 2.0                     # Focus parameter — down-weights easy examples
    USE_FOCAL_LOSS: bool = True                  # Use Focal Loss instead of CrossEntropy

@dataclass
class StrategyConfig:
    """Settings for trading strategy and evaluation"""
    TARGET_PROFIT_PCT: float = 1.0              # Fixed TP% (fallback)
    STOP_LOSS_PCT: float = 1.0                  # Fixed SL% (fallback) - Enforcing 1:1 ratio
    USE_DYNAMIC_TP_SL_LABELS: bool = False      # Disabled when USE_ORACLE_LABELS=True
    TP_ATR_MULTIPLIER: float = 1.25
    SL_ATR_MULTIPLIER: float = 0.65
    LABEL_TP_PCT_MIN: float = 0.5              # Minimum 0.5% TP for 6-hour window
    LABEL_TP_PCT_MAX: float = 10.0
    LABEL_SL_PCT_MIN: float = 0.8              # Minimum 0.8% SL — realistic for 6h crypto noise
    LABEL_SL_PCT_MAX: float = 5.0

    # -----------------------------------------------------------------------
    # Risk Management & Execution Rules
    # -----------------------------------------------------------------------
    AI_CONFIDENCE_THRESHOLD: float = 0.50       # Higher threshold to filter out low-conviction noise
    MIN_REWARD_RISK_RATIO: float = 1.0           # Enforce minimum 1:1 R:R before taking a trade
    SLIPPAGE_PCT: float = 0.05                   # 5bps per leg (realistic for crypto)
    PARALLEL_SLOTS: int = 1                      # One trade at a time
    KELLY_FRACTION: float = 0.5                  # Half-Kelly for conservative capital growth

    # -----------------------------------------------------------------------
    # Oracle Labeling Engine
    # Uses ACTUAL future price data to create high-quality training labels.
    # -----------------------------------------------------------------------
    USE_ORACLE_LABELS: bool = True
    ORACLE_TP_CAPTURE_RATIO: float = 0.70       # 70% capture as requested
    ORACLE_SL_CAPTURE_RATIO: float = 0.50       # 50% adverse excursion
    ORACLE_MIN_TP_PCT: float = 0.8              # Increased floor to 0.8% to avoid market noise
    ORACLE_MAX_TP_PCT: float = 5.0
    ORACLE_MIN_SL_PCT: float = 0.8              # Minimum 0.8% stop to survive normal volatility
    ORACLE_MAX_SL_PCT: float = 5.0
    ORACLE_MIN_RR: float = 1.0                  # Keep 1:1 RR
    ORACLE_MIN_UPSIDE_PCT: float = 0.8          # Require 0.8% move
    ORACLE_MIN_DOWNSIDE_PCT: float = 0.8        # Require 0.8% move

    INITIAL_CAPITAL_USD: float = 1000.0
    RISK_PER_TRADE_PCT_OF_EQUITY: float = 1.0   # Risk 1% equity per trade
    MAX_POSITION_NOTIONAL_PCT_OF_EQUITY: float = 0.95
    ROUND_TRIP_FEE_PCT: float = 0.1

@dataclass
class TradingConfig:
    """Global configuration object — single source of truth for all hyperparameters"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # Auto-detect best compute device
    @property
    def DEVICE(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

# Singleton instance — import this everywhere
config = TradingConfig()
