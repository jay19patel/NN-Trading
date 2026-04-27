# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch
import os

@dataclass
class DataConfig:
    """Settings for data fetching and caching"""
    SYMBOLS: list[str] = ("BTCUSD","ETHUSD") # List of symbols to train on
    INTERVAL: str = "15m"             # Timeframe bars ki length (e.g. 1m, 5m, 15m, 1h)
    TOTAL_DAYS: int = 300       # History length; must exceed VAL_DAYS + TEST_DAYS + warmup for stable training
    CACHE_VALID_MINS: int = 120       # Feature cache kitne minutes tak valid rahega (2 hours)

@dataclass
class FeatureConfig:
    """Settings for feature engineering and pruning"""
    LOOKAHEAD_BARS: int = 200         # Labeling ke liye futures bars (192 = 2 Days in 15m)
    PRUNING_THRESHOLD: float = 0.01   # Noise features ko hatane ke liye correlation threshold

@dataclass
class ModelConfig:
    """Transformer Architecture Hyperparameters"""
    HIDDEN_DIM: int = 64              # Model ki internal dimension complexity
    NUM_HEADS: int = 4                # Attention heads ki count
    NUM_LAYERS: int = 2               # Transformer blocks kitne layer honge
    DROPOUT: float = 0.2              # Overfitting se bachne ke liye dropout rate
    SEQ_LEN: int = 200                 # Sequence window size (Kitne pichle bars model dekhega)

@dataclass
class TrainingConfig:
    """Settings for the AI training process"""
    # Chronological holdouts: train -> val -> test (oldest -> newest). Critical for time series.
    VAL_DAYS: int = 4                 # Validation window before the test window (tune early stopping / LR)
    TEST_DAYS: int = 10               # Last kitne dino ka data final evaluation ke liye alag rakna he
    EPOCHS: int = 100                  # Kitni baar model pura data repeat karke sikhega
    RL_FINE_TUNE_EPOCHS: int = 20      # Extra epochs for phase 2 RL fine tuning
    EARLY_STOP_PATIENCE: int = 30      # Stop if validation loss does not improve for this many epochs
    BATCH_SIZE: int = 64              # Ek baar me kitne samples process honge
    LEARNING_RATE: float = 0.001      # Weights update karne ki speed
    RL_LEARNING_RATE: float = 0.0001  # Lower LR for RL phase
    WEIGHT_DECAY: float = 0.01        # Weight regularization taaki pattern stable rahe
    USE_WEIGHTED_SAMPLER: bool = True # Balance direction classes during training (can distort train metrics vs live distribution)
    
    # Loss weights
    LOSS_ALPHA: float = 1.0           # Signal loss weight
    LOSS_BETA: float = 0.5            # Sizing loss weight
    LOSS_GAMMA: float = 0.3           # PnL penalty weight (used heavily in phase 2)

@dataclass
class StrategyConfig:
    """Settings for trading strategy and evaluation"""
    TARGET_PROFIT_PCT: float = 3.0    # Fixed TP% when USE_DYNAMIC_TP_SL_LABELS is False
    STOP_LOSS_PCT: float = 1.0        # Fixed SL% when USE_DYNAMIC_TP_SL_LABELS is False
    AI_CONFIDENCE_THRESHOLD: float = 0.3

    # ATR-scaled dynamic TP/SL for labels, regression targets, and path simulation
    USE_DYNAMIC_TP_SL_LABELS: bool = True
    TP_ATR_MULTIPLIER: float = 1.25
    SL_ATR_MULTIPLIER: float = 0.65
    LABEL_TP_PCT_MIN: float = 0.25
    LABEL_TP_PCT_MAX: float = 12.0
    LABEL_SL_PCT_MIN: float = 0.15
    LABEL_SL_PCT_MAX: float = 6.0

    INITIAL_CAPITAL_USD: float = 1000.0
    RISK_PER_TRADE_PCT_OF_EQUITY: float = 1.0
    MAX_POSITION_NOTIONAL_PCT_OF_EQUITY: float = 0.95
    ROUND_TRIP_FEE_PCT: float = 0.1

@dataclass
class TradingConfig:
    """Global configuration object"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    
    # Auto-detect device for compute (MPS for Mac M1/M2, CUDA for NVIDIA, CPU for default)
    @property
    def DEVICE(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

# Create a singleton instance for project-wide use
config = TradingConfig()
