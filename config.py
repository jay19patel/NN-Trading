# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch
import os


def interval_to_minutes(interval: str) -> int:
    """Convert a timeframe like 15m, 1h, or 1d into minutes."""
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    raise ValueError(f"Unsupported interval format: {interval}")


def bars_per_day(interval: str) -> int:
    return (24 * 60) // interval_to_minutes(interval)


@dataclass
class DataConfig:
    """Settings for data fetching and caching"""
    SYMBOLS: list[str] = ("BTCUSD", "ETHUSD")  # List of symbols to train on
    INTERVAL: str = "15m"                       # Scalping-oriented base timeframe
    TOTAL_DAYS: int = 365                        # 1 full year of history for robust training
    CACHE_VALID_MINS: int = 120                  # Feature cache validity (2 hours)
    FEATURE_CACHE_VERSION: int = 7

@dataclass
class FeatureConfig:
    """Settings for feature engineering and pruning"""
    LOOKAHEAD_BARS: int = 8                      # 8 bars = 2h on 15m data
    PRUNING_THRESHOLD: float = 0.03              # Spearman correlation pruning floor
    USE_CURATED_FEATURES: bool = True            # Keep high-signal, interpretable features only
    CURATED_FEATURES: tuple[str, ...] = (
        # Trend / regime
        "price_to_ema_20",
        "price_to_ema_50",
        "price_to_ema_100",
        "price_to_ema_200",
        "price_to_sma_50",
        "price_to_sma_200",
        "ema_5_10_cross_pct",
        "ema_10_20_cross_pct",
        "supertrend_direction",
        "supertrend_distance",
        "ADX_14",
        "directional_bias",
        "bars_since_flip",
        "efficiency_ratio",
        "trend_persistence",
        "trend_strength",
        "is_trending",
        "is_choppy",
        # Momentum
        "RSI_14",
        "MACD_hist_pct",
        "ROC_10",
        "ROC_20",
        "WilliamsR_14",
        "Stoch_K",
        # Volatility / breakout context
        "ATR_pct",
        "BB_width",
        "BB_position",
        "volatility_10",
        "volatility_20",
        "chop_index",
        "range_compression",
        # Volume / participation
        "volume_ratio",
        "volume_zscore_20",
        "volume_change_5",
        "OBV_change_20",
        "CMF",
        "MFI",
        # Price action
        "return_1",
        "return_5",
        "return_10",
        "daily_range",
        "range_pct",
        "wick_imbalance",
        "wick_to_body",
        "close_position",
        "candle_strength",
        "buy_pressure",
        "stop_hunt_proxy",
        # Statistical extremes
        "zscore_20",
        "percentile_rank_20",
        "skew_20",
        "surprise",
        # Explicit pattern/setup features
        "trend_pullback_long_setup",
        "trend_pullback_short_setup",
        "breakout_long_setup",
        "breakout_short_setup",
        "reversal_long_setup",
        "reversal_short_setup",
        "long_setup_score",
        "short_setup_score",
        "setup_score_spread",
        "compression_breakout_score",
    )
    # Auto-pruning thresholds (applied after scaling in training_utils)
    VARIANCE_FLOOR: float = 0.01                 # Drop features with var < 0.01 after StandardScaler
    CORRELATION_CEILING: float = 0.95            # Drop one of each pair with |corr| > 0.95

@dataclass
class ModelConfig:
    """Transformer Architecture Hyperparameters"""
    HIDDEN_DIM: int = 48                         # Smaller model for curated feature set
    NUM_HEADS: int = 4                           # Attention heads (48 / 4 = 12 dim per head)
    NUM_LAYERS: int = 2                          # 2 layers for pattern depth
    DROPOUT: float = 0.30                        # Regularization without drowning signal
    SEQ_LEN: int = 32                            # 32 bars = 8h lookback on 15m data

@dataclass
class TrainingConfig:
    """Settings for the AI training process"""
    # Chronological holdouts: train -> val -> test (oldest -> newest)
    VAL_DAYS: int = 15                           # Increased from 4 — more reliable validation signal
    TEST_DAYS: int = 30                          # 1 month test window
    EPOCHS: int = 50                             # Increased epochs for deeper learning
    RL_FINE_TUNE_EPOCHS: int = 0                 # Disabled by default until a real path-based RL target exists
    EARLY_STOP_PATIENCE: int = 10                # Increased patience for slower LR
    BATCH_SIZE: int = 512                        # M4 GPU efficient batch size
    LEARNING_RATE: float = 0.00008               # Slightly faster LR for smaller model
    RL_LEARNING_RATE: float = 0.00002            # Phase 2 RL: slower to preserve Phase 1 knowledge
    WEIGHT_DECAY: float = 0.001                  # REDUCED from 0.01: was killing gradients
    USE_WEIGHTED_SAMPLER: bool = False            # Replaced by Focal Loss (no class weights needed)
    CLASS_WEIGHT_POWER: float = 0.75              # Strong enough to learn rare setups without exploding
    MIN_CLASS_WEIGHT: float = 0.75
    MAX_CLASS_WEIGHT: float = 2.25

    # Loss weights
    LOSS_ALPHA: float = 1.0                      # Signal (direction) loss weight — primary objective
    LOSS_BETA: float = 0.3                       # Sizing loss weight — secondary
    LOSS_GAMMA: float = 0.0                      # Disabled by default: current training is supervised

    # Focal Loss hyperparameters
    FOCAL_GAMMA: float = 1.25                    # Focus on rare setup mistakes, but not too aggressively
    USE_FOCAL_LOSS: bool = True                  # Use Focal Loss instead of CrossEntropy

@dataclass
class StrategyConfig:
    """Settings for trading strategy and evaluation"""
    TARGET_PROFIT_PCT: float = 0.75             # Fallback TP large enough to clear fees/slippage
    STOP_LOSS_PCT: float = 0.35                 # Fallback SL for controlled invalidation
    USE_DYNAMIC_TP_SL_LABELS: bool = False      # Disabled when USE_ORACLE_LABELS=True
    TP_ATR_MULTIPLIER: float = 1.25
    SL_ATR_MULTIPLIER: float = 0.65
    LABEL_TP_PCT_MIN: float = 0.45              # Avoid labeling trades that costs can erase
    LABEL_TP_PCT_MAX: float = 2.50
    LABEL_SL_PCT_MIN: float = 0.25              # Avoid unrealistically tight noisy stops
    LABEL_SL_PCT_MAX: float = 1.25

    # -----------------------------------------------------------------------
    # Risk Management & Execution Rules
    # -----------------------------------------------------------------------
    AI_CONFIDENCE_THRESHOLD: float = 0.50       # Trade only when the classifier has real separation
    MIN_REWARD_RISK_RATIO: float = 1.60         # Require enough edge after costs
    SLIPPAGE_PCT: float = 0.05                   # 5bps per leg (realistic for crypto)
    PARALLEL_SLOTS: int = 1                      # Keep one slot until edge quality improves
    KELLY_FRACTION: float = 0.3                  # More conservative sizing
    MIN_DIRECTIONAL_EDGE: float = 0.12          # buy/sell prob must beat neutral by a larger margin
    EXECUTION_TP_SCALE: float = 1.00            # Keep TP cost-aware; do not shrink below label intent
    EXECUTION_SL_SCALE: float = 1.00            # Keep SL aligned with trained labels
    MAX_DAILY_TRADES: int = 4
    COOLDOWN_BARS: int = 2
    MAX_CONSECUTIVE_LOSSES: int = 3
    DAILY_STOP_LOSS_PCT: float = 2.0
    BREAK_EVEN_AFTER_R: float = 0.80
    TRAIL_STOP_AFTER_R: float = 1.20
    TRAIL_STOP_R_MULTIPLE: float = 0.60

    # -----------------------------------------------------------------------
    # Oracle Labeling Engine
    # Uses ACTUAL future price data to create high-quality training labels.
    # -----------------------------------------------------------------------
    USE_ORACLE_LABELS: bool = True
    ORACLE_TP_CAPTURE_RATIO: float = 0.50       # Capture a tradable move, not dust after costs
    ORACLE_SL_CAPTURE_RATIO: float = 0.25       # Keep stops controlled but not micro-noisy
    ORACLE_MIN_TP_PCT: float = 0.45
    ORACLE_MAX_TP_PCT: float = 2.50
    ORACLE_MIN_SL_PCT: float = 0.25
    ORACLE_MAX_SL_PCT: float = 1.25
    ORACLE_MIN_RR: float = 1.45
    ORACLE_MIN_UPSIDE_PCT: float = 0.45
    ORACLE_MIN_DOWNSIDE_PCT: float = 0.45
    REQUIRE_PATTERN_SETUP_FOR_ORACLE: bool = True
    MIN_ORACLE_SETUP_SCORE: float = 0.25

    # Production acceptance gates. These do not make profit guaranteed; they block
    # production-style output unless the most recent held-out month already proved it.
    MIN_PRODUCTION_TEST_GROWTH_PCT: float = 10.0
    MIN_PRODUCTION_TRADES: int = 10
    MIN_VALIDATION_PRECISION: float = 0.55
    MIN_VALIDATION_EXPECTANCY_PCT: float = 0.05
    COST_BUFFER_PCT: float = 0.10

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
