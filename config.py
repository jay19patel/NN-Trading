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
    # ? BUMPED FOR ACCELERATION + CROSS-ASSET FEATURE EXPANSION (V11)
    FEATURE_CACHE_VERSION: int = 11

@dataclass
class FeatureConfig:
    """Settings for feature engineering and pruning"""
    LOOKAHEAD_BARS: int = 12                     # Path resolution window
    PRUNING_THRESHOLD: float = 0.03              # Spearman correlation pruning floor
    USE_CURATED_FEATURES: bool = True            # Keep high-signal, interpretable features only
    # ? CURATED LIST — MUST BE DIRECTIONAL OR REGIME-CONDITIONED.
    # ? PURE-MAGNITUDE FEATURES (volatility_*, BB_width, ATR_pct, ...) ARE INCLUDED ONLY
    # ? IF THEY HAVE A SIGNED COUNTERPART (signed_atr_pct, signed_bb_width) ALSO PRESENT.
    CURATED_FEATURES: tuple[str, ...] = (
        # Trend / regime — all directional
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
        # Volatility (kept as REGIME context — ALWAYS PAIRED WITH SIGNED VERSION BELOW)
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
        # ? NEW DELTA-OF-DELTA (ACCELERATION) FEATURES — added by add_acceleration_features
        "return_accel_5",
        "return_accel_10",
        "return_accel_smooth",
        "rsi_velocity",
        "rsi_accel",
        "vol_velocity",
        "vol_accel",
        "macd_velocity",
        "macd_accel",
        # ? NEW CROSS-ASSET / REGIME / SIGNED-MAGNITUDE FEATURES — fix the directional flip
        "htf_return_pct",
        "htf_trend_sign",
        "vol_regime_ratio",
        "signed_volatility_10",
        "signed_atr_pct",
        "signed_bb_width",
        "signed_volume_ratio",
        "rsi_excess",
        "above_htf_ema",
        "htf_ema_distance",
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
    # ? BULLETPROOF RE-TUNE: 4 LAYERS WAS OVERFITTING ON ~30K WINDOWS.
    # ? 3 LAYERS WITH WIDER FFN AND HIGHER DROPOUT GENERALIZES BETTER.
    HIDDEN_DIM: int = 128
    NUM_HEADS: int = 8
    NUM_LAYERS: int = 3                          # Was 4 — reduced to combat memorization
    DROPOUT: float = 0.30                        # Was 0.25 — increased for stronger regularization
    SEQ_LEN: int = 48

@dataclass
class TrainingConfig:
    """Settings for the AI training process"""
    # Chronological holdouts: train -> val -> test (oldest -> newest)
    VAL_DAYS: int = 20
    TEST_DAYS: int = 30
    # ? MORE EPOCHS LET ONE-CYCLE SCHEDULER FINISH; EARLY-STOP STILL CAPTURES BEST COMPOSITE CHECKPOINT.
    EPOCHS: int = 15
    # ? PHASE 2 RL OFTEN DOES NOT IMPROVE HOLDOUT PRECISION — KEEP OFF UNTIL SUPERVISED MODEL IS STRONG.
    RL_FINE_TUNE_EPOCHS: int = 0
    EARLY_STOP_PATIENCE: int = 15                # Reduced from 20 — combined with composite score
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 0.00006
    RL_LEARNING_RATE: float = 0.00002
    WEIGHT_DECAY: float = 0.0015                 # Bumped from 0.001 — more L2 against memorization
    USE_WEIGHTED_SAMPLER: bool = False
    CLASS_WEIGHT_POWER: float = 0.85
    MIN_CLASS_WEIGHT: float = 0.60
    MAX_CLASS_WEIGHT: float = 3.0

    # Loss weights
    LOSS_ALPHA: float = 1.0
    LOSS_BETA: float = 0.5
    LOSS_GAMMA: float = 0.0                      # Must be >0 with RL_FINE_TUNE_EPOCHS >0 for Phase 2
    # ? NEW — STRENGTH OF THE SIGNAL/SIZING CONSISTENCY LOSS COMPONENT
    LOSS_CONSISTENCY: float = 0.10

    # Focal Loss hyperparameters
    FOCAL_GAMMA: float = 2.2
    USE_FOCAL_LOSS: bool = True
    # ? EXTRA LOSS WHEN TARGET IS NEUTRAL BUT PREDICTION IS BUY OR SELL (BOOSTS TRADING PRECISION / WIN-RATE PATH).
    FOCAL_NEUTRAL_VIOLATION_SCALE: float = 1.42
    # ? AFTER INVERSE-FREQ WEIGHTS: SLIGHTLY UP NEUTRAL, DOWN LONG/SHORT TO REDUCE FALSE SIGNALS.
    CLASS_WEIGHT_NEUTRAL_SCALE: float = 1.12
    CLASS_WEIGHT_DIRECTIONAL_SCALE: float = 0.91

    # -----------------------------------------------------------------------
    # Speed / throughput (same math; faster wall-clock)
    # -----------------------------------------------------------------------
    # ? CHUNK SIZE FOR VAL/TEST FORWARD PASSES — LARGER = FEWER KERNEL LAUNCHES (GPU)
    EVAL_CHUNK_SIZE: int = 4096
    # ? DATALOADER WORKERS: -1 = AUTO (0 ON MPS/MACOS; MIN(8, CPU_COUNT) ON CUDA LINUX)
    DATALOADER_NUM_WORKERS: int = -1
    DATALOADER_PERSISTENT_WORKERS: bool = True
    # ? EXPERIMENTAL: TORCH.COMPILE (PYTORCH 2+) — OFTEN SKIPPED ON MPS; SAFE DEFAULT OFF
    USE_TORCH_COMPILE: bool = False
    # ? CUDA ONLY — MATMUL TF32 FOR ~2X SPEED ON AMPERE+ WITH NEGLIGIBLE IMPACT FOR THIS SCALE
    CUDA_MATMUL_TF32: bool = True
    CUDNN_BENCHMARK: bool = True

@dataclass
class StrategyConfig:
    """Settings for trading strategy and evaluation"""
    TARGET_PROFIT_PCT: float = 0.70             # Slightly more realistic for 15m
    STOP_LOSS_PCT: float = 0.40                 # Slightly wider stop for noise
    USE_DYNAMIC_TP_SL_LABELS: bool = False      # Disabled when USE_ORACLE_LABELS=True
    TP_ATR_MULTIPLIER: float = 1.25
    SL_ATR_MULTIPLIER: float = 0.75
    LABEL_TP_PCT_MIN: float = 0.35              # Lowered to capture more moves
    LABEL_TP_PCT_MAX: float = 2.50
    LABEL_SL_PCT_MIN: float = 0.25              # Avoid unrealistically tight noisy stops
    LABEL_SL_PCT_MAX: float = 1.50

    # -----------------------------------------------------------------------
    # Risk Management & Execution Rules
    # ? CONFIDENCE GATE USES MAX SOFTMAX — NOT A CALIBRATED WIN PROBABILITY.
    # ? RAW MODEL OUTPUT IS OFTEN FLAT (MAX ~0.42–0.55); 0.60 BLOCKED EVERY TRADE IN PRACTICE.
    # ? USE INFERENCE_DIRECTION_TEMPERATURE < 1 TO SHARPEN + THRESHOLD ~0.44–0.50 FOR RESEARCH RUNS.
    # -----------------------------------------------------------------------
    AI_CONFIDENCE_THRESHOLD: float = 0.46
    # ? SOFTMAX SHARPENING AT INFERENCE ONLY (LOGITS / T, T<1). TRAINING UNCHANGED.
    INFERENCE_DIRECTION_TEMPERATURE: float = 0.78
    MIN_REWARD_RISK_RATIO: float = 1.50         # Higher quality bar; calibrator pushes above
    SLIPPAGE_PCT: float = 0.05                   # 5bps per leg (realistic for crypto)
    PARALLEL_SLOTS: int = 2                      # Allow 2 trades for diversification
    KELLY_FRACTION: float = 0.3                  # More conservative sizing
    MIN_DIRECTIONAL_EDGE: float = 0.10          # Bumped from 0.05 — only act on true edge
    # ? ARGMAX IS OFTEN NOISY (E.G. 0.34/0.33/0.33). REQUIRE TOP CLASS TO BEAT RUNNER-UP.
    MIN_SOFTMAX_MARGIN: float = 0.06
    # ? PER ROW: BREAK-EVEN WIN RATE = (SL + COST) / (TP + SL); SKIP IF MAX SOFTMAX < THAT + BUFFER.
    # ? (SOFTMAX IS NOT A CALIBRATED P, BUT BELOW THIS BAR EVEN OPTIMISTIC EV IS NEGATIVE.)
    BREAKEVEN_CONFIDENCE_GATE: bool = True
    BREAKEVEN_CONFIDENCE_BUFFER: float = 0.03
    EXECUTION_TP_SCALE: float = 1.00
    EXECUTION_SL_SCALE: float = 1.00
    MAX_DAILY_TRADES: int = 6
    COOLDOWN_BARS: int = 2
    MAX_CONSECUTIVE_LOSSES: int = 3              # Tightened from 4 — kill streaks early
    DAILY_STOP_LOSS_PCT: float = 2.5             # Tightened from 3.0 — capital preservation
    BREAK_EVEN_AFTER_R: float = 0.80
    TRAIL_STOP_AFTER_R: float = 1.20
    TRAIL_STOP_R_MULTIPLE: float = 0.60

    # -----------------------------------------------------------------------
    # Oracle Labeling Engine
    # Uses ACTUAL future price data to create high-quality training labels.
    # -----------------------------------------------------------------------
    USE_ORACLE_LABELS: bool = True
    ORACLE_TP_CAPTURE_RATIO: float = 0.50       # Capture a tradable move, not dust after costs
    ORACLE_SL_CAPTURE_RATIO: float = 0.25       # Tighten stops for higher RR labels
    ORACLE_MIN_TP_PCT: float = 0.45
    ORACLE_MAX_TP_PCT: float = 2.50
    ORACLE_MIN_SL_PCT: float = 0.25
    ORACLE_MAX_SL_PCT: float = 1.25
    ORACLE_MIN_RR: float = 1.60                 # Increased for high-quality signal learning
    ORACLE_MIN_UPSIDE_PCT: float = 0.45         
    ORACLE_MIN_DOWNSIDE_PCT: float = 0.45       
    REQUIRE_PATTERN_SETUP_FOR_ORACLE: bool = True 
    MIN_ORACLE_SETUP_SCORE: float = 0.10        # Relaxed pattern score requirement

    # Production acceptance gates. These do not make profit guaranteed; they block
    # production-style output unless the most recent held-out month already proved it.
    MIN_PRODUCTION_TEST_GROWTH_PCT: float = 5.0
    MIN_PRODUCTION_TRADES: int = 5
    # ? VALIDATION PRECISION FLOOR RAISED TO BREAK-EVEN+: only ship a calibrated
    # ? threshold when the validation slice precision exceeds this value.
    MIN_VALIDATION_PRECISION: float = 0.48      # Aligned with sharpened val probs + noisy short horizon
    MIN_VALIDATION_EXPECTANCY_PCT: float = 0.02  # Allow calibration to find a threshold vs always BLOCKED
    COST_BUFFER_PCT: float = 0.05               # Reduced buffer

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
