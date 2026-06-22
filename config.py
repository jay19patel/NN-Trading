# -*- coding: utf-8 -*-
"""
Central configuration for the Hudu labeling pipeline.
All numeric hyperparameters live here — change them once and every module picks them up.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # Labeling behavior
    LABELING_MODE: str = "fixed_rule"
    ENTRY_MODE: str = "next_open"
    USE_SMA_FILTER: bool = False
    USE_TECHNICAL_CONFIRMATION_FILTER: bool = True

    # Oracle labeler look-ahead window (bars).
    # Raised 48 → 96 (24h) so 5×ATR TP has more time to develop.
    LOOKAHEAD_BARS: int = 96

    # ATR smoothing period (Wilder's EWM)
    ATR_LENGTH: int = 14

    # Fixed-rule TP / SL multipliers.
    # TP=5×ATR, SL=1×ATR → ~5:1 gross RR, break-even ~17%.
    # After 0.2% round-trip cost: net TP≈1.3%, net SL≈0.5% → RR 2.6:1.
    # Break-even accuracy = 27.8%. Even at 30% the strategy has positive EV.
    FIXED_TP_ATR_MULTIPLIER: float = 5.0
    FIXED_SL_ATR_MULTIPLIER: float = 1.0

    # Grid of TP / SL ATR multipliers tested per bar
    TP_ATR_MULTIPLIERS: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    SL_ATR_MULTIPLIERS: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.75])

    # Reward-to-risk filter — combos below this ratio are skipped
    ORACLE_MIN_RR: float = 1.5

    # Hard floor / ceiling on TP and SL expressed as % of entry price.
    MIN_ATR_TARGET_PCT: float = 1.0
    MAX_ATR_TARGET_PCT: float = 5.0
    MIN_ATR_STOP_PCT: float = 0.25
    MAX_ATR_STOP_PCT: float = 2.0

    # Post-label confirmation filter.
    MIN_CONFIRMATION_SCORE: float = 4.0
    MIN_CONFIRMATION_EDGE: float = 2.0
    MIN_CONFIRMED_RETURN_PCT: float = 0.0


@dataclass
class TestingConfig:
    # Round-trip exchange fee (maker+taker, %)
    ROUND_TRIP_FEE_PCT: float = 0.1

    # One-way slippage estimate (%)
    SLIPPAGE_PCT: float = 0.05

    # Signal firing thresholds.
    # With fixed focal-loss bug, the model now learns NEUTRAL properly so
    # T will be much lower (closer to 1.0–1.5) and more bars will exceed 0.35.
    SIGNAL_MARGIN_THRESHOLD: float = 0.03   # best-class must beat NEUTRAL by this
    AI_CONFIDENCE_THRESHOLD: float = 0.35   # absolute floor on best-class prob
    DIRECTION_SPREAD_MIN: float = 0.05      # |p_long - p_short| floor — ensures model
                                            # isn't flip-flopping between directions


@dataclass
class MLTrainingConfig:
    """Controls the ML model training stage (train_model.py)."""
    TASK_MODE: str = "multiclass"

    # Chronological split (no shuffle). Test = 1 - TRAIN_FRAC - VAL_FRAC.
    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15
    RANDOM_STATE: int = 42

    FEATURE_SELECTION: str = "importance_top_k"
    TOP_K_FEATURES: int = 40


@dataclass
class MLBacktestConfig:
    """Controls the ML model backtest stage (model_backtest.py)."""
    INITIAL_CAPITAL: float = 1000.0
    CONFIDENCE_THRESHOLD: float = 0.60


@dataclass
class NNConfig:
    """Causal-Transformer model + training knobs.

    The sequence model reads the last WINDOW_SIZE bars of features (24h of
    context at WINDOW_SIZE=96 on 15m).
    """
    # Architecture — wider (128→192) and deeper (3→4 layers).
    # DROPOUT lowered 0.35→0.20: model was underfitting (val macro-F1 ≈ random),
    # not overfitting, so less regularisation is correct.
    HIDDEN_DIM: int = 192       # model width (must be divisible by NUM_HEADS)
    NUM_LAYERS: int = 4         # transformer encoder depth
    NUM_HEADS: int = 4          # attention heads
    DROPOUT: float = 0.20
    WINDOW_SIZE: int = 96       # past candles fed to the model (96×15m = 24h)
    MAX_SEQ_LEN: int = 256      # positional-encoding capacity (>= WINDOW_SIZE)

    # Training — more epochs + patience for the larger model to converge.
    # NOTE: set EPOCHS=10 only for quick testing. Use 150 for a real training run.
    EPOCHS: int = 15
    LR: float = 1e-4
    WEIGHT_DECAY: float = 2e-3
    BATCH_SIZE: int = 512
    EARLY_STOP_PATIENCE: int = 35  # epochs without val macro-F1 gain → stop
    LABEL_SMOOTHING: float = 0.0
    RANDOM_STATE: int = 42


@dataclass
class AppConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    ml_training: MLTrainingConfig = field(default_factory=MLTrainingConfig)
    ml_backtest: MLBacktestConfig = field(default_factory=MLBacktestConfig)
    nn: NNConfig = field(default_factory=NNConfig)


cfg = AppConfig()
