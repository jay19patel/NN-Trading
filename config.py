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
    # Raised 20 → 48 (12h) so the bigger targets below have time to develop.
    LOOKAHEAD_BARS: int = 48

    # ATR smoothing period (Wilder's EWM)
    ATR_LENGTH: int = 14

    # Fixed-rule TP / SL multipliers (RR kept ~2:1).
    # Economics fix: median 15m ATR is only ~0.27%, so the old TP=1.0×ATR
    # (~0.27%) was 75% eaten by the 0.2% round-trip fee. Bigger multipliers aim
    # for ~1%+ moves where the fee is a small fraction of the target.
    FIXED_TP_ATR_MULTIPLIER: float = 3.0
    FIXED_SL_ATR_MULTIPLIER: float = 1.5

    # Grid of TP / SL ATR multipliers tested per bar
    TP_ATR_MULTIPLIERS: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    SL_ATR_MULTIPLIERS: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.75])

    # Reward-to-risk filter — combos below this ratio are skipped
    ORACLE_MIN_RR: float = 1.5

    # Hard floor / ceiling on TP and SL expressed as % of entry price.
    # Floors raised so a trade only fires when the target is large enough for
    # the 0.2% fee to be a small fraction (TP floor 0.8% → fee is ~25% of TP
    # worst case, vs 75% before).
    MIN_ATR_TARGET_PCT: float = 0.8
    MAX_ATR_TARGET_PCT: float = 5.0
    MIN_ATR_STOP_PCT: float = 0.4
    MAX_ATR_STOP_PCT: float = 3.0

    # Post-label confirmation filter.
    # Oracle labels are first generated from future trade outcome, then only
    # labels confirmed by causal indicators on the signal candle are retained.
    MIN_CONFIRMATION_SCORE: float = 4.0
    MIN_CONFIRMATION_EDGE: float = 2.0
    MIN_CONFIRMED_RETURN_PCT: float = 0.0


@dataclass
class TestingConfig:
    # Round-trip exchange fee (maker+taker, %)
    ROUND_TRIP_FEE_PCT: float = 0.1

    # One-way slippage estimate (%)
    SLIPPAGE_PCT: float = 0.05

    # Signal firing thresholds — only trade high-confidence setups.
    # Model accuracy at conf≥0.40 is ~56%; at conf 0.30-0.40 it's ~30% (below random).
    # Low thresholds generate many trades but all at random accuracy → guaranteed loss.
    SIGNAL_MARGIN_THRESHOLD: float = 0.08   # best-class must beat NEUTRAL by this
    AI_CONFIDENCE_THRESHOLD: float = 0.40   # absolute floor on best-class prob


@dataclass
class MLTrainingConfig:
    """Controls the ML model training stage (train_model.py)."""
    # Prediction task.
    #   "binary"     → LONG vs SHORT only (NEUTRAL bars dropped for training)
    #   "multiclass" → LONG / NEUTRAL / SHORT  (model learns when to STAY OUT —
    #                  essential: only ~11% of bars are real setups, so the
    #                  NEUTRAL class is what stops the strategy over-trading)
    TASK_MODE: str = "multiclass"

    # Chronological split (no shuffle). Test = 1 - TRAIN_FRAC - VAL_FRAC.
    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15
    RANDOM_STATE: int = 42

    # Feature selection / filter.
    #   "importance_top_k" → keep the TOP_K_FEATURES by model importance
    #   "all"              → use every leakage-safe causal feature
    FEATURE_SELECTION: str = "importance_top_k"
    TOP_K_FEATURES: int = 40


@dataclass
class MLBacktestConfig:
    """Controls the ML model backtest stage (model_backtest.py)."""
    INITIAL_CAPITAL: float = 1000.0

    # Only trade when the winning TRADE-class probability is at least this high
    # AND beats NEUTRAL. Higher → fewer, higher-conviction trades.
    CONFIDENCE_THRESHOLD: float = 0.60


@dataclass
class NNConfig:
    """Causal-Transformer model + training knobs (train_nn.py / nn_backtest.py).

    The sequence model reads the last WINDOW_SIZE bars of features (24h of
    context at WINDOW_SIZE=96 on 15m) instead of a single bar — this is the
    edge the single-bar sklearn model could not capture.
    """
    # Architecture
    HIDDEN_DIM: int = 128       # model width (must be divisible by NUM_HEADS)
    NUM_LAYERS: int = 3         # transformer encoder depth
    NUM_HEADS: int = 4          # attention heads
    DROPOUT: float = 0.35
    WINDOW_SIZE: int = 96       # past candles fed to the model (96×15m = 24h)
    MAX_SEQ_LEN: int = 256      # positional-encoding capacity (>= WINDOW_SIZE)

    # Training
    EPOCHS: int = 80
    LR: float = 1e-4
    WEIGHT_DECAY: float = 2e-3
    BATCH_SIZE: int = 512
    EARLY_STOP_PATIENCE: int = 20  # epochs without val macro-F1 gain → stop
    LABEL_SMOOTHING: float = 0.10
    RANDOM_STATE: int = 42

    # Reuses the chronological split + confidence gating from the other ML
    # configs so the held-out test window matches model_backtest exactly.


@dataclass
class AppConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    ml_training: MLTrainingConfig = field(default_factory=MLTrainingConfig)
    ml_backtest: MLBacktestConfig = field(default_factory=MLBacktestConfig)
    nn: NNConfig = field(default_factory=NNConfig)


cfg = AppConfig()
