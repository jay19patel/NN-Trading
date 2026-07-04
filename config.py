# -*- coding: utf-8 -*-
from dataclasses import dataclass, field

LOOKAHEAD_BARS: int = 20
MAX_RETURN_PCT: float = 5.0


@dataclass
class MLTrainingConfig:
    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15
    RANDOM_STATE: int = 42


@dataclass
class NNConfig:
    # HIDDEN_DIM must be divisible by NUM_HEADS
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 3
    NUM_HEADS: int = 4
    DROPOUT: float = 0.20
    WINDOW_SIZE: int = 60       # past candles fed to model (60×15m = 15h context)
    MAX_SEQ_LEN: int = 256

    EPOCHS: int = 50
    LR: float = 2e-4
    WEIGHT_DECAY: float = 2e-3
    BATCH_SIZE: int = 1024
    EARLY_STOP_PATIENCE: int = 10
    RANDOM_STATE: int = 42

    # Multi-task loss weights
    DIR_LOSS_WEIGHT: float = 1.0    # direction classification (BCE)
    MAG_LOSS_WEIGHT: float = 0.5    # upside/downside magnitude regression (Huber)


@dataclass
class BacktestConfig:
    INITIAL_CAPITAL: float = 100.0      # USD
    CONFIDENCE_THRESHOLD: float = 0.60  # trade only when max(p_up, 1-p_up) >= this
    MIN_PREDICTED_MOVE: float = 0.60    # % — skip signals with tiny predicted moves
    TP_FRACTION: float = 0.80           # take-profit = predicted move × this
    SL_FRACTION: float = 0.50           # stop-loss   = predicted move × this
    FEE_PCT: float = 0.05               # taker fee per side (Delta Exchange)
    SLIPPAGE_PCT: float = 0.02          # per side
    POSITION_FRACTION: float = 0.95     # fraction of equity deployed per trade
    MAX_HOLD_BARS: int = LOOKAHEAD_BARS # time exit after this many bars

    # ── Breakout bracket mode (direction-less) ────────────────────────────────
    BREAKOUT_MIN_MOVE: float = 1.00     # % — arm bracket only when predicted move >= this
    BREAKOUT_ARM_FRACTION: float = 0.30 # trigger distance = predicted move × this (each side)
    BREAKOUT_ARM_WINDOW: int = 8        # bars to wait for a trigger before cancelling


@dataclass
class AppConfig:
    ml_training: MLTrainingConfig = field(default_factory=MLTrainingConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


cfg = AppConfig()
