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

    EPOCHS: int = 5
    LR: float = 1e-4
    WEIGHT_DECAY: float = 2e-3
    BATCH_SIZE: int = 1024
    EARLY_STOP_PATIENCE: int = 30
    RANDOM_STATE: int = 42


@dataclass
class AppConfig:
    ml_training: MLTrainingConfig = field(default_factory=MLTrainingConfig)
    nn: NNConfig = field(default_factory=NNConfig)


cfg = AppConfig()
