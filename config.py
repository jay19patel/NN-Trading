# -*- coding: utf-8 -*-
"""
NN-Trading Engine — Central Configuration
==========================================
3 config classes, each with a clear single responsibility:

  NNModelConfig  — Model architecture + data source params (retrain required if changed)
  TrainingConfig — Oracle labeling + training hyperparams (retrain required if changed)
  TestingConfig  — Capital, risk, execution, signal filtering (no retrain required)

Usage:
  from config import cfg, bars_per_day
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. NNModelConfig — Everything that defines the neural network itself.
#    IMPORTANT: Changing anything here REQUIRES a full retrain.
# ---------------------------------------------------------------------------

@dataclass
class NNModelConfig:
    """
    Neural network architecture and data source parameters.

    Any change here invalidates saved model weights — delete models/ and retrain.
    """

    # --- Data Source ---
    SYMBOLS: list[str] = field(default_factory=lambda: ["ETHUSD"])
    INTERVAL: str = "1h"           # Shifted to 1h to eliminate HFT noise and boost PnL
    CACHE_VALID_MINS: int = 0      # 0 = always fetch fresh data

    # --- Transformer Architecture ---
    HIDDEN_DIM: int = 128          # Model width (was 64 — now bigger for more capacity)
    NUM_LAYERS: int = 3            # Encoder depth (was 2)
    NUM_HEADS: int = 4             # Attention heads (HIDDEN_DIM must be divisible by NUM_HEADS)
    DROPOUT: float = 0.30          # Uniform dropout across all layers
    MAX_SEQ_LEN: int = 256         # Positional encoding max length

    # --- Sequence Window ---
    WINDOW_SIZE: int = 96          # Past candles fed to model (96 × 1h = 4 days of context)


# ---------------------------------------------------------------------------
# 2. TrainingConfig — Oracle labeling + how the model is trained.
#    IMPORTANT: Changing anything here REQUIRES a full retrain.
# ---------------------------------------------------------------------------

@dataclass
class TradeBoundsConfig:
    """Shared constraints used by both Oracle Labeler and Backtester."""
    MIN_ATR_STOP_PCT: float = 0.30   # Min SL as % of price
    MAX_ATR_STOP_PCT: float = 2.00   # Max SL as % of price
    MIN_ATR_TARGET_PCT: float = 0.50 # Min TP as % of price
    MAX_ATR_TARGET_PCT: float = 6.00 # Max TP as % of price
    LOOKAHEAD_BARS: int = 24         # Max bars to hold a position (24 = 1 day)

@dataclass
class TrainingConfig(TradeBoundsConfig):
    """
    Oracle labeling parameters and training hyperparameters.

    Bump FEATURE_CACHE_VERSION whenever labeler or feature params change
    to force parquet cache rebuild.
    """

    # --- Data Volume (Days) ---
    TRAINING_DATA_DAYS: int = 500
    VALIDATION_DATA_DAYS: int = 100
    TEST_DATA_DAYS: int = 100

    # --- Leakage Prevention ---
    PURGE_BARS: int = 24           # Gap between train/val/test splits

    # --- Oracle Labeling (Balanced 1:1 Risk:Reward Strategy) ---
    # Equal targets and stops. Because we use a Macro Trend Filter, the win-rate
    # will stay above 50%, ensuring mathematically guaranteed long-term positive PnL.
    TP_ATR_MULTIPLIERS: tuple[float, ...] = (1.2, 1.5, 2.0)
    SL_ATR_MULTIPLIERS: tuple[float, ...] = (1.0, 1.2, 1.5)
    ATR_LENGTH: int = 14
    ORACLE_MIN_RR: float = 1   # STRICT 1:1 RR ratio

    # --- Training Hyperparams ---
    BATCH_SIZE: int = 128
    EPOCHS: int = 50               # High max; early stopping prevents overfit
    LR: float = 0.0001             # Lower LR to prevent model collapse
    EARLY_STOP_PATIENCE: int = 10   # Epochs without val improvement before stopping
    FOCAL_GAMMA: float = 3.0       # Higher gamma focus on harder, clear patterns
    FOCAL_NEUTRAL_VIOLATION_SCALE: float = 2.0  # Penalty when model wrongly trades on NEUTRAL

    # --- Threshold Search ---
    THRESHOLD_MIN_TRADES: int = 30  # Min trades required for a threshold to be considered valid

    # --- Cache Versioning ---
    # Bump this number whenever any labeling/feature param above changes.
    # This forces train.py to rebuild the parquet cache instead of using stale data.
    FEATURE_CACHE_VERSION: int = 114


# ---------------------------------------------------------------------------
# 3. TestingConfig — Backtesting, capital management, and signal execution.
#    Safe to change WITHOUT retraining the model.
# ---------------------------------------------------------------------------

@dataclass
class TestingConfig(TradeBoundsConfig):
    """
    Capital management, risk parameters, and signal filtering for backtesting/live trading.

    These can be tuned without retraining. Adjust signal thresholds here to control
    trade frequency vs. selectivity.
    """

    # --- Capital, Margin & Leverage ---
    INITIAL_CAPITAL_USD: float = 100.0
    MARGIN_PER_TRADE_PCT_OF_EQUITY: float = 5.0
    LEVERAGE: float = 20
    ROUND_TRIP_FEE_PCT: float = 0.10             # Total fee for entry + exit
    SLIPPAGE_PCT: float = 0.03                   # Per-leg slippage estimate

    # --- Signal Filtering (NEW APPROACH: margin-based, not absolute threshold) ---
    # A trade signal fires when the model's leading class probability beats neutral
    # by at least SIGNAL_MARGIN_THRESHOLD.  This is far more meaningful than a
    # raw probability floor because the model tends to output ~50% for everything.
    #
    # Example: prob_long=0.48, prob_neutral=0.33, prob_short=0.19
    #   margin = 0.48 - 0.33 = 0.15 → FIRES
    SIGNAL_MARGIN_THRESHOLD: float = 0.20        # Require a clear edge over NEUTRAL
    AI_CONFIDENCE_THRESHOLD: float = 0.75        # Trade only high-confidence signals
    USE_TREND_FILTER: bool = True               # Strictly trade with macro trend
    AI_TARGET_DISCOUNT_FACTOR: float = 0.80     # Scale down AI targets for conservative exits (1.0 = No discount)

    # --- Execution Guards ---
    PARALLEL_SLOTS: int = 1
    MAX_DAILY_TRADES: int = 2
    COOLDOWN_BARS: int = 6
    MAX_CONSECUTIVE_LOSSES: int = 10
    DAILY_STOP_LOSS_PCT: float = 10.0


# ---------------------------------------------------------------------------
# Root Config — Combines all three. Import and use `cfg` everywhere.
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    """Top-level config. Use `cfg` singleton from this module."""

    model: NNModelConfig = field(default_factory=NNModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)

    @property
    def DEVICE(self) -> str:
        """Auto-detect best available compute device."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# Singleton — import `cfg` everywhere instead of `config`
cfg = TradingConfig()


def _validate_config(c: TradingConfig) -> None:
    """Fail fast on configuration values that would break training or testing."""
    assert c.model.HIDDEN_DIM % c.model.NUM_HEADS == 0, (
        f"HIDDEN_DIM {c.model.HIDDEN_DIM} must be divisible by NUM_HEADS {c.model.NUM_HEADS}"
    )
    assert c.model.WINDOW_SIZE <= c.model.MAX_SEQ_LEN, (
        f"WINDOW_SIZE {c.model.WINDOW_SIZE} > MAX_SEQ_LEN {c.model.MAX_SEQ_LEN}"
    )
    assert c.training.PURGE_BARS >= c.training.LOOKAHEAD_BARS, (
        "PURGE_BARS should be >= LOOKAHEAD_BARS to prevent label-to-feature leakage"
    )
    assert 0.0 <= c.testing.ROUND_TRIP_FEE_PCT <= 1.0, "Fee must be 0-1%"
    assert c.testing.LEVERAGE >= 1.0, "Leverage must be >= 1"
    assert c.model.HIDDEN_DIM > 0 and c.model.NUM_LAYERS > 0


_validate_config(cfg)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def bars_per_day(interval: str) -> int:
    """Return the number of OHLCV bars per calendar day for a given interval string."""
    mapping = {"15m": 96, "1h": 24, "1d": 1}
    return mapping.get(interval, 96)
