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
    INTERVAL: str = "15m"          # 15-minute for better signal quality (was 5m - too noisy)
    CACHE_VALID_MINS: int = 0      # 0 = always fetch fresh data

    # --- Transformer Architecture ---
    HIDDEN_DIM: int = 128          # Model width (was 64 — now bigger for more capacity)
    NUM_LAYERS: int = 3            # Encoder depth (was 2)
    NUM_HEADS: int = 4             # Attention heads (HIDDEN_DIM must be divisible by NUM_HEADS)
    DROPOUT: float = 0.20          # Lower dropout for 5m data (noisier, less regularization needed)
    MAX_SEQ_LEN: int = 256         # Positional encoding max length

    # --- Sequence Window ---
    WINDOW_SIZE: int = 96          # Past candles fed to model (96 × 15m = 24 hours of context)


# ---------------------------------------------------------------------------
# 2. TrainingConfig — Oracle labeling + how the model is trained.
#    IMPORTANT: Changing anything here REQUIRES a full retrain.
# ---------------------------------------------------------------------------

@dataclass
class TradeBoundsConfig:
    """Shared constraints used by both Oracle Labeler and Backtester."""
    MIN_ATR_STOP_PCT: float = 0.20   # Min SL as % of price (15m sweet spot)
    MAX_ATR_STOP_PCT: float = 0.80   # Max SL as % of price (15m sweet spot)
    MIN_ATR_TARGET_PCT: float = 0.40 # Min TP as % of price (15m sweet spot)
    MAX_ATR_TARGET_PCT: float = 1.50 # Max TP as % of price (15m sweet spot)
    LOOKAHEAD_BARS: int = 16         # Max bars to hold a position (16 × 15m = 4 hours)

@dataclass
class TrainingConfig(TradeBoundsConfig):
    """
    Oracle labeling parameters and training hyperparameters.

    Bump FEATURE_CACHE_VERSION whenever labeler or feature params change
    to force parquet cache rebuild.
    """

    # --- Data Volume (Days) ---
    TRAINING_DATA_DAYS: int = 300
    VALIDATION_DATA_DAYS: int = 60
    TEST_DATA_DAYS: int = 60

    # --- Leakage Prevention ---
    PURGE_BARS: int = 32           # Gap between train/val/test splits (2× LOOKAHEAD_BARS)

    # --- Oracle Labeling (Scalping-optimized targets) ---
    # Tighter ATR multipliers for smaller scalping moves
    TP_ATR_MULTIPLIERS: tuple[float, ...] = (0.8, 1.0, 1.2)  # Wider targets for better RR
    SL_ATR_MULTIPLIERS: tuple[float, ...] = (0.3, 0.4, 0.5)
    ATR_LENGTH: int = 14
    ORACLE_MIN_RR: float = 2.0   # Higher minimum RR for scalping quality (was 1.5)

    # --- Training Hyperparams ---
    BATCH_SIZE: int = 128
    EPOCHS: int = 50               # High max; early stopping prevents overfit
    LR: float = 0.0001             # Lower LR to prevent model collapse
    EARLY_STOP_PATIENCE: int = 10   # Epochs without val improvement before stopping
    FOCAL_GAMMA: float = 4.0       # Higher gamma focus on harder, clear patterns (was 3.0)
    FOCAL_NEUTRAL_VIOLATION_SCALE: float = 3.0  # Stronger penalty for wrong trades (was 2.0)

    # --- Threshold Search ---
    THRESHOLD_MIN_TRADES: int = 30  # Min trades required for a threshold to be considered valid

    # --- Cache Versioning ---
    # Bump this number whenever any labeling/feature param above changes.
    # This forces train.py to rebuild the parquet cache instead of using stale data.
    # v115: Converted to 5m scalping system with microstructure features and SMA20 filter
    # v116: Improved Oracle RR ratio (2.0) and wider TP targets for better win rate
    # v117: Switch to 15m interval - better signal/noise ratio than 5m
    FEATURE_CACHE_VERSION: int = 117


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
    MARGIN_PER_TRADE_PCT_OF_EQUITY: float = 5.0  # Base margin (overridden by dynamic sizing)
    LEVERAGE: float = 5                          # Lower leverage for 5m scalping (was 20)
    ROUND_TRIP_FEE_PCT: float = 0.05             # Total fee for entry + exit (maker orders)
    SLIPPAGE_PCT: float = 0.01                   # Per-leg slippage estimate (tighter for 5m)

    # --- Signal Filtering (margin-based) ---
    # A trade signal fires when the model's leading class probability beats neutral
    # by at least SIGNAL_MARGIN_THRESHOLD.
    SIGNAL_MARGIN_THRESHOLD: float = 0.20        # Higher threshold for quality (was 0.10)
    AI_CONFIDENCE_THRESHOLD: float = 0.60        # Higher absolute floor for quality (was 0.45)
    USE_TREND_FILTER: bool = True                # Enable SMA20 trend filter
    AI_TARGET_DISCOUNT_FACTOR: float = 1.0       # No discount for 5m (was 0.80)

    # --- Dynamic Position Sizing (Pattern Confidence Engine) ---
    BASE_MARGIN_PCT: float = 2.0                 # Minimum margin for low-confidence trades
    MAX_MARGIN_PCT: float = 8.0                  # Maximum margin for high-confidence trades

    # --- Execution Guards ---
    PARALLEL_SLOTS: int = 1
    MAX_DAILY_TRADES: int = 20                   # Higher limit for scalping (was 2)
    COOLDOWN_BARS: int = 3                       # Shorter cooldown for 5m (was 6)
    MAX_CONSECUTIVE_LOSSES: int = 5              # Tighter stop for scalping (was 10)
    DAILY_STOP_LOSS_PCT: float = 3.0             # Tighter daily stop for scalping (was 10.0)


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
    assert c.training.PURGE_BARS >= c.training.LOOKAHEAD_BARS * 2, (
        "PURGE_BARS should be >= 2x LOOKAHEAD_BARS to prevent label leakage"
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
    mapping = {"5m": 288, "15m": 96, "1h": 24, "1d": 1}
    return mapping.get(interval, 96)
