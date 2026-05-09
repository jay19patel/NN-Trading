# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import os

@dataclass
class DataConfig:
    SYMBOLS: list[str] = ("ETHUSD",)    # Trading pair (e.g. ETHUSD, BTCUSD)
    INTERVAL: str = "15m"              # Timeframe (15m, 1h, etc.)
    CACHE_VALID_MINS: int = 0          # 0 means rebuild data every time for 100% accuracy

@dataclass
class FeatureConfig:
    LOOKAHEAD_BARS: int = 48           # Increased to 12 hours for better trend capture
    PURGE_BARS: int = 24               
    FEATURE_CACHE_VERSION: int = 100   

@dataclass
class StrategyConfig:
    # --- Capital & Risk ---
    INITIAL_CAPITAL_USD: float = 50.0  
    RISK_PER_TRADE_PCT_OF_EQUITY: float = 2.0  
    MAX_POSITION_NOTIONAL_PCT_OF_EQUITY: float = 0.95  
    ROUND_TRIP_FEE_PCT: float = 0.05   
    SLIPPAGE_PCT: float = 0.02         
    ATR_LENGTH: int = 14               

    # --- AI Sizing Boundaries ---
    MIN_ATR_STOP_PCT: float = 0.80     # Slightly reduced for better entry
    MAX_ATR_STOP_PCT: float = 2.50     
    MIN_ATR_TARGET_PCT: float = 1.00   
    MAX_ATR_TARGET_PCT: float = 6.00   

    # --- Oracle Filter Settings ---
    ORACLE_MIN_RR: float = 1.5         # Realistic RR for 15m timeframe

    # --- Execution Rules ---
    PARALLEL_SLOTS: int = 1            # Ek saath kitne trades open ho sakte hain
    MAX_DAILY_TRADES: int = 15         # Din mein maximum kitne trades lene hain
    COOLDOWN_BARS: int = 8             # Ek trade ke baad kitni der rukna hai (8 bars = 2 hours)
    MAX_CONSECUTIVE_LOSSES: int = 2    # Kitne loss ke baad trading band karni hai session ke liye
    DAILY_STOP_LOSS_PCT: float = 4.0   # Pura din ka maximum loss limit

    # --- Oracle Labeling (Automatic target generation) ---
    TP_ATR_MULTIPLIERS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0) # Targets dhoondhne ki range
    SL_ATR_MULTIPLIERS: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5) # Stop loss dhoondhne ki range

    # --- Thresholds ---
    AI_CONFIDENCE_THRESHOLD: float = 0.40 # Agar JSON missing hai, toh kitni confidence par trade lena hai

@dataclass
class ModelConfig:
    HIDDEN_DIM: int = 64               # Neural network ki width (layers ki power)
    NUM_HEADS: int = 4                 # Transformer attention heads
    NUM_LAYERS: int = 2                # Kitni gehri layers honi chahiye (Noisy data ke liye 2 best hai)
    DROPOUT: float = 0.40              # Overfitting rokne ke liye kitne neurons randomly band karne hain
    MAX_SEQ_LEN: int = 128             # Maximum context memory
    WINDOW_SIZE: int = 30              # Pichli kitni candles dekh kar pattern pehchanna hai (30 bars = 7.5 hours)

@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 128              # Ek saath kitna data GPU ko dena hai
    EPOCHS: int = 50                   # Pura data kitni baar model ko dikhana hai
    LR: float = 0.0003                 # Seekhne ki raftaar (Slower = Deeper pattern learning)
    FOCAL_NEUTRAL_VIOLATION_SCALE: float = 2.0  # Galat signals ko punish karne ki power
    
    # --- Data Volume (Days) ---
    TRAINING_DATA_DAYS: int = 500      # 1.5 years of history for deep pattern learning
    VALIDATION_DATA_DAYS: int = 50     # History for threshold tuning
    TEST_DATA_DAYS: int = 30           # Final evaluation on the most recent month
    
    EARLY_STOP_PATIENCE: int = 3       # Agar 3 epochs tak loss kam na ho, toh training rok do
    SHORT_OBJECTIVE: str = "three_class_long_neutral_short" # Model ka main goal
    THRESHOLD_MIN_TRADES: int = 50     # Threshold search ke liye minimum kitne trades hone chahiye

@dataclass
class TradingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def DEVICE(self):
        import torch
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

config = TradingConfig()

def bars_per_day(interval: str) -> int:
    inv = {"15m": 96, "1h": 24, "1d": 1}
    return inv.get(interval, 96)
