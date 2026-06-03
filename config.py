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
    USE_SMA_FILTER: bool = True

    # Oracle labeler look-ahead window (bars)
    LOOKAHEAD_BARS: int = 20

    # ATR smoothing period (Wilder's EWM)
    ATR_LENGTH: int = 14

    # Fixed-rule TP / SL multipliers for realistic label analysis
    FIXED_TP_ATR_MULTIPLIER: float = 1.0
    FIXED_SL_ATR_MULTIPLIER: float = 0.5

    # Grid of TP / SL ATR multipliers tested per bar
    TP_ATR_MULTIPLIERS: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    SL_ATR_MULTIPLIERS: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.75])

    # Reward-to-risk filter — combos below this ratio are skipped
    ORACLE_MIN_RR: float = 1.5

    # Hard floor / ceiling on TP and SL expressed as % of entry price
    MIN_ATR_TARGET_PCT: float = 0.2
    MAX_ATR_TARGET_PCT: float = 5.0
    MIN_ATR_STOP_PCT: float = 0.1
    MAX_ATR_STOP_PCT: float = 3.0


@dataclass
class TestingConfig:
    # Round-trip exchange fee (maker+taker, %)
    ROUND_TRIP_FEE_PCT: float = 0.1

    # One-way slippage estimate (%)
    SLIPPAGE_PCT: float = 0.05


@dataclass
class AppConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)


cfg = AppConfig()
