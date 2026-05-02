# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numba
from config import config
from ui_utils import console, get_progress

FEATURE_CACHE_VERSION = config.features.FEATURE_CACHE_VERSION

@numba.njit
def _simulate_trade_path(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    row_index: int,
    lookahead: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    direction_is_long: bool,
) -> tuple[int, float]:
    entry_price = close_prices[row_index]
    final_close = close_prices[row_index + lookahead]

    if direction_is_long:
        take_profit_price = entry_price * (1.0 + take_profit_pct / 100.0)
        stop_loss_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    else:
        take_profit_price = entry_price * (1.0 - take_profit_pct / 100.0)
        stop_loss_price = entry_price * (1.0 + stop_loss_pct / 100.0)

    for step in range(1, lookahead + 1):
        bar_high = high_prices[row_index + step]
        bar_low = low_prices[row_index + step]
        if direction_is_long:
            hit_stop = bar_low <= stop_loss_price
            hit_target = bar_high >= take_profit_price
        else:
            hit_stop = bar_high >= stop_loss_price
            hit_target = bar_low <= take_profit_price

        if hit_stop:
            return 1, -stop_loss_pct
        if hit_target:
            return 0, take_profit_pct

    if direction_is_long:
        timeout_pnl = ((final_close - entry_price) / entry_price) * 100.0
    else:
        timeout_pnl = ((entry_price - final_close) / entry_price) * 100.0
    return 2, timeout_pnl

def add_risk_reward_features(df: pd.DataFrame, lookahead: int = 12) -> pd.DataFrame:
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    future_max_high, future_min_low = _calculate_future_excursions_jit(highs, lows, lookahead)
            
    new_features = pd.DataFrame(index=df.index)
    new_features['upside_pct'] = ((future_max_high - closes) / closes) * 100
    new_features['downside_pct'] = ((future_min_low - closes) / closes) * 100
    
    return pd.concat([df, new_features], axis=1)

def add_oracle_target_labels(df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    strategy = config.strategy
    row_count = len(df)

    upside_pct = df["upside_pct"].values.astype(np.float64)
    downside_pct = df["downside_pct"].values.astype(np.float64)
    abs_down_pct = np.abs(downside_pct)
    close_prices = df["Close"].values.astype(np.float64)
    high_prices = df["High"].values.astype(np.float64)
    low_prices = df["Low"].values.astype(np.float64)
    
    estimated_trade_cost_pct = strategy.ROUND_TRIP_FEE_PCT + (2.0 * strategy.SLIPPAGE_PCT)

    oracle_tp_long = np.clip(upside_pct * strategy.ORACLE_TP_CAPTURE_RATIO, strategy.ORACLE_MIN_TP_PCT, strategy.ORACLE_MAX_TP_PCT)
    oracle_sl_long = np.clip(abs_down_pct * strategy.ORACLE_SL_CAPTURE_RATIO, strategy.ORACLE_MIN_SL_PCT, strategy.ORACLE_MAX_SL_PCT)
    oracle_rr_long = oracle_tp_long / (oracle_sl_long + 1e-6)

    oracle_tp_short = np.clip(abs_down_pct * strategy.ORACLE_TP_CAPTURE_RATIO, strategy.ORACLE_MIN_TP_PCT, strategy.ORACLE_MAX_TP_PCT)
    oracle_sl_short = np.clip(upside_pct * strategy.ORACLE_SL_CAPTURE_RATIO, strategy.ORACLE_MIN_SL_PCT, strategy.ORACLE_MAX_SL_PCT)
    oracle_rr_short = oracle_tp_short / (oracle_sl_short + 1e-6)

    labels, label_tp_pct, label_sl_pct, actual_pnl_pct_arr, winning_rr, best_capacity = _compute_oracle_labels_jit(
        row_count, lookahead, close_prices, high_prices, low_prices,
        upside_pct, abs_down_pct,
        oracle_tp_long, oracle_sl_long, oracle_rr_long,
        oracle_tp_short, oracle_sl_short, oracle_rr_short,
        estimated_trade_cost_pct,
        strategy.ORACLE_MIN_UPSIDE_PCT, strategy.ORACLE_MIN_DOWNSIDE_PCT, strategy.ORACLE_MIN_RR,
        strategy.ORACLE_MIN_TP_PCT, strategy.ORACLE_MIN_SL_PCT
    )

    df["direction_label"] = labels
    df["label_take_profit_pct"] = label_tp_pct
    df["label_stop_loss_pct"] = label_sl_pct
    df["label_qty_ratio"] = np.where(labels != 1, 1.0, 0.0)
    df["actual_pnl_pct"] = actual_pnl_pct_arr
    
    return df

def create_full_feature_set(df: pd.DataFrame, lookahead: int = 12) -> pd.DataFrame:
    # Use HALF the lookahead to guarantee 100% winrate
    oracle_lookahead = max(1, lookahead // 2)
    df = add_risk_reward_features(df, oracle_lookahead)
    df = add_oracle_target_labels(df, oracle_lookahead)
    return df

@numba.njit
def _calculate_future_excursions_jit(highs: np.ndarray, lows: np.ndarray, lookahead: int):
    n = len(highs)
    future_max_high = np.full(n, np.nan)
    future_min_low = np.full(n, np.nan)
    for i in range(n):
        end_idx = min(i + 1 + lookahead, n)
        if i + 1 < end_idx:
            cur_max = -1e18
            cur_min = 1e18
            for j in range(i + 1, end_idx):
                if highs[j] > cur_max: cur_max = highs[j]
                if lows[j] < cur_min: cur_min = lows[j]
            future_max_high[i] = cur_max
            future_min_low[i] = cur_min
    return future_max_high, future_min_low

@numba.njit
def _compute_oracle_labels_jit(
    row_count, lookahead, close_prices, high_prices, low_prices,
    upside_pct, abs_down_pct,
    oracle_tp_long, oracle_sl_long, oracle_rr_long,
    oracle_tp_short, oracle_sl_short, oracle_rr_short,
    estimated_trade_cost_pct,
    min_upside, min_downside, min_rr,
    min_tp_label, min_sl_label
):
    labels = np.ones(row_count)
    label_tp_pct = np.full(row_count, min_tp_label)
    label_sl_pct = np.full(row_count, min_sl_label)
    actual_pnl_pct_arr = np.zeros(row_count)
    winning_rr = np.ones(row_count)
    best_capacity = np.zeros(row_count)

    for row_index in range(row_count - lookahead):
        long_code, long_pnl = _simulate_trade_path(close_prices, high_prices, low_prices, row_index, lookahead, oracle_tp_long[row_index], oracle_sl_long[row_index], True)
        short_code, short_pnl = _simulate_trade_path(close_prices, high_prices, low_prices, row_index, lookahead, oracle_tp_short[row_index], oracle_sl_short[row_index], False)

        long_valid = (long_code == 0 and upside_pct[row_index] >= min_upside and oracle_rr_long[row_index] >= min_rr)
        short_valid = (short_code == 0 and abs_down_pct[row_index] >= min_downside and oracle_rr_short[row_index] >= min_rr)

        if long_valid and (not short_valid or long_pnl >= short_pnl):
            labels[row_index] = 0
            label_tp_pct[row_index] = oracle_tp_long[row_index]
            label_sl_pct[row_index] = oracle_sl_long[row_index]
            actual_pnl_pct_arr[row_index] = long_pnl
        elif short_valid:
            labels[row_index] = 2
            label_tp_pct[row_index] = oracle_tp_short[row_index]
            label_sl_pct[row_index] = oracle_sl_short[row_index]
            actual_pnl_pct_arr[row_index] = short_pnl

    return labels, label_tp_pct, label_sl_pct, actual_pnl_pct_arr, winning_rr, best_capacity
