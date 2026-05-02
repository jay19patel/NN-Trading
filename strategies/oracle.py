# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numba
from config import config
from strategies.base import BaseStrategy

@numba.njit
def _simulate_trade_path_with_entry(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    row_index: int,
    lookahead: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    direction_is_long: bool,
    entry_price: float,
) -> tuple[int, float]:
    if direction_is_long:
        tp_price = entry_price * (1 + take_profit_pct / 100.0)
        sl_price = entry_price * (1 - stop_loss_pct / 100.0)
    else:
        tp_price = entry_price * (1 - take_profit_pct / 100.0)
        sl_price = entry_price * (1 + stop_loss_pct / 100.0)

    for step in range(1, lookahead + 1):
        if row_index + step >= len(close_prices): break
        bar_high = high_prices[row_index + step]
        bar_low = low_prices[row_index + step]
        
        hit_tp = False
        hit_sl = False
        
        if direction_is_long:
            if bar_high >= tp_price: hit_tp = True
            if bar_low <= sl_price: hit_sl = True
        else:
            if bar_low <= tp_price: hit_tp = True
            if bar_high >= sl_price: hit_sl = True

        if hit_sl:
            return 1, -stop_loss_pct
        if hit_tp:
            return 0, take_profit_pct

    final_close = close_prices[min(row_index + lookahead, len(close_prices)-1)]
    if direction_is_long:
        timeout_pnl = ((final_close - entry_price) / entry_price) * 100.0
    else:
        timeout_pnl = ((entry_price - final_close) / entry_price) * 100.0
    return 2, timeout_pnl

@numba.njit
def _calculate_future_excursions_jit(highs, lows, lookahead):
    n = len(highs)
    future_max_high = np.zeros(n)
    future_min_low = np.zeros(n)
    for i in range(n):
        end_idx = min(i + lookahead + 1, n)
        if end_idx > i + 1:
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
    upside_adj, downside_adj,
    oracle_tp_long, oracle_sl_long, oracle_rr_long,
    oracle_tp_short, oracle_sl_short, oracle_rr_short,
    estimated_trade_cost_pct_frac,
    slippage_fraction,
    min_rr
):
    labels = np.ones(row_count)
    label_tp_pct = np.full(row_count, 1.0)
    label_sl_pct = np.full(row_count, 0.5)
    
    cost_pct = estimated_trade_cost_pct_frac * 100.0
    
    for row_index in range(row_count - lookahead):
        entry_long = close_prices[row_index] * (1 + slippage_fraction)
        entry_short = close_prices[row_index] * (1 - slippage_fraction)
        
        long_code, long_pnl = _simulate_trade_path_with_entry(close_prices, high_prices, low_prices, row_index, lookahead, oracle_tp_long[row_index], oracle_sl_long[row_index], True, entry_long)
        short_code, short_pnl = _simulate_trade_path_with_entry(close_prices, high_prices, low_prices, row_index, lookahead, oracle_tp_short[row_index], oracle_sl_short[row_index], False, entry_short)

        long_valid = (long_code == 0 and (oracle_tp_long[row_index] - cost_pct) > 0.01 and oracle_rr_long[row_index] >= min_rr)
        short_valid = (short_code == 0 and (oracle_tp_short[row_index] - cost_pct) > 0.01 and oracle_rr_short[row_index] >= min_rr)

        if long_valid and (not short_valid or long_pnl >= short_pnl):
            labels[row_index] = 0
            label_tp_pct[row_index] = oracle_tp_long[row_index]
            label_sl_pct[row_index] = oracle_sl_long[row_index]
        elif short_valid:
            labels[row_index] = 2
            label_tp_pct[row_index] = oracle_tp_short[row_index]
            label_sl_pct[row_index] = oracle_sl_short[row_index]
            
    return labels, label_tp_pct, label_sl_pct

class OracleStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "Oracle"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lookahead = config.features.LOOKAHEAD_BARS
        strategy = config.strategy
        
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        row_count = len(df)
        
        future_max_high, future_min_low = _calculate_future_excursions_jit(highs, lows, lookahead)
        
        slippage_fraction = strategy.SLIPPAGE_PCT / 100.0
        entry_price_long = closes * (1 + slippage_fraction)
        entry_price_short = closes * (1 - slippage_fraction)
        
        upside_adj = ((future_max_high - entry_price_long) / entry_price_long) * 100.0
        downside_adj = ((entry_price_short - future_min_low) / entry_price_short) * 100.0
        
        oracle_tp_long = np.maximum(upside_adj * strategy.ORACLE_TP_CAPTURE_RATIO, strategy.ORACLE_MIN_TP_PCT)
        oracle_sl_long = oracle_tp_long / 2.0
        
        oracle_tp_short = np.maximum(downside_adj * strategy.ORACLE_TP_CAPTURE_RATIO, strategy.ORACLE_MIN_TP_PCT)
        oracle_sl_short = oracle_tp_short / 2.0

        oracle_rr_long = oracle_tp_long / oracle_sl_long
        oracle_rr_short = oracle_tp_short / oracle_sl_short

        oracle_lookahead = lookahead // 2 
        
        labels, label_tp_pct, label_sl_pct = _compute_oracle_labels_jit(
            row_count, oracle_lookahead, closes, highs, lows,
            upside_adj, downside_adj,
            oracle_tp_long, oracle_sl_long, oracle_rr_long,
            oracle_tp_short, oracle_sl_short, oracle_rr_short,
            strategy.ROUND_TRIP_FEE_PCT / 100.0 + (2.0 * strategy.SLIPPAGE_PCT / 100.0),
            slippage_fraction,
            strategy.ORACLE_MIN_RR
        )
        
        df['ai_verdict'] = labels
        df['ai_take_profit_pct'] = label_tp_pct
        df['ai_stop_loss_pct'] = label_sl_pct
        df['ai_qty_ratio'] = 1.0
        df['ai_confidence'] = 1.0
        df['ai_directional_edge'] = 1.0
        
        return df
