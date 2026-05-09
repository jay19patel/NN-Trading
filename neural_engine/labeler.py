# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
try:
    import numba
except ModuleNotFoundError:
    class _NumbaFallback:
        @staticmethod
        def njit(func=None, **_kwargs):
            if func is None:
                return lambda wrapped: wrapped
            return func
    numba = _NumbaFallback()

from config import config

def _causal_atr_pct(df: pd.DataFrame, length: int) -> np.ndarray:
    """ATR percentage using only current and historical bars."""
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    atr_pct = (atr / df["Close"]) * 100.0
    return atr_pct.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)

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
    entry_price: float,
) -> tuple[int, float, int]:
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
            return 1, -stop_loss_pct, step
        if hit_tp:
            return 0, take_profit_pct, step

    final_close = close_prices[min(row_index + lookahead, len(close_prices)-1)]
    if direction_is_long:
        timeout_pnl = ((final_close - entry_price) / entry_price) * 100.0
    else:
        timeout_pnl = ((entry_price - final_close) / entry_price) * 100.0
    return 2, timeout_pnl, lookahead

@numba.njit
def _compute_oracle_labels_jit(
    row_count, lookahead, close_prices, high_prices, low_prices,
    atr_pct, tp_multipliers, sl_multipliers,
    estimated_trade_cost_pct_frac,
    slippage_fraction,
    min_rr,
    min_target_pct,
    max_target_pct,
    min_stop_pct,
    max_stop_pct
):
    labels = np.ones(row_count)
    label_tp_pct = np.full(row_count, 1.0)
    label_sl_pct = np.full(row_count, 0.5)
    label_expected_return_pct = np.zeros(row_count)
    label_time_to_target = np.full(row_count, lookahead, dtype=np.float64)
    label_magnitude = np.zeros(row_count)
    
    cost_pct = estimated_trade_cost_pct_frac * 100.0
    
    for row_index in range(row_count - lookahead):
        entry_long = close_prices[row_index] * (1 + slippage_fraction)
        entry_short = close_prices[row_index] * (1 - slippage_fraction)
        
        best_label = 1
        best_tp = 1.0
        best_sl = 0.5
        best_net_return = 0.0
        best_time = lookahead
        best_magnitude = 0.0

        current_atr_pct = atr_pct[row_index]
        if current_atr_pct <= 0.0: continue

        for tp_i in range(len(tp_multipliers)):
            tp_pct = current_atr_pct * tp_multipliers[tp_i]
            if tp_pct < min_target_pct: tp_pct = min_target_pct
            elif tp_pct > max_target_pct: tp_pct = max_target_pct
            
            for sl_i in range(len(sl_multipliers)):
                sl_pct = current_atr_pct * sl_multipliers[sl_i]
                if sl_pct < min_stop_pct: sl_pct = min_stop_pct
                elif sl_pct > max_stop_pct: sl_pct = max_stop_pct
                if tp_pct / max(sl_pct, 1e-9) < min_rr: continue

                l_code, l_pnl, l_time = _simulate_trade_path(close_prices, high_prices, low_prices, row_index, lookahead, tp_pct, sl_pct, True, entry_long)
                s_code, s_pnl, s_time = _simulate_trade_path(close_prices, high_prices, low_prices, row_index, lookahead, tp_pct, sl_pct, False, entry_short)

                l_net = l_pnl - cost_pct
                s_net = s_pnl - cost_pct

                if l_code == 0 and l_net > best_net_return:
                    best_label, best_tp, best_sl, best_net_return, best_time, best_magnitude = 0, tp_pct, sl_pct, l_net, l_time, tp_pct
                if s_code == 0 and s_net > best_net_return:
                    best_label, best_tp, best_sl, best_net_return, best_time, best_magnitude = 2, tp_pct, sl_pct, s_net, s_time, -tp_pct

        if best_net_return > 0.0:
            labels[row_index] = best_label
            label_tp_pct[row_index] = best_tp
            label_sl_pct[row_index] = best_sl
            label_expected_return_pct[row_index] = best_net_return
            label_time_to_target[row_index] = best_time
            label_magnitude[row_index] = best_magnitude

    return labels, label_tp_pct, label_sl_pct, label_expected_return_pct, label_time_to_target, label_magnitude

class OracleLabeler:
    """Generates ground-truth labels by looking ahead at future price action."""
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = config.strategy
        lookahead = config.features.LOOKAHEAD_BARS
        
        atr_pct = _causal_atr_pct(df, strategy.ATR_LENGTH)
        labels, tp_pct, sl_pct, exp_ret, time_to_target, magnitude = _compute_oracle_labels_jit(
            len(df), lookahead, df['Close'].values, df['High'].values, df['Low'].values,
            atr_pct, np.array(strategy.TP_ATR_MULTIPLIERS), np.array(strategy.SL_ATR_MULTIPLIERS),
            strategy.ROUND_TRIP_FEE_PCT/100.0 + (2.0*strategy.SLIPPAGE_PCT/100.0),
            strategy.SLIPPAGE_PCT/100.0, strategy.ORACLE_MIN_RR,
            strategy.MIN_ATR_TARGET_PCT, strategy.MAX_ATR_TARGET_PCT,
            strategy.MIN_ATR_STOP_PCT, strategy.MAX_ATR_STOP_PCT
        )
        
        df['direction_label'] = labels
        df['take_profit_pct'] = tp_pct
        df['stop_loss_pct'] = sl_pct
        df['expected_return_pct'] = exp_ret
        df['time_to_target'] = time_to_target
        df['magnitude_label'] = magnitude
        return df
