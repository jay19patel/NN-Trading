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
from strategies.base import BaseStrategy


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
    label_r_multiple = np.zeros(row_count)
    
    cost_pct = estimated_trade_cost_pct_frac * 100.0
    
    for row_index in range(row_count - lookahead):
        entry_long = close_prices[row_index] * (1 + slippage_fraction)
        entry_short = close_prices[row_index] * (1 - slippage_fraction)
        
        best_label = 1
        best_tp = 1.0
        best_sl = 0.5
        best_net_return = 0.0
        best_r_multiple = 0.0

        current_atr_pct = atr_pct[row_index]
        if current_atr_pct <= 0.0:
            continue

        for tp_i in range(len(tp_multipliers)):
            tp_pct = current_atr_pct * tp_multipliers[tp_i]
            if tp_pct < min_target_pct:
                tp_pct = min_target_pct
            if tp_pct > max_target_pct:
                tp_pct = max_target_pct
            for sl_i in range(len(sl_multipliers)):
                sl_pct = current_atr_pct * sl_multipliers[sl_i]
                if sl_pct < min_stop_pct:
                    sl_pct = min_stop_pct
                if sl_pct > max_stop_pct:
                    sl_pct = max_stop_pct
                rr = tp_pct / max(sl_pct, 1e-9)
                if rr < min_rr:
                    continue

                long_code, long_pnl = _simulate_trade_path_with_entry(
                    close_prices, high_prices, low_prices, row_index, lookahead,
                    tp_pct, sl_pct, True, entry_long
                )
                short_code, short_pnl = _simulate_trade_path_with_entry(
                    close_prices, high_prices, low_prices, row_index, lookahead,
                    tp_pct, sl_pct, False, entry_short
                )

                long_net = long_pnl - cost_pct
                short_net = short_pnl - cost_pct

                if long_code == 0 and long_net > best_net_return:
                    best_label = 0
                    best_tp = tp_pct
                    best_sl = sl_pct
                    best_net_return = long_net
                    best_r_multiple = long_net / max(sl_pct, 1e-9)

                if short_code == 0 and short_net > best_net_return:
                    best_label = 2
                    best_tp = tp_pct
                    best_sl = sl_pct
                    best_net_return = short_net
                    best_r_multiple = short_net / max(sl_pct, 1e-9)

        if best_net_return > 0.0:
            labels[row_index] = best_label
            label_tp_pct[row_index] = best_tp
            label_sl_pct[row_index] = best_sl
            label_expected_return_pct[row_index] = best_net_return
            label_r_multiple[row_index] = best_r_multiple

    return labels, label_tp_pct, label_sl_pct, label_expected_return_pct, label_r_multiple

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
        
        slippage_fraction = strategy.SLIPPAGE_PCT / 100.0
        atr_pct = _causal_atr_pct(df, strategy.ATR_LENGTH)
        tp_multipliers = np.array(strategy.TP_ATR_MULTIPLIERS, dtype=np.float64)
        sl_multipliers = np.array(strategy.SL_ATR_MULTIPLIERS, dtype=np.float64)
        
        labels, label_tp_pct, label_sl_pct, label_expected_return_pct, label_r_multiple = _compute_oracle_labels_jit(
            row_count, lookahead, closes, highs, lows,
            atr_pct, tp_multipliers, sl_multipliers,
            strategy.ROUND_TRIP_FEE_PCT / 100.0 + (2.0 * strategy.SLIPPAGE_PCT / 100.0),
            slippage_fraction,
            strategy.ORACLE_MIN_RR,
            strategy.MIN_ATR_TARGET_PCT,
            strategy.MAX_ATR_TARGET_PCT,
            strategy.MIN_ATR_STOP_PCT,
            strategy.MAX_ATR_STOP_PCT
        )
        
        df['ai_verdict'] = labels
        df['ai_take_profit_pct'] = label_tp_pct
        df['ai_stop_loss_pct'] = label_sl_pct
        df['ai_qty_ratio'] = 1.0
        df['ai_confidence'] = 1.0
        df['ai_directional_edge'] = 1.0
        df['ai_expected_return_pct'] = label_expected_return_pct
        df['ai_r_multiple'] = label_r_multiple
        
        return df
