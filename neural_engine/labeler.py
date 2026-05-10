# -*- coding: utf-8 -*-
"""
Oracle Labeler — Ground-truth label generation via future price simulation.
===========================================================================
For each bar the labeler looks LOOKAHEAD_BARS into the future and tests all
combinations of TP × SL ATR multipliers for both LONG and SHORT.

The combination that:
  1. Achieves TP_HIT (not SL_HIT) within the lookahead window
  2. Has the highest net return (after fees & slippage)
  3. Meets the minimum reward:risk ratio

...wins the label.  If no combination passes, the bar is labelled NEUTRAL (1).

Labels:
  0 — LONG  (best profitable long opportunity found)
  1 — NEUTRAL (no profitable trade found within lookahead)
  2 — SHORT (best profitable short opportunity found)
"""
import numpy as np
import pandas as pd

try:
    import numba
except ModuleNotFoundError:
    # Graceful fallback when numba is unavailable
    class _NumbaFallback:  # type: ignore[no-redef]
        @staticmethod
        def njit(func=None, **_kwargs):
            if func is None:
                return lambda wrapped: wrapped
            return func

    numba = _NumbaFallback()  # type: ignore[assignment]

from config import cfg


# ---------------------------------------------------------------------------
# ATR Calculation (causal — uses only past data)
# ---------------------------------------------------------------------------

def _causal_atr_pct(df: pd.DataFrame, length: int) -> np.ndarray:
    """
    Compute ATR as a percentage of Close price using only historical bars.

    Uses EWM (exponentially weighted moving average) with alpha = 1/length,
    which is equivalent to Wilder's smoothing method used in the original ATR.
    """
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


# ---------------------------------------------------------------------------
# Numba-JIT trade path simulation
# ---------------------------------------------------------------------------

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
    """
    Walk forward bar-by-bar from row_index and return the first outcome:
      - 0 = TP_HIT  (take profit reached)
      - 1 = SL_HIT  (stop loss reached — SL checked first if both hit same bar)
      - 2 = TIMEOUT (neither hit within lookahead)

    Returns: (outcome_code, pnl_pct, bars_to_outcome)
    """
    if direction_is_long:
        tp_price = entry_price * (1.0 + take_profit_pct / 100.0)
        sl_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    else:
        tp_price = entry_price * (1.0 - take_profit_pct / 100.0)
        sl_price = entry_price * (1.0 + stop_loss_pct / 100.0)

    for step in range(1, lookahead + 1):
        if row_index + step >= len(close_prices):
            break
        bar_high = high_prices[row_index + step]
        bar_low = low_prices[row_index + step]

        if direction_is_long:
            hit_sl = bar_low <= sl_price
            hit_tp = bar_high >= tp_price
        else:
            hit_sl = bar_high >= sl_price
            hit_tp = bar_low <= tp_price

        # Stop loss takes priority when both are hit in the same bar
        if hit_sl:
            return 1, -stop_loss_pct, step
        if hit_tp:
            return 0, take_profit_pct, step

    # Timeout — measure actual PnL at the final bar
    final_close = close_prices[min(row_index + lookahead, len(close_prices) - 1)]
    if direction_is_long:
        timeout_pnl = ((final_close - entry_price) / entry_price) * 100.0
    else:
        timeout_pnl = ((entry_price - final_close) / entry_price) * 100.0
    return 2, timeout_pnl, lookahead


@numba.njit
def _compute_oracle_labels_jit(
    row_count: int,
    lookahead: int,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    atr_pct: np.ndarray,
    tp_multipliers: np.ndarray,
    sl_multipliers: np.ndarray,
    estimated_trade_cost_pct_frac: float,
    slippage_fraction: float,
    min_rr: float,
    min_target_pct: float,
    max_target_pct: float,
    min_stop_pct: float,
    max_stop_pct: float,
    sma50: np.ndarray,
    atr_ratio: np.ndarray,
):
    """
    Core Oracle labeling loop — fully JIT-compiled for speed.

    For every bar tests all TP×SL combos in both directions and keeps the
    combination with the highest net positive return.
    """
    labels = np.ones(row_count)                           # default NEUTRAL
    label_tp_pct = np.full(row_count, 0.8)
    label_sl_pct = np.full(row_count, 0.4)
    label_expected_return_pct = np.zeros(row_count)
    label_time_to_target = np.full(row_count, float(lookahead))
    label_magnitude = np.zeros(row_count)

    cost_pct = estimated_trade_cost_pct_frac * 100.0

    for row_index in range(row_count - lookahead):
        current_atr = atr_pct[row_index]
        if current_atr <= 0.0:
            continue

        entry_long = close_prices[row_index] * (1.0 + slippage_fraction)
        entry_short = close_prices[row_index] * (1.0 - slippage_fraction)

        best_label = 1          # NEUTRAL default
        best_tp = 0.8
        best_sl = 0.4
        best_net_return = 0.0
        best_time = float(lookahead)
        best_magnitude = 0.0

        for tp_i in range(len(tp_multipliers)):
            tp_pct = current_atr * tp_multipliers[tp_i]
            if tp_pct < min_target_pct:
                tp_pct = min_target_pct
            elif tp_pct > max_target_pct:
                tp_pct = max_target_pct

            for sl_i in range(len(sl_multipliers)):
                sl_pct = current_atr * sl_multipliers[sl_i]
                if sl_pct < min_stop_pct:
                    sl_pct = min_stop_pct
                elif sl_pct > max_stop_pct:
                    sl_pct = max_stop_pct

                # Skip combos that don't meet minimum reward:risk
                rr = tp_pct / max(sl_pct, 1e-9)
                if rr < min_rr:
                    continue

                # Volatility Filter: Skip if current ATR is much higher than average (noisy regime)
                if atr_ratio[row_index] > 1.5:
                    continue

                # Test LONG (Only if Price > SMA50)
                if close_prices[row_index] > sma50[row_index]:
                    l_code, l_pnl, l_time = _simulate_trade_path(
                        close_prices, high_prices, low_prices,
                        row_index, lookahead, tp_pct, sl_pct, True, entry_long,
                    )
                    l_net = l_pnl - cost_pct
                    if l_code == 0 and l_net > best_net_return:
                        best_label, best_tp, best_sl = 0, tp_pct, sl_pct
                        best_net_return, best_time, best_magnitude = l_net, float(l_time), tp_pct

                # Test SHORT (Only if Price < SMA50)
                if close_prices[row_index] < sma50[row_index]:
                    s_code, s_pnl, s_time = _simulate_trade_path(
                        close_prices, high_prices, low_prices,
                        row_index, lookahead, tp_pct, sl_pct, False, entry_short,
                    )
                    s_net = s_pnl - cost_pct
                    if s_code == 0 and s_net > best_net_return:
                        best_label, best_tp, best_sl = 2, tp_pct, sl_pct
                        best_net_return, best_time, best_magnitude = s_net, float(s_time), -tp_pct

        if best_net_return > 0.0:
            labels[row_index] = best_label
            label_tp_pct[row_index] = best_tp
            label_sl_pct[row_index] = best_sl
            label_expected_return_pct[row_index] = best_net_return
            label_time_to_target[row_index] = best_time
            label_magnitude[row_index] = best_magnitude

    return labels, label_tp_pct, label_sl_pct, label_expected_return_pct, label_time_to_target, label_magnitude


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OracleLabeler:
    """
    Generates ground-truth trading labels by simulating future price action.

    Design rationale:
      - ATR-based TP/SL: volatility-adaptive — small targets in quiet markets,
        larger in volatile ones.  This makes the strategy auto-scaling.
      - Tighter multipliers (0.8-1.5×ATR TP, 0.4-0.75×ATR SL): smaller targets
        are hit more often within the shorter lookahead window, producing more
        LONG/SHORT labels and fewer NEUTRAL → better class balance → better model.
      - Oracle sees the future — it is NOT used at inference time.  Its sole
        purpose is to create unambiguous training targets.
    """

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate df with oracle trading labels.

        Adds columns:
          direction_label       int   0=LONG, 1=NEUTRAL, 2=SHORT
          take_profit_pct       float optimal TP as % of entry
          stop_loss_pct         float optimal SL as % of entry
          expected_return_pct   float net return after costs
          time_to_target        float bars until TP was hit (or LOOKAHEAD)
          magnitude_label       float signed move magnitude (positive=long, negative=short)
        """
        training = cfg.training
        lookahead = training.LOOKAHEAD_BARS

        atr_pct = _causal_atr_pct(df, training.ATR_LENGTH)

        cost_frac = (
            cfg.testing.ROUND_TRIP_FEE_PCT / 100.0
            + 2.0 * cfg.testing.SLIPPAGE_PCT / 100.0
        )

        labels, tp_pct, sl_pct, exp_ret, time_to_target, magnitude = _compute_oracle_labels_jit(
            len(df),
            lookahead,
            df["Close"].values,
            df["High"].values,
            df["Low"].values,
            atr_pct,
            np.array(training.TP_ATR_MULTIPLIERS, dtype=np.float64),
            np.array(training.SL_ATR_MULTIPLIERS, dtype=np.float64),
            cost_frac,
            cfg.testing.SLIPPAGE_PCT / 100.0,
            training.ORACLE_MIN_RR,
            training.MIN_ATR_TARGET_PCT,
            training.MAX_ATR_TARGET_PCT,
            training.MIN_ATR_STOP_PCT,
            training.MAX_ATR_STOP_PCT,
            df["sma_50"].values if "sma_50" in df.columns else df["Close"].rolling(50).mean().fillna(df["Close"]).values,
            (df["atr"] / df["atr"].rolling(100).mean()).fillna(1.0).values,
        )

        df["direction_label"] = labels
        df["take_profit_pct"] = tp_pct
        df["stop_loss_pct"] = sl_pct
        df["expected_return_pct"] = exp_ret
        df["time_to_target"] = time_to_target
        df["magnitude_label"] = magnitude
        return df
