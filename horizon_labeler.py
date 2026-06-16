# -*- coding: utf-8 -*-
"""
Horizon labeler: augments a raw OHLCV DataFrame with forward-simulation targets.

New columns added by generate():
  mfe_up_pct             – max favorable excursion (up) from next-bar open, in %
  mae_down_pct           – max adverse excursion (down) from next-bar open, in %
  bars_to_peak           – 1-based bar index of the highest high in the lookahead
  horizon_direction_label – LONG=0 / NEUTRAL=1 / SHORT=2 via fixed-ATR triple-barrier
  horizon_label_valid    – True for bars that have a full H-bar lookahead window

Design constraints:
  - ATR computation is causal (no look-ahead into the future).
  - Future OHLC is accessed ONLY inside the forward-simulation windows — it never
    appears in the feature matrix.
  - Entry assumed at the NEXT bar's Open (bar t+1 after signal bar t).
  - SL hit takes priority when both TP and SL touch on the same bar.
  - Fully vectorised with NumPy — no Python-level bar loop.

Usage:
    from horizon_labeler import HorizonLabeler

    labeler = HorizonLabeler()
    df_raw   = pd.read_csv("data/labeled_BTCUSD_15m.csv", index_col=0, parse_dates=True)
    df_augmented = labeler.generate(df_raw)          # adds 5 new columns
    df_valid = df_augmented[df_augmented["horizon_label_valid"]].copy()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import cfg

LONG    = 0
NEUTRAL = 1
SHORT   = 2

# Hard cap on MFE/MAE so regression targets stay in a bounded range.
MAX_MFE_PCT: float = 3.0


class HorizonLabeler:
    def __init__(
        self,
        lookahead_bars: int | None = None,
        tp_atr_multiplier: float = 1.5,
        sl_atr_multiplier: float = 0.75,
        atr_length: int | None = None,
        min_target_pct: float = 0.10,
        max_mfe_pct: float = MAX_MFE_PCT,
    ) -> None:
        self.lookahead_bars = lookahead_bars or cfg.training.LOOKAHEAD_BARS
        self.tp_atr_mult    = tp_atr_multiplier
        self.sl_atr_mult    = sl_atr_multiplier
        self.atr_length     = atr_length or cfg.training.ATR_LENGTH
        self.min_target_pct = min_target_pct
        self.max_mfe_pct    = max_mfe_pct

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _causal_atr_pct(self, df: pd.DataFrame) -> np.ndarray:
        """Wilder ATR as % of Close — causal, no future look-ahead."""
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(
            alpha=1.0 / self.atr_length,
            adjust=False,
            min_periods=self.atr_length,
        ).mean()
        atr_pct = (atr / df["Close"]) * 100.0
        return (
            atr_pct.replace([np.inf, -np.inf], np.nan)
                   .ffill()
                   .fillna(0.0)
                   .to_numpy(dtype=np.float64)
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add horizon label columns to a FULL (unfiltered) OHLCV DataFrame.

        Pass the raw CSV before any label_valid filtering so the labeler can
        reach into the last H bars as lookahead context.

        Parameters
        ----------
        df : DataFrame with Open, High, Low, Close columns (and any extras).

        Returns
        -------
        A copy of df with five appended columns.
        """
        out  = df.copy()
        n    = len(out)
        H    = self.lookahead_bars

        high    = out["High"].to_numpy(dtype=np.float64)
        low     = out["Low"].to_numpy(dtype=np.float64)
        opens   = out["Open"].to_numpy(dtype=np.float64)
        atr_pct = self._causal_atr_pct(out)

        # ── Vectorised MFE / MAE ─────────────────────────────────────────────
        # Pad the arrays so we can always extract a window of length H.
        pad    = np.full(H, np.nan, dtype=np.float64)
        ph     = np.concatenate([high, pad])    # (n+H,)
        pl     = np.concatenate([low,  pad])    # (n+H,)

        # wins_high[i] = high[i+1 : i+1+H]  (the H bars AFTER signal bar i)
        # sliding_window_view(ph, H) has shape (n+1, H); drop the first row.
        wins_high = np.lib.stride_tricks.sliding_window_view(ph, H)[1 : n + 1]  # (n, H)
        wins_low  = np.lib.stride_tricks.sliding_window_view(pl, H)[1 : n + 1]  # (n, H)

        # Entry price = Open of bar i+1  (last bar has no next bar → NaN)
        entry_prices          = np.empty(n, dtype=np.float64)
        entry_prices[: n - 1] = opens[1:]
        entry_prices[n - 1]   = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            max_high = np.nanmax(wins_high, axis=1)   # (n,)
            min_low  = np.nanmin(wins_low,  axis=1)   # (n,)

            mfe_raw = np.where(
                entry_prices > 0,
                (max_high - entry_prices) / entry_prices * 100.0,
                0.0,
            )
            mae_raw = np.where(
                entry_prices > 0,
                (entry_prices - min_low) / entry_prices * 100.0,
                0.0,
            )

        mfe_up_pct   = np.clip(np.nan_to_num(mfe_raw, nan=0.0), 0.0, self.max_mfe_pct).astype(np.float32)
        mae_down_pct = np.clip(np.nan_to_num(mae_raw, nan=0.0), 0.0, self.max_mfe_pct).astype(np.float32)

        # Argmax of the highest high (1-based bar index within the lookahead).
        # Replace NaN pads with -inf so they're never chosen as the argmax.
        wins_filled  = np.where(np.isnan(wins_high), -np.inf, wins_high)
        bars_to_peak = (np.argmax(wins_filled, axis=1) + 1).astype(np.float32)

        # ── Vectorised triple-barrier direction label ─────────────────────────
        # tp_pct / sl_pct per bar from causal ATR (floored at min_target_pct).
        tp_pct = np.maximum(atr_pct * self.tp_atr_mult, self.min_target_pct)   # (n,)
        sl_pct = np.maximum(atr_pct * self.sl_atr_mult, self.min_target_pct * 0.5)

        ep = entry_prices[:, None]   # (n, 1) for broadcasting against (n, H)

        # Price levels for the four barrier lines:
        #   TP_short < SL_long < entry < SL_short < TP_long
        tp_long_price  = ep * (1.0 + tp_pct[:, None] / 100.0)
        sl_long_price  = ep * (1.0 - sl_pct[:, None] / 100.0)
        tp_short_price = ep * (1.0 - tp_pct[:, None] / 100.0)
        sl_short_price = ep * (1.0 + sl_pct[:, None] / 100.0)

        with np.errstate(invalid="ignore"):
            long_tp_hit  = wins_high >= tp_long_price   # (n, H) bool
            long_sl_hit  = wins_low  <= sl_long_price
            short_tp_hit = wins_low  <= tp_short_price
            short_sl_hit = wins_high >= sl_short_price

        # First bar index where each condition fires; H if it never fires.
        def _first_hit(mask: np.ndarray) -> np.ndarray:
            """First bar index (0-based within window) that is True; H if never."""
            any_hit = mask.any(axis=1)
            idx     = np.argmax(mask, axis=1)
            return np.where(any_hit, idx, H).astype(np.int32)

        ltp_bar = _first_hit(long_tp_hit)
        lsl_bar = _first_hit(long_sl_hit)
        stp_bar = _first_hit(short_tp_hit)
        ssl_bar = _first_hit(short_sl_hit)

        # LONG: long TP hits before long SL, and before (or tied with) short TP.
        # SHORT: short TP hits before short SL, and strictly before long TP.
        long_wins  = (ltp_bar < lsl_bar) & (ltp_bar <= stp_bar)
        short_wins = (stp_bar < ssl_bar) & (stp_bar < ltp_bar)

        direction = np.full(n, NEUTRAL, dtype=np.int64)
        direction[long_wins]  = LONG
        direction[short_wins] = SHORT

        # ── Valid flag ────────────────────────────────────────────────────────
        # Bar i is valid when bars i+1 … i+H all exist in the original data.
        label_valid = np.zeros(n, dtype=bool)
        if n > H + 1:
            label_valid[: n - H - 1] = True

        # ── Direction-aware mfe_pct ───────────────────────────────────────────
        # For LONG:    max(high[t+1..t+H] - entry) / entry * 100  (upward)
        # For SHORT:   max(entry - low[t+1..t+H]) / entry * 100   (downward)
        # For NEUTRAL: max of both directions (potential)
        mfe_pct = np.where(
            direction == LONG,  mfe_up_pct,
            np.where(direction == SHORT, mae_down_pct,
                     np.maximum(mfe_up_pct, mae_down_pct))
        ).astype(np.float32)

        out["mfe_pct"]                  = mfe_pct
        out["mfe_up_pct"]               = mfe_up_pct    # kept for diagnostics
        out["mae_down_pct"]             = mae_down_pct  # kept for diagnostics
        out["bars_to_peak"]             = bars_to_peak
        out["horizon_direction_label"]  = direction
        out["horizon_label_valid"]      = label_valid

        return out
