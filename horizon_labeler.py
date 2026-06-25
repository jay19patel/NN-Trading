# -*- coding: utf-8 -*-
"""
Horizon labeler: next-N-candle max return targets.

Columns added by generate():
  upside_max_return   – max(High[i+1..i+N] - Close[i]) / Close[i] * 100  (always ≥ 0)
  downside_max_return – max(Close[i] - Low[i+1..i+N])  / Close[i] * 100  (always ≥ 0)
  max_return          – signed: +upside if upside ≥ downside, else -downside
  horizon_label_valid – True for bars that have a full N-bar lookahead window

Example:
  Close = 100, max High in next 20 bars = 150 → upside_max_return = 50.0
  Close = 100, min Low  in next 20 bars = 50  → downside_max_return = 50.0
  If upside > downside: max_return = +50.0, else max_return = -50.0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LOOKAHEAD_BARS


class HorizonLabeler:
    def __init__(self, lookahead_bars: int = LOOKAHEAD_BARS) -> None:
        self.lookahead_bars = lookahead_bars

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        H = self.lookahead_bars

        close = out["Close"].to_numpy(dtype=np.float64)
        high  = out["High"].to_numpy(dtype=np.float64)
        low   = out["Low"].to_numpy(dtype=np.float64)

        # Pad so every bar can form a window of length H
        pad_h = np.full(H, np.nan, dtype=np.float64)
        pad_l = np.full(H, np.nan, dtype=np.float64)
        ph = np.concatenate([high, pad_h])
        pl = np.concatenate([low, pad_l])

        # wins_high[i] = high[i+1 .. i+H]  (H bars AFTER bar i)
        wins_high = np.lib.stride_tricks.sliding_window_view(ph, H)[1 : n + 1]  # (n, H)
        wins_low  = np.lib.stride_tricks.sliding_window_view(pl, H)[1 : n + 1]  # (n, H)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_high = np.nanmax(wins_high, axis=1)   # (n,)
            min_low  = np.nanmin(wins_low,  axis=1)   # (n,)

            ref = close
            upside_raw   = np.where(ref > 0, (max_high - ref) / ref * 100.0, 0.0)
            downside_raw = np.where(ref > 0, (ref - min_low)  / ref * 100.0, 0.0)

        upside   = np.clip(np.nan_to_num(upside_raw,   nan=0.0), 0.0, 100.0).astype(np.float32)
        downside = np.clip(np.nan_to_num(downside_raw, nan=0.0), 0.0, 100.0).astype(np.float32)

        # Signed: positive = upside dominant, negative = downside dominant
        max_return = np.where(upside >= downside, upside.astype(np.float64), -downside.astype(np.float64)).astype(np.float32)

        # Bar i is valid when bars i+1 … i+H all exist in the data
        label_valid = np.zeros(n, dtype=bool)
        if n > H:
            label_valid[: n - H] = True

        out["upside_max_return"]   = upside
        out["downside_max_return"] = downside
        out["max_return"]          = max_return
        out["horizon_label_valid"] = label_valid

        return out
