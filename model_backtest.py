# -*- coding: utf-8 -*-
"""
Real-world backtest of the trained direction model.

This is the honest, out-of-sample test: the model is evaluated ONLY on the
chronological test window — bars it never saw during training. We walk those
bars one at a time, exactly like live trading would:

  1. Ask the model for a direction + confidence on the current (closed) bar.
  2. If confidence >= THRESHOLD and we are flat, enter at the NEXT bar's open
     (no look-ahead — we act on the next bar, not the signal bar).
  3. Set an ATR-based take-profit / stop-loss (same multipliers the labeler
     uses, from config.py) and walk forward bar-by-bar until TP, SL, or a
     LOOKAHEAD_BARS timeout — using realistic intrabar High/Low touches.
  4. Apply round-trip fees + slippage to every trade.
  5. One position at a time (no overlap), starting from $100.

Reported against a buy-&-hold benchmark over the same window so you can see
whether the model actually adds anything.

Usage
-----
    uv run python train_model.py      # first, to train + save the model
    uv run python model_backtest.py
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from config import cfg
from set_label import LONG, NEUTRAL, SHORT
from train_model import MODEL_PATH, feature_matrix, load_valid, split_bounds

# ── Config ────────────────────────────────────────────────────────────────────
# Tunable knobs live in config.py (cfg.ml_backtest). Paths stay here.
INITIAL_CAPITAL: float = cfg.ml_backtest.INITIAL_CAPITAL
CONFIDENCE_THRESHOLD: float = cfg.ml_backtest.CONFIDENCE_THRESHOLD
TRADE_LOG_PATH: str = "data/model_trade_log.csv"

console = Console()


# ── Trade simulation ──────────────────────────────────────────────────────────

def _clip(value: float, lo: float, hi: float) -> float:
    return max(min(value, hi), lo)


def _simulate_trade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_index: int,
    entry_price: float,
    side: int,
    lookahead: int,
    tp_pct: float,
    sl_pct: float,
) -> tuple[str, float, int]:
    """Walk forward from the entry bar until TP/SL/timeout. Returns
    (reason, gross_return_pct, bars_held)."""
    if side == LONG:
        tp_price = entry_price * (1.0 + tp_pct / 100.0)
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
    else:
        tp_price = entry_price * (1.0 - tp_pct / 100.0)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)

    last = min(entry_index + lookahead, len(close) - 1)
    for step in range(1, lookahead + 1):
        bar = entry_index + step
        if bar >= len(close):
            break
        if side == LONG:
            if low[bar] <= sl_price:
                return "SL", -sl_pct, step
            if high[bar] >= tp_price:
                return "TP", tp_pct, step
        else:
            if high[bar] >= sl_price:
                return "SL", -sl_pct, step
            if low[bar] <= tp_price:
                return "TP", tp_pct, step

    exit_price = close[last]
    if side == LONG:
        gross = (exit_price - entry_price) / entry_price * 100.0
    else:
        gross = (entry_price - exit_price) / entry_price * 100.0
    return "TIMEOUT", gross, max(last - entry_index, 1)


def _atr_pct(df: pd.DataFrame) -> np.ndarray:
    if "natr" in df.columns:
        s = df["natr"]
    elif "atr" in df.columns:
        s = df["atr"] / df["Close"] * 100.0
    else:
        raise ValueError("Need 'natr' or 'atr' column to size TP/SL.")
    return s.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_model_backtest(
    df_test: pd.DataFrame,
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    proba_neutral: np.ndarray,
    threshold: float,
    capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, list[float]]:
    training = cfg.training
    lookahead = int(training.LOOKAHEAD_BARS)
    cost_pct = cfg.testing.ROUND_TRIP_FEE_PCT + 2.0 * cfg.testing.SLIPPAGE_PCT

    open_ = df_test["Open"].to_numpy(dtype=np.float64)
    high = df_test["High"].to_numpy(dtype=np.float64)
    low = df_test["Low"].to_numpy(dtype=np.float64)
    close = df_test["Close"].to_numpy(dtype=np.float64)
    atr = _atr_pct(df_test)
    index = df_test.index

    equity = float(capital)
    equity_curve: list[float] = [equity]
    records: list[dict] = []
    busy_until = -1
    n = len(df_test)

    for i in range(n):
        flat = i > busy_until
        # Decide the winning class across LONG / NEUTRAL / SHORT. We only enter
        # when a TRADE class (LONG/SHORT) wins AND clears the threshold — if
        # NEUTRAL wins (or nothing is confident enough) we stay flat. This is
        # the "don't trade" gate that the binary model lacked.
        side = LONG if proba_long[i] >= proba_short[i] else SHORT
        trade_conf = max(proba_long[i], proba_short[i])
        wins_over_neutral = trade_conf >= proba_neutral[i]
        conf = trade_conf
        entry_bar = i + 1  # act on the NEXT bar's open — no look-ahead

        if flat and wins_over_neutral and conf >= threshold and entry_bar < n and atr[i] > 0.0:
            entry_price = open_[entry_bar]
            tp_pct = _clip(atr[i] * float(training.FIXED_TP_ATR_MULTIPLIER),
                           float(training.MIN_ATR_TARGET_PCT), float(training.MAX_ATR_TARGET_PCT))
            sl_pct = _clip(atr[i] * float(training.FIXED_SL_ATR_MULTIPLIER),
                           float(training.MIN_ATR_STOP_PCT), float(training.MAX_ATR_STOP_PCT))
            reason, gross, bars = _simulate_trade(
                high, low, close, entry_bar, entry_price, side, lookahead, tp_pct, sl_pct
            )
            net = gross - cost_pct
            pnl = equity * net / 100.0
            equity += pnl
            busy_until = entry_bar + bars

            records.append({
                "timestamp": index[i],
                "direction": "BUY" if side == LONG else "SELL",
                "confidence": float(conf),
                "entry_price": float(entry_price),
                "tp_pct": float(tp_pct),
                "sl_pct": float(sl_pct),
                "exit_reason": reason,
                "gross_return_pct": float(gross),
                "net_return_pct": float(net),
                "pnl_dollar": float(pnl),
                "bars_held": int(bars),
                "equity_after": float(equity),
            })

        equity_curve.append(equity)

    return pd.DataFrame(records), equity_curve


# ── Stats ─────────────────────────────────────────────────────────────────────

def summarize(trades: pd.DataFrame, equity_curve: list[float], df_test: pd.DataFrame,
              threshold: float) -> dict:
    eq = np.asarray(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.where(peak > 0, peak, 1.0) - 1.0) * 100.0

    n = len(trades)
    wins = int((trades["net_return_pct"] > 0).sum()) if n else 0
    gp = float(trades.loc[trades["net_return_pct"] > 0, "pnl_dollar"].sum()) if n else 0.0
    gl = float(trades.loc[trades["net_return_pct"] < 0, "pnl_dollar"].sum()) if n else 0.0

    bh_start = float(df_test["Close"].iloc[0])
    bh_end = float(df_test["Close"].iloc[-1])
    bh_ret = (bh_end / bh_start - 1.0) * 100.0 if bh_start else 0.0

    return {
        "threshold": threshold,
        "start": eq[0],
        "end": float(eq[-1]),
        "total_ret_pct": (float(eq[-1]) / eq[0] - 1.0) * 100.0,
        "buy_hold_ret_pct": bh_ret,
        "n_trades": n,
        "n_buy": int((trades["direction"] == "BUY").sum()) if n else 0,
        "n_sell": int((trades["direction"] == "SELL").sum()) if n else 0,
        "win_rate_pct": (wins / n * 100.0) if n else 0.0,
        "profit_factor": (gp / abs(gl)) if gl != 0 else float("inf"),
        "max_dd_pct": float(dd.min()),
        "avg_ret_pct": float(trades["net_return_pct"].mean()) if n else 0.0,
        "avg_conf": float(trades["confidence"].mean()) if n else 0.0,
        "tp_hits": int((trades["exit_reason"] == "TP").sum()) if n else 0,
        "sl_hits": int((trades["exit_reason"] == "SL").sum()) if n else 0,
        "timeout_hits": int((trades["exit_reason"] == "TIMEOUT").sum()) if n else 0,
        "avg_bars": float(trades["bars_held"].mean()) if n else 0.0,
        "test_bars": len(df_test),
    }


def print_summary(s: dict) -> None:
    t = Table(
        title=("[bold cyan]Model Backtest[/bold cyan]  —  $100, out-of-sample test window  "
               f"(conf ≥ {s['threshold']:.2f})"),
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Metric", style="bold", min_width=26)
    t.add_column("Value", justify="right", min_width=18)

    t.add_row("Test window bars", str(s["test_bars"]))
    t.add_row("Start capital", f"${s['start']:.2f}")
    t.add_row("End equity", f"${s['end']:.2f}")
    col = "green" if s["total_ret_pct"] >= 0 else "red"
    t.add_row("Strategy return", f"[{col}]{s['total_ret_pct']:+.2f}%[/{col}]")
    bcol = "green" if s["buy_hold_ret_pct"] >= 0 else "red"
    t.add_row("Buy & Hold return", f"[{bcol}]{s['buy_hold_ret_pct']:+.2f}%[/{bcol}]")
    edge = s["total_ret_pct"] - s["buy_hold_ret_pct"]
    ecol = "green" if edge >= 0 else "red"
    t.add_row("Edge vs Buy & Hold", f"[{ecol}]{edge:+.2f}%[/{ecol}]")
    t.add_section()
    t.add_row("Total trades", str(s["n_trades"]))
    t.add_row("  BUY / SELL", f"{s['n_buy']} / {s['n_sell']}")
    t.add_row("Win rate", f"{s['win_rate_pct']:.1f}%")
    pf = s["profit_factor"]
    t.add_row("Profit factor", f"{pf:.3f}" if pf != float("inf") else "∞")
    t.add_row("Max drawdown", f"{s['max_dd_pct']:.2f}%")
    t.add_row("Avg return / trade", f"{s['avg_ret_pct']:.3f}%")
    t.add_row("Avg confidence", f"{s['avg_conf']:.3f}")
    t.add_section()
    t.add_row("TP / SL / Timeout", f"{s['tp_hits']} / {s['sl_hits']} / {s['timeout_hits']}")
    t.add_row("Avg bars held", f"{s['avg_bars']:.1f}")
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Hudu — Real-World Model Backtest[/bold blue]")

    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]Model not found at {MODEL_PATH}. Run `uv run python train_model.py` first.[/red]")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    classes = list(bundle["classes"])

    # Rebuild the SAME chronological test window the model was held out from.
    df = load_valid()
    _, val_end = split_bounds(len(df))
    df_test = df.iloc[val_end:].copy()
    console.print(f"\n[bold]Test window[/bold] {df_test.index[0]} → {df_test.index[-1]}  "
                  f"({len(df_test)} bars)")

    X_test = feature_matrix(df_test, features)
    proba = model.predict_proba(X_test)
    col = {c: j for j, c in enumerate(classes)}
    zeros = np.zeros(len(df_test))
    proba_long = proba[:, col[LONG]] if LONG in col else zeros
    proba_short = proba[:, col[SHORT]] if SHORT in col else zeros
    # In binary models there is no NEUTRAL column → 0, so the trade class always
    # wins the NEUTRAL comparison and only the threshold gates entries.
    proba_neutral = proba[:, col[NEUTRAL]] if NEUTRAL in col else zeros

    with console.status("Running model backtest..."):
        trades, equity_curve = run_model_backtest(
            df_test, proba_long, proba_short, proba_neutral, CONFIDENCE_THRESHOLD
        )

    stats = summarize(trades, equity_curve, df_test, CONFIDENCE_THRESHOLD)
    print_summary(stats)

    if not trades.empty:
        trades.to_csv(TRADE_LOG_PATH, index=False)
        console.print(f"  [dim]Trade log saved → {TRADE_LOG_PATH}[/dim]")
    else:
        console.print("  [yellow]No trades fired — try lowering CONFIDENCE_THRESHOLD.[/yellow]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
