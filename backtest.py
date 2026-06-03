# -*- coding: utf-8 -*-
"""
Event-driven backtest on oracle-labeled 15m OHLCV data.

What this does:
  - Loads the labeled CSV produced by app.py
  - Walks every bar chronologically with $100 starting capital
  - One position at a time — no overlapping trades
  - For every trade fired, records which indicator conditions were True
  - Produces:
      1. Overall equity summary
      2. Per-condition P&L breakdown ($ weighted) sorted by discrimination power
      3. Direction guide — which conditions are stronger for BUY vs SELL

Key insight:
  separation_score = |BUY% - SELL%| for a condition.
  High score → condition separates BUY from SELL clearly.
  Low score  → condition fires similarly on both sides.

Usage:
    uv run python backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from config import cfg
from set_label import (
    LONG,
    MA_STATUS_COLUMNS,
    NEUTRAL,
    SHORT,
    TECHNICAL_CONDITION_COLUMNS,
)

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 100.0
CSV_PATH: str = "data/labeled_BTCUSD_15m.csv"

# Labeling no longer uses SMA/EMA/HMA as hard direction gates, so conditions
# are compared as normal signal-candle evidence.
LABEL_RULE_ARTIFACTS: frozenset[str] = frozenset()

console = Console()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = {"direction_label", "net_return_pct", "bars_to_exit"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Walk bars chronologically. One trade at a time (no overlap).

    P&L is applied at the signal bar using the pre-computed net_return_pct
    (already includes fees + slippage from the labeler). The actual trade
    entry is at next open — this simplification affects only the intrarade
    equity display, not the final result.

    Returns:
        trade_log : DataFrame — one row per executed trade with all conditions
        equity_curve : list[float] — equity after each bar (length = n+1)
    """
    all_cond_cols = [
        c for c in list(MA_STATUS_COLUMNS) + list(TECHNICAL_CONDITION_COLUMNS)
        if c in df.columns
    ]

    arr_label  = df["direction_label"].to_numpy(dtype=np.int32)
    arr_net    = df["net_return_pct"].to_numpy(dtype=np.float64)
    arr_bars   = df["bars_to_exit"].to_numpy(dtype=np.float64)
    arr_valid  = (
        df["label_valid"].to_numpy(dtype=bool)
        if "label_valid" in df.columns
        else np.ones(len(df), dtype=bool)
    )
    arr_reason = (
        df["exit_reason"].to_numpy()
        if "exit_reason" in df.columns
        else np.full(len(df), "NONE")
    )

    equity: float = capital
    equity_curve: list[float] = [capital]
    records: list[dict] = []
    busy_until: int = -1

    for i in range(len(df)):
        label     = int(arr_label[i])
        can_enter = (i > busy_until) and (label != NEUTRAL) and bool(arr_valid[i])

        if can_enter:
            net_pct    = float(arr_net[i])
            bars       = max(int(arr_bars[i]), 1)
            pnl        = equity * (net_pct / 100.0)
            equity    += pnl
            busy_until = i + bars

            rec: dict = {
                "bar_index":      i,
                "timestamp":      df.index[i],
                "direction":      label,
                "direction_name": "BUY" if label == LONG else "SELL",
                "net_return_pct": net_pct,
                "pnl_dollar":     pnl,
                "exit_reason":    str(arr_reason[i]),
                "bars_held":      bars,
                "equity_after":   equity,
            }
            for col in all_cond_cols:
                rec[col] = bool(df[col].iloc[i])
            records.append(rec)

        equity_curve.append(equity)

    return pd.DataFrame(records), equity_curve


# ── Per-condition strategy backtest ───────────────────────────────────────────

def _condition_atr_pct(df: pd.DataFrame) -> np.ndarray:
    if "natr" in df.columns:
        return df["natr"].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)
    if "atr" in df.columns:
        return (df["atr"] / df["Close"] * 100.0).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / cfg.training.ATR_LENGTH, adjust=False, min_periods=cfg.training.ATR_LENGTH).mean()
    return (atr / df["Close"] * 100.0).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)


def _clip_pct(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def _simulate_fixed_trade(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    row_index: int,
    lookahead: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    direction_name: str,
    entry_price: float,
) -> tuple[str, float, int]:
    if direction_name == "BUY":
        tp_price = entry_price * (1.0 + take_profit_pct / 100.0)
        sl_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    else:
        tp_price = entry_price * (1.0 - take_profit_pct / 100.0)
        sl_price = entry_price * (1.0 + stop_loss_pct / 100.0)

    max_exit_index = min(row_index + lookahead, len(close_prices) - 1)
    for step in range(1, lookahead + 1):
        bar_index = row_index + step
        if bar_index >= len(close_prices):
            break
        high = high_prices[bar_index]
        low = low_prices[bar_index]

        if direction_name == "BUY":
            hit_sl = low <= sl_price
            hit_tp = high >= tp_price
        else:
            hit_sl = high >= sl_price
            hit_tp = low <= tp_price

        if hit_sl:
            return "SL", -stop_loss_pct, step
        if hit_tp:
            return "TP", take_profit_pct, step

    exit_price = close_prices[max_exit_index]
    if direction_name == "BUY":
        gross_return_pct = (exit_price - entry_price) / entry_price * 100.0
    else:
        gross_return_pct = (entry_price - exit_price) / entry_price * 100.0
    return "TIMEOUT", gross_return_pct, max(lookahead, 1)


def _run_condition_side_backtest(
    df: pd.DataFrame,
    condition: str,
    direction_name: str,
    capital: float = INITIAL_CAPITAL,
) -> dict[str, float | int | str]:
    training = cfg.training
    lookahead = int(training.LOOKAHEAD_BARS)
    trade_cost_pct = cfg.testing.ROUND_TRIP_FEE_PCT + 2.0 * cfg.testing.SLIPPAGE_PCT
    slippage_fraction = cfg.testing.SLIPPAGE_PCT / 100.0

    open_prices = df["Open"].to_numpy(dtype=np.float64)
    close_prices = df["Close"].to_numpy(dtype=np.float64)
    high_prices = df["High"].to_numpy(dtype=np.float64)
    low_prices = df["Low"].to_numpy(dtype=np.float64)
    atr_pct = _condition_atr_pct(df)
    condition_values = df[condition].fillna(False).astype(bool).to_numpy()
    valid_values = (
        df["label_valid"].fillna(False).astype(bool).to_numpy()
        if "label_valid" in df.columns
        else np.ones(len(df), dtype=bool)
    )

    equity = float(capital)
    peak = equity
    max_dd = 0.0
    busy_until = -1
    trades = 0
    wins = 0
    tp_hits = 0
    sl_hits = 0
    timeout_hits = 0
    total_return_pct = 0.0
    total_bars = 0
    tp_values: list[float] = []
    sl_values: list[float] = []

    for row_index in range(len(df) - lookahead):
        if row_index <= busy_until or not valid_values[row_index] or not condition_values[row_index]:
            continue

        current_atr = float(atr_pct[row_index])
        if current_atr <= 0.0:
            continue

        if training.ENTRY_MODE == "next_open":
            entry_bar_index = row_index + 1
            if entry_bar_index >= len(df):
                continue
            base_entry = open_prices[entry_bar_index]
        else:
            base_entry = close_prices[row_index]

        tp_pct = _clip_pct(
            current_atr * float(training.FIXED_TP_ATR_MULTIPLIER),
            float(training.MIN_ATR_TARGET_PCT),
            float(training.MAX_ATR_TARGET_PCT),
        )
        sl_pct = _clip_pct(
            current_atr * float(training.FIXED_SL_ATR_MULTIPLIER),
            float(training.MIN_ATR_STOP_PCT),
            float(training.MAX_ATR_STOP_PCT),
        )
        entry_price = base_entry * (1.0 + slippage_fraction if direction_name == "BUY" else 1.0 - slippage_fraction)

        reason, gross_pct, bars = _simulate_fixed_trade(
            close_prices=close_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            row_index=row_index,
            lookahead=lookahead,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
            direction_name=direction_name,
            entry_price=entry_price,
        )
        net_pct = gross_pct - trade_cost_pct
        pnl = equity * net_pct / 100.0
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity / peak - 1.0) * 100.0)

        trades += 1
        wins += int(net_pct > 0.0)
        tp_hits += int(reason == "TP")
        sl_hits += int(reason == "SL")
        timeout_hits += int(reason == "TIMEOUT")
        total_return_pct += net_pct
        total_bars += int(bars)
        tp_values.append(tp_pct)
        sl_values.append(sl_pct)
        busy_until = row_index + max(int(bars), 1)

    total_ret_pct = (equity / capital - 1.0) * 100.0 if capital else 0.0
    return {
        "trades": trades,
        "end_capital": equity,
        "pnl_dollar": equity - capital,
        "total_return_pct": total_ret_pct,
        "win_rate_pct": wins / trades * 100.0 if trades else 0.0,
        "avg_net_return_pct": total_return_pct / trades if trades else 0.0,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "timeout_hits": timeout_hits,
        "max_dd_pct": max_dd,
        "avg_bars": total_bars / trades if trades else 0.0,
        "avg_tp_pct": float(np.mean(tp_values)) if tp_values else 0.0,
        "avg_sl_pct": float(np.mean(sl_values)) if sl_values else 0.0,
    }


def condition_breakdown(df: pd.DataFrame, capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """
    For every condition, run two independent strategies:
      - BUY whenever this condition is true
      - SELL whenever this condition is true

    This is intentionally separate from the oracle-label summary. It tests
    whether the condition itself is useful as an entry trigger.
    """
    all_conds = [
        c for c in list(MA_STATUS_COLUMNS) + list(TECHNICAL_CONDITION_COLUMNS)
        if c in df.columns
    ]
    if not all_conds:
        return pd.DataFrame()

    rows: list[dict] = []
    for cond in all_conds:
        buy = _run_condition_side_backtest(df, cond, "BUY", capital=capital)
        sell = _run_condition_side_backtest(df, cond, "SELL", capital=capital)
        buy_return = float(buy["total_return_pct"])
        sell_return = float(sell["total_return_pct"])
        edge = abs(buy_return - sell_return)
        if buy_return > sell_return:
            best_for = "BUY"
            best_return = buy_return
        elif sell_return > buy_return:
            best_for = "SELL"
            best_return = sell_return
        else:
            best_for = "NEUTRAL"
            best_return = buy_return

        coverage_pct = max(int(buy["trades"]), int(sell["trades"])) / max(len(df), 1) * 100.0
        confidence_score = edge * min(max(int(buy["trades"]), int(sell["trades"])) / 100.0, 1.0)

        rows.append({
            "condition": cond,
            "best_for": best_for,
            "edge": edge,
            "return_edge_pct": edge,
            "confidence_score": confidence_score,
            "coverage_pct": coverage_pct,
            "total_trades": max(int(buy["trades"]), int(sell["trades"])),
            "best_return_pct": best_return,
            "buy_trades": int(buy["trades"]),
            "buy_pnl_dollar": float(buy["pnl_dollar"]),
            "buy_return_pct": buy_return,
            "buy_win_rate_pct": float(buy["win_rate_pct"]),
            "buy_tp_hits": int(buy["tp_hits"]),
            "buy_sl_hits": int(buy["sl_hits"]),
            "buy_timeout_hits": int(buy["timeout_hits"]),
            "buy_max_dd_pct": float(buy["max_dd_pct"]),
            "sell_trades": int(sell["trades"]),
            "sell_pnl_dollar": float(sell["pnl_dollar"]),
            "sell_return_pct": sell_return,
            "sell_win_rate_pct": float(sell["win_rate_pct"]),
            "sell_tp_hits": int(sell["tp_hits"]),
            "sell_sl_hits": int(sell["sl_hits"]),
            "sell_timeout_hits": int(sell["timeout_hits"]),
            "sell_max_dd_pct": float(sell["max_dd_pct"]),
            "avg_tp_pct": (float(buy["avg_tp_pct"]) + float(sell["avg_tp_pct"])) / 2.0,
            "avg_sl_pct": (float(buy["avg_sl_pct"]) + float(sell["avg_sl_pct"])) / 2.0,
            "is_artifact": cond in LABEL_RULE_ARTIFACTS,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["confidence_score", "edge", "total_trades"], ascending=False)
        .reset_index(drop=True)
    )


# ── Stats helpers ─────────────────────────────────────────────────────────────

def equity_stats(trades: pd.DataFrame, equity_curve: list[float]) -> dict:
    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq / np.where(peak > 0, peak, 1.0) - 1.0) * 100.0

    n = len(trades)
    wins = int((trades["pnl_dollar"] > 0).sum()) if n else 0
    gp   = float(trades[trades["pnl_dollar"] > 0]["pnl_dollar"].sum()) if n else 0.0
    gl   = float(trades[trades["pnl_dollar"] < 0]["pnl_dollar"].sum()) if n else 0.0

    n_buy  = int((trades["direction"] == LONG).sum())  if n else 0
    n_sell = int((trades["direction"] == SHORT).sum()) if n else 0

    buy_wins  = int((trades[trades["direction"] == LONG]["pnl_dollar"] > 0).sum())  if n_buy  else 0
    sell_wins = int((trades[trades["direction"] == SHORT]["pnl_dollar"] > 0).sum()) if n_sell else 0

    tp_hits      = int((trades["exit_reason"] == "TP").sum())      if n else 0
    sl_hits      = int((trades["exit_reason"] == "SL").sum())      if n else 0
    timeout_hits = int((trades["exit_reason"] == "TIMEOUT").sum()) if n else 0

    return {
        "start":           equity_curve[0],
        "end":             equity_curve[-1],
        "total_ret_pct":   (equity_curve[-1] / equity_curve[0] - 1.0) * 100.0,
        "n_trades":        n,
        "n_buy":           n_buy,
        "n_sell":          n_sell,
        "wins":            wins,
        "losses":          n - wins,
        "win_rate_pct":    wins / n * 100.0   if n else 0.0,
        "buy_win_rate":    buy_wins  / n_buy  * 100.0 if n_buy  else 0.0,
        "sell_win_rate":   sell_wins / n_sell * 100.0 if n_sell else 0.0,
        "profit_factor":   gp / abs(gl) if gl != 0 else float("inf"),
        "max_dd_pct":      float(dd.min()),
        "avg_ret_pct":     float(trades["net_return_pct"].mean()) if n else 0.0,
        "tp_hits":         tp_hits,
        "sl_hits":         sl_hits,
        "timeout_hits":    timeout_hits,
        "avg_bars_held":   float(trades["bars_held"].mean()) if n else 0.0,
    }


# ── Rich printers ─────────────────────────────────────────────────────────────

def print_backtest_summary(s: dict) -> None:
    t = Table(
        title="[bold cyan]Backtest Summary[/bold cyan]  —  $100 starting capital",
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Metric",  style="bold", min_width=28)
    t.add_column("Value",   justify="right", min_width=18)

    t.add_row("Start capital",    f"${s['start']:.2f}")
    t.add_row("End equity",       f"${s['end']:.2f}")
    col = "green" if s["total_ret_pct"] >= 0 else "red"
    t.add_row("Total return",     f"[{col}]{s['total_ret_pct']:+.2f}%[/{col}]")
    t.add_section()
    t.add_row("Total trades",     str(s["n_trades"]))
    t.add_row("  BUY  trades",    str(s["n_buy"]))
    t.add_row("  SELL trades",    str(s["n_sell"]))
    t.add_row("Wins / Losses",    f"{s['wins']} / {s['losses']}")
    t.add_row("Win rate (all)",   f"{s['win_rate_pct']:.1f}%")
    t.add_row("Win rate (BUY)",   f"{s['buy_win_rate']:.1f}%")
    t.add_row("Win rate (SELL)",  f"{s['sell_win_rate']:.1f}%")
    pf = s["profit_factor"]
    t.add_row("Profit factor",    f"{pf:.3f}" if pf != float("inf") else "∞")
    t.add_row("Max drawdown",     f"{s['max_dd_pct']:.2f}%")
    t.add_row("Avg return/trade", f"{s['avg_ret_pct']:.3f}%")
    t.add_section()
    t.add_row("TP exits",         str(s["tp_hits"]))
    t.add_row("SL exits",         str(s["sl_hits"]))
    t.add_row("Timeout exits",    str(s["timeout_hits"]))
    t.add_row("Avg bars held",    f"{s['avg_bars_held']:.1f}")

    console.print(t)


def _sep_color(sep: float) -> str:
    if sep >= 60:
        return "green"
    if sep >= 30:
        return "yellow"
    return "red"


def _direction_note(row: "pd.Series") -> str:
    if bool(row["is_artifact"]):
        return "[dim]Label rule artifact, ignore for direction choice[/dim]"
    sep = float(row["edge"])
    dominant = str(row["best_for"])
    buy_return = float(row["buy_return_pct"])
    sell_return = float(row["sell_return_pct"])
    lead = f"{dominant}: BUY {buy_return:+.1f}% | SELL {sell_return:+.1f}%"
    if sep >= 60:
        return f"[green]Best for {lead}[/green]"
    if sep >= 30:
        return f"[yellow]Moderate for {lead}[/yellow]"
    return f"[red]Weak compare: BUY {buy_return:+.1f}% | SELL {sell_return:+.1f}%[/red]"


def _best_for_cell(row: "pd.Series") -> str:
    if bool(row["is_artifact"]):
        return "[dim]ARTIFACT[/dim]"
    dominant = str(row["best_for"])
    if dominant == "BUY":
        color = "green"
    elif dominant == "SELL":
        color = "red"
    else:
        color = "yellow"
    return f"[{color}]{dominant}[/{color}]"


def print_condition_breakdown(bd: pd.DataFrame) -> None:
    if bd.empty:
        return
    training = cfg.training
    rr = float(training.FIXED_TP_ATR_MULTIPLIER) / max(float(training.FIXED_SL_ATR_MULTIPLIER), 1e-9)
    cost = cfg.testing.ROUND_TRIP_FEE_PCT + 2.0 * cfg.testing.SLIPPAGE_PCT

    t = Table(
        title=(
            "[bold cyan]Per-Condition Strategy Backtest[/bold cyan]  "
            f"capital=${INITIAL_CAPITAL:.0f}, TP={training.FIXED_TP_ATR_MULTIPLIER:g}xATR, "
            f"SL={training.FIXED_SL_ATR_MULTIPLIER:g}xATR, RR={rr:.2f}, "
            f"lookahead={training.LOOKAHEAD_BARS}, cost={cost:.2f}%"
        ),
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Condition",   style="bold", no_wrap=True, min_width=32)
    t.add_column("Best For",     justify="center", min_width=8)
    t.add_column("Score",        justify="right", min_width=7)
    t.add_column("Coverage",     justify="right", min_width=8)
    t.add_column("BUY Trades",   justify="right", min_width=10)
    t.add_column("BUY P&L",      justify="right", min_width=9)
    t.add_column("BUY Ret",      justify="right", min_width=8)
    t.add_column("BUY Win",      justify="right", min_width=7)
    t.add_column("BUY TP/SL/T",   justify="right", min_width=10)
    t.add_column("BUY DD",       justify="right", min_width=7)
    t.add_column("SELL Trades",  justify="right", min_width=11)
    t.add_column("SELL P&L",     justify="right", min_width=9)
    t.add_column("SELL Ret",     justify="right", min_width=8)
    t.add_column("SELL Win",     justify="right", min_width=8)
    t.add_column("SELL TP/SL/T",  justify="right", min_width=11)
    t.add_column("SELL DD",      justify="right", min_width=7)
    t.add_column("Edge",        justify="right", min_width=7)
    t.add_column("Compare",     min_width=28)

    for _, row in bd.iterrows():
        sep = float(row["edge"])
        sc = _sep_color(sep)
        label = row["condition"] + (" [dim]⚠[/dim]" if row["is_artifact"] else "")

        buy_pnl = float(row["buy_pnl_dollar"])
        sell_pnl = float(row["sell_pnl_dollar"])
        buy_pnl_str = f"[green]${buy_pnl:+.2f}[/green]" if buy_pnl >= 0 else f"[red]${buy_pnl:+.2f}[/red]"
        sell_pnl_str = f"[green]${sell_pnl:+.2f}[/green]" if sell_pnl >= 0 else f"[red]${sell_pnl:+.2f}[/red]"

        t.add_row(
            label,
            _best_for_cell(row),
            f"{row['confidence_score']:.1f}",
            f"{row['coverage_pct']:.1f}%",
            str(int(row["buy_trades"])),
            buy_pnl_str,
            f"{row['buy_return_pct']:+.1f}%",
            f"{row['buy_win_rate_pct']:.1f}%",
            f"{int(row['buy_tp_hits'])}/{int(row['buy_sl_hits'])}/{int(row['buy_timeout_hits'])}",
            f"{row['buy_max_dd_pct']:.1f}%",
            str(int(row["sell_trades"])),
            sell_pnl_str,
            f"{row['sell_return_pct']:+.1f}%",
            f"{row['sell_win_rate_pct']:.1f}%",
            f"{int(row['sell_tp_hits'])}/{int(row['sell_sl_hits'])}/{int(row['sell_timeout_hits'])}",
            f"{row['sell_max_dd_pct']:.1f}%",
            f"[{sc}]{sep:.1f}[/{sc}]",
            _direction_note(row),
        )

    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Hudu — $100 Backtest + Condition Breakdown[/bold blue]")

    console.print(f"\n[bold]Loading[/bold] {CSV_PATH}")
    df = _load(CSV_PATH)

    n_labeled = int((df["direction_label"] != NEUTRAL).sum())
    n_valid   = int(df.get("label_valid", pd.Series(True, index=df.index)).sum()) if "label_valid" in df.columns else len(df)
    console.print(f"  {len(df)} bars  |  {n_labeled} labeled trades  |  {n_valid} valid rows")

    # ── Backtest ──────────────────────────────────────────────────────────────
    with console.status("Running backtest..."):
        trades, equity_curve = run_backtest(df, capital=INITIAL_CAPITAL)

    stats = equity_stats(trades, equity_curve)
    print_backtest_summary(stats)

    # ── Per-condition breakdown ───────────────────────────────────────────────
    with console.status("Computing per-condition breakdown..."):
        bd = condition_breakdown(df, capital=INITIAL_CAPITAL)

    print_condition_breakdown(bd)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not trades.empty:
        trades.to_csv("data/trade_log.csv", index=False)
        console.print("  [dim]Trade log saved     → data/trade_log.csv[/dim]")

    if not bd.empty:
        bd.to_csv("data/condition_breakdown.csv", index=False)
        console.print("  [dim]Condition breakdown → data/condition_breakdown.csv[/dim]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
