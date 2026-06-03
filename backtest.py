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
      3. ML feature guide — which columns to use vs exclude

Key insight:
  separation_score = |BUY% - SELL%| for a condition.
  High score → condition separates BUY from SELL → useful for ML.
  Low score  → condition fires equally on both sides → skip for ML.

  CAUTION: price_above_sma20 has a perfect score of 100 but ONLY because
  the labeling filter forces all BUY labels to have price > SMA20. It is a
  label-definition artifact, NOT a market signal. Treat it as a hard regime
  rule, not a learnable feature.

Usage:
    uv run python backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

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

# Conditions that are artifacts of the SMA labeling filter.
# Their perfect discrimination score is tautological — the label rule says
# "only LONG when price > SMA20", so price_above_sma20 is always True for
# BUY labels. DO NOT use as a learnable feature.
SMA_FILTER_ARTIFACTS: frozenset[str] = frozenset({"price_above_sma20"})

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


# ── Per-condition breakdown ───────────────────────────────────────────────────

def condition_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """
    For every indicator condition, compute:
      - How many BUY / SELL trades fired while condition was True
      - Total dollar P&L
      - Win rate
      - Separation score = |BUY% - SELL%|
        High score → condition discriminates direction → USE for ML
        Low score  → fires equally on both sides → SKIP for ML
    """
    if trades.empty:
        return pd.DataFrame()

    all_conds = [
        c for c in list(MA_STATUS_COLUMNS) + list(TECHNICAL_CONDITION_COLUMNS)
        if c in trades.columns
    ]
    buy_df  = trades[trades["direction"] == LONG]
    sell_df = trades[trades["direction"] == SHORT]
    n_buy   = len(buy_df)
    n_sell  = len(sell_df)

    rows: list[dict] = []
    for cond in all_conds:
        buy_sub  = buy_df[buy_df[cond] == True]
        sell_sub = sell_df[sell_df[cond] == True]

        buy_pct  = len(buy_sub)  / n_buy  * 100.0 if n_buy  else 0.0
        sell_pct = len(sell_sub) / n_sell * 100.0 if n_sell else 0.0
        sep      = abs(buy_pct - sell_pct)

        for dir_val, dir_name, subset, n_dir in (
            (LONG,  "BUY",  buy_sub,  n_buy),
            (SHORT, "SELL", sell_sub, n_sell),
        ):
            n   = len(subset)
            pnl = float(subset["pnl_dollar"].sum())              if n else 0.0
            wr  = float((subset["pnl_dollar"] > 0).mean() * 100) if n else 0.0
            avg = float(subset["net_return_pct"].mean())          if n else 0.0
            pct = float(n / n_dir * 100.0)                        if n_dir else 0.0

            rows.append({
                "condition":        cond,
                "direction":        dir_name,
                "trades":           n,
                "pct_of_direction": pct,
                "pnl_dollar":       pnl,
                "win_rate_pct":     wr,
                "avg_return_pct":   avg,
                "separation_score": sep,
                "is_artifact":      cond in SMA_FILTER_ARTIFACTS,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("separation_score", ascending=False)
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


def _ml_note(row: "pd.Series") -> str:
    if bool(row["is_artifact"]):
        return "[dim]⚠ Label filter artifact — tautological[/dim]"
    sep = float(row["separation_score"])
    direction = row["direction"]
    if sep >= 60:
        bias = "BUY" if direction == "BUY" else "SELL"
        return f"[green]Strong {bias} discriminator → USE for ML[/green]"
    if sep >= 30:
        return "[yellow]Moderate signal → useful context feature[/yellow]"
    return "[red]Weak — both directions similar → SKIP[/red]"


def print_condition_breakdown(bd: pd.DataFrame) -> None:
    if bd.empty:
        return

    t = Table(
        title="[bold cyan]Per-Condition Performance[/bold cyan]  ($ weighted, sorted by separation score)",
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Condition",   style="bold", no_wrap=True, min_width=32)
    t.add_column("Dir",         no_wrap=True, min_width=5)
    t.add_column("Trades",      justify="right", min_width=7)
    t.add_column("% of Dir",    justify="right", min_width=9)
    t.add_column("P&L ($)",     justify="right", min_width=10)
    t.add_column("Win %",       justify="right", min_width=7)
    t.add_column("Avg Ret %",   justify="right", min_width=9)
    t.add_column("Sep Score",   justify="right", min_width=9)
    t.add_column("ML Note",     min_width=42)

    prev_cond: str | None = None
    for _, row in bd.iterrows():
        cond = row["condition"]
        if prev_cond is not None and cond != prev_cond:
            t.add_section()
        prev_cond = cond

        sep     = float(row["separation_score"])
        sc      = _sep_color(sep)
        pnl     = float(row["pnl_dollar"])
        pnl_str = (
            f"[green]${pnl:+.2f}[/green]"
            if pnl >= 0 else
            f"[red]${pnl:+.2f}[/red]"
        )
        dir_col = "green" if row["direction"] == "BUY" else "red"
        label   = cond + (" [dim]⚠[/dim]" if row["is_artifact"] else "")

        t.add_row(
            label,
            f"[{dir_col}]{row['direction']}[/{dir_col}]",
            str(int(row["trades"])),
            f"{row['pct_of_direction']:.1f}%",
            pnl_str,
            f"{row['win_rate_pct']:.1f}%",
            f"{row['avg_return_pct']:.3f}%",
            f"[{sc}]{sep:.1f}[/{sc}]",
            _ml_note(row),
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
        bd = condition_breakdown(trades)

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
