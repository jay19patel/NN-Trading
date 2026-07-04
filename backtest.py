# -*- coding: utf-8 -*-
"""
Realistic backtest — trade the model like a real account.

Rules (sab config.py → BacktestConfig me):
  - Start capital $100, ek waqt me ek hi position, compounding.
  - Signal tab hi jab confidence >= threshold AUR predicted move >= minimum.
  - Entry agli candle ke OPEN par (no lookahead cheating).
  - Take-profit / stop-loss predicted magnitude se; timeout ke baad market exit.
  - Same candle me TP aur SL dono touch ho to SL mana jata hai (conservative).
  - Har trade par fees + slippage kat'ti hai (dono side).

Run:
    uv run python backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rich import box
from rich.console import Console
from rich.table import Table

from config import cfg
from evaluator import evaluate_model, print_evaluation, predict_batches
from model import DirectionMagnitudeModel
from trainer import build_dataset, make_windows, _device, MODEL_PATH

console = Console()

CHART_PATH = "models/backtest_analysis.png"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> tuple[DirectionMagnitudeModel, dict]:
    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = DirectionMagnitudeModel(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


# ── Simulation ────────────────────────────────────────────────────────────────

def run_backtest(
    df:        pd.DataFrame,   # df_valid — OHLC rows aligned with positions
    positions: np.ndarray,     # test bar indices into df (chronological)
    p_up:      np.ndarray,     # per test bar
    up_pred:   np.ndarray,
    dn_pred:   np.ndarray,
    y_up:      np.ndarray,     # actual labels (for direction-check reporting)
    y_dn:      np.ndarray,
) -> tuple[pd.DataFrame, list[float]]:
    bt = cfg.backtest
    open_  = df["Open"].to_numpy(dtype=np.float64)
    high   = df["High"].to_numpy(dtype=np.float64)
    low    = df["Low"].to_numpy(dtype=np.float64)
    close  = df["Close"].to_numpy(dtype=np.float64)
    times  = df.index

    cost_pct = 2.0 * (bt.FEE_PCT + bt.SLIPPAGE_PCT) / 100.0   # round-trip cost

    equity        = bt.INITIAL_CAPITAL
    equity_curve  = [equity]
    trades: list[dict] = []

    pos_by_bar = {int(t): j for j, t in enumerate(positions)}
    n_bars     = len(df)

    bar = int(positions[0])
    last_bar = int(positions[-1])

    while bar <= last_bar:
        j = pos_by_bar.get(bar)
        if j is None:
            bar += 1
            continue

        conf     = max(p_up[j], 1.0 - p_up[j])
        is_long  = p_up[j] >= 0.5
        pred_mag = up_pred[j] if is_long else dn_pred[j]

        # ── Signal filter ─────────────────────────────────────────────────────
        if conf < bt.CONFIDENCE_THRESHOLD or pred_mag < bt.MIN_PREDICTED_MOVE or bar + 1 >= n_bars:
            bar += 1
            continue

        # ── Open position at next candle open ─────────────────────────────────
        entry_bar   = bar + 1
        entry_price = open_[entry_bar]
        tp_pct      = pred_mag * bt.TP_FRACTION / 100.0
        sl_pct      = pred_mag * bt.SL_FRACTION / 100.0
        if is_long:
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)
        else:
            tp_price = entry_price * (1.0 - tp_pct)
            sl_price = entry_price * (1.0 + sl_pct)

        exit_bar, exit_price, exit_reason = None, None, None
        last_hold = min(entry_bar + bt.MAX_HOLD_BARS - 1, n_bars - 1)
        for k in range(entry_bar, last_hold + 1):
            if is_long:
                if low[k] <= sl_price:            # SL checked first — conservative
                    exit_bar, exit_price, exit_reason = k, sl_price, "SL"
                    break
                if high[k] >= tp_price:
                    exit_bar, exit_price, exit_reason = k, tp_price, "TP"
                    break
            else:
                if high[k] >= sl_price:
                    exit_bar, exit_price, exit_reason = k, sl_price, "SL"
                    break
                if low[k] <= tp_price:
                    exit_bar, exit_price, exit_reason = k, tp_price, "TP"
                    break
        if exit_bar is None:
            exit_bar, exit_price, exit_reason = last_hold, close[last_hold], "TIME"

        # ── PnL ───────────────────────────────────────────────────────────────
        raw_ret  = (exit_price - entry_price) / entry_price * (1.0 if is_long else -1.0)
        net_ret  = raw_ret - cost_pct
        notional = equity * bt.POSITION_FRACTION
        pnl_usd  = notional * net_ret
        equity  += pnl_usd
        equity_curve.append(equity)

        actual_up_dominant = y_up[j] >= y_dn[j]
        trades.append({
            "entry_time":  times[entry_bar],
            "exit_time":   times[exit_bar],
            "side":        "LONG" if is_long else "SHORT",
            "confidence":  conf,
            "pred_move":   pred_mag,
            "achieved":    (y_up[j] if is_long else y_dn[j]),
            "dir_correct": bool(is_long == actual_up_dominant),
            "exit_reason": exit_reason,
            "hold_bars":   exit_bar - entry_bar + 1,
            "raw_ret_pct": raw_ret * 100.0,
            "net_ret_pct": net_ret * 100.0,
            "pnl_usd":     pnl_usd,
            "equity":      equity,
        })

        bar = exit_bar + 1     # no new signal while position open

        if equity <= 1.0:      # account blown
            break

    return pd.DataFrame(trades), equity_curve


# ── Reporting ─────────────────────────────────────────────────────────────────

def _max_drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(((curve - peak) / peak).min())


def print_backtest_report(trades: pd.DataFrame, equity_curve: list[float],
                          df: pd.DataFrame, positions: np.ndarray) -> None:
    bt = cfg.backtest
    start_t, end_t = df.index[int(positions[0])], df.index[int(positions[-1])]
    days = max((end_t - start_t).total_seconds() / 86400.0, 1.0)

    close = df["Close"].to_numpy(dtype=np.float64)
    bh_ret = (close[int(positions[-1])] / close[int(positions[0])] - 1.0) * 100.0

    console.rule("[bold magenta]BACKTEST — $100 REAL-TRADING SIMULATION[/bold magenta]")
    console.print(
        f"\n  [bold]Period:[/bold] {start_t.strftime('%Y-%m-%d')} → {end_t.strftime('%Y-%m-%d')} "
        f"({days:.0f} days, test set — model ne ye data kabhi nahi dekha)\n"
        f"  [bold]Rules:[/bold] confidence ≥ {bt.CONFIDENCE_THRESHOLD:.2f}, "
        f"predicted move ≥ {bt.MIN_PREDICTED_MOVE:.2f}%, "
        f"TP = {bt.TP_FRACTION:.0%} × pred, SL = {bt.SL_FRACTION:.0%} × pred, "
        f"fees {bt.FEE_PCT}%/side + slippage {bt.SLIPPAGE_PCT}%/side\n"
    )

    if trades.empty:
        console.print("[red]  Koi trade trigger nahi hua — thresholds bahut strict hain.[/red]")
        return

    net   = trades["net_ret_pct"].to_numpy()
    wins  = trades[trades["pnl_usd"] > 0]
    losses = trades[trades["pnl_usd"] <= 0]
    gross_win  = wins["pnl_usd"].sum()
    gross_loss = abs(losses["pnl_usd"].sum())
    final_eq   = equity_curve[-1]
    curve      = np.asarray(equity_curve)

    # ── Capital summary ───────────────────────────────────────────────────────
    ret_pct = (final_eq / bt.INITIAL_CAPITAL - 1.0) * 100.0
    color   = "green" if ret_pct > 0 else "red"
    t = Table(title="[bold]CAPITAL — paisa grow hua ya nahi?[/bold]", box=box.HEAVY)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Start capital", f"${bt.INITIAL_CAPITAL:.2f}")
    t.add_row("Final capital", f"[{color}]${final_eq:.2f}[/{color}]")
    t.add_row("Total return", f"[{color}]{ret_pct:+.2f}%[/{color}]")
    t.add_row("Buy & hold (same period)", f"{bh_ret:+.2f}%")
    t.add_row("Max drawdown", f"{_max_drawdown(curve)*100:.2f}%")
    t.add_row("Trades", f"{len(trades):,}  ({len(trades)/days:.1f}/day)")
    console.print(t)

    # ── Trade stats ───────────────────────────────────────────────────────────
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    trade_sharpe = net.mean() / net.std() * np.sqrt(len(trades) / days * 365) if net.std() > 0 else 0.0
    t = Table(title="[bold]TRADE STATS[/bold]", box=box.ROUNDED)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Win rate", f"{len(wins)/len(trades)*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    t.add_row("Avg win / avg loss", f"{wins['net_ret_pct'].mean():+.2f}% / {losses['net_ret_pct'].mean():+.2f}%"
              if len(wins) and len(losses) else "—")
    t.add_row("Expectancy per trade", f"{net.mean():+.3f}%")
    t.add_row("Profit factor", f"{pf:.2f}")
    t.add_row("Sharpe (annualized, approx)", f"{trade_sharpe:.2f}")
    t.add_row("Avg hold", f"{trades['hold_bars'].mean():.1f} bars (×15m)")
    console.print(t)

    # ── Exit breakdown ────────────────────────────────────────────────────────
    t = Table(title="[bold]EXIT BREAKDOWN — trades kaise band hue[/bold]", box=box.ROUNDED)
    t.add_column("Exit", style="bold")
    t.add_column("Trades", justify="right")
    t.add_column("Win rate", justify="right")
    t.add_column("Avg net return", justify="right")
    for reason in ("TP", "SL", "TIME"):
        sub = trades[trades["exit_reason"] == reason]
        if sub.empty:
            t.add_row(reason, "0", "—", "—")
            continue
        t.add_row(reason, f"{len(sub):,}",
                  f"{(sub['pnl_usd'] > 0).mean()*100:.1f}%",
                  f"{sub['net_ret_pct'].mean():+.2f}%")
    console.print(t)

    # ── Direction accuracy on trades ─────────────────────────────────────────
    t = Table(title="[bold]DIRECTION ON TRADES — jo bola wahi hua?[/bold]", box=box.ROUNDED)
    t.add_column("Side", style="bold")
    t.add_column("Trades", justify="right")
    t.add_column("Direction sahi", justify="right")
    t.add_column("Avg predicted move", justify="right")
    t.add_column("Median achieved move", justify="right")
    for side in ("LONG", "SHORT"):
        sub = trades[trades["side"] == side]
        if sub.empty:
            t.add_row(side, "0", "—", "—", "—")
            continue
        t.add_row(side, f"{len(sub):,}",
                  f"{sub['dir_correct'].mean()*100:.1f}%",
                  f"{sub['pred_move'].mean():.2f}%",
                  f"{sub['achieved'].median():.2f}%")
    all_acc = trades["dir_correct"].mean() * 100
    t.add_row("[bold]ALL[/bold]", f"{len(trades):,}", f"[bold]{all_acc:.1f}%[/bold]",
              f"{trades['pred_move'].mean():.2f}%", f"{trades['achieved'].median():.2f}%")
    console.print(t)

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    tm = trades.copy()
    tm["month"] = tm["exit_time"].dt.tz_localize(None).dt.to_period("M").astype(str)
    t = Table(title="[bold]MONTHLY — har mahine ka result[/bold]", box=box.ROUNDED)
    t.add_column("Month", style="bold")
    t.add_column("Trades", justify="right")
    t.add_column("Win rate", justify="right")
    t.add_column("Return", justify="right")
    t.add_column("Equity (end)", justify="right")
    for month, sub in tm.groupby("month"):
        frac  = cfg.backtest.POSITION_FRACTION
        m_ret = (np.prod(1.0 + frac * sub["net_ret_pct"].to_numpy() / 100.0) - 1.0) * 100.0
        c = "green" if m_ret > 0 else "red"
        t.add_row(month, f"{len(sub):,}",
                  f"{(sub['pnl_usd'] > 0).mean()*100:.0f}%",
                  f"[{c}]{m_ret:+.2f}%[/{c}]",
                  f"${sub['equity'].iloc[-1]:.2f}")
    console.print(t)

    # ── Last few trades ───────────────────────────────────────────────────────
    t = Table(title="[bold]LAST 10 TRADES[/bold]", box=box.SIMPLE_HEAVY)
    for col in ("Entry", "Side", "Conf", "Pred", "Achieved", "Exit", "Net %", "Equity"):
        t.add_column(col, justify="right")
    for _, r in trades.tail(10).iterrows():
        c = "green" if r["pnl_usd"] > 0 else "red"
        t.add_row(
            r["entry_time"].strftime("%m-%d %H:%M"),
            r["side"],
            f"{r['confidence']:.2f}",
            f"{r['pred_move']:.2f}%",
            f"{r['achieved']:.2f}%",
            r["exit_reason"],
            f"[{c}]{r['net_ret_pct']:+.2f}%[/{c}]",
            f"${r['equity']:.2f}",
        )
    console.print(t)


def plot_backtest(trades: pd.DataFrame, equity_curve: list[float]) -> None:
    if trades.empty:
        return
    DARK, PANEL = "#0e1117", "#1a1f2e"
    GREEN, RED, BLUE, ORANGE, GRAY = "#00d26a", "#ff4757", "#3d84ff", "#ffa502", "#8a8f9e"

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle("Backtest — $100 Real-Trading Simulation", fontsize=16, color="white",
                 fontweight="bold", y=0.98)
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    ax1, ax2, ax3, ax4 = axes
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    curve = np.asarray(equity_curve)

    # ── Panel 1: Equity curve ────────────────────────────────────────────────
    color = GREEN if curve[-1] >= curve[0] else RED
    ax1.plot(curve, color=color, linewidth=1.2)
    ax1.axhline(curve[0], color=GRAY, linewidth=0.8, linestyle="--", label="Start $100")
    ax1.fill_between(np.arange(len(curve)), curve, curve[0], alpha=0.12, color=color)
    ax1.set_xlabel("Trade #", color=GRAY, fontsize=10)
    ax1.set_ylabel("Equity ($)", color=GRAY, fontsize=10)
    ax1.set_title("Equity Curve (capital growth)", color="white", fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    peak = np.maximum.accumulate(curve)
    dd   = (curve - peak) / peak * 100.0
    ax2.fill_between(np.arange(len(dd)), dd, 0, color=RED, alpha=0.5)
    ax2.set_xlabel("Trade #", color=GRAY, fontsize=10)
    ax2.set_ylabel("Drawdown %", color=GRAY, fontsize=10)
    ax2.set_title("Drawdown", color="white", fontsize=11, pad=8)

    # ── Panel 3: Trade PnL histogram ─────────────────────────────────────────
    net = trades["net_ret_pct"].to_numpy()
    bins = np.linspace(net.min(), net.max(), 40)
    ax3.hist(net[net > 0],  bins=bins, color=GREEN, alpha=0.75, label="Wins")
    ax3.hist(net[net <= 0], bins=bins, color=RED,   alpha=0.75, label="Losses")
    ax3.axvline(0, color=GRAY, linewidth=0.8)
    ax3.set_xlabel("Net return per trade %", color=GRAY, fontsize=10)
    ax3.set_ylabel("Trades", color=GRAY, fontsize=10)
    ax3.set_title("Per-Trade PnL Distribution", color="white", fontsize=11, pad=8)
    ax3.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 4: Monthly returns ─────────────────────────────────────────────
    tm = trades.copy()
    tm["month"] = tm["exit_time"].dt.tz_localize(None).dt.to_period("M").astype(str)
    frac = cfg.backtest.POSITION_FRACTION
    monthly = tm.groupby("month")["net_ret_pct"].apply(
        lambda r: (np.prod(1.0 + frac * r.to_numpy() / 100.0) - 1.0) * 100.0
    )
    colors = [GREEN if v > 0 else RED for v in monthly.to_numpy()]
    ax4.bar(monthly.index, monthly.to_numpy(), color=colors, alpha=0.85)
    ax4.axhline(0, color=GRAY, linewidth=0.8)
    ax4.set_ylabel("Return %", color=GRAY, fontsize=10)
    ax4.set_title("Monthly Returns", color="white", fontsize=11, pad=8)
    ax4.tick_params(axis="x", rotation=45)

    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    console.print(f"\n  [dim]Chart → {CHART_PATH}[/dim]\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Backtest — Direction + Magnitude Model[/bold blue]")
    device = _device()
    console.print(f"\n[bold]Device:[/bold] {device}")

    df_valid, _scaler, Xs, y_up, y_dn, _y_dir, (_tr, _va, test_pos) = build_dataset()
    model, ckpt = load_model(device)
    W = ckpt["window"]

    # ── 1. Prediction scorecard on full test set ─────────────────────────────
    X_seq   = make_windows(Xs, test_pos, W)
    metrics = evaluate_model(model, X_seq, y_up[test_pos], y_dn[test_pos], device)
    print_evaluation(metrics, title="MODEL SCORECARD — TEST SET")

    # ── 2. Trading simulation ────────────────────────────────────────────────
    p_up, up_pred, dn_pred = metrics["_p_up"], metrics["_up_pred"], metrics["_dn_pred"]
    trades, equity_curve = run_backtest(
        df_valid, test_pos, p_up, up_pred, dn_pred, y_up[test_pos], y_dn[test_pos]
    )
    print_backtest_report(trades, equity_curve, df_valid, test_pos)
    plot_backtest(trades, equity_curve)

    if not trades.empty:
        trades.to_csv("models/backtest_trades.csv", index=False)
        console.print("  [dim]Trade log → models/backtest_trades.csv[/dim]")

    console.rule("[bold green]Backtest Complete[/bold green]")


if __name__ == "__main__":
    main()
