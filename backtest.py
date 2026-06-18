# -*- coding: utf-8 -*-
"""
Walk-forward backtest of the Quantile Transformer.

Run:
    uv run python backtest.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.table import Table

from config import cfg
from horizon_labeler import MAX_MFE_PCT
from model import QuantileTradingModel
from set_label import LONG, NEUTRAL, SHORT
from trainer import CSV_PATH, MODEL_PATH, feature_matrix, split_bounds

console        = Console()
TRADE_LOG_PATH = "data/trade_log.csv"

SIGNAL_MARGIN = cfg.testing.SIGNAL_MARGIN_THRESHOLD
CONF_FLOOR    = cfg.testing.AI_CONFIDENCE_THRESHOLD


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_valid(path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "label_valid" in df.columns:
        df = df[df["label_valid"].fillna(False).astype(bool)].copy()
    return df


# ── Trade simulation ──────────────────────────────────────────────────────────

def _simulate_trade(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    entry_index: int, entry_price: float, side: int,
    lookahead: int, tp_pct: float, sl_pct: float,
) -> tuple[str, float, int]:
    """Walk forward bar-by-bar until TP / SL / timeout. SL wins if both hit same bar."""
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
    gross = (
        (exit_price - entry_price) / entry_price * 100.0 if side == LONG
        else (entry_price - exit_price) / entry_price * 100.0
    )
    return "TIMEOUT", gross, max(last - entry_index, 1)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _summarize(trades: pd.DataFrame, equity: list[float], df_test: pd.DataFrame) -> dict:
    eq   = np.asarray(equity)
    peak = np.maximum.accumulate(eq)
    dd   = (eq / np.where(peak > 0, peak, 1.0) - 1.0) * 100.0
    n    = len(trades)
    wins = int((trades["net_return_pct"] > 0).sum()) if n else 0
    gp   = float(trades.loc[trades["net_return_pct"] > 0, "pnl_dollar"].sum()) if n else 0.0
    gl   = float(trades.loc[trades["net_return_pct"] < 0, "pnl_dollar"].sum()) if n else 0.0
    bh   = (float(df_test["Close"].iloc[-1]) / float(df_test["Close"].iloc[0]) - 1.0) * 100.0

    bar_ret = pd.Series(equity).pct_change().dropna()
    bars_per_year = 96 * 365   # 15m bars
    sharpe = float(bar_ret.mean() / (bar_ret.std() + 1e-9) * (bars_per_year ** 0.5))

    # Annualised return (CAGR)
    days = len(df_test) / 96.0   # trading days equivalent
    years = days / 365.0
    cagr = ((float(eq[-1]) / float(eq[0])) ** (1.0 / max(years, 1e-6)) - 1.0) * 100.0 if years > 0 else 0.0

    return {
        "start": float(eq[0]), "end": float(eq[-1]),
        "total_ret_pct": (float(eq[-1]) / eq[0] - 1.0) * 100.0,
        "cagr_pct": cagr,
        "buy_hold_ret_pct": bh, "n_trades": n,
        "n_buy":  int((trades["direction"] == "BUY").sum())  if n else 0,
        "n_sell": int((trades["direction"] == "SELL").sum()) if n else 0,
        "win_rate_pct": (wins / n * 100.0) if n else 0.0,
        "profit_factor": (gp / abs(gl)) if gl != 0 else float("inf"),
        "max_dd_pct": float(dd.min()), "sharpe": sharpe,
        "avg_ret_pct": float(trades["net_return_pct"].mean()) if n else 0.0,
        "avg_conf":    float(trades["confidence"].mean())     if n else 0.0,
        "tp_hits":     int((trades["exit_reason"] == "TP").sum())      if n else 0,
        "sl_hits":     int((trades["exit_reason"] == "SL").sum())      if n else 0,
        "timeout_hits":int((trades["exit_reason"] == "TIMEOUT").sum()) if n else 0,
        "avg_bars":    float(trades["bars_held"].mean()) if n else 0.0,
        "test_bars":   len(df_test),
        "test_days":   round(days, 1),
        "equity_curve": eq.tolist(),
    }


def _print_summary(s: dict) -> None:
    t = Table(title="[bold cyan]Virtual Trading — $1,000 Simulation[/bold cyan]", box=box.ROUNDED, show_lines=True)
    t.add_column("Metric", style="bold", min_width=28)
    t.add_column("Value",  justify="right", min_width=20)

    t.add_row("Test period (bars)", f"{s['test_bars']:,}  ({s['test_days']:.0f} days)")
    t.add_row("Start capital",  f"${s['start']:,.2f}")
    t.add_row("End equity",     f"${s['end']:,.2f}")
    pnl = s["end"] - s["start"]
    pcol = "green" if pnl >= 0 else "red"
    t.add_row("P&L (dollar)",   f"[{pcol}]{pnl:+,.2f}[/{pcol}]")
    col = "green" if s["total_ret_pct"] >= 0 else "red"
    t.add_row("Total return",   f"[{col}]{s['total_ret_pct']:+.2f}%[/{col}]")
    ccol = "green" if s["cagr_pct"] >= 0 else "red"
    t.add_row("CAGR (annualised)", f"[{ccol}]{s['cagr_pct']:+.2f}%[/{ccol}]")
    bcol = "green" if s["buy_hold_ret_pct"] >= 0 else "red"
    t.add_row("Buy & Hold return", f"[{bcol}]{s['buy_hold_ret_pct']:+.2f}%[/{bcol}]")
    ecol = "green" if (s["total_ret_pct"] - s["buy_hold_ret_pct"]) >= 0 else "red"
    t.add_row("Alpha vs B&H",  f"[{ecol}]{s['total_ret_pct']-s['buy_hold_ret_pct']:+.2f}%[/{ecol}]")
    t.add_section()
    t.add_row("Total trades",  str(s["n_trades"]))
    t.add_row("  BUY / SELL",  f"{s['n_buy']} / {s['n_sell']}")
    t.add_row("Win rate",      f"{s['win_rate_pct']:.1f}%")
    pf = s["profit_factor"]
    t.add_row("Profit factor", f"{pf:.3f}" if pf != float("inf") else "∞")
    t.add_row("Max drawdown",  f"{s['max_dd_pct']:.2f}%")
    t.add_row("Sharpe (ann.)", f"{s['sharpe']:.3f}")
    t.add_row("Avg ret/trade", f"{s['avg_ret_pct']:.3f}%")
    t.add_row("Avg confidence",f"{s['avg_conf']:.3f}")
    t.add_section()
    t.add_row("TP / SL / Timeout", f"{s['tp_hits']} / {s['sl_hits']} / {s['timeout_hits']}")
    t.add_row("Avg bars held",     f"{s['avg_bars']:.1f}")
    console.print(t)


def _print_trade_journal(trades: pd.DataFrame) -> None:
    """Print each trade with running equity — the virtual trading journal."""
    if trades.empty:
        return
    t = Table(title="[bold yellow]Trade Journal[/bold yellow]", box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("#",        justify="right",  style="dim",   width=4)
    t.add_column("Date",     justify="left",                  width=22)
    t.add_column("Dir",      justify="center",                width=6)
    t.add_column("Conf",     justify="right",                 width=6)
    t.add_column("TP%",      justify="right",                 width=7)
    t.add_column("SL%",      justify="right",                 width=7)
    t.add_column("Exit",     justify="center",                width=8)
    t.add_column("Net%",     justify="right",                 width=8)
    t.add_column("P&L $",    justify="right",                 width=10)
    t.add_column("Equity $", justify="right",                 width=11)

    for i, row in trades.iterrows():
        net = float(row["net_return_pct"])
        pnl = float(row["pnl_dollar"])
        eq  = float(row["equity_after"])
        reason = str(row["exit_reason"])
        direction = str(row["direction"])
        color = "green" if net > 0 else "red"
        exit_color = "green" if reason == "TP" else ("red" if reason == "SL" else "yellow")
        t.add_row(
            str(int(i) + 1),
            str(row["timestamp"])[:19],
            f"[cyan]{direction}[/cyan]",
            f"{float(row['confidence']):.2f}",
            f"{float(row['tp_pct']):.2f}",
            f"{float(row['sl_pct']):.2f}",
            f"[{exit_color}]{reason}[/{exit_color}]",
            f"[{color}]{net:+.2f}%[/{color}]",
            f"[{color}]{pnl:+.2f}[/{color}]",
            f"${eq:,.2f}",
        )
    console.print(t)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(
    model: QuantileTradingModel, Xs: np.ndarray,
    test_start: int, window: int, device: torch.device, max_mfe_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    Xs_t = torch.from_numpy(Xs)
    all_probs, all_mq = [], []
    for i in range(0, len(Xs) - test_start, 512):
        chunk = list(range(test_start + i, min(test_start + i + 512, len(Xs))))
        seqs  = torch.stack([Xs_t[t - window + 1 : t + 1] for t in chunk]).to(device)
        probs = model.calibrated_direction_probs(seqs)
        mq    = model(seqs)["move_q"]
        all_probs.append(probs.cpu().numpy())
        all_mq.append(mq.cpu().numpy() * max_mfe_pct)
    return np.concatenate(all_probs), np.concatenate(all_mq)


# ── Walk-forward ──────────────────────────────────────────────────────────────

def _run_backtest(
    df_test: pd.DataFrame, probs: np.ndarray, move_q: np.ndarray,
    lookahead: int,
) -> tuple[pd.DataFrame, list[float]]:
    high  = df_test["High"].to_numpy(dtype=np.float64)
    low   = df_test["Low"].to_numpy(dtype=np.float64)
    close = df_test["Close"].to_numpy(dtype=np.float64)
    open_ = df_test["Open"].to_numpy(dtype=np.float64)
    natr  = (
        df_test["natr"].to_numpy(dtype=np.float64) if "natr" in df_test.columns
        else (df_test["atr"] / df_test["Close"] * 100.0).to_numpy(dtype=np.float64)
    )
    index    = df_test.index
    cost_pct = cfg.testing.ROUND_TRIP_FEE_PCT + 2.0 * cfg.testing.SLIPPAGE_PCT
    capital  = float(cfg.ml_backtest.INITIAL_CAPITAL)
    equity: list[float] = [capital]
    records: list[dict] = []
    busy_until = -1
    n = len(df_test)

    for i in range(n):
        flat      = i > busy_until
        entry_bar = i + 1
        if not flat or entry_bar >= n or natr[i] <= 0.0:
            equity.append(capital)
            continue

        p_long = float(probs[i, LONG])
        p_neu  = float(probs[i, NEUTRAL])
        p_short= float(probs[i, SHORT])
        conf   = max(p_long, p_short)
        side   = LONG if p_long >= p_short else SHORT
        margin = conf - p_neu

        if margin < SIGNAL_MARGIN or conf < CONF_FLOOR:
            equity.append(capital)
            continue

        # ATR-based TP / SL
        tp_pct = float(np.clip(
            natr[i] * cfg.training.FIXED_TP_ATR_MULTIPLIER,
            cfg.training.MIN_ATR_TARGET_PCT, cfg.training.MAX_ATR_TARGET_PCT,
        ))
        sl_pct = float(np.clip(
            natr[i] * cfg.training.FIXED_SL_ATR_MULTIPLIER,
            cfg.training.MIN_ATR_STOP_PCT, cfg.training.MAX_ATR_STOP_PCT,
        ))

        entry_price = float(open_[entry_bar])
        q10, q50, q90 = float(move_q[i, 0]), float(move_q[i, 1]), float(move_q[i, 2])

        reason, gross, bars = _simulate_trade(
            high, low, close, entry_bar, entry_price, side, lookahead, tp_pct, sl_pct
        )
        net      = gross - cost_pct
        pnl      = capital * net / 100.0
        capital += pnl
        busy_until = entry_bar + bars

        records.append({
            "timestamp": str(index[i]),
            "direction": "BUY" if side == LONG else "SELL",
            "confidence": conf, "entry_price": entry_price,
            "tp_pct": tp_pct, "sl_pct": sl_pct,
            "exit_reason": reason, "gross_return_pct": gross,
            "net_return_pct": net, "pnl_dollar": pnl, "bars_held": bars,
            "move_q10_pct": q10, "move_q50_pct": q50, "move_q90_pct": q90,
            "equity_after": capital,
        })
        equity.append(capital)

    return pd.DataFrame(records), equity


# ── Verdict ───────────────────────────────────────────────────────────────────

def _print_verdict(s: dict) -> None:
    """Print a plain-language verdict on model + backtest quality."""
    n = s["n_trades"]
    ret = s["total_ret_pct"]
    wr  = s["win_rate_pct"]
    pf  = s["profit_factor"]
    cagr = s["cagr_pct"]
    bh   = s["buy_hold_ret_pct"]
    alpha = ret - bh

    # Grade each dimension
    def _grade(val, good, ok):
        if val >= good: return "[green]GOOD[/green]"
        if val >= ok:   return "[yellow]OK[/yellow]"
        return "[red]WEAK[/red]"

    t = Table(title="[bold magenta]Model & Strategy Verdict[/bold magenta]", box=box.ROUNDED, show_lines=True)
    t.add_column("Dimension",  style="bold", min_width=24)
    t.add_column("Value",      justify="right", min_width=14)
    t.add_column("Grade",      justify="center", min_width=8)
    t.add_column("Comment",    min_width=32)

    trade_grade = _grade(n, 20, 10)
    t.add_row("Total trades",       str(n), trade_grade,
              "Need 20+ for reliable stats" if n < 20 else "Sufficient sample")

    wr_grade = _grade(wr, 45, 38)
    t.add_row("Win rate",       f"{wr:.1f}%", wr_grade,
              "Need >38% for 2:1 R:R break-even")

    pf_grade = _grade(pf if pf != float("inf") else 99, 1.5, 1.0)
    t.add_row("Profit factor", f"{pf:.3f}" if pf != float("inf") else "∞", pf_grade,
              ">1.5 = consistent edge")

    ret_grade = _grade(ret, 5.0, 0.0)
    t.add_row("Total return",  f"{ret:+.2f}%", ret_grade,
              f"Over {s['test_days']:.0f} days")

    cagr_grade = _grade(cagr, 10.0, 0.0)
    t.add_row("CAGR",          f"{cagr:+.2f}%", cagr_grade,
              "Annualised return")

    alpha_grade = _grade(alpha, 3.0, 0.0)
    t.add_row("Alpha vs B&H",  f"{alpha:+.2f}%", alpha_grade,
              "Edge over passive holding")

    console.print(t)

    # Plain-language summary
    if n < 10:
        verdict = "[yellow]INSUFFICIENT DATA[/yellow] — too few trades. Lower [bold]CONF_FLOOR[/bold] in config.py (try 0.35) to fire more signals."
    elif pf >= 1.5 and wr >= 38 and alpha > 0:
        verdict = "[green]POSITIVE EDGE[/green] — model shows consistent profit and beats Buy & Hold."
    elif alpha > 0 and pf >= 1.0:
        verdict = "[yellow]MARGINAL EDGE[/yellow] — model beats B&H but profit factor needs improvement."
    else:
        verdict = "[red]NO EDGE[/red] — strategy loses to Buy & Hold. Model direction accuracy too low (~30%); retrain or adjust features."

    console.print(f"\n  Verdict: {verdict}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Quantile Transformer — Backtest[/bold blue]")

    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]Model not found: {MODEL_PATH}. Run trainer.py first.[/red]")
        return

    bundle      = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    features    = bundle["features"]
    window      = int(bundle["window"])
    mean        = bundle["scaler_mean"]
    scale       = bundle["scaler_scale"]
    temperature = float(bundle.get("temperature", 1.0))
    lookahead   = int(bundle.get("lookahead_bars", cfg.training.LOOKAHEAD_BARS))
    max_mfe_pct = float(bundle.get("max_mfe_pct", MAX_MFE_PCT))

    device = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Restore architecture from bundle so model always matches the saved checkpoint.
    cfg.nn.HIDDEN_DIM = int(bundle.get("hidden_dim", cfg.nn.HIDDEN_DIM))
    cfg.nn.NUM_LAYERS = int(bundle.get("num_layers", cfg.nn.NUM_LAYERS))
    cfg.nn.NUM_HEADS  = int(bundle.get("num_heads",  cfg.nn.NUM_HEADS))
    cfg.nn.DROPOUT    = float(bundle.get("dropout",  cfg.nn.DROPOUT))

    model  = QuantileTradingModel(input_dim=bundle["input_dim"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.temperature.fill_(temperature)
    console.print(f"  [dim]T={temperature:.3f}[/dim]")

    df = _load_valid()
    _, val_end = split_bounds(len(df))
    df_test    = df.iloc[val_end:].copy()
    console.print(f"\n  Test: {df_test.index[0]} → {df_test.index[-1]}  ({len(df_test)} bars)")

    X  = feature_matrix(df, features).fillna(0.0).to_numpy(dtype=np.float32)
    Xs = np.nan_to_num(((X - mean) / scale).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    with console.status("Inference..."):
        probs, move_q = _predict(model, Xs, val_end, window, device, max_mfe_pct)

    trades, equity = _run_backtest(df_test, probs, move_q, lookahead)
    stats = _summarize(trades, equity, df_test)
    _print_summary(stats)

    if not trades.empty:
        _print_trade_journal(trades.reset_index(drop=True))

        # Save equity curve for charting
        os.makedirs("data", exist_ok=True)
        eq_df = pd.DataFrame({
            "timestamp": df_test.index,
            "equity":    stats["equity_curve"][1:],  # [0] = initial capital, [1:] aligns bar-by-bar
        })
        eq_df.to_csv("data/equity_curve.csv", index=False)
        console.print(f"  [dim]Equity curve → data/equity_curve.csv[/dim]")

    if not trades.empty:
        trades.to_csv(TRADE_LOG_PATH, index=False)
        console.print(f"  [dim]Trade log → {TRADE_LOG_PATH}[/dim]")
    else:
        console.print(
            f"\n  [yellow]No trades fired. Lower CONF_FLOOR ({CONF_FLOOR}) "
            f"or SIGNAL_MARGIN ({SIGNAL_MARGIN}) in config.py.[/yellow]"
        )

    _print_verdict(stats)
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
