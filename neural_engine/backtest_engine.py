# -*- coding: utf-8 -*-
"""
Backtesting Engine
==================
Loads a trained model and runs a paper-trading simulation on recent test data.

Signal firing logic (NEW — margin-based):
  LONG  fires when: (prob_long  - prob_neutral) >= long_signal_margin
  SHORT fires when: (prob_short - prob_neutral) >= short_signal_margin

This is superior to an absolute probability threshold because the model
outputs ~33% per class on uncertain bars (three-class softmax), so a raw
threshold of 0.50 would almost never fire.  A margin threshold of 0.08
instead measures "how much more confident is the model in LONG vs NEUTRAL?"
"""
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import cfg
from engine.data_handler import fetch_data
from engine.backtester import run_paper_portfolio_on_signals
from neural_engine.feature_utils import add_technical_indicators, get_feature_columns
from neural_engine.model import MultiHeadTradingModel
from ui_utils import console

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main backtest entry point
# ---------------------------------------------------------------------------

def run_backtest(symbol: str = "ETHUSD") -> None:
    """Run paper-trading backtest for a trained model on the most recent test period."""
    console.print(Panel(
        f"[bold cyan]BACKTESTING: {symbol}",
        subtitle="Advanced Neural Analytics Engine",
        expand=False,
    ))

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    total_days = cfg.training.TEST_DATA_DAYS + 45
    df = fetch_data(symbol=symbol, total_days=total_days, interval=cfg.model.INTERVAL)
    if df.empty:
        console.print("[red]No data found.[/red]")
        return

    # ── 2. Feature engineering ────────────────────────────────────────────
    feature_cols = get_feature_columns()
    df_features = add_technical_indicators(df.copy())

    # ── 3. Load model ─────────────────────────────────────────────────────
    model_path = Path("models/trading_model.pth")
    if not model_path.exists():
        console.print(f"[red]Model file {model_path} not found. Run --train first.[/red]")
        return

    device = cfg.DEVICE
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    param_count = sum(param.numel() for param in model.parameters())

    # ── 4. Load scaler ────────────────────────────────────────────────────
    try:
        mean = np.load("models/scaler_mean.npy")
        scale = np.load("models/scaler_scale.npy")
    except FileNotFoundError:
        console.print("[red]Scaler files not found. Run --train first.[/red]")
        return

    # ── 5. Generate predictions ───────────────────────────────────────────
    feature_data = df_features[feature_cols].values
    feature_data = (feature_data - mean) / (scale + 1e-8)

    window_size = cfg.model.WINDOW_SIZE
    if len(feature_data) < window_size + 1:
        console.print("[red]Not enough data for window size.[/red]")
        return

    # CRITICAL FIX: Use [:-1] to create windows that predict the NEXT bar
    # This prevents same-bar leakage (using bar t's data to predict bar t)
    from numpy.lib.stride_tricks import sliding_window_view
    X_seq = sliding_window_view(feature_data[:-1], (window_size, feature_data.shape[1])).squeeze(1)

    probs_list, sizing_list, mag_list, time_list = [], [], [], []
    model.eval()  # Ensure eval mode — disables dropout for deterministic inference
    with torch.no_grad():
        for i in range(0, len(X_seq), 1024):
            batch = torch.from_numpy(X_seq[i : i + 1024].copy()).float().to(device)
            out = model(batch)
            probs_list.append(torch.softmax(out["direction"], dim=1).cpu().numpy())
            sizing_list.append(out["sizing"].cpu().numpy())
            mag_list.append(out["magnitude"].cpu().numpy())
            time_list.append(out["time"].cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    sizing = np.concatenate(sizing_list, axis=0)
    magnitude = np.concatenate(mag_list, axis=0)
    time_pred = np.concatenate(time_list, axis=0)

    # Pad with WINDOW_SIZE zeros at the start (no predictions for first window_size bars)
    # Now predictions align correctly: window[0:95] predicts row 96, etc.
    pad_len = window_size
    probs = np.vstack([np.zeros((pad_len, 3)), probs])
    sizing = np.vstack([np.zeros((pad_len, 3)), sizing])
    magnitude = np.vstack([np.zeros((pad_len, 1)), magnitude])
    time_pred = np.vstack([np.zeros((pad_len, 1)), time_pred])

    # ── 6. Decode model outputs ───────────────────────────────────────────
    max_tp = cfg.testing.MAX_ATR_TARGET_PCT
    max_sl = cfg.testing.MAX_ATR_STOP_PCT

    df_features["ai_prob_long"] = probs[:, 0]
    df_features["ai_prob_neutral"] = probs[:, 1]
    df_features["ai_prob_short"] = probs[:, 2]
    df_features["ai_confidence"] = np.maximum(probs[:, 0], probs[:, 2])
    df_features["ai_take_profit_pct"] = sizing[:, 1] * max_tp
    df_features["ai_stop_loss_pct"] = sizing[:, 2] * max_sl
    df_features["ai_qty_ratio"] = sizing[:, 0]
    df_features["predicted_magnitude"] = magnitude[:, 0] * max_tp
    df_features["predicted_time_bars"] = time_pred[:, 0] * cfg.training.LOOKAHEAD_BARS

    # ── 7. Apply margin-based signal logic ────────────────────────────────
    long_margin = cfg.testing.SIGNAL_MARGIN_THRESHOLD
    short_margin = cfg.testing.SIGNAL_MARGIN_THRESHOLD

    logger.info(f"Signal thresholds (Master Config) — long_margin: {long_margin:.2f}, short_margin: {short_margin:.2f}")

    # Compute probability margins
    long_margin_arr = probs[:, 0] - probs[:, 1]    # prob_long - prob_neutral
    short_margin_arr = probs[:, 2] - probs[:, 1]   # prob_short - prob_neutral

    # Default verdict: NEUTRAL (1)
    df_features["ai_verdict"] = 1

    abs_floor = cfg.testing.AI_CONFIDENCE_THRESHOLD

    # LONG signal: margin AND absolute floor
    long_signal = (long_margin_arr >= long_margin) & (probs[:, 0] >= abs_floor)
    # SHORT signal: margin AND absolute floor AND no stronger opposing LONG signal
    short_signal = (
        (short_margin_arr >= short_margin)
        & (probs[:, 2] >= abs_floor)
        & ~long_signal  # LONG takes priority if both fire
    )

    # Apply macro trend filter to prevent fighting the market
    if getattr(cfg.testing, "USE_TREND_FILTER", True):
        # Use the same EMA-regime feature family that the model sees.
        trend_long = df_features["dist_ema_50"].values > 0
        trend_short = df_features["dist_ema_50"].values < 0
        long_signal = long_signal & trend_long
        short_signal = short_signal & trend_short

    df_features.loc[long_signal, "ai_verdict"] = 0
    df_features.loc[short_signal, "ai_verdict"] = 2

    long_count = long_signal.sum()
    short_count = short_signal.sum()
    logger.info(f"Signals generated — LONG: {long_count}, SHORT: {short_count}, NEUTRAL: {(~long_signal & ~short_signal).sum()}")

    # ── 8. Run backtester ─────────────────────────────────────────────────
    from config import bars_per_day
    test_bars = int(cfg.training.TEST_DATA_DAYS * bars_per_day(cfg.model.INTERVAL))
    df_test = df_features.iloc[-test_bars:].copy()
    eligible_rows = max(len(df_features) - window_size + 1, 0)

    final_df, trades, summary = run_paper_portfolio_on_signals(
        panel=df_test,
        symbol=symbol,
        initial_capital_usd=cfg.testing.INITIAL_CAPITAL_USD,
        margin_per_trade_pct_of_equity=cfg.testing.MARGIN_PER_TRADE_PCT_OF_EQUITY,
        leverage=cfg.testing.LEVERAGE,
        round_trip_fee_pct=cfg.testing.ROUND_TRIP_FEE_PCT,
    )
    # Calculate additional metrics
    initial = cfg.testing.INITIAL_CAPITAL_USD
    pnl = summary.get("total_pnl_net_usd", 0)
    roi_pct = (pnl / initial) * 100.0 if initial > 0 else 0

    # Calculate CAGR (Compound Annual Growth Rate)
    if summary.get("duration_days", 0) > 0:
        years = summary["duration_days"] / 365.0
        final_equity = initial + pnl
        cagr = ((final_equity / initial) ** (1 / years) - 1) * 100
        summary["cagr_pct"] = round(cagr, 2)
    else:
        summary["cagr_pct"] = 0.0

    # Calculate Buy & Hold benchmark
    if not df_test.empty and len(df_test) > 1:
        buy_hold_return = ((df_test["Close"].iloc[-1] - df_test["Close"].iloc[0]) / df_test["Close"].iloc[0]) * 100
        summary["buy_hold_return_pct"] = round(buy_hold_return, 2)
        summary["excess_return_vs_buy_hold"] = round(roi_pct - buy_hold_return, 2)
    else:
        summary["buy_hold_return_pct"] = 0.0
        summary["excess_return_vs_buy_hold"] = 0.0

    # Calculate Calmar Ratio (Return / Max Drawdown)
    max_dd = abs(summary.get("max_drawdown_pct", 0))
    if max_dd > 0:
        summary["calmar_ratio"] = round(abs(roi_pct / max_dd), 2)
    else:
        summary["calmar_ratio"] = 0.0

    summary.update({
        "raw_rows": len(df),
        "feature_rows": len(df_features),
        "eligible_prediction_rows": eligible_rows,
        "test_rows": len(df_test),
        "feature_count": len(feature_cols),
        "window_size": window_size,
        "model_parameters": param_count,
        "interval": cfg.model.INTERVAL,
        "signal_long_count": int(long_count),
        "signal_short_count": int(short_count),
        "signal_neutral_count": int((~long_signal & ~short_signal).sum()),
        "signal_margin_threshold": cfg.testing.SIGNAL_MARGIN_THRESHOLD,
        "confidence_threshold": cfg.testing.AI_CONFIDENCE_THRESHOLD,
    })

    # ── 9. Print results ──────────────────────────────────────────────────
    _print_professional_backtest_summary(symbol, summary, trades)

    if trades:
        os.makedirs("data", exist_ok=True)
        trades_df = pd.DataFrame([vars(t) for t in trades])
        trades_df.to_csv(f"data/backtest_trades_{symbol}.csv", index=False)


# ---------------------------------------------------------------------------
# Rich output formatting
# ---------------------------------------------------------------------------

def _money(value: float) -> str:
    """Format signed dollar values with color for Rich tables."""
    if value >= 0:
        return f"[green]+${value:,.2f}[/]"
    return f"[red]-${abs(value):,.2f}[/]"


def _format_pf(value: float) -> str:
    """Format profit factor, including infinite values."""
    if np.isinf(value):
        return "∞"
    return f"{value:.2f}"


def _print_professional_backtest_summary(symbol: str, summary: dict, trades: list) -> None:
    """Render a rich console summary of backtest results."""
    time_range = Text.assemble(
        ("Period: ", "bold white"),
        (f"{summary.get('start_time', 'N/A')} ", "cyan"),
        ("to ", "bold white"),
        (f"{summary.get('end_time', 'N/A')}", "cyan"),
        (f" ({summary.get('duration_days', 0):.1f} Days)", "dim"),
    )
    console.print(time_range)
    console.print("")

    initial = cfg.testing.INITIAL_CAPITAL_USD
    pnl = summary.get("total_pnl_net_usd", 0)
    roi_pct = (pnl / initial) * 100.0 if initial > 0 else 0

    long_trades = sum(1 for t in trades if getattr(t, "side", "").upper() == "LONG")
    short_trades = sum(1 for t in trades if getattr(t, "side", "").upper() == "SHORT")
    long_stats = summary.get("long", {})
    short_stats = summary.get("short", {})

    # ── Performance table ─────────────────────────────────────────────────
    perf_table = Table(
        title="[bold cyan]Core Strategy Performance",
        show_header=True,
        header_style="bold magenta",
        box=box.MINIMAL,
    )
    perf_table.add_column("Metric", style="dim")
    perf_table.add_column("Value", justify="right", style="bold")
    perf_table.add_row("Initial Capital", f"${initial:,.2f}")
    perf_table.add_row(
        "Net Profit (PnL)",
        f"[green]+${pnl:,.2f}[/]" if pnl >= 0 else f"[red]-${abs(pnl):,.2f}[/]",
    )
    perf_table.add_row("Final Equity", f"${initial + pnl:,.2f}")
    perf_table.add_row(
        "ROI (%)",
        f"[green]{roi_pct:+.2f}%[/]" if roi_pct >= 0 else f"[red]{roi_pct:+.2f}%[/]",
    )
    # NEW: CAGR metric
    cagr = summary.get("cagr_pct", 0)
    perf_table.add_row(
        "CAGR (%)",
        f"[green]{cagr:+.2f}%[/]" if cagr >= 0 else f"[red]{cagr:+.2f}%[/]",
    )
    # NEW: Buy & Hold comparison
    buy_hold = summary.get("buy_hold_return_pct", 0)
    perf_table.add_row(
        "Buy & Hold (%)",
        f"[green]{buy_hold:+.2f}%[/]" if buy_hold >= 0 else f"[red]{buy_hold:+.2f}%[/]",
    )
    # NEW: Excess return
    excess = summary.get("excess_return_vs_buy_hold", 0)
    perf_table.add_row(
        "Excess Return (%)",
        f"[green]{excess:+.2f}%[/]" if excess >= 0 else f"[red]{excess:+.2f}%[/]",
    )
    perf_table.add_row("Total Trades", str(summary.get("trade_count", 0)))
    perf_table.add_row("  └─ Long Trades", f"[cyan]{long_trades}[/]")
    perf_table.add_row("  └─ Short Trades", f"[magenta]{short_trades}[/]")
    perf_table.add_row("Win Rate", f"{summary.get('win_rate_pct', 0):.2f}%")

    # ── Trade analytics table ─────────────────────────────────────────────
    analytics_table = Table(
        title="[bold yellow]Trade Analytics",
        show_header=True,
        header_style="bold yellow",
        box=box.MINIMAL,
    )
    analytics_table.add_column("Parameter", style="dim")
    analytics_table.add_column("Value", justify="right", style="bold")
    analytics_table.add_row("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")
    analytics_table.add_row("Gross PnL", f"{_money(summary.get('total_pnl_gross_usd', 0))}")
    analytics_table.add_row("Fees Paid", f"[red]-${summary.get('total_fees_usd', 0):,.2f}[/]")
    analytics_table.add_row("Winners Total", f"[green]+${summary.get('positive_pnl_usd', 0):,.2f}[/]")
    analytics_table.add_row("Losers Total", f"[red]${summary.get('negative_pnl_usd', 0):,.2f}[/]")
    analytics_table.add_row("Avg Win", f"[green]${summary.get('avg_win_usd', 0):,.2f}[/]")
    analytics_table.add_row("Avg Loss", f"[red]-${abs(summary.get('avg_loss_usd', 0)):,.2f}[/]")
    analytics_table.add_row("Expectancy/Trade", f"{_money(summary.get('expectancy_usd', 0))}")
    analytics_table.add_row("Avg Holding", f"{summary.get('avg_holding_bars', 0):.1f} Bars")
    analytics_table.add_row("Trades / Day", f"{summary.get('trades_per_day', 0):.2f}")
    analytics_table.add_row("Max Drawdown", f"[red]{summary.get('max_drawdown_pct', 0):.2f}%[/]")
    analytics_table.add_row("Sharpe", f"{summary.get('sharpe_ratio', 0):.2f}")
    # NEW: Calmar Ratio
    analytics_table.add_row("Calmar Ratio", f"{summary.get('calmar_ratio', 0):.2f}")

    # ── Direction breakdown table ────────────────────────────────────────
    direction_table = Table(
        title="[bold magenta]Long vs Short",
        show_header=True,
        header_style="bold magenta",
        box=box.MINIMAL,
    )
    direction_table.add_column("Side", style="dim")
    direction_table.add_column("Trades", justify="right", style="bold")
    direction_table.add_column("Win Rate", justify="right", style="bold")
    direction_table.add_column("Net PnL", justify="right", style="bold")
    direction_table.add_column("PF", justify="right", style="bold")
    direction_table.add_row(
        "LONG",
        str(long_stats.get("trades", 0)),
        f"{long_stats.get('win_rate_pct', 0):.2f}%",
        _money(long_stats.get("net_pnl_usd", 0)),
        _format_pf(long_stats.get("profit_factor", 0)),
    )
    direction_table.add_row(
        "SHORT",
        str(short_stats.get("trades", 0)),
        f"{short_stats.get('win_rate_pct', 0):.2f}%",
        _money(short_stats.get("net_pnl_usd", 0)),
        _format_pf(short_stats.get("profit_factor", 0)),
    )

    # ── Model/data context table ─────────────────────────────────────────
    context_table = Table(
        title="[bold green]Model & Data",
        show_header=True,
        header_style="bold green",
        box=box.MINIMAL,
    )
    context_table.add_column("Metric", style="dim")
    context_table.add_column("Value", justify="right", style="bold")
    context_table.add_row("Interval", str(summary.get("interval", cfg.model.INTERVAL)))
    context_table.add_row("Features", str(summary.get("feature_count", 0)))
    context_table.add_row("Window", f"{summary.get('window_size', 0)} bars")
    context_table.add_row("Model Params", f"{summary.get('model_parameters', 0):,}")
    context_table.add_row("Raw Rows", f"{summary.get('raw_rows', 0):,}")
    context_table.add_row("Feature Rows", f"{summary.get('feature_rows', 0):,}")
    context_table.add_row("Predictable Rows", f"{summary.get('eligible_prediction_rows', 0):,}")
    context_table.add_row("Test Rows", f"{summary.get('test_rows', 0):,}")
    context_table.add_row("Signals L/S/N", (
        f"{summary.get('signal_long_count', 0):,} / "
        f"{summary.get('signal_short_count', 0):,} / "
        f"{summary.get('signal_neutral_count', 0):,}"
    ))
    context_table.add_row("Conf / Margin", (
        f"{summary.get('confidence_threshold', 0):.2f} / "
        f"{summary.get('signal_margin_threshold', 0):.2f}"
    ))

    # ── Signal quality table ──────────────────────────────────────────────
    rel_table = Table(
        title="[bold blue]Signal Quality",
        show_header=True,
        header_style="bold blue",
        box=box.MINIMAL,
    )
    rel_table.add_column("Metric", style="dim")
    rel_table.add_column("Value", justify="right", style="bold")

    win_confs = [getattr(t, "ai_confidence", 0) for t in trades if getattr(t, "pnl_net_usd", 0) > 0]
    loss_confs = [getattr(t, "ai_confidence", 0) for t in trades if getattr(t, "pnl_net_usd", 0) <= 0]
    avg_win_conf = np.mean(win_confs) * 100 if win_confs else 0.0
    avg_loss_conf = np.mean(loss_confs) * 100 if loss_confs else 0.0
    gap = avg_win_conf - avg_loss_conf

    rel_table.add_row("Avg Win Confidence", f"{avg_win_conf:.1f}%")
    rel_table.add_row("Avg Loss Confidence", f"{avg_loss_conf:.1f}%")
    rel_table.add_row(
        "Reliability Gap",
        f"[green]{gap:+.1f}%[/]" if gap > 0 else f"[red]{gap:+.1f}%[/]",
    )
    rel_table.add_row("Max Win Streak", str(summary.get("max_win_streak", 0)))
    rel_table.add_row("Max Loss Streak", str(summary.get("max_loss_streak", 0)))

    console.print(Columns([perf_table, analytics_table, direction_table, rel_table, context_table]))
    console.print("")

    if not trades:
        return

    trade_table = Table(
        title="Recent Trade History (AI Performance Details)",
        show_header=True,
        header_style="bold green",
    )
    trade_table.add_column("#", justify="center", style="dim")
    trade_table.add_column("Side", justify="center")
    trade_table.add_column("Qty", justify="right", style="bold cyan")
    trade_table.add_column("Entry", justify="right")
    trade_table.add_column("Exit", justify="right")
    trade_table.add_column("Conf %", justify="right", style="bold yellow")
    trade_table.add_column("TP %", justify="right", style="green")
    trade_table.add_column("SL %", justify="right", style="red")
    trade_table.add_column("Dur", justify="right")
    trade_table.add_column("Status", justify="center")
    trade_table.add_column("Max %", justify="right", style="green")
    trade_table.add_column("Min %", justify="right", style="red")
    trade_table.add_column("Net PnL ($)", justify="right")

    start_idx = max(0, len(trades) - 20)
    for i, t in enumerate(trades[start_idx:]):
        pnl_val = getattr(t, "pnl_net_usd", 0)
        res_style = "green" if pnl_val > 0 else "red"
        side = getattr(t, "side", "N/A")
        side_color = "cyan" if side == "LONG" else "magenta"
        trade_table.add_row(
            str(start_idx + i + 1),
            f"[{side_color}]{side}[/]",
            f"{getattr(t, 'quantity', 0):.4f}",
            f"{getattr(t, 'entry_price', 0):.2f}",
            f"{getattr(t, 'exit_price', 0):.2f}",
            f"{getattr(t, 'ai_confidence', 0) * 100:.1f}%",
            f"{getattr(t, 'take_profit_pct', 0):.2f}%",
            f"{getattr(t, 'stop_loss_pct', 0):.2f}%",
            f"{getattr(t, 'holding_bars', 0)}b",
            f"[{res_style}]{getattr(t, 'exit_reason', 'N/A')}[/]",
            f"{getattr(t, 'mfe_pct', 0):.2f}%",
            f"{getattr(t, 'mae_pct', 0):.2f}%",
            f"[{res_style}]{pnl_val:+.2f}[/]",
        )
    console.print(trade_table)


if __name__ == "__main__":
    run_backtest("ETHUSD")
