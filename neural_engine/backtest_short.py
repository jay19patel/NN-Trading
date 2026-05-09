# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
import logging
import os
import json
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

from engine.data_handler import fetch_data
from neural_engine.feature_utils import add_technical_indicators, get_feature_columns
from engine.backtester import run_paper_portfolio_on_signals
from neural_engine.model import MultiHeadTradingModel
from config import config
from ui_utils import console

logger = logging.getLogger(__name__)

def run_backtest(symbol: str = "ETHUSD"):
    """Run backtest for the model on a specific symbol."""
    console.print(Panel(f"[bold cyan]BACKTESTING: {symbol}", subtitle="Advanced Neural Analytics Engine", expand=False))
    
    # 1. Fetch Data
    total_days = config.training.TEST_DATA_DAYS + 10
    df = fetch_data(symbol=symbol, total_days=total_days, interval=config.data.INTERVAL)
    if df.empty:
        console.print("[red]No data found.")
        return

    # 2. Feature Engineering
    feature_cols = get_feature_columns()
    df_features = add_technical_indicators(df.copy())
    
    # 3. Load Model and Scaler
    model_path = Path("models/short_model_eth.pth")
    if not model_path.exists():
        console.print(f"[red]Model file {model_path} not found. Please train first.")
        return
        
    device = config.DEVICE
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    try:
        mean = np.load("models/scaler_mean.npy")
        scale = np.load("models/scaler_scale.npy")
    except FileNotFoundError:
        console.print("[red]Scaler files not found. Please train first.")
        return

    # 4. Generate Predictions
    feature_data = df_features[feature_cols].values
    feature_data = (feature_data - mean) / (scale + 1e-8)
    
    window_size = config.model.WINDOW_SIZE
    if len(feature_data) < window_size:
        console.print("[red]Not enough data for window size.")
        return
        
    from numpy.lib.stride_tricks import sliding_window_view
    X_seq = sliding_window_view(feature_data, (window_size, feature_data.shape[1])).squeeze(1)
    
    probs_list, sizing_list, mag_list, time_list = [], [], [], []
    
    with torch.no_grad():
        for i in range(0, len(X_seq), 1024):
            batch = torch.from_numpy(X_seq[i:i+1024].copy()).float().to(device)
            out = model(batch)
            probs_list.append(torch.softmax(out["direction"], dim=1).cpu().numpy())
            sizing_list.append(out["sizing"].cpu().numpy())
            mag_list.append(out["magnitude"].cpu().numpy())
            time_list.append(out["time"].cpu().numpy())
            
    probs = np.concatenate(probs_list, axis=0)
    sizing = np.concatenate(sizing_list, axis=0)
    magnitude = np.concatenate(mag_list, axis=0)
    time_pred = np.concatenate(time_list, axis=0)
    
    pad_len = window_size - 1
    probs = np.vstack([np.zeros((pad_len, 3)), probs])
    sizing = np.vstack([np.zeros((pad_len, 3)), sizing])
    magnitude = np.vstack([np.zeros((pad_len, 1)), magnitude])
    time_pred = np.vstack([np.zeros((pad_len, 1)), time_pred])
    
    df_features["ai_verdict"] = 1
    df_features["ai_confidence"] = np.maximum(probs[:, 0], probs[:, 2])
    df_features["ai_prob_long"] = probs[:, 0]
    df_features["ai_prob_short"] = probs[:, 2]
    df_features["ai_take_profit_pct"] = sizing[:, 1] * config.strategy.MAX_ATR_TARGET_PCT
    df_features["ai_stop_loss_pct"] = sizing[:, 2] * config.strategy.MAX_ATR_STOP_PCT
    df_features["ai_qty_ratio"] = sizing[:, 0]
    df_features["predicted_magnitude"] = magnitude[:, 0] * config.strategy.MAX_ATR_TARGET_PCT
    df_features["predicted_time_bars"] = time_pred[:, 0] * config.features.LOOKAHEAD_BARS
    
    long_threshold = 0.4
    short_threshold = 0.4
    if os.path.exists("models/short_thresholds.json"):
        with open("models/short_thresholds.json", "r") as f:
            t_data = json.load(f)
            long_threshold = t_data.get("long_probability_threshold", 0.4)
            short_threshold = t_data.get("short_probability_threshold", 0.4)
            
    df_features.loc[df_features["ai_prob_long"] > long_threshold, "ai_verdict"] = 0
    df_features.loc[df_features["ai_prob_short"] > short_threshold, "ai_verdict"] = 2
    
    # 5. Run Backtester
    test_bars = int(config.training.TEST_DATA_DAYS * (24 if config.data.INTERVAL == "1h" else 96))
    df_test = df_features.iloc[-test_bars:].copy()
    
    final_df, trades, summary = run_paper_portfolio_on_signals(
        panel=df_test,
        symbol=symbol,
        initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
        risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
        max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
        round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT
    )
    
    # 6. Premium Console Output
    _print_professional_backtest_summary(symbol, summary, trades)
    
    if trades:
        os.makedirs("data", exist_ok=True)
        trades_df = pd.DataFrame([vars(t) for t in trades])
        trades_df.to_csv(f"data/backtest_trades_{symbol}.csv", index=False)

def _print_professional_backtest_summary(symbol: str, summary: dict, trades: list):
    """Prints a simplified, professional backtest summary."""
    # Time Range Header
    time_range = Text.assemble(
        ("Period: ", "bold white"),
        (f"{summary.get('start_time', 'N/A')} ", "cyan"),
        ("to ", "bold white"),
        (f"{summary.get('end_time', 'N/A')}", "cyan"),
        (f" ({summary.get('duration_days', 0):.1f} Days)", "dim")
    )
    console.print(time_range)
    console.print("")

    # --- Calculate New Metrics ---
    initial = config.strategy.INITIAL_CAPITAL_USD
    pnl = summary.get('total_pnl_net_usd', 0)
    roi_pct = (pnl / initial) * 100.0 if initial > 0 else 0
    
    long_trades = sum(1 for t in trades if getattr(t, 'side', '').upper() == 'LONG' or (isinstance(t, dict) and str(t.get('side', '')).upper() == 'LONG'))
    short_trades = sum(1 for t in trades if getattr(t, 'side', '').upper() == 'SHORT' or (isinstance(t, dict) and str(t.get('side', '')).upper() == 'SHORT'))

    # 1. Main Performance Table
    from rich import box
    perf_table = Table(title="[bold cyan]Core Strategy Performance", show_header=True, header_style="bold magenta", box=box.MINIMAL)
    perf_table.add_column("Metric", style="dim")
    perf_table.add_column("Value", justify="right", style="bold")

    perf_table.add_row("Initial Capital", f"${initial:,.2f}")
    perf_table.add_row("Net Profit (PnL)", f"[green]+${pnl:,.2f}[/]" if pnl >= 0 else f"[red]-${abs(pnl):,.2f}[/]")
    perf_table.add_row("Final Equity", f"${initial + pnl:,.2f}")
    perf_table.add_row("ROI (%)", f"[green]{roi_pct:+.2f}%[/]" if roi_pct >= 0 else f"[red]{roi_pct:+.2f}%[/]")
    perf_table.add_row("Total Trades", f"{summary.get('trade_count', 0)}")
    perf_table.add_row("  └─ Long Trades", f"[cyan]{long_trades}[/]")
    perf_table.add_row("  └─ Short Trades", f"[magenta]{short_trades}[/]")
    perf_table.add_row("Win Rate", f"{summary.get('win_rate_pct', 0):.2f}%")

    # 2. Analytics Table
    analytics_table = Table(title="[bold yellow]Trade Analytics", show_header=True, header_style="bold yellow", box=box.MINIMAL)
    analytics_table.add_column("Parameter", style="dim")
    analytics_table.add_column("Value", justify="right", style="bold")
    
    analytics_table.add_row("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")
    analytics_table.add_row("Avg Win", f"[green]${summary.get('avg_win_usd', 0):,.2f}[/]")
    analytics_table.add_row("Avg Loss", f"[red]-${abs(summary.get('avg_loss_usd', 0)):,.2f}[/]")
    analytics_table.add_row("Avg Holding", f"{summary.get('avg_holding_bars', 0):.1f} Bars")
    analytics_table.add_row("Max Drawdown", f"[red]{summary.get('max_drawdown_pct', 0):.2f}%[/]")

    # 3. Model Quality Table
    rel_table = Table(title="[bold blue]Signal Quality", show_header=True, header_style="bold blue", box=box.MINIMAL)
    rel_table.add_column("Metric", style="dim")
    rel_table.add_column("Value", justify="right", style="bold")
    
    win_confs = [getattr(t, 'ai_confidence', 0) for t in trades if getattr(t, 'pnl_net_usd', 0) > 0]
    loss_confs = [getattr(t, 'ai_confidence', 0) for t in trades if getattr(t, 'pnl_net_usd', 0) <= 0]
    avg_win_conf = np.mean(win_confs) * 100 if win_confs else 0
    avg_loss_conf = np.mean(loss_confs) * 100 if loss_confs else 0
    
    rel_table.add_row("Avg Win Confidence", f"{avg_win_conf:.1f}%")
    rel_table.add_row("Avg Loss Confidence", f"{avg_loss_conf:.1f}%")
    rel_table.add_row("Reliability Gap", f"{avg_win_conf - avg_loss_conf:+.1f}%")

    console.print(Columns([perf_table, analytics_table, rel_table]))
    console.print("")

    if trades:
        trade_table = Table(title="Recent Trade History (AI Performance Details)", show_header=True, header_style="bold green")
        trade_table.add_column("#", justify="center", style="dim")
        trade_table.add_column("Side", justify="center")
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("Conf %", justify="right", style="bold yellow")
        trade_table.add_column("TP %", justify="right", style="green")
        trade_table.add_column("SL %", justify="right", style="red")
        trade_table.add_column("Dur", justify="right")
        trade_table.add_column("Outcome", justify="center")
        trade_table.add_column("Reason", style="bold italic")
        trade_table.add_column("Net PnL ($)", justify="right")

        start_idx = max(0, len(trades) - 20)
        for i, t in enumerate(trades[start_idx:]):
            pnl_val = getattr(t, 'pnl_net_usd', 0)
            res_style = "green" if pnl_val > 0 else "red"
            side = getattr(t, 'side', 'N/A')
            side_color = "cyan" if side == "LONG" else "magenta"
            
            trade_table.add_row(
                str(start_idx + i + 1),
                f"[{side_color}]{side}[/]",
                f"{getattr(t, 'entry_price', 0):.2f}",
                f"{getattr(t, 'exit_price', 0):.2f}",
                f"{getattr(t, 'ai_confidence', 0)*100:.1f}%",
                f"{getattr(t, 'take_profit_pct', 0):.2f}%",
                f"{getattr(t, 'stop_loss_pct', 0):.2f}%",
                f"{getattr(t, 'holding_bars', 0)}b",
                f"[{res_style}]{getattr(t, 'outcome', 'N/A')}[/]",
                getattr(t, 'exit_reason', 'N/A'),
                f"[{res_style}]{pnl_val:+.2f}[/]"
            )
        console.print(trade_table)

if __name__ == "__main__":
    run_backtest("ETHUSD")
