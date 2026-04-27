# -*- coding: utf-8 -*-
import time
import os
import warnings
import torch
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import config
from data_gathering import fetch_data
from features import create_full_feature_set
from paper_trading import (
    describe_last_candle_and_model_outputs,
    run_paper_portfolio_on_signals,
)
from training_utils import (
    build_target_arrays,
    create_sequences,
    prepare_multi_symbol_data,
    predict_model_outputs_for_single_window,
    train_model,
    run_inference_with_confidence_filter,
)
from evaluation_metrics import sequence_count_for_split
from ui_utils import console, print_banner, Table

warnings.filterwarnings("ignore")


def _bars_per_15m_day() -> int:
    return (24 * 60) // 15


def evaluate_ai_verdicts(dataframe: pd.DataFrame, lookahead_bars: int) -> pd.DataFrame:
    """
    Path simulation using model-predicted TP%/SL% per row (fallback to fixed strategy % if missing).

    Same-bar touch: stop is assumed first (conservative). Populates ai_return_fraction for PnL math.
    """
    dataframe = dataframe.copy()
    if "ai_take_profit_pct" not in dataframe.columns:
        dataframe["ai_take_profit_pct"] = config.strategy.TARGET_PROFIT_PCT
    if "ai_stop_loss_pct" not in dataframe.columns:
        dataframe["ai_stop_loss_pct"] = config.strategy.STOP_LOSS_PCT

    dataframe["ai_outcome"] = "NONE"
    dataframe["ai_return_fraction"] = 0.0

    close_prices = dataframe["Close"].values
    high_prices = dataframe["High"].values
    low_prices = dataframe["Low"].values
    verdict_values = dataframe["ai_verdict"].values
    target_pct_per_row = dataframe["ai_take_profit_pct"].values.astype(np.float64)
    stop_pct_per_row = dataframe["ai_stop_loss_pct"].values.astype(np.float64)

    row_count = len(dataframe)
    outcomes = ["NONE"] * row_count
    return_fractions = [0.0] * row_count

    for row_index in range(row_count - lookahead_bars):
        verdict = verdict_values[row_index]
        if verdict == 1:
            continue

        entry_price = close_prices[row_index]
        target_pct = float(target_pct_per_row[row_index])
        stop_pct = float(stop_pct_per_row[row_index])

        path_outcome = "TIMEOUT"
        if verdict == 0:
            take_profit_price = entry_price * (1 + target_pct / 100)
            stop_loss_price = entry_price * (1 - stop_pct / 100)
            for step in range(1, lookahead_bars + 1):
                bar_high = high_prices[row_index + step]
                bar_low = low_prices[row_index + step]
                hit_stop = bar_low <= stop_loss_price
                hit_target = bar_high >= take_profit_price
                if hit_stop and hit_target:
                    path_outcome = "FAILED"
                    break
                if hit_stop:
                    path_outcome = "FAILED"
                    break
                if hit_target:
                    path_outcome = "SUCCESS"
                    break
        elif verdict == 2:
            take_profit_price = entry_price * (1 - target_pct / 100)
            stop_loss_price = entry_price * (1 + stop_pct / 100)
            for step in range(1, lookahead_bars + 1):
                bar_high = high_prices[row_index + step]
                bar_low = low_prices[row_index + step]
                hit_stop = bar_high >= stop_loss_price
                hit_target = bar_low <= take_profit_price
                if hit_stop and hit_target:
                    path_outcome = "FAILED"
                    break
                if hit_stop:
                    path_outcome = "FAILED"
                    break
                if hit_target:
                    path_outcome = "SUCCESS"
                    break
        else:
            continue

        outcomes[row_index] = path_outcome
        exit_close = close_prices[row_index + lookahead_bars]
        if path_outcome == "SUCCESS":
            return_fraction = target_pct / 100.0
        elif path_outcome == "FAILED":
            return_fraction = -stop_pct / 100.0
        elif verdict == 0:
            return_fraction = (exit_close - entry_price) / entry_price
        else:
            return_fraction = (entry_price - exit_close) / entry_price

        return_fractions[row_index] = return_fraction

    dataframe["ai_outcome"] = outcomes
    dataframe["ai_return_fraction"] = return_fractions
    return dataframe


def main() -> None:
    device = config.DEVICE
    symbols = config.data.SYMBOLS
    interval = config.data.INTERVAL

    print_banner(
        "GLOBAL AI TRADING SYSTEM",
        f"Symbols: {', '.join(symbols)} | Device: {device}",
    )

    symbol_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        console.print(f"[info]Fetching data for [bold]{symbol}[/bold]...[/info]")
        market_frame = fetch_data(
            symbol=symbol, total_days=config.data.TOTAL_DAYS, interval=interval
        )
        if market_frame.empty:
            console.print(f"[error]No data for {symbol}. Skipping.[/error]")
            continue

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        cache_name = os.path.join(data_dir, f"features_cache_{symbol}_{interval}.csv")
        lookahead = config.features.LOOKAHEAD_BARS
        if os.path.exists(cache_name):
            cache_age_minutes = (time.time() - os.path.getmtime(cache_name)) / 60
            if cache_age_minutes < config.data.CACHE_VALID_MINS:
                console.print(f"  Using cache for {symbol}")
                features_frame = pd.read_csv(cache_name, index_col=0, parse_dates=True)

                # Invalidate if oracle columns missing (stale ATR-based cache)
                oracle_cols_missing = "oracle_tp_pct" not in features_frame.columns
                label_cols_missing  = "label_take_profit_pct" not in features_frame.columns
                if oracle_cols_missing or label_cols_missing:
                    reason = "oracle labels" if oracle_cols_missing else "dynamic TP/SL labels"
                    console.print(
                        f"  [warning]Cache missing {reason}; rebuilding {symbol}...[/warning]"
                    )
                    features_frame = create_full_feature_set(market_frame, lookahead=lookahead)
                    features_frame.to_csv(cache_name)
            else:
                console.print(f"  Rebuilding features for {symbol}...")
                features_frame = create_full_feature_set(market_frame, lookahead=lookahead)
                features_frame.to_csv(cache_name)
        else:
            features_frame = create_full_feature_set(market_frame, lookahead=lookahead)
            features_frame.to_csv(cache_name)

        symbol_frames[symbol] = features_frame

    if not symbol_frames:
        console.print("[error]No symbol data available. Exiting.[/error]")
        return

    (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        feature_names,
        feature_scaler,
    ) = prepare_multi_symbol_data(
        symbol_frames,
        test_days=config.training.TEST_DAYS,
        val_days=config.training.VAL_DAYS,
    )

    console.print(
        f"\n[highlight]Train/Val/Test tensors: {len(train_X)} / {len(val_X)} / {len(test_X)} windows[/highlight]"
    )
    console.print(f"[info]{len(feature_names)} input features | seq_len={config.model.SEQ_LEN}[/info]")

    trained_model, test_metrics = train_model(
        train_X,
        train_y,
        device,
        len(feature_names),
        val_features=val_X,
        val_targets=val_y,
        test_features=test_X,
        test_targets=test_y,
        epochs=config.training.EPOCHS,
    )
    
    # Save the trained model
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "trading_model_v1.pt")
    torch.save(trained_model.state_dict(), model_path)
    console.print(f"\n[bold green]💾 Model weights saved successfully to {model_path}[/bold green]")

    if test_metrics:
        console.print("\n[highlight]Held-out test metrics (direction + regression)[/highlight]")
        for metric_key, metric_value in test_metrics.items():
            if metric_key == "classification_report_text":
                console.print(metric_value)
            elif metric_key == "confusion_matrix_3x3":
                console.print(f"Confusion matrix [Buy, Neut, Sell]: {metric_value}")
            else:
                console.print(f"  {metric_key}: {metric_value}")

    console.print("\n[highlight]Per-symbol trading outcomes on last test window[/highlight]")
    predictions_frame = run_inference_with_confidence_filter(trained_model, test_X, device)

    bars_per_day = _bars_per_15m_day()
    test_row_count = config.training.TEST_DAYS * bars_per_day
    sequences_per_symbol = sequence_count_for_split(test_row_count, config.model.SEQ_LEN)

    summary_table = Table(
        title="Global model — summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Symbol", justify="left", style="bold")
    summary_table.add_column("Signals", justify="right")
    summary_table.add_column("Win rate (TP first)", justify="right")
    summary_table.add_column("W / L", justify="right")
    summary_table.add_column("Paper $", justify="right")

    # Create results directory
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    # Global tracking
    global_initial_capital = 0.0
    global_final_capital = 0.0
    global_total_fees = 0.0
    global_total_trades = 0
    global_total_wins = 0
    global_total_losses = 0

    sequence_offset = 0
    all_trade_records = []
    
    # For plotting
    equity_curves = {}
    symbol_panels_data = {} # To store panels for detailed plotting
    
    for symbol in symbols:
        if symbol not in symbol_frames:
            continue

        symbol_predictions = predictions_frame.iloc[
            sequence_offset : sequence_offset + sequences_per_symbol
        ]
        symbol_panel = symbol_frames[symbol].iloc[-sequences_per_symbol:].copy()
        sequence_offset += sequences_per_symbol

        symbol_panel["ai_verdict"] = symbol_predictions["ai_verdict"].values
        symbol_panel["ai_confidence"] = symbol_predictions["ai_confidence"].values
        symbol_panel["ai_qty_ratio"] = symbol_predictions["ai_qty_ratio"].values
        symbol_panel["ai_take_profit_pct"] = symbol_predictions["ai_take_profit_pct"].values
        symbol_panel["ai_stop_loss_pct"] = symbol_predictions["ai_stop_loss_pct"].values

        symbol_panel = evaluate_ai_verdicts(
            symbol_panel,
            lookahead_bars=config.features.LOOKAHEAD_BARS,
        )

        symbol_panel, trade_records, paper_summary = run_paper_portfolio_on_signals(
            symbol_panel,
            symbol=symbol,
            initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
            risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
            max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
            round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT,
        )
        all_trade_records.extend(trade_records)

        buy_rows = symbol_panel[symbol_panel["ai_verdict"] == 0]
        sell_rows = symbol_panel[symbol_panel["ai_verdict"] == 2]
        signal_rows = pd.concat([buy_rows, sell_rows])
        resolved = signal_rows[signal_rows["ai_outcome"].isin(["SUCCESS", "FAILED"])]
        wins = int((resolved["ai_outcome"] == "SUCCESS").sum())
        losses = int((resolved["ai_outcome"] == "FAILED").sum())
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

        summary_table.add_row(
            symbol,
            str(len(signal_rows)),
            f"{win_rate:.1f}%",
            f"{wins} / {losses}",
            f"{paper_summary['final_equity_usd']:.2f}",
        )

        detail_table = Table(
            title=f"{symbol} — last {config.training.TEST_DAYS} days (strategy filter)",
            show_header=True,
            header_style="bold magenta",
        )
        detail_table.add_column("Metric", justify="left")
        detail_table.add_column("Buy", justify="right", style="buy")
        detail_table.add_column("Sell", justify="right", style="sell")
        detail_table.add_column("Total", justify="right", style="bold")

        buy_wins = int((buy_rows["ai_outcome"] == "SUCCESS").sum())
        buy_losses = int((buy_rows["ai_outcome"] == "FAILED").sum())
        sell_wins = int((sell_rows["ai_outcome"] == "SUCCESS").sum())
        sell_losses = int((sell_rows["ai_outcome"] == "FAILED").sum())

        detail_table.add_row("Signals", str(len(buy_rows)), str(len(sell_rows)), str(len(signal_rows)))
        detail_table.add_row("Wins", str(buy_wins), str(sell_wins), str(wins))
        detail_table.add_row("Losses", str(buy_losses), str(sell_losses), str(losses))
        detail_table.add_row(
            "Paper final $",
            "—",
            "—",
            f"{paper_summary['final_equity_usd']:.2f}",
        )
        initial_cap = config.strategy.INITIAL_CAPITAL_USD
        final_cap = paper_summary['final_equity_usd']
        net_pnl_pct = ((final_cap - initial_cap) / initial_cap) * 100
        total_fees = paper_summary.get('total_fees_usd', 0)
        fees_pct = (total_fees / initial_cap) * 100

        detail_table.add_row(
            "Total Trades",
            "—",
            "—",
            str(paper_summary['trade_count']),
        )
        detail_table.add_row(
            "Net PnL %",
            "—",
            "—",
            f"{net_pnl_pct:.2f}%",
        )
        detail_table.add_row(
            "Total Fees %",
            "—",
            "—",
            f"{fees_pct:.2f}%",
        )
        detail_table.add_row(
            "Loss→next win",
            "—",
            "—",
            str(paper_summary.get("loss_then_win_count", 0)),
        )
        console.print(detail_table)
        console.print(
            f"[info]{symbol} paper book: trades={paper_summary['trade_count']} | "
            f"fees≈${paper_summary.get('total_fees_usd', 0):.2f} | "
            f"net PnL≈${paper_summary.get('total_pnl_net_usd', 0):.2f}[/info]"
        )
        console.print("")
        
        # Track for global
        global_initial_capital += config.strategy.INITIAL_CAPITAL_USD
        global_final_capital += paper_summary['final_equity_usd']
        global_total_fees += paper_summary.get('total_fees_usd', 0)
        global_total_trades += paper_summary['trade_count']
        global_total_wins += wins
        global_total_losses += losses
        
        # Track for plotting
        equity_curves[symbol] = {
            "dates": symbol_panel.index if isinstance(symbol_panel.index, pd.DatetimeIndex) else list(range(len(symbol_panel))),
            "equity": symbol_panel["paper_equity_curve"].values
        }
        symbol_panels_data[symbol] = symbol_panel.copy()

    # Add global row
    global_win_rate = (global_total_wins / (global_total_wins + global_total_losses) * 100) if (global_total_wins + global_total_losses) > 0 else 0.0
    global_net_pnl_pct = ((global_final_capital - global_initial_capital) / global_initial_capital * 100) if global_initial_capital > 0 else 0.0
    global_fees_pct = (global_total_fees / global_initial_capital * 100) if global_initial_capital > 0 else 0.0
    
    summary_table.add_row(
        "GLOBAL TOTAL",
        str(global_total_trades),
        f"{global_win_rate:.1f}%",
        f"{global_total_wins} / {global_total_losses}",
        f"${global_final_capital:.2f}",
        style="bold yellow"
    )
    
    console.print(summary_table)
    
    global_stats_table = Table(title="Global Portfolio Performance", show_header=True, header_style="bold green")
    global_stats_table.add_column("Metric", justify="left")
    global_stats_table.add_column("Value", justify="right", style="bold")
    
    global_stats_table.add_row("Total Initial Capital", f"${global_initial_capital:.2f}")
    global_stats_table.add_row("Total Final Capital", f"${global_final_capital:.2f}")
    global_stats_table.add_row("Global Net PnL %", f"{global_net_pnl_pct:.2f}%")
    global_stats_table.add_row("Total Fees Paid", f"${global_total_fees:.2f}")
    global_stats_table.add_row("Fees as % of Capital", f"{global_fees_pct:.2f}%")
    global_stats_table.add_row("Total Trades Taken", str(global_total_trades))
    
    console.print(global_stats_table)

    if all_trade_records:
        df_trades = pd.DataFrame([vars(t) for t in all_trade_records])
        trades_csv_path = os.path.join(results_dir, "paper_trading_results.csv")
        df_trades.to_csv(trades_csv_path, index=False)
        console.print(f"\n[bold green]✅ Saved detailed trade analysis to {trades_csv_path} ({len(df_trades)} trades)[/bold green]")
        
    # Generate Advanced Visualizations
    from plotly.subplots import make_subplots
    
    # 1. Portfolio Equity Curve (Individual + Global Total)
    fig_equity = go.Figure()
    
    # Calculate Global Total Equity Curve
    first_sym = symbols[0]
    if first_sym in equity_curves:
        total_equity_curve = np.zeros_like(equity_curves[first_sym]["equity"])
        dates = equity_curves[first_sym]["dates"]
        
        for sym, data in equity_curves.items():
            fig_equity.add_trace(go.Scatter(x=data["dates"], y=data["equity"], mode='lines', name=f"{sym} Equity", line=dict(width=1.5, dash='dot')))
            # Add to total if shapes match
            if len(data["equity"]) == len(total_equity_curve):
                total_equity_curve += data["equity"]
        
        # Add the BOLD Global Total line
        fig_equity.add_trace(go.Scatter(x=dates, y=total_equity_curve, mode='lines', name="TOTAL PORTFOLIO", line=dict(color='gold', width=4)))

    fig_equity.update_layout(
        title="Portfolio Capital Growth (Individual & Global Total)",
        xaxis_title="Date/Time", yaxis_title="Account Equity (USD)",
        template="plotly_dark", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_equity.write_html(os.path.join(results_dir, "portfolio_equity_curve.html"))

    # 2. Detailed Trade Visualizer (Price + Markers)
    for sym, panel in symbol_panels_data.items():
        fig_trades = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, subplot_titles=(f"{sym} Price & Trades", "Equity Growth"),
                                   row_width=[0.3, 0.7])
        
        # Price Trace
        fig_trades.add_trace(go.Scatter(x=panel.index, y=panel['Close'], name='Price', line=dict(color='gray', width=1)), row=1, col=1)
        
        # Add Trade Markers
        buys = panel[panel['ai_verdict'] == 0]
        sells = panel[panel['ai_verdict'] == 2]
        
        fig_trades.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy Signal',
                                        marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        fig_trades.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell Signal',
                                        marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)
        
        # Equity Trace on Subplot 2
        fig_trades.add_trace(go.Scatter(x=panel.index, y=panel['paper_equity_curve'], name='Equity', line=dict(color='cyan')), row=2, col=1)
        
        fig_trades.update_layout(height=800, title_text=f"{sym} Detailed Trade Analytics", template="plotly_dark")
        fig_trades.write_html(os.path.join(results_dir, f"{sym}_detailed_analytics.html"))

    console.print(f"\n[bold cyan]📊 All visualizations saved to the '{results_dir}' folder:[/bold cyan]")
    console.print(f"  - portfolio_equity_curve.html (Individual + TOTAL)")
    for sym in symbols:
        console.print(f"  - {sym}_detailed_analytics.html (Price + Buy/Sell Markers)")
        
    console.print(
        f"\n[info]Model evaluation completed on {config.training.TEST_DAYS} held-out days. | lookahead={config.features.LOOKAHEAD_BARS} bars | "
        f"paper start=${config.strategy.INITIAL_CAPITAL_USD:.0f} "
        f"(risk {config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY}% equity / trade)[/info]"
    )

    demo_symbol = next((symbol for symbol in symbols if symbol in symbol_frames), None)
    if demo_symbol is not None:
        test_row_count = config.training.TEST_DAYS * bars_per_day
        cleaned_tail = (
            symbol_frames[demo_symbol]
            .iloc[-test_row_count:]
            .copy()
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )
        scaled_matrix = feature_scaler.transform(cleaned_tail[feature_names])
        tail_targets = build_target_arrays(cleaned_tail)
        demo_windows, _ = create_sequences(
            scaled_matrix, tail_targets, sequence_length=config.model.SEQ_LEN
        )
        last_feature_window = demo_windows[-1]
        last_candle_row = cleaned_tail.iloc[-1]
        model_output_map = predict_model_outputs_for_single_window(
            trained_model, last_feature_window, device
        )
        console.print("\n[highlight]Production-style preview (last test candle)[/highlight]")
        console.print(
            describe_last_candle_and_model_outputs(last_candle_row, model_output_map)
        )
        console.print(
            "\n[info]Online 'learn from last mistake' is not applied inside one training run; "
            "the model learns average TP/SL from labels. Use periodic retraining or a "
            "separate calibration layer for live adaptation.[/info]\n"
        )


if __name__ == "__main__":
    main()
