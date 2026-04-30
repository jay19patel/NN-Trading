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

    calibrated_thresholds = None
    if test_metrics:
        calibrated_thresholds = test_metrics.pop("calibrated_thresholds", None)
        console.print("\n[highlight]Held-out test metrics (direction + regression)[/highlight]")
        for metric_key, metric_value in test_metrics.items():
            if metric_key == "classification_report_text":
                console.print(metric_value)
            elif metric_key == "confusion_matrix_3x3":
                console.print(f"Confusion matrix [Buy, Neut, Sell]: {metric_value}")
            else:
                console.print(f"  {metric_key}: {metric_value}")

    console.print("\n[highlight]Per-symbol trading outcomes on last test window[/highlight]")
    predictions_frame = run_inference_with_confidence_filter(
        trained_model, test_X, device, calibrated_thresholds=calibrated_thresholds
    )

    bars_per_day = _bars_per_15m_day()
    test_row_count = config.training.TEST_DAYS * bars_per_day
    sequences_per_symbol = sequence_count_for_split(test_row_count, config.model.SEQ_LEN)

    summary_table = Table(
        title="Global model — summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Symbol", justify="left", style="bold")
    summary_table.add_column("Trades", justify="right")
    summary_table.add_column("Win %", justify="right")
    summary_table.add_column("W / L", justify="right")
    summary_table.add_column("Fees $", justify="right", style="red")
    summary_table.add_column("Net PnL $", justify="right")
    summary_table.add_column("Equity $", justify="right", style="bold green")

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
        actual_len = len(symbol_predictions)
        symbol_panel = symbol_frames[symbol].iloc[-actual_len:].copy()
        sequence_offset += sequences_per_symbol

        symbol_panel["ai_verdict"] = symbol_predictions["ai_verdict"].values
        symbol_panel["ai_confidence"] = symbol_predictions["ai_confidence"].values
        symbol_panel["ai_qty_ratio"] = symbol_predictions["ai_qty_ratio"].values
        symbol_panel["ai_take_profit_pct"] = symbol_predictions["ai_take_profit_pct"].values * 0.70 # 70% safety factor as requested
        symbol_panel["ai_stop_loss_pct"] = symbol_predictions["ai_stop_loss_pct"].values

        # Skip evaluate_ai_verdicts as run_paper_portfolio_on_signals now handles path simulation sequentially
        # symbol_panel = evaluate_ai_verdicts(symbol_panel, lookahead_bars=config.features.LOOKAHEAD_BARS)

        symbol_panel, trade_records, paper_summary = run_paper_portfolio_on_signals(
            symbol_panel,
            symbol=symbol,
            initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
            risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
            max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
            round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT,
        )
        all_trade_records.extend(trade_records)

        # Use sequential trade results for summary — use .get() to handle 0-trade case
        wins = paper_summary.get("wins", 0)
        losses = paper_summary.get("losses", 0)
        win_rate = paper_summary.get("win_rate_pct", 0.0)
        total_trades = paper_summary.get("trade_count", 0)

        summary_table.add_row(
            symbol,
            str(total_trades),
            f"{win_rate:.1f}%",
            f"{wins} / {losses}",
            f"{paper_summary.get('total_fees_usd', 0):.2f}",
            f"{paper_summary.get('total_pnl_net_usd', 0):.2f}",
            f"{paper_summary['final_equity_usd']:.2f}",
        )

        console.print("")

        # Track for global
        global_initial_capital += config.strategy.INITIAL_CAPITAL_USD
        global_final_capital += paper_summary['final_equity_usd']
        global_total_fees += paper_summary.get('total_fees_usd', 0)
        global_total_trades += paper_summary.get('trade_count', 0)
        global_total_wins += wins
        global_total_losses += losses

        # Track for plotting — handle case where paper_equity_curve may not exist (0 trades)
        if "paper_equity_curve" in symbol_panel.columns:
            equity_curve_vals = symbol_panel["paper_equity_curve"].values
        else:
            equity_curve_vals = [config.strategy.INITIAL_CAPITAL_USD] * len(symbol_panel)
        equity_curves[symbol] = {
            "dates": symbol_panel.index if isinstance(symbol_panel.index, pd.DatetimeIndex) else list(range(len(symbol_panel))),
            "equity": equity_curve_vals,
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
        f"{global_total_fees:.2f}",
        f"{(global_final_capital - global_initial_capital):.2f}",
        f"{global_final_capital:.2f}",
        style="bold yellow"
    )

    
    console.print(summary_table)



    if all_trade_records:
        df_trades = pd.DataFrame([vars(t) for t in all_trade_records])
        
        # 1. Format CSV data: Convert fraction to % and round everything for readability
        df_trades['return_pct'] = (df_trades.get('return_fraction', 0) * 100).round(2)
        if 'return_fraction' in df_trades:
            df_trades = df_trades.drop(columns=['return_fraction'])
            
        # Round percentages and model outputs
        for col in ['take_profit_pct', 'stop_loss_pct', 'ai_confidence', 'ai_qty_ratio']:
            if col in df_trades:
                df_trades[col] = df_trades[col].astype(float).round(2)
                
        # Round monetary and price values
        monetary_cols = ['entry_price', 'exit_price', 'notional_usd', 'pnl_before_fees_usd', 
                         'fees_usd', 'pnl_net_usd', 'capital_before_usd', 'equity_after_usd']
        for col in monetary_cols:
            if col in df_trades:
                df_trades[col] = df_trades[col].astype(float).round(2)
                
        # Crypto quantities need more precision
        if 'quantity' in df_trades:
            df_trades['quantity'] = df_trades['quantity'].astype(float).round(6)
            
        trades_csv_path = os.path.join(results_dir, "paper_trading_results.csv")
        df_trades.to_csv(trades_csv_path, index=False)
        
        trades_csv_path = os.path.join(results_dir, "paper_trading_results.csv")
        df_trades.to_csv(trades_csv_path, index=False)
        console.print(f"\n[bold green]✅ Saved formatted CSV to {trades_csv_path} ({len(df_trades)} trades)[/bold green]")

        
    # -----------------------------------------------------------------------
    # Final Summary Dashboard
    # -----------------------------------------------------------------------
    final_summary_table = Table(title="[bold gold]Final System Summary[/bold gold]", show_header=True, header_style="bold magenta")
    final_summary_table.add_column("Parameter", style="cyan")
    final_summary_table.add_column("Value", style="bold green")

    final_summary_table.add_row("Training Epochs", str(config.training.EPOCHS))
    final_summary_table.add_row("Training History Days", str(config.data.TOTAL_DAYS))
    final_summary_table.add_row("Backtest Testing Days", str(config.training.TEST_DAYS))
    final_summary_table.add_row("Total Initial Capital", f"${global_initial_capital:.2f}")
    final_summary_table.add_row("Total Final Capital", f"${global_final_capital:.2f}")
    final_summary_table.add_row("Net Profit/Loss", f"${(global_final_capital - global_initial_capital):.2f}")
    final_summary_table.add_row("Percentage Growth", f"{global_net_pnl_pct:.2f}%")
    final_summary_table.add_row("Total Trades (Global)", str(global_total_trades))
    final_summary_table.add_row("Overall Win Rate", f"{global_win_rate:.1f}%")
    final_summary_table.add_row("Total Fees Paid", f"${global_total_fees:.2f}")

    console.print("\n")
    console.print(final_summary_table)

        
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
