# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from config import config
from ui_utils import console
from engine.data_handler import get_data_for_symbols
from engine.backtester import run_paper_portfolio_on_signals, print_rich_summary
from engine.visuals import save_backtest_results
from strategies.oracle import OracleStrategy
from strategies.random import RandomStrategy
from strategies.short_nn_strategy import ShortNNStrategy

def run_strategy_backtest(strategy, all_candles):
    console.rule(f"[bold cyan]{strategy.name.upper()} STRATEGY BACKTEST")
    
    all_trade_records = []
    equity_curves = {}
    
    for symbol, df in all_candles.items():
        console.print(f"Processing {symbol} with {strategy.name}...")
        
        # 1. Generate Signals
        df_with_signals = strategy.generate_signals(df.copy())
        
        # 2. Run Backtest
        final_df, trade_records, summary = run_paper_portfolio_on_signals(
            panel=df_with_signals,
            symbol=symbol,
            initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
            risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
            max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
            round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT
        )
        
        all_trade_records.extend(trade_records)
        equity_curves[symbol] = final_df["paper_equity_curve"].values
        
        # Professional Console Summary
        print_rich_summary(strategy.name, symbol, summary)

    # 3. Save results to results/<strategy_name>/
    save_backtest_results(all_trade_records, equity_curves, all_candles, strategy.name)

def ensure_model_trained():
    """Checks if the Short NN model exists, if not, runs training."""
    model_path = "models/short_model_eth.pth"
    if not os.path.exists(model_path):
        console.rule("[bold yellow]TRAINING SHORT-ONLY MODEL")
        console.print("[info]Model not found. Initializing auto-training sequence...[/info]")
        from training_short_only.train import train_short_model
        train_short_model()
        console.print("[success]✅ Training Complete![/success]\n")

def main():
    console.clear()
    console.rule("[bold magenta]ETHUSD AI TRADING ENGINE")
    
    # Ensure the short model is ready
    ensure_model_trained()
    
    symbols = config.data.SYMBOLS
    days = config.data.TOTAL_DAYS
    interval = config.data.INTERVAL
    
    # 1. Fetch Data Once
    all_candles = get_data_for_symbols(symbols, days, interval)
    
    # 2. Run different strategies
    # Load model-based strategies here so they can find the files we just trained
    model_path = "models/short_model_eth.pth"
    mean_path = "models/scaler_mean.npy"
    scale_path = "models/scaler_scale.npy"
    
    strategies_to_run = [
        OracleStrategy(),
        ShortNNStrategy(model_path, mean_path, scale_path, threshold_path="models/short_thresholds.json"),
        RandomStrategy()
    ]
    
    for strategy in strategies_to_run:
        run_strategy_backtest(strategy, all_candles)

    # 4. Final summary
    console.print("\n[bold green]───────────────────────── ALL SIMULATIONS COMPLETE ──────────────────────────[/bold green]")
    console.print("Check the [bold]results/[/bold] folder for raw data.")
    
    # 5. Start Dashboard Server
    console.print("\n[bold cyan]🚀 Starting Dynamic Analytics Dashboard...[/bold cyan]")
    console.print("[info]Access your results at: [bold underline]http://127.0.0.1:5001[/bold underline][/info]\n")
    
    from web.app import app
    app.run(debug=False, port=5001)

if __name__ == "__main__":
    main()
