# -*- coding: utf-8 -*-
import os
import pandas as pd
from config import config
from ui_utils import console
from engine.data_handler import get_data_for_symbols
from engine.backtester import run_paper_portfolio_on_signals
from engine.visuals import save_backtest_results
from strategies.oracle import OracleStrategy
from strategies.random import RandomStrategy

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
        from engine.backtester import print_rich_summary
        print_rich_summary(strategy.name, symbol, summary)

    # 3. Save results to results/<strategy_name>/
    save_backtest_results(all_trade_records, equity_curves, all_candles, strategy.name)

def main():
    console.clear()
    console.rule("[bold magenta]MULTI-STRATEGY TRADING ENGINE")
    
    symbols = config.data.SYMBOLS
    days = config.data.TOTAL_DAYS
    interval = config.data.INTERVAL
    
    # 1. Fetch Data Once
    all_candles = get_data_for_symbols(symbols, days, interval)
    
    # 2. Run different strategies
    strategies_to_run = [
        OracleStrategy(),
        RandomStrategy()
    ]
    
    for strategy in strategies_to_run:
        run_strategy_backtest(strategy, all_candles)

    console.rule("[bold green]ALL SIMULATIONS COMPLETE")
    console.print(f"Check the [bold cyan]results/[/bold cyan] folder for strategy reports.")

if __name__ == "__main__":
    main()
