import os
import sys
import pandas as pd

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.data_handler import fetch_data
from engine.backtester import run_paper_portfolio_on_signals
from strategies.short_nn_strategy import ShortNNStrategy
from config import config
from ui_utils import console

def run_short_backtest():
    symbol = config.data.SYMBOLS[0]
    days = config.data.TOTAL_DAYS
    interval = config.data.INTERVAL
    
    # 1. Fetch Data
    console.print(f"[info]Fetching last {days} days for testing (unseen data)...[/info]")
    df = fetch_data(symbol, days, interval)
    if df.empty:
        console.print("[error]No data for backtest.[/error]")
        return
        
    # 2. Initialize Strategy
    model_path = "models/short_model_eth.pth"
    mean_path = "models/scaler_mean.npy"
    scale_path = "models/scaler_scale.npy"
    
    if not os.path.exists(model_path):
        console.print("[error]Model not found. Run training first.[/error]")
        return
        
    strategy = ShortNNStrategy(model_path, mean_path, scale_path)
    
    # 3. Generate Signals
    console.print(f"[info]Generating signals using {strategy.name}...[/info]")
    df_with_signals = strategy.generate_signals(df)
    
    # 4. Run Backtest
    console.print(f"[info]Running backtest for {symbol}...[/info]")
    from engine.backtester import run_paper_portfolio_on_signals, print_rich_summary
    from engine.visuals import save_backtest_results
    
    final_df, trade_ledger, summary = run_paper_portfolio_on_signals(
        panel=df_with_signals,
        symbol=symbol,
        initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
        risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
        max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
        round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT
    )
    
    # 5. Save results for UI
    equity_curves = {symbol: final_df["paper_equity_curve"].values}
    all_candles = {symbol: df}
    save_backtest_results(trade_ledger, equity_curves, all_candles, strategy.name)
    
    # 6. Show Summary
    print_rich_summary(strategy.name, symbol, summary)

if __name__ == "__main__":
    run_short_backtest()
