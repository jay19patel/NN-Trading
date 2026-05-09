# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from rich.panel import Panel

from engine.data_handler import fetch_data
from neural_engine.labeler import OracleLabeler
from engine.backtester import run_paper_portfolio_on_signals
from config import config
from ui_utils import console

def run_oracle_perfect_backtest(symbol: str = "ETHUSD"):
    """Run backtest using perfect Oracle labels instead of AI predictions."""
    console.print(Panel(f"[bold gold1]ORACLE PERFECT BACKTEST: {symbol}", subtitle="Maximum Theoretical Profitability (100% Accuracy)", expand=False))
    
    # 1. Fetch Data for Test Period
    # We take the last 30 days to match your current test config
    total_days = config.training.TEST_DATA_DAYS + 10
    df = fetch_data(symbol=symbol, total_days=total_days, interval=config.data.INTERVAL)
    if df.empty:
        console.print("[red]No data found.")
        return

    # 2. Generate Oracle Labels (The 'Perfect' signals)
    labeler = OracleLabeler()
    df_labeled = labeler.generate_labels(df.copy())
    
    # 3. Convert Labels to Signal Format for Backtester
    # Oracle's Class 2 is SHORT, Class 0 is LONG, Class 1 is NEUTRAL
    df_labeled["ai_verdict"] = df_labeled["direction_label"]
    df_labeled["ai_confidence"] = 1.0  # Oracle is 100% sure
    df_labeled["ai_take_profit_pct"] = df_labeled["take_profit_pct"]
    df_labeled["ai_stop_loss_pct"] = df_labeled["stop_loss_pct"]
    df_labeled["ai_qty_ratio"] = 1.0   # Full size
    
    # 4. Run Backtester on the signals
    test_bars = int(config.training.TEST_DATA_DAYS * (24 if config.data.INTERVAL == "1h" else 96))
    df_test = df_labeled.iloc[-test_bars:].copy()
    
    final_df, trades, summary = run_paper_portfolio_on_signals(
        panel=df_test,
        symbol=symbol,
        initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
        risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
        max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
        round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT
    )
    
    # 5. Display Results
    from neural_engine.backtest_short import _print_professional_backtest_summary
    _print_professional_backtest_summary(symbol, summary, trades)

if __name__ == "__main__":
    run_oracle_perfect_backtest("ETHUSD")
