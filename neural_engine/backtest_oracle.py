# -*- coding: utf-8 -*-
"""
Oracle Backtester — Backtesting with perfect future knowledge.
==============================================================
Uses the OracleLabeler to generate signals that "know" the future,
resulting in a theoretical 100% win rate. Used for label verification.
"""
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from rich.text import Text
from rich.columns import Columns

from config import cfg
from ui_utils import console
from engine.data_handler import fetch_data
from neural_engine.feature_utils import add_technical_indicators
from neural_engine.labeler import OracleLabeler
from engine.backtester import run_paper_portfolio_on_signals
from neural_engine.backtest_engine import _print_professional_backtest_summary

logger = logging.getLogger("OracleBacktest")

def run_oracle_backtest(symbol: str) -> None:
    """Run a backtest using oracle labels instead of model predictions."""
    console.print(Text.assemble(
        ("\n", ""),
        ("╭─────────────────────╮\n", "bold yellow"),
        ("│  ORACLE BACKTEST    │\n", "bold yellow"),
        ("╰─ Perfect Knowledge ─╯", "bold yellow")
    ))

    # 1. Fetch data
    total_days = cfg.training.TEST_DATA_DAYS + 10
    df = fetch_data(symbol=symbol, total_days=total_days, interval=cfg.model.INTERVAL)
    if df.empty:
        console.print("[red]No data found.[/red]")
        return

    # 2. Add technical indicators (needed for ATR based labels)
    df_features = add_technical_indicators(df.copy())

    # 3. Generate Oracle Labels
    labeler = OracleLabeler()
    df_oracle = labeler.generate_labels(df_features)

    # 4. Map Oracle labels to backtester format
    # backtester expects: ai_verdict (0=long, 1=neutral, 2=short), ai_confidence, ai_take_profit_pct, ai_stop_loss_pct
    df_oracle["ai_verdict"] = df_oracle["direction_label"]
    df_oracle["ai_confidence"] = 1.0  # Oracle is 100% confident
    df_oracle["ai_take_profit_pct"] = df_oracle["take_profit_pct"]
    df_oracle["ai_stop_loss_pct"] = df_oracle["stop_loss_pct"]
    df_oracle["ai_qty_ratio"] = 1.0

    # 5. Run backtester on the test period
    bars_per_interval = 24 if cfg.model.INTERVAL == "1h" else 96 if cfg.model.INTERVAL == "15m" else 288 # 5m
    test_bars = int(cfg.training.TEST_DATA_DAYS * bars_per_interval)
    df_test = df_oracle.iloc[-test_bars:].copy()

    final_df, trades, summary = run_paper_portfolio_on_signals(
        panel=df_test,
        symbol=symbol,
        initial_capital_usd=cfg.testing.INITIAL_CAPITAL_USD,
        margin_per_trade_pct_of_equity=cfg.testing.MARGIN_PER_TRADE_PCT_OF_EQUITY,
        leverage=cfg.testing.LEVERAGE,
        round_trip_fee_pct=cfg.testing.ROUND_TRIP_FEE_PCT,
    )

    # 6. Print results
    _print_professional_backtest_summary(symbol, summary, trades)

    if trades:
        os.makedirs("data", exist_ok=True)
        trades_df = pd.DataFrame([vars(t) for t in trades])
        trades_df.to_csv(f"data/oracle_trades_{symbol}.csv", index=False)
        console.print(f"\n[green]Oracle trades saved to data/oracle_trades_{symbol}.csv[/green]")

if __name__ == "__main__":
    run_oracle_backtest("ETHUSD")
