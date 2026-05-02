# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from config import config

def save_backtest_results(trade_records: list, equity_curves: dict, all_candles: dict, strategy_name: str, results_root: str = "results"):
    """Saves raw backtest data as JSON for the Flask dashboard."""
    results_dir = os.path.join(results_root, strategy_name.lower())
    os.makedirs(results_dir, exist_ok=True)
    
    df_trades = pd.DataFrame()
    trades_json = []
    if trade_records:
        df_trades = pd.DataFrame([vars(t) for t in trade_records])
        df_trades['return_pct'] = (df_trades['return_fraction'] * 100).round(2)
        df_trades = df_trades.round(4)
        
        # Save CSV for legacy/audit
        csv_path = os.path.join(results_dir, "trade_history.csv")
        df_trades.to_csv(csv_path, index=False)
        trades_json = df_trades.to_dict(orient='records')

    # Prepare Equity Curves
    initial_total = config.strategy.INITIAL_CAPITAL_USD * len(equity_curves)
    portfolio_equity = []
    global_timeline = []
    
    if equity_curves:
        # Get the timeline from the longest candle set (they should all be synchronized but just in case)
        first_sym = list(all_candles.keys())[0]
        global_timeline = all_candles[first_sym].index.astype(str).tolist()
        
        max_len = max(len(c) for c in equity_curves.values())
        synchronized_curves = []
        for s, c in equity_curves.items():
            if len(c) < max_len:
                c = np.pad(c, (0, max_len - len(c)), mode='edge')
            synchronized_curves.append(c)
        portfolio_equity = np.sum(synchronized_curves, axis=0).tolist()

    equity_json = { "GLOBAL": portfolio_equity }
    for sym, curve in equity_curves.items():
        equity_json[sym] = curve.tolist()

    # Prepare Candles
    candles_json = {}
    for sym, df in all_candles.items():
        mini_df = df[['Open', 'High', 'Low', 'Close']].copy()
        mini_df['Date'] = df.index.astype(str)
        candles_json[sym] = mini_df.to_dict(orient='records')

    # Save everything into a single JSON for the Flask app to read
    master_data = {
        "strategy_name": strategy_name,
        "trades": trades_json,
        "equity": equity_json,
        "candles": candles_json,
        "timeline": global_timeline,
        "stats": {
            "total_pnl": float(df_trades['pnl_net_usd'].sum()) if not df_trades.empty else 0.0,
            "total_trades": len(df_trades),
            "initial_capital": float(initial_total),
            "final_equity": portfolio_equity[-1] if portfolio_equity else float(initial_total)
        }
    }

    json_path = os.path.join(results_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f)
    
    print(f"✅ Saved {strategy_name} raw results to {json_path}")
