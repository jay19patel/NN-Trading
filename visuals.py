# -*- coding: utf-8 -*-
import os
import pandas as pd
import plotly.graph_objects as go
from config import config

def save_backtest_results(trade_records: list, equity_curves: dict, results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Save Trade CSV
    if trade_records:
        df_trades = pd.DataFrame([vars(t) for t in trade_records])
        # Format for readability
        df_trades['return_pct'] = (df_trades['return_fraction'] * 100).round(2)
        df_trades = df_trades.round(4)
        csv_path = os.path.join(results_dir, "trade_history.csv")
        df_trades.to_csv(csv_path, index=False)
        print(f"✅ Saved trade history to {csv_path}")

    # 2. Plot Equity Curve
    if equity_curves:
        fig = go.Figure()
        for symbol, data in equity_curves.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(data))), 
                y=data, 
                mode='lines', 
                name=f"{symbol} Equity"
            ))
        
        fig.update_layout(
            title="Perfect Oracle Capital Growth",
            xaxis_title="Time (Bars)",
            yaxis_title="Equity (USD)",
            template="plotly_dark",
            legend_title="Symbols"
        )
        
        chart_path = os.path.join(results_dir, "equity_curve.html")
        fig.write_html(chart_path)
        print(f"📈 Saved equity curve chart to {chart_path}")
