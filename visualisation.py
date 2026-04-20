# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def get_top_differentiators(df: pd.DataFrame, top_n: int = 15):
    """
    Automatically scans ALL numeric columns to find which features 
    best differentiate BUY signals from SELL signals.
    Returns a ranked DataFrame of insights.
    """
    if 'direction_label' not in df.columns:
        return pd.DataFrame()

    # Numeric columns only, excluding targets and technical IDs
    excluded = ['time', 'direction_label', 'upside_pct', 'downside_pct', 'future_drawdown_pct', 'label', 'index', 'ai_verdict']
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    
    # Calculate means and std
    grouped = df.groupby('direction_label')[cols].mean()
    total_std = df[cols].std()
    
    if 0.0 not in grouped.index or 2.0 not in grouped.index:
        return pd.DataFrame()
        
    # Differentiation Score: Normalized difference between Buy and Sell
    diff_score = abs(grouped.loc[0.0] - grouped.loc[2.0]) / (total_std + 1e-6)
    
    results = pd.DataFrame({
        'Feature': cols,
        'Buy Mean': grouped.loc[0.0].values,
        'Sell Mean': grouped.loc[2.0].values,
        'Diff Score': diff_score.values
    })
    
    return results.sort_values(by='Diff Score', ascending=False).head(top_n)

def print_signal_insights(df: pd.DataFrame):
    """
    Scans all columns and prints the top most important market conditions for signals.
    """
    top_diffs = get_top_differentiators(df, top_n=20)
    
    if top_diffs.empty:
        print("\nNo significant signal differentiators found.")
        return

    print("\n" + "="*70)
    print("🏆 TOP 20 MARKET CONDITIONS (INSIGHT ENGINE)")
    print("="*70)
    print(top_diffs[['Feature', 'Buy Mean', 'Sell Mean', 'Diff Score']].to_string(index=False))
    print("="*70 + "\n")

def plot_trading_analysis(df: pd.DataFrame, symbol: str = "Stock Data"):
    """
    Generate a professional trading dashboard with 6 panes.
    Rows: Price, Volume, MACD, RSI, Trend/Vol, and a Dynamic Insights Table.
    """
    # Get top 12 differentiators for the chart table
    insights = get_top_differentiators(df, top_n=12)

    # Create subplots: 6 rows
    fig = make_subplots(
        rows=6, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.015, 
        subplot_titles=(
            f'Price Analysis: {symbol}', 'Volume', 'MACD', 'RSI', 
            'Trend & Volatility Strength', '🏆 TOP SIGNAL INSIGHTS (Pattern Recognition)'
        ),
        row_heights=[0.35, 0.08, 0.12, 0.12, 0.13, 0.2],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]]
    )

    # 1. Main Price Chart (Row 1)
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Overlays
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='cyan', width=1), name='EMA 20'), row=1, col=1)
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='rgba(173,216,230,0.1)'), name='BB', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='rgba(173,216,230,0.1)'), fill='tonexty', showlegend=False), row=1, col=1)

    # 📊 Actual Signals
    if 'direction_label' in df.columns:
        buys = df[df['direction_label'] == 0]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='Actual BUY'), row=1, col=1)
        sells = df[df['direction_label'] == 2]
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Actual SELL'), row=1, col=1)

    # 🤖 AI Verdict Predictions
    if 'ai_verdict' in df.columns:
        # Successful AI Buys (Hit TP first)
        ai_buys = df[df['ai_verdict'] == 0]
        fig.add_trace(go.Scatter(x=ai_buys.index, y=ai_buys['Low']*0.97, mode='markers', marker=dict(symbol='circle', size=9, color='#00ff00', line=dict(width=1, color='white')), name='🤖 AI BUY Prediction'), row=1, col=1)
        
        # Failed AI Trades (Hit SL first)
        if 'ai_outcome' in df.columns:
            failed = df[(df['ai_outcome'] == 'FAILED')]
            fig.add_trace(go.Scatter(x=failed.index, y=df.loc[failed.index, 'Close'], mode='markers', marker=dict(symbol='x', size=10, color='yellow'), name='❌ Trade STOPPED OUT'), row=1, col=1)
            
        ai_sells = df[df['ai_verdict'] == 2]
        fig.add_trace(go.Scatter(x=ai_sells.index, y=ai_sells['High']*1.03, mode='markers', marker=dict(symbol='circle', size=9, color='#ff3333', line=dict(width=1, color='white')), name='🤖 AI SELL Prediction'), row=1, col=1)

    # 2. Volume (Row 2)
    vol_colors = np.where(df['Close'] >= df['Open'], 'green', 'red')
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='Volume'), row=2, col=1)

    # 3. MACD (Row 3)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Hist'), row=3, col=1)

    # 4. RSI (Row 4)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=1.5), name='RSI'), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    # 5. Trend/Vol Strength (Row 5)
    if 'ADX_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX_14'], line=dict(color='white', width=1.5), name='ADX'), row=5, col=1)
    if 'ATR_pct' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR_pct'], line=dict(color='red', width=1, dash='dot'), name='ATR %'), row=5, col=1)

    # 6. Insight Table
    if not insights.empty:
        fig.add_trace(
            go.Table(
                header=dict(values=["<b>Indicator Feature</b>", "<b>Buy Mean</b>", "<b>Sell Mean</b>", "<b>Diff Score</b>"], fill_color='rgba(100,100,100,0.5)', align='left'),
                cells=dict(values=[insights['Feature'], insights['Buy Mean'].round(4), insights['Sell Mean'].round(4), insights['Diff Score'].round(2)], fill_color='rgba(50,50,50,0.5)', align='left')
            ),
            row=6, col=1
        )

    fig.update_layout(template='plotly_dark', height=1400, hovermode='x unified', dragmode='pan')
    fig.show(config={'scrollZoom': True, 'displayModeBar': True})
