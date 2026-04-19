# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_trading_analysis(df: pd.DataFrame, symbol: str = "Stock Data"):
    """
    Generate a comprehensive, interactive trading dashboard using Plotly.
    Includes Price, Volume, MACD, and RSI with technical indicators and signals.
    """
    # Create subplots: 4 rows, shared x-axis
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=(f'Price Analysis: {symbol}', 'Volume', 'MACD', 'RSI'),
        row_heights=[0.5, 0.15, 0.175, 0.175]
    )

    # 1. Main Price Chart (Row 1)
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', showlegend=True
        ),
        row=1, col=1
    )

    # EMAs
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='cyan', width=1), name='EMA 20'), row=1, col=1)
    if 'EMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='yellow', width=1), name='EMA 50'), row=1, col=1)

    # Bollinger Bands
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='rgba(173, 216, 230, 0.2)'), name='BB Upper', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='rgba(173, 216, 230, 0.2)'), fill='tonexty', name='BB Lower', showlegend=False), row=1, col=1)

    # Supertrend
    if 'supertrend' in df.columns:
        # We can color it based on direction if direction exists
        color = 'magenta'
        fig.add_trace(go.Scatter(x=df.index, y=df['supertrend'], line=dict(color=color, width=1.5, dash='dash'), name='Supertrend'), row=1, col=1)

    # Buy/Sell Signals
    if 'direction_label' in df.columns:
        # Buy signals (Label 0)
        buys = df[df['direction_label'] == 0]
        fig.add_trace(
            go.Scatter(
                x=buys.index, y=buys['Low'] * 0.99, 
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'),
                name='BUY Signal'
            ),
            row=1, col=1
        )
        # Sell signals (Label 2)
        sells = df[df['direction_label'] == 2]
        fig.add_trace(
            go.Scatter(
                x=sells.index, y=sells['High'] * 1.01, 
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                name='SELL Signal'
            ),
            row=1, col=1
        )

    # 2. Volume (Row 2)
    colors = ['green' if df['Close'][i] >= df['Open'][i] else 'red' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'),
        row=2, col=1
    )

    # 3. MACD (Row 3)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='orange', width=1), name='Signal'), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Hist'), row=3, col=1)

    # 4. RSI (Row 4)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=1), name='RSI'), row=4, col=1)
        # Overbought/Oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    # Update Layout for a "TradingView" feel
    fig.update_layout(
        template='plotly_dark',
        title=f'Advanced Trading Analysis Dashboard: {symbol}',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',  # Shows all indicator values for the same time on hover
        dragmode='pan',         # Default to pan for smoother navigation
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # 5. Advanced Interactivity (Spikelines / Crosshairs)
    fig.update_xaxes(
        showspikes=True, 
        spikecolor="gray", 
        spikethickness=1, 
        spikesnap="cursor", 
        spikemode="across",
        spikedash="dash"
    )
    fig.update_yaxes(
        showspikes=True, 
        spikecolor="gray", 
        spikethickness=1, 
        spikesnap="cursor", 
        spikemode="across",
        spikedash="dash"
    )

    # Final touch: Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)

    # Show chart with scrollZoom enabled
    fig.show(config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    })
