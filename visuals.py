# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import plotly.graph_objects as go
from config import config

def save_backtest_results(trade_records: list, equity_curves: dict, all_candles: dict, results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Save Trade CSV
    df_trades = pd.DataFrame()
    if trade_records:
        df_trades = pd.DataFrame([vars(t) for t in trade_records])
        df_trades['return_pct'] = (df_trades['return_fraction'] * 100).round(2)
        df_trades = df_trades.round(4)
        csv_path = os.path.join(results_dir, "trade_history.csv")
        df_trades.to_csv(csv_path, index=False)
        print(f"✅ Saved trade history to {csv_path}")

    # 2. Build the Custom Advanced HTML Report
    html_content = _generate_custom_platform_html(df_trades, equity_curves, all_candles)
    
    report_path = os.path.join(results_dir, "info.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"🚀 Custom Analytics Platform saved to {report_path}")

def _generate_custom_platform_html(df_trades: pd.DataFrame, equity_curves: dict, all_candles: dict) -> str:
    total_pnl = df_trades['pnl_net_usd'].sum() if not df_trades.empty else 0
    win_rate = (len(df_trades[df_trades['pnl_net_usd'] > 0]) / len(df_trades) * 100) if not df_trades.empty else 0
    total_trades = len(df_trades)
    
    # Equity Curve JSON
    equity_data = {}
    for sym, curve in equity_curves.items():
        equity_data[sym] = curve.tolist()

    # Candles JSON
    candles_json = {}
    for sym, df in all_candles.items():
        mini_df = df[['Open', 'High', 'Low', 'Close']].copy()
        mini_df['Date'] = df.index.astype(str)
        candles_json[sym] = mini_df.to_dict(orient='records')

    # Trades JSON
    trades_json = df_trades.to_dict(orient='records') if not df_trades.empty else []

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oracle Pro - Advanced Trading View</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --accent: #58a6ff;
            --text: #c9d1d9;
            --success: #3fb950;
            --danger: #f85149;
            --border: #30363d;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            margin: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .navbar {{
            height: 60px;
            background: var(--card-bg);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            flex-shrink: 0;
            z-index: 100;
        }}
        .main-content {{
            display: flex;
            flex-grow: 1;
            overflow: hidden;
        }}
        .sidebar {{
            width: 320px;
            background: var(--card-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }}
        .chart-area {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg);
            padding: 10px;
            gap: 10px;
            overflow: hidden;
            position: relative;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 15px;
            border-bottom: 1px solid var(--border);
        }}
        .stat-item {{
            background: #ffffff05;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-val {{ font-size: 16px; font-weight: 800; color: #fff; }}
        .stat-lab {{ font-size: 10px; opacity: 0.5; margin-top: 2px; }}

        .trades-container {{
            flex-grow: 1;
            overflow-y: auto;
        }}
        .trade-card {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: 0.2s;
        }}
        .trade-card:hover {{ background: #ffffff08; }}
        .trade-card.active {{ background: #58a6ff22; border-left: 4px solid var(--accent); }}
        
        .badge {{
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 800;
        }}
        .long-badge {{ background: #3fb95022; color: var(--success); }}
        .short-badge {{ background: #f8514922; color: var(--danger); }}
        
        .chart-window {{
            background: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border);
            flex-grow: 1;
            position: relative;
        }}
        .equity-mini {{
            height: 150px;
            background: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border);
            flex-shrink: 0;
        }}

        /* Corner Overlay */
        .trade-overlay {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 240px;
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 15px;
            z-index: 50;
            display: none;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }}
        .overlay-title {{ font-weight: 800; color: var(--accent); font-size: 14px; margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 5px; }}
        .overlay-row {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; }}
        .overlay-val {{ font-weight: 600; color: #fff; }}

        .chart-hint {{
            position: absolute;
            bottom: 50px;
            left: 20px;
            font-size: 10px;
            opacity: 0.4;
            pointer-events: none;
            color: #fff;
        }}
        
        ::-webkit-scrollbar {{ width: 4px; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); }}
        
        .btn-action {{
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid var(--border);
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: 0.2s;
        }}
        .btn-action:hover {{ background: #30363d; border-color: #8b949e; }}
    </style>
</head>
<body>

<div class="navbar">
    <div style="display:flex; align-items:center; gap:15px;">
        <span style="font-weight: 800; color: var(--accent); font-size: 20px;">ORACLE PRO</span>
        <select id="symbolPicker" onchange="loadSymbol(this.value)" style="background:#0d1117; color:#fff; border:1px solid var(--border); border-radius:4px; padding:4px 8px;">
            {_generate_symbol_options(all_candles)}
        </select>
        <button class="btn-action" onclick="resetChart()">Reset View</button>
    </div>
    <div style="display:flex; gap:30px; font-size: 13px;">
        <div>PNL: <span style="color:var(--success); font-weight:800;">${total_pnl:,.2f}</span></div>
        <div>WIN RATE: <span style="color:var(--accent); font-weight:800;">{win_rate:.1f}%</span></div>
    </div>
</div>

<div class="main-content">
    <div class="sidebar">
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-val">{total_trades}</div>
                <div class="stat-lab">TOTAL TRADES</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">${config.strategy.INITIAL_CAPITAL_USD}</div>
                <div class="stat-lab">CAPITAL</div>
            </div>
        </div>
        <div class="trades-container" id="tradesList">
            {_generate_trade_cards(df_trades)}
        </div>
    </div>
    
    <div class="chart-area">
        <div class="trade-overlay" id="tradeOverlay">
            <div class="overlay-title">TRADE DETAILS</div>
            <div class="overlay-row"><span>Entry:</span> <span class="overlay-val" id="ov-entry"></span></div>
            <div class="overlay-row"><span>Target:</span> <span class="overlay-val" id="ov-target" style="color:var(--success)"></span></div>
            <div class="overlay-row"><span>Stoploss:</span> <span class="overlay-val" id="ov-sl" style="color:var(--danger)"></span></div>
            <div class="overlay-row"><span>Net PnL:</span> <span class="overlay-val" id="ov-pnl"></span></div>
            <div class="overlay-row"><span>Time:</span> <span class="overlay-val" id="ov-time"></span></div>
        </div>
        <div class="chart-window" id="mainChart"></div>
        <div class="chart-hint">Tip: Drag X/Y axis to stretch. Use mouse wheel to zoom. Drag chart to pan.</div>
        <div class="equity-mini" id="equityChart"></div>
    </div>
</div>

<script>
    const candlesData = {json.dumps(candles_json)};
    const tradesData = {json.dumps(trades_json)};
    const equityData = {json.dumps(equity_data)};

    let currentSym = '';
    let selectedTradeIdx = -1;

    // --- 1. EQUITY CHART ---
    const equityTraces = [];
    for(const s in equityData) {{
        equityTraces.push({{
            y: equityData[s], name: s, type: 'scatter', mode: 'lines',
            line: {{ width: 2, color: '#58a6ff' }},
            fill: 'tozeroy', fillcolor: 'rgba(88, 166, 255, 0.1)'
        }});
    }}
    Plotly.newPlot('equityChart', equityTraces, {{
        title: {{ text: 'Equity Curve', font: {{ size: 11, color: '#8b949e' }} }},
        margin: {{ t: 30, b: 20, l: 40, r: 20 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark',
        xaxis: {{ showgrid: false, fixedrange: true }},
        yaxis: {{ gridcolor: '#30363d', fixedrange: true }}
    }}, {{displayModeBar: false}});

    // --- 2. MAIN CANDLESTICK CHART ---
    function loadSymbol(symbol, zoomTrade = null) {{
        currentSym = symbol;
        const df = candlesData[symbol];
        
        const trace = {{
            x: df.map(c => c.Date),
            open: df.map(c => c.Open),
            high: df.map(c => c.High),
            low: df.map(c => c.Low),
            close: df.map(c => c.Close),
            type: 'candlestick',
            name: symbol,
            increasing: {{ line: {{ color: '#3fb950' }} }},
            decreasing: {{ line: {{ color: '#f85149' }} }}
        }};

        const layout = {{
            title: {{ text: symbol + ' - Performance View', font: {{ color: '#fff', size: 14 }} }},
            dragmode: 'pan',
            template: 'plotly_dark',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 40, b: 60, l: 60, r: 10 }},
            xaxis: {{ 
                rangeslider: {{ visible: false }},
                gridcolor: '#30363d',
                fixedrange: false, // Allows stretching
                automargin: true
            }},
            yaxis: {{ 
                gridcolor: '#30363d',
                fixedrange: false, // Allows stretching
                automargin: true
            }},
            shapes: [],
            annotations: []
        }};

        const overlay = document.getElementById('tradeOverlay');

        if (zoomTrade) {{
            const start = Math.max(0, zoomTrade.entry_index - 30);
            const end = Math.min(df.length - 1, zoomTrade.entry_index + 100);
            layout.xaxis.range = [df[start].Date, df[end].Date];
            
            // Auto-scale Y-axis for zoom
            const visibleCandles = df.slice(start, end);
            const highs = visibleCandles.map(c => c.High);
            const lows = visibleCandles.map(c => c.Low);
            const minY = Math.min(...lows);
            const maxY = Math.max(...highs);
            const padding = (maxY - minY) * 0.2;
            layout.yaxis.range = [minY - padding, maxY + padding];

            // Add Trade Visuals
            layout.shapes.push({{
                type: 'line', x0: zoomTrade.entry_datetime, x1: zoomTrade.exit_datetime,
                y0: zoomTrade.entry_price, y1: zoomTrade.entry_price,
                line: {{ color: '#58a6ff', width: 2, dash: 'dot' }}
            }});
            
            layout.annotations.push({{
                x: zoomTrade.entry_datetime, y: zoomTrade.entry_price,
                text: 'BUY ENTRY', showarrow: true, arrowhead: 2, font: {{ color: '#58a6ff' }}
            }});
            
            layout.annotations.push({{
                x: zoomTrade.exit_datetime, y: zoomTrade.exit_price,
                text: zoomTrade.pnl_net_usd > 0 ? 'WIN' : 'LOSS',
                showarrow: true, arrowhead: 2,
                font: {{ color: zoomTrade.pnl_net_usd > 0 ? '#3fb950' : '#f85149' }}
            }});

            overlay.style.display = 'block';
            document.getElementById('ov-entry').innerText = '$' + zoomTrade.entry_price.toFixed(2);
            document.getElementById('ov-target').innerText = '$' + zoomTrade.tp_price.toFixed(2);
            document.getElementById('ov-sl').innerText = '$' + zoomTrade.sl_price.toFixed(2);
            document.getElementById('ov-pnl').innerText = '$' + zoomTrade.pnl_net_usd.toFixed(2);
            document.getElementById('ov-pnl').style.color = zoomTrade.pnl_net_usd > 0 ? 'var(--success)' : 'var(--danger)';
            document.getElementById('ov-time').innerText = zoomTrade.entry_datetime.split(' ')[1];
        }} else {{
            overlay.style.display = 'none';
        }}

        Plotly.newPlot('mainChart', [trace], layout, {{
            scrollZoom: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d'],
            responsive: true
        }});
    }}

    function selectTrade(idx) {{
        const cards = document.querySelectorAll('.trade-card');
        if (selectedTradeIdx === idx) {{
            selectedTradeIdx = -1;
            cards[idx].classList.remove('active');
            loadSymbol(currentSym);
        }} else {{
            selectedTradeIdx = idx;
            const trade = tradesData[idx];
            document.getElementById('symbolPicker').value = trade.symbol;
            loadSymbol(trade.symbol, trade);
            cards.forEach(c => c.classList.remove('active'));
            cards[idx].classList.add('active');
        }}
    }}

    function resetChart() {{
        selectedTradeIdx = -1;
        document.querySelectorAll('.trade-card').forEach(c => c.classList.remove('active'));
        loadSymbol(currentSym);
    }}

    // Initial load
    const first = Object.keys(candlesData)[0];
    if (first) loadSymbol(first);

    document.querySelectorAll('.trade-card').forEach((c, i) => {{
        c.onclick = () => selectTrade(i);
    }});
</script>

</body>
</html>
"""

def _generate_trade_cards(df: pd.DataFrame) -> str:
    if df.empty: return "<div style='padding:20px; opacity:0.5;'>No trades</div>"
    cards = []
    for i, row in df.iterrows():
        pnl_col = "var(--success)" if row['pnl_net_usd'] > 0 else "var(--danger)"
        badge = "long-badge" if row['side'] == 'LONG' else "short-badge"
        cards.append(f"""
            <div class="trade-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:800; color:#fff;">{row['symbol']}</span>
                    <span style="color:{pnl_col}; font-weight:800;">${row['pnl_net_usd']:.2f}</span>
                </div>
                <div style="margin-top:5px; font-size:11px; display:flex; flex-direction:column; gap:3px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span><span class="badge {badge}">{row['side']}</span></span>
                        <span style="opacity:0.5;">{row['entry_datetime'].split(' ')[1]}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; opacity:0.7; font-size:10px;">
                        <span>TP: ${row['tp_price']:.2f}</span>
                        <span>SL: ${row['sl_price']:.2f}</span>
                    </div>
                </div>
            </div>
        """)
    return "".join(cards)

def _generate_symbol_options(all_candles: dict) -> str:
    return "".join([f'<option value="{s}">{s}</option>' for s in all_candles.keys()])
