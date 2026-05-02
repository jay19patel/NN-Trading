# -*- coding: utf-8 -*-
import os
import json
import numpy as np
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

    # 2. Build the Advanced tabbed HTML Report
    html_content = _generate_tabbed_platform_html(df_trades, equity_curves, all_candles)
    
    report_path = os.path.join(results_dir, "info.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"🚀 Custom Analytics Platform saved to {report_path}")

def _generate_tabbed_platform_html(df_trades: pd.DataFrame, equity_curves: dict, all_candles: dict) -> str:
    # --- STATISTICS CALCULATIONS ---
    total_pnl = df_trades['pnl_net_usd'].sum() if not df_trades.empty else 0
    total_trades = len(df_trades)
    
    # Global PnL and Growth
    initial_total = config.strategy.INITIAL_CAPITAL_USD * len(equity_curves)
    final_total = 0
    
    # Portfolio Equity Curve Calculation
    portfolio_equity = []
    if equity_curves:
        max_len = max(len(c) for c in equity_curves.values())
        synchronized_curves = []
        for s, c in equity_curves.items():
            final_total += c[-1]
            if len(c) < max_len:
                c = np.pad(c, (0, max_len - len(c)), mode='edge')
            synchronized_curves.append(c)
        portfolio_equity = np.sum(synchronized_curves, axis=0).tolist()

    total_growth_pct = ((final_total - initial_total) / initial_total * 100) if initial_total > 0 else 0

    # Buy/Sell Stats
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']
    
    long_wins = len(long_trades[long_trades['pnl_net_usd'] > 0])
    short_wins = len(short_trades[short_trades['pnl_net_usd'] > 0])
    
    # Max Drawdown Calculation for Portfolio
    mdd_pct = 0
    if portfolio_equity:
        pe = np.array(portfolio_equity)
        peak = pe[0]
        max_dd = 0
        for val in pe:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
        mdd_pct = max_dd * 100

    # JSON Data
    equity_json = { "GLOBAL": portfolio_equity }
    for sym, curve in equity_curves.items():
        equity_json[sym] = curve.tolist()

    candles_json = {}
    for sym, df in all_candles.items():
        mini_df = df[['Open', 'High', 'Low', 'Close']].copy()
        mini_df['Date'] = df.index.astype(str)
        candles_json[sym] = mini_df.to_dict(orient='records')

    trades_json = df_trades.to_dict(orient='records') if not df_trades.empty else []

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oracle Pro - Premium Analytics</title>
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
            height: 50px;
            background: var(--card-bg);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            flex-shrink: 0;
        }}
        .tabs-container {{
            display: flex;
            gap: 10px;
            padding: 10px 20px;
            background: #0d1117;
            border-bottom: 1px solid var(--border);
        }}
        .tab-btn {{
            padding: 8px 16px;
            background: transparent;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: 0.2s;
        }}
        .tab-btn:hover {{ background: #ffffff05; }}
        .tab-btn.active {{
            background: var(--accent);
            color: #000;
            border-color: var(--accent);
        }}
        .tab-content {{
            display: none;
            flex-grow: 1;
            overflow: hidden;
        }}
        .tab-content.active {{
            display: flex;
        }}
        
        /* Terminal Tab */
        .sidebar {{
            width: 300px;
            background: var(--card-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }}
        .main-view {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            padding: 10px;
            gap: 10px;
            position: relative;
        }}
        
        /* Analytics Tab */
        .analytics-view {{
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card .val {{ font-size: 18px; font-weight: 800; color: #fff; }}
        .stat-card .lab {{ font-size: 10px; opacity: 0.5; text-transform: uppercase; margin-top: 4px; }}
        
        .chart-container {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
        }}
        
        /* Trades List */
        .trade-card {{ padding: 10px 15px; border-bottom: 1px solid var(--border); cursor: pointer; }}
        .trade-card:hover {{ background: #ffffff05; }}
        .trade-card.active {{ background: #58a6ff15; border-left: 4px solid var(--accent); }}
        .badge {{ padding: 2px 5px; border-radius: 3px; font-size: 9px; font-weight: 800; }}
        .long-badge {{ background: #3fb95022; color: var(--success); }}
        .short-badge {{ background: #f8514922; color: var(--danger); }}
        
        .trade-overlay {{
            position: absolute; top: 15px; right: 15px; width: 220px;
            background: rgba(22, 27, 34, 0.95); border: 1px solid var(--accent);
            border-radius: 8px; padding: 12px; z-index: 50; display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
        
        ::-webkit-scrollbar {{ width: 4px; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); }}
    </style>
</head>
<body>

<div class="navbar">
    <div style="font-weight: 800; color: var(--accent); font-size: 16px;">ORACLE PRO ANALYTICS</div>
    <div style="font-size: 12px;">Total Portfolio PnL: <span style="color:var(--success); font-weight:800;">${total_pnl:,.2f}</span></div>
</div>

<div class="tabs-container">
    <button class="tab-btn active" onclick="switchTab('trading')">Trading Terminal</button>
    <button class="tab-btn" onclick="switchTab('analytics')">Portfolio Analytics</button>
</div>

<!-- TAB 1: TERMINAL -->
<div id="tradingTab" class="tab-content active">
    <div class="sidebar">
        <div style="padding: 12px 15px; font-size: 11px; font-weight: 800; border-bottom: 1px solid var(--border);">SYMBOL</div>
        <select id="symbolPicker" onchange="loadSymbol(this.value)" style="margin: 8px 15px; background:#0d1117; color:#fff; border:1px solid var(--border); border-radius:4px; padding:5px; font-size:12px;">
            {_generate_symbol_options(all_candles)}
        </select>
        <div style="padding: 8px 15px; font-size: 10px; opacity: 0.5; border-bottom: 1px solid var(--border);">TRADES ({total_trades})</div>
        <div class="trades-container" style="overflow-y: auto; flex-grow: 1;">
            {_generate_trade_cards(df_trades)}
        </div>
    </div>
    <div class="main-view">
        <div class="trade-overlay" id="tradeOverlay">
            <div style="font-weight:800; font-size:11px; color:var(--accent); margin-bottom:8px; border-bottom:1px solid var(--border); padding-bottom:4px;">TRADE INFO</div>
            <div id="overlayContent" style="font-size:11px; display:flex; flex-direction:column; gap:5px;"></div>
        </div>
        <div id="mainChart" style="flex-grow: 1; background: var(--card-bg); border-radius: 6px; border: 1px solid var(--border);"></div>
    </div>
</div>

<!-- TAB 2: ANALYTICS -->
<div id="analyticsTab" class="tab-content">
    <div class="analytics-view">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="val" style="color:var(--success)">{total_growth_pct:.2f}%</div>
                <div class="lab">Total Combined Growth</div>
            </div>
            <div class="stat-card">
                <div class="val" style="color:var(--danger)">{mdd_pct:.2f}%</div>
                <div class="lab">Portfolio Max Drawdown</div>
            </div>
            <div class="stat-card">
                <div class="val">{total_trades}</div>
                <div class="lab">Total Portfolio Trades</div>
            </div>
            <div class="stat-card">
                <div class="val">{((long_wins+short_wins)/max(1, total_trades)*100):.1f}%</div>
                <div class="lab">Portfolio Win Rate</div>
            </div>
        </div>

        <div class="chart-container">
            <div id="bigEquityChart" style="height: 450px;"></div>
        </div>
        
        <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
            <div class="stat-card">
                <div class="val" style="color:var(--success)">{long_wins} / {len(long_trades)}</div>
                <div class="lab">Long Wins / Total Longs</div>
            </div>
            <div class="stat-card">
                <div class="val" style="color:var(--success)">{short_wins} / {len(short_trades)}</div>
                <div class="lab">Short Wins / Total Shorts</div>
            </div>
        </div>
    </div>
</div>

<script>
    const candlesData = {json.dumps(candles_json)};
    const tradesData = {json.dumps(trades_json)};
    const equityData = {json.dumps(equity_json)};

    let currentSym = '';
    let selectedTradeIdx = -1;

    function switchTab(tab) {{
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        if (tab === 'trading') {{
            document.querySelector('.tab-btn[onclick*="trading"]').classList.add('active');
            document.getElementById('tradingTab').classList.add('active');
            Plotly.Plots.resize('mainChart');
        }} else {{
            document.querySelector('.tab-btn[onclick*="analytics"]').classList.add('active');
            document.getElementById('analyticsTab').classList.add('active');
            renderBigEquity();
        }}
    }}

    function renderBigEquity() {{
        const traces = [];
        // Global Portfolio Line First (Heavier)
        if (equityData["GLOBAL"]) {{
            traces.push({{
                y: equityData["GLOBAL"], name: "TOTAL PORTFOLIO", type: 'scatter', mode: 'lines',
                line: {{ width: 4, color: '#fff' }},
                fill: 'tozeroy', fillcolor: 'rgba(255, 255, 255, 0.05)'
            }});
        }}
        // Individual Symbol Lines
        for (const s in equityData) {{
            if (s === "GLOBAL") continue;
            traces.push({{
                y: equityData[s], name: s + " GROWTH", type: 'scatter', mode: 'lines',
                line: {{ width: 2 }},
                opacity: 0.7
            }});
        }}
        Plotly.newPlot('bigEquityChart', traces, {{
            title: {{ text: 'Compounded Growth Breakdown', font: {{ size: 14, color: '#fff' }} }},
            template: 'plotly_dark',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {{ gridcolor: '#30363d', title: 'Timeline (Bars)' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Equity (USD)' }},
            margin: {{ t: 50, b: 50, l: 70, r: 30 }},
            legend: {{ orientation: 'h', y: -0.2 }}
        }});
    }}

    function loadSymbol(symbol, zoomTrade = null) {{
        currentSym = symbol;
        const df = candlesData[symbol];
        const trace = {{
            x: df.map(c => c.Date),
            open: df.map(c => c.Open), high: df.map(c => c.High),
            low: df.map(c => c.Low), close: df.map(c => c.Close),
            type: 'candlestick', name: symbol,
            increasing: {{ line: {{ color: '#3fb950' }} }},
            decreasing: {{ line: {{ color: '#f85149' }} }}
        }};

        const layout = {{
            dragmode: 'pan', template: 'plotly_dark',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 10, b: 30, l: 50, r: 10 }},
            xaxis: {{ rangeslider: {{ visible: false }}, gridcolor: '#30363d' }},
            yaxis: {{ gridcolor: '#30363d' }},
            shapes: [], annotations: []
        }};

        if (zoomTrade) {{
            const start = Math.max(0, zoomTrade.entry_index - 20);
            const end = Math.min(df.length - 1, zoomTrade.entry_index + 80);
            layout.xaxis.range = [df[start].Date, df[end].Date];
            
            // Tight Y scaling
            const window = df.slice(start, end);
            const high = Math.max(...window.map(c => c.High));
            const low = Math.min(...window.map(c => c.Low));
            const pad = (high - low) * 0.15;
            layout.yaxis.range = [low - pad, high + pad];

            layout.shapes.push({{
                type: 'line', x0: zoomTrade.entry_datetime, x1: zoomTrade.exit_datetime,
                y0: zoomTrade.entry_price, y1: zoomTrade.entry_price,
                line: {{ color: '#58a6ff', width: 2, dash: 'dot' }}
            }});
            
            document.getElementById('tradeOverlay').style.display = 'block';
            document.getElementById('overlayContent').innerHTML = `
                <div style="display:flex; justify-content:space-between;"><span>Price:</span> <span style="color:#fff;">$${{zoomTrade.entry_price.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Target:</span> <span style="color:var(--success);">$${{zoomTrade.tp_price.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Stop:</span> <span style="color:var(--danger);">$${{zoomTrade.sl_price.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between; font-weight:800; border-top:1px solid var(--border); padding-top:4px; margin-top:4px;"><span>NET PNL:</span> <span style="color:${{zoomTrade.pnl_net_usd > 0 ? 'var(--success)' : 'var(--danger)'}};">$${{zoomTrade.pnl_net_usd.toFixed(2)}}</span></div>
            `;
        }} else {{
            document.getElementById('tradeOverlay').style.display = 'none';
        }}

        Plotly.newPlot('mainChart', [trace], layout, {{ scrollZoom: true, displayModeBar: true }});
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

    // Initial load
    const first = Object.keys(candlesData)[0];
    if (first) loadSymbol(first);
    document.querySelectorAll('.trade-card').forEach((c, i) => c.onclick = () => selectTrade(i));
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
                    <span style="font-weight:800; color:#fff; font-size:12px;">{row['symbol']}</span>
                    <span style="color:{pnl_col}; font-weight:800; font-size:12px;">${row['pnl_net_usd']:.2f}</span>
                </div>
                <div style="margin-top:4px; font-size:10px; display:flex; justify-content:space-between; opacity:0.6;">
                    <span><span class="badge {badge}">{row['side']}</span></span>
                    <span>{row['entry_datetime'].split(' ')[1]}</span>
                </div>
            </div>
        """)
    return "".join(cards)

def _generate_symbol_options(all_candles: dict) -> str:
    return "".join([f'<option value="{s}">{s}</option>' for s in all_candles.keys()])
