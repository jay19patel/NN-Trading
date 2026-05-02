# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import config

def save_backtest_results(trade_records: list, equity_curves: dict, all_candles: dict, strategy_name: str, results_root: str = "results"):
    results_dir = os.path.join(results_root, strategy_name.lower())
    os.makedirs(results_dir, exist_ok=True)
    
    df_trades = pd.DataFrame()
    if trade_records:
        df_trades = pd.DataFrame([vars(t) for t in trade_records])
        df_trades['return_pct'] = (df_trades['return_fraction'] * 100).round(2)
        df_trades = df_trades.round(4)
        csv_path = os.path.join(results_dir, "trade_history.csv")
        df_trades.to_csv(csv_path, index=False)
        print(f"✅ Saved {strategy_name} trade history to {csv_path}")

    html_content = _generate_tabbed_platform_html(df_trades, equity_curves, all_candles, strategy_name)
    report_path = os.path.join(results_dir, "info.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"🚀 {strategy_name} Analytics Platform saved to {report_path}")

def _generate_tabbed_platform_html(df_trades: pd.DataFrame, equity_curves: dict, all_candles: dict, strategy_name: str) -> str:
    total_pnl = df_trades['pnl_net_usd'].sum() if not df_trades.empty else 0
    total_trades = len(df_trades)
    
    initial_total = config.strategy.INITIAL_CAPITAL_USD * len(equity_curves)
    final_total = 0
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

    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']
    long_wins = len(long_trades[long_trades['pnl_net_usd'] > 0])
    short_wins = len(short_trades[short_trades['pnl_net_usd'] > 0])
    
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
    <title>Oracle Pro - {strategy_name} Analytics</title>
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
        }}
        .tab-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
        .tab-content {{ display: none; flex-grow: 1; overflow: hidden; }}
        .tab-content.active {{ display: flex; }}
        
        .sidebar {{
            width: 300px;
            background: var(--card-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }}
        .sidebar-header {{
            padding: 12px 15px;
            font-size: 11px;
            font-weight: 800;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .filter-section {{
            padding: 10px 15px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .filter-group {{ display: flex; gap: 5px; }}
        .filter-btn {{
            flex: 1;
            padding: 6px;
            font-size: 10px;
            background: #21262d;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text);
            cursor: pointer;
            font-weight: 600;
        }}
        .filter-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
        
        .main-view {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            padding: 10px;
            gap: 10px;
            position: relative;
        }}
        
        .trade-card {{ padding: 12px 15px; border-bottom: 1px solid var(--border); cursor: pointer; display: flex; flex-direction: column; gap: 4px; }}
        .trade-card.hidden {{ display: none; }}
        .trade-card:hover {{ background: #ffffff05; }}
        .trade-card.active {{ background: #58a6ff15; border-left: 4px solid var(--accent); }}
        .badge {{ padding: 2px 5px; border-radius: 3px; font-size: 9px; font-weight: 800; }}
        .long-badge {{ background: #3fb95022; color: var(--success); }}
        .short-badge {{ background: #f8514922; color: var(--danger); }}
        
        .trade-overlay {{
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            width: 360px; background: rgba(22, 27, 34, 0.98); border: 1px solid var(--accent);
            border-radius: 12px; padding: 15px; z-index: 50; display: none;
            box-shadow: 0 12px 48px rgba(0,0,0,0.9);
            backdrop-filter: blur(10px);
        }}
        
        ::-webkit-scrollbar {{ width: 4px; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); }}
    </style>
</head>
<body>

<div class="navbar">
    <div style="font-weight: 800; color: var(--accent); font-size: 16px;">{strategy_name.upper()} TERMINAL</div>
    <div style="font-size: 12px;">Net Portfolio PnL: <span style="color:var(--success); font-weight:800;">${total_pnl:,.2f}</span></div>
</div>

<div class="tabs-container">
    <button class="tab-btn active" onclick="switchTab('trading')">Terminal View</button>
    <button class="tab-btn" onclick="switchTab('analytics')">Portfolio Metrics</button>
</div>

<div id="tradingTab" class="tab-content active">
    <div class="sidebar">
        <div class="sidebar-header">Filters & Sorting</div>
        <div class="filter-section">
            <div style="font-size: 10px; opacity: 0.5;">SYMBOL</div>
            <select id="symbolPicker" onchange="updateFilters()" style="width:100%; background:#0d1117; color:#fff; border:1px solid var(--border); border-radius:4px; padding:6px; font-size:12px;">
                <option value="ALL">ALL SYMBOLS</option>
                {_generate_symbol_options(all_candles)}
            </select>
            
            <div style="font-size: 10px; opacity: 0.5; margin-top:5px;">SIDE</div>
            <div class="filter-group">
                <button id="filter-ALL" class="filter-btn active" onclick="setSideFilter('ALL')">ALL</button>
                <button id="filter-LONG" class="filter-btn" onclick="setSideFilter('LONG')">LONG</button>
                <button id="filter-SHORT" class="filter-btn" onclick="setSideFilter('SHORT')">SHORT</button>
            </div>
            
            <div style="font-size: 10px; opacity: 0.5; margin-top:5px;">SORT BY</div>
            <div class="filter-group">
                <button id="sort-TIME" class="filter-btn active" onclick="setSort('TIME')">TIME</button>
                <button id="sort-PNL" class="filter-btn" onclick="setSort('PNL')">PNL</button>
            </div>
        </div>

        <div id="tradesContainer" class="trades-container" style="overflow-y: auto; flex-grow: 1;">
            {_generate_trade_cards(df_trades)}
        </div>
    </div>
    <div class="main-view">
        <div id="mainChart" style="flex-grow: 1; background: var(--card-bg); border-radius: 6px; border: 1px solid var(--border);"></div>
        
        <div class="trade-overlay" id="tradeOverlay">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; border-bottom:1px solid var(--border); padding-bottom:5px;">
                <span style="font-weight:800; font-size:11px; color:var(--accent);">EXECUTION DETAILS</span>
                <span style="font-size:14px; color:#fff; cursor:pointer; font-weight:800;" onclick="selectTrade(selectedTradeIdx)">×</span>
            </div>
            <div id="overlayContent" style="font-size:11px; display:flex; flex-direction:column; gap:6px;"></div>
        </div>
    </div>
</div>

<div id="analyticsTab" class="tab-content">
    <div class="analytics-view">
        <div class="stats-grid">
            <div class="stat-card"><div class="val" style="color:var(--success)">{total_growth_pct:.2f}%</div><div class="lab">Portfolio Growth</div></div>
            <div class="stat-card"><div class="val" style="color:var(--danger)">{mdd_pct:.2f}%</div><div class="lab">Max Drawdown</div></div>
            <div class="stat-card"><div class="val">{total_trades}</div><div class="lab">Total Trades</div></div>
            <div class="stat-card"><div class="val">{((long_wins+short_wins)/max(1, total_trades)*100):.1f}%</div><div class="lab">Win Rate</div></div>
        </div>
        <div style="background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 15px;">
            <div id="bigEquityChart" style="height: 400px;"></div>
        </div>
    </div>
</div>

<script>
    const candlesData = {json.dumps(candles_json)};
    let tradesData = {json.dumps(trades_json)};
    const equityData = {json.dumps(equity_json)};

    let currentSym = 'ALL';
    let currentSide = 'ALL';
    let currentSort = 'TIME';
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

    function setSideFilter(side) {{
        currentSide = side;
        document.querySelectorAll('[id^="filter-"]').forEach(b => b.classList.remove('active'));
        document.getElementById('filter-' + side).classList.add('active');
        updateFilters();
    }}

    function setSort(sort) {{
        currentSort = sort;
        document.querySelectorAll('[id^="sort-"]').forEach(b => b.classList.remove('active'));
        document.getElementById('sort-' + sort).classList.add('active');
        updateFilters();
    }}

    function updateFilters() {{
        currentSym = document.getElementById('symbolPicker').value;
        const cards = document.querySelectorAll('.trade-card');
        
        // 1. Filtering
        cards.forEach(c => {{
            const sym = c.getAttribute('data-symbol');
            const side = c.getAttribute('data-side');
            const matchSym = (currentSym === 'ALL' || currentSym === sym);
            const matchSide = (currentSide === 'ALL' || currentSide === side);
            
            if (matchSym && matchSide) {{
                c.classList.remove('hidden');
            }} else {{
                c.classList.add('hidden');
            }}
        }});
        
        // 2. Sorting
        const container = document.getElementById('tradesContainer');
        const visibleCards = Array.from(cards);
        visibleCards.sort((a, b) => {{
            if (currentSort === 'TIME') {{
                return b.getAttribute('data-time').localeCompare(a.getAttribute('data-time'));
            }} else {{
                return parseFloat(b.getAttribute('data-pnl')) - parseFloat(a.getAttribute('data-pnl'));
            }}
        }});
        visibleCards.forEach(c => container.appendChild(c));
        
        // Update Chart if symbol changed
        if (currentSym !== 'ALL') {{
            loadSymbol(currentSym);
        }}
    }}

    function renderBigEquity() {{
        const traces = [];
        if (equityData["GLOBAL"]) traces.push({{ y: equityData["GLOBAL"], name: "PORTFOLIO", type: 'scatter', mode: 'lines', line: {{ width: 4, color: '#fff' }}, fill: 'tozeroy', fillcolor: 'rgba(255, 255, 255, 0.05)' }});
        for (const s in equityData) {{ if (s === "GLOBAL") continue; traces.push({{ y: equityData[s], name: s, type: 'scatter', mode: 'lines', line: {{ width: 2 }}, opacity: 0.6 }}); }}
        Plotly.newPlot('bigEquityChart', traces, {{ title: 'Global Growth', template: 'plotly_dark', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {{ t: 40, b: 60, l: 60, r: 20 }}, legend: {{ orientation: 'h', y: -0.2 }} }});
    }}

    function loadSymbol(symbol, zoomTrade = null) {{
        const df = candlesData[symbol] || candlesData[Object.keys(candlesData)[0]];
        const traces = [{{ x: df.map(c => c.Date), open: df.map(c => c.Open), high: df.map(c => c.High), low: df.map(c => c.Low), close: df.map(c => c.Close), type: 'candlestick', name: symbol, increasing: {{ line: {{ color: '#3fb950' }} }}, decreasing: {{ line: {{ color: '#f85149' }} }} }}];
        const layout = {{ dragmode: 'pan', template: 'plotly_dark', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {{ t: 10, b: 30, l: 50, r: 10 }}, xaxis: {{ rangeslider: {{ visible: false }}, gridcolor: '#30363d' }}, yaxis: {{ gridcolor: '#30363d' }}, shapes: [], annotations: [] }};

        if (zoomTrade) {{
            const start = Math.max(0, zoomTrade.entry_index - 30);
            const end = Math.min(df.length - 1, zoomTrade.entry_index + 100);
            layout.xaxis.range = [df[start].Date, df[end].Date];
            const window = df.slice(start, end);
            const high = Math.max(...window.map(c => c.High));
            const low = Math.min(...window.map(c => c.Low));
            const pad = (high - low) * 0.2;
            layout.yaxis.range = [low - pad, high + pad];
            
            traces.push({{ x: [zoomTrade.entry_datetime], y: [zoomTrade.entry_price], mode: 'markers+text', type: 'scatter', name: 'ENTRY', marker: {{ symbol: 'triangle-right', size: 12, color: '#58a6ff' }}, text: ['ENTRY'], textposition: 'top center' }});
            traces.push({{ x: [zoomTrade.exit_datetime], y: [zoomTrade.exit_price], mode: 'markers+text', type: 'scatter', name: 'EXIT', marker: {{ symbol: 'x', size: 10, color: '#fff' }}, text: ['EXIT'], textposition: 'bottom center' }});

            layout.shapes.push({{ type: 'line', x0: zoomTrade.entry_datetime, x1: zoomTrade.exit_datetime, y0: zoomTrade.entry_price, y1: zoomTrade.entry_price, line: {{ color: '#58a6ff', width: 2, dash: 'dot' }} }});
            layout.shapes.push({{ type: 'line', x0: zoomTrade.entry_datetime, x1: zoomTrade.exit_datetime, y0: zoomTrade.tp_price, y1: zoomTrade.tp_price, line: {{ color: '#3fb950', width: 2, dash: 'dash' }} }});
            layout.shapes.push({{ type: 'line', x0: zoomTrade.entry_datetime, x1: zoomTrade.exit_datetime, y0: zoomTrade.sl_price, y1: zoomTrade.sl_price, line: {{ color: '#f85149', width: 2, dash: 'dash' }} }});

            document.getElementById('tradeOverlay').style.display = 'block';
            document.getElementById('overlayContent').innerHTML = `
                <div style="display:flex; justify-content:space-between;"><span>Execution Price:</span> <span style="color:#fff; font-weight:600;">$${{zoomTrade.entry_price.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Quantity:</span> <span style="color:var(--accent); font-weight:800;">${{zoomTrade.quantity.toFixed(4)}} units</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Position Size:</span> <span style="color:#fff;">$${{zoomTrade.notional_usd.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between; margin-top:5px;"><span>Target Goal:</span> <span style="color:var(--success); font-weight:800;">$${{zoomTrade.tp_price.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Stop Loss:</span> <span style="color:var(--danger); font-weight:800;">$${{zoomTrade.sl_price.toFixed(2)}}</span></div>
                <div style="display:flex; justify-content:space-between; border-top:1px solid var(--border); padding-top:6px; margin-top:6px; font-size:12px;"><span>PNL REALIZED:</span> <span style="color:${{zoomTrade.pnl_net_usd > 0 ? 'var(--success)' : 'var(--danger)'}}; font-weight:800;">$${{zoomTrade.pnl_net_usd.toFixed(2)}}</span></div>
            `;
        }} else {{
            document.getElementById('tradeOverlay').style.display = 'none';
        }}
        Plotly.newPlot('mainChart', traces, layout, {{ scrollZoom: true, displayModeBar: true }});
    }}

    function selectTrade(tradeId) {{
        const cards = document.querySelectorAll('.trade-card');
        const card = Array.from(cards).find(c => c.getAttribute('data-id') == tradeId);
        const idx = tradesData.findIndex(t => t.entry_datetime + t.symbol == tradeId); // unique enough
        
        if (selectedTradeIdx === idx) {{
            selectedTradeIdx = -1;
            cards.forEach(c => c.classList.remove('active'));
            if (currentSym !== 'ALL') loadSymbol(currentSym);
        }} else {{
            selectedTradeIdx = idx;
            const trade = tradesData[idx];
            document.getElementById('symbolPicker').value = trade.symbol;
            loadSymbol(trade.symbol, trade);
            cards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        }}
    }}

    // Initial load
    updateFilters();
    const first = Object.keys(candlesData)[0];
    if (first) loadSymbol(first);
</script>

</body>
</html>
"""

def _generate_trade_cards(df: pd.DataFrame) -> str:
    if df.empty: return "<div style='padding:20px; opacity:0.5;'>No history</div>"
    cards = []
    for i, row in df.iterrows():
        pnl_col = "var(--success)" if row['pnl_net_usd'] > 0 else "var(--danger)"
        badge = "long-badge" if row['side'] == 'LONG' else "short-badge"
        trade_id = f"{row['entry_datetime']}{row['symbol']}"
        cards.append(f"""
            <div class="trade-card" data-symbol="{row['symbol']}" data-side="{row['side']}" data-time="{row['entry_datetime']}" data-pnl="{row['pnl_net_usd']}" data-id="{trade_id}" onclick="selectTrade('{trade_id}')">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:800; color:#fff; font-size:12px;">{row['symbol']}</span>
                    <span style="color:{pnl_col}; font-weight:800; font-size:12px;">${row['pnl_net_usd']:.2f}</span>
                </div>
                <div style="margin-top:4px; font-size:10px; display:flex; justify-content:space-between; opacity:0.6;">
                    <span><span class="badge {badge}">{row['side']}</span> (Qty: {row['quantity']:.4f})</span>
                    <span>{row['entry_datetime'].split(' ')[1]}</span>
                </div>
            </div>
        """)
    return "".join(cards)

def _generate_symbol_options(all_candles: dict) -> str:
    return "".join([f'<option value="{s}">{s}</option>' for s in all_candles.keys()])
