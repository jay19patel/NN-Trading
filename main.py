# -*- coding: utf-8 -*-
import os
import pandas as pd
from config import config
from data_gathering import fetch_data
from features import create_full_feature_set
from paper_trading import run_paper_portfolio_on_signals
from ui_utils import console, print_banner, Table
from visuals import save_backtest_results

def main() -> None:
    symbols = config.data.SYMBOLS
    interval = config.data.INTERVAL
    total_days = config.data.TOTAL_DAYS

    print_banner(
        "PERFECT ORACLE BACKTESTER",
        f"Symbols: {', '.join(symbols)} | Days: {total_days} | Interval: {interval}",
    )

    summary_table = Table(title="Oracle Results Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Symbol", justify="left", style="bold")
    summary_table.add_column("Trades", justify="right")
    summary_table.add_column("Win %", justify="right")
    summary_table.add_column("Net PnL $", justify="right")
    summary_table.add_column("Equity $", justify="right", style="bold green")

    global_initial = 0.0
    global_final = 0.0
    
    all_trade_records = []
    equity_curves = {}

    for symbol in symbols:
        console.print(f"\n[info]Processing [bold]{symbol}[/bold]...[/info]")
        df = fetch_data(symbol=symbol, total_days=total_days, interval=interval)
        if df.empty:
            continue

        console.print(f"Generating perfect Oracle labels for {symbol}...")
        df = create_full_feature_set(df, lookahead=config.features.LOOKAHEAD_BARS)

        # Map Oracle labels to backtester expectations
        df["ai_verdict"] = df["direction_label"]
        df["ai_take_profit_pct"] = df["label_take_profit_pct"]
        df["ai_stop_loss_pct"] = df["label_stop_loss_pct"]
        df["ai_qty_ratio"] = df["label_qty_ratio"]
        df["ai_confidence"] = 1.0 
        df["ai_directional_edge"] = 1.0 

        console.print(f"Running backtest on {symbol}...")
        df, trade_records, summary = run_paper_portfolio_on_signals(
            df,
            symbol=symbol,
            initial_capital_usd=config.strategy.INITIAL_CAPITAL_USD,
            risk_per_trade_pct_of_equity=config.strategy.RISK_PER_TRADE_PCT_OF_EQUITY,
            max_notional_pct_of_equity=config.strategy.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY,
            round_trip_fee_pct=config.strategy.ROUND_TRIP_FEE_PCT,
        )
        
        all_trade_records.extend(trade_records)
        equity_curves[symbol] = df["paper_equity_curve"].values

        # --- SYMBOL SPECIFIC REPORT ---
        df_trades = pd.DataFrame([vars(t) for t in trade_records])
        if not df_trades.empty:
            # Calculate trades per day
            df_trades['date'] = pd.to_datetime(df_trades['entry_datetime']).dt.date
            num_days = len(df_trades['date'].unique())
            trades_per_day = len(df_trades) / num_days if num_days > 0 else 0
            
            wins_sym = df_trades[df_trades['pnl_net_usd'] > 0]
            total_fees = df_trades['fees_usd'].sum()
            total_slippage = df_trades['slippage_usd'].sum()
            gross_pnl = df_trades['pnl_gross_usd'].sum()
            
            console.print(f"\n[bold cyan]📊 {symbol} DETAILED INSIGHTS (REAL-WORLD MODE)[/bold cyan]")
            sym_table = Table(show_header=False, box=None)
            sym_table.add_row("Total Trades:", f"{len(df_trades)}")
            sym_table.add_row("Avg Trades/Day:", f"{trades_per_day:.1f}")
            sym_table.add_row("Win Rate:", f"[bold green]{(len(wins_sym)/len(df_trades)*100):.1f}%[/bold green]")
            sym_table.add_row("-" * 30, "-" * 10)
            sym_table.add_row("Gross PnL:", f"[green]${gross_pnl:.2f}[/green]")
            sym_table.add_row("Total Fees Paid:", f"[red]${total_fees:.2f}[/red]")
            sym_table.add_row("Total Slippage Loss:", f"[red]${total_slippage:.2f}[/red]")
            sym_table.add_row("Net PnL (Final):", f"[bold green]${summary['total_pnl_net_usd']:.2f}[/bold green]")
            console.print(sym_table)

        summary_table.add_row(
            symbol,
            str(summary["trade_count"]),
            f"{summary['win_rate_pct']:.1f}%",
            f"{summary['total_pnl_net_usd']:.2f}",
            f"{summary['final_equity_usd']:.2f}",
        )

        global_initial += config.strategy.INITIAL_CAPITAL_USD
        global_final += summary['final_equity_usd']

    console.print("\n")
    console.print(summary_table)

    growth = ((global_final - global_initial) / global_initial) * 100
    console.print(f"\n[bold gold]TOTAL CAPITAL GROWTH: {growth:.2f}%[/bold gold]")
    
    # Save results
    save_backtest_results(all_trade_records, equity_curves)
    
    # --- GLOBAL DEEP DIVE REPORT ---
    if all_trade_records:
        df_all_trades = pd.DataFrame([vars(t) for t in all_trade_records])
        
        total_trades = len(df_all_trades)
        wins = df_all_trades[df_all_trades['pnl_net_usd'] > 0]
        losses = df_all_trades[df_all_trades['pnl_net_usd'] <= 0]
        
        global_win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        total_fees = df_all_trades['fees_usd'].sum()
        total_slippage = df_all_trades['slippage_usd'].sum()
        max_profit = df_all_trades['pnl_net_usd'].max()
        
        console.print("\n" + "="*50)
        console.print("[bold gold]🌍 GLOBAL REAL-WORLD SUMMARY (ALL SYMBOLS)[/bold gold]")
        console.print("="*50)
        
        insight_table = Table(show_header=False, box=None)
        insight_table.add_row("Total Combined Trades:", f"[bold]{total_trades}[/bold]")
        insight_table.add_row("Total Wins:", f"[green]{len(wins)}[/green]")
        insight_table.add_row("Total Losses:", f"[red]{len(losses)}[/red]")
        insight_table.add_row("Global Win Rate:", f"[bold cyan]{global_win_rate:.1f}%[/bold cyan]")
        insight_table.add_row("-" * 30, "-" * 10)
        insight_table.add_row("Total Portfolio Fees:", f"[red]${total_fees:.2f}[/red]")
        insight_table.add_row("Total Portfolio Slippage:", f"[red]${total_slippage:.2f}[/red]")
        insight_table.add_row("Global Max Single Profit:", f"[green]${max_profit:.2f}[/green]")
        
        interval_mins = 1 if config.data.INTERVAL == "1m" else 15
        hours = (config.features.LOOKAHEAD_BARS * interval_mins) / 60
        insight_table.add_row("Look-ahead Window:", f"{config.features.LOOKAHEAD_BARS} candles ({hours:.1f} hours)")
        
        console.print(insight_table)
        console.print("="*50)
        
    console.print(f"\n[info]Simulation complete. Results saved in 'results' folder.[/info]\n")

if __name__ == "__main__":
    main()
