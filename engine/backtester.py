# -*- coding: utf-8 -*-
"""
Paper portfolio simulation: risk-based sizing from predicted stop distance and PnL from path outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from config import config


@dataclass
class PaperTradeRecord:
    """One closed or timed-out paper trade for audit and feedback stats."""

    symbol: str
    entry_datetime: str
    entry_index: int
    exit_datetime: str
    exit_price: float
    side: str
    entry_price: float
    quantity: float
    notional_usd: float
    take_profit_pct: float
    stop_loss_pct: float
    tp_price: float
    sl_price: float
    ai_confidence: float
    ai_qty_ratio: float
    outcome: str
    return_fraction: float
    pnl_gross_usd: float
    slippage_usd: float
    fees_usd: float
    pnl_net_usd: float
    capital_before_usd: float
    equity_after_usd: float
    holding_bars: int = 0
    holding_duration_mins: float = 0.0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0

def compute_position_size_for_risk_budget(
    equity_usd: float,
    entry_price: float,
    stop_loss_pct: float,
    risk_budget_fraction_of_equity: float,
    max_notional_fraction_of_equity: float,
    qty_ratio: float = 1.0,
) -> tuple[float, float]:
    """
    Size like a discretionary trader: risk a fixed fraction of equity to the stop distance,
    then scale by the AI's confidence-based qty_ratio.

    Returns (quantity, notional_usd). Quantity is in coin units; notional is entry_price * quantity.
    """
    if entry_price <= 0 or equity_usd <= 0:
        return 0.0, 0.0
    stop_distance_fraction = max(float(stop_loss_pct) / 100.0, 1e-5)
    dollars_at_risk_per_unit = entry_price * stop_distance_fraction
    risk_budget_usd = equity_usd * risk_budget_fraction_of_equity
    
    # Scale initial budget by AI's requested quantity ratio
    quantity = (risk_budget_usd / dollars_at_risk_per_unit) * qty_ratio
    
    notional_usd = quantity * entry_price
    max_notional_usd = equity_usd * max_notional_fraction_of_equity
    if notional_usd > max_notional_usd and max_notional_usd > 0:
        scale = max_notional_usd / notional_usd
        quantity *= scale
        notional_usd = quantity * entry_price
    return float(quantity), float(notional_usd)

def run_paper_portfolio_on_signals(
    panel: pd.DataFrame,
    symbol: str,
    initial_capital_usd: float,
    risk_per_trade_pct_of_equity: float,
    max_notional_pct_of_equity: float,
    round_trip_fee_pct: float,
) -> tuple[pd.DataFrame, List[PaperTradeRecord], dict]:
    """
    Advanced walk-forward backtester:
    1. Supports PARALLEL_SLOTS (multiple concurrent trades).
    2. Applies SLIPPAGE_PCT to entry and exit prices.
    3. Uses AI-predicted TP% and SL% for each entry.
    4. Records exit time and price with realistic fee/slippage logic.
    """
    risk_fraction = risk_per_trade_pct_of_equity / 100.0
    max_notional_fraction = max_notional_pct_of_equity
    fee_fraction_per_leg = (round_trip_fee_pct / 100.0) / 2.0
    slippage_fraction = config.strategy.SLIPPAGE_PCT / 100.0
    lookahead_limit = config.features.LOOKAHEAD_BARS
    max_slots = config.strategy.PARALLEL_SLOTS

    working_frame = panel.sort_index().copy()
    equity_usd = float(initial_capital_usd)
    trade_records: List[PaperTradeRecord] = []
    equity_by_bar: List[float] = []

    # List of active trades (up to PARALLEL_SLOTS)
    active_trades: List[dict] = []

    # Daily trade tracking
    last_day: str | None = None
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    cooldown_bars_left: int = 0
    daily_start_equity = equity_usd
    trading_locked_for_day = False
    max_daily_trades = config.strategy.MAX_DAILY_TRADES
    cooldown_bars = config.strategy.COOLDOWN_BARS
    max_consecutive_losses = config.strategy.MAX_CONSECUTIVE_LOSSES
    daily_stop_loss_fraction = config.strategy.DAILY_STOP_LOSS_PCT / 100.0
    break_even_after_r = config.strategy.BREAK_EVEN_AFTER_R
    trail_stop_after_r = config.strategy.TRAIL_STOP_AFTER_R
    trail_stop_r_multiple = config.strategy.TRAIL_STOP_R_MULTIPLE

    # Interval in minutes
    interval_str = config.data.INTERVAL
    interval_mins = 1.0
    if interval_str.endswith('m'): interval_mins = float(interval_str[:-1])
    elif interval_str.endswith('h'): interval_mins = float(interval_str[:-1]) * 60
    elif interval_str.endswith('d'): interval_mins = float(interval_str[:-1]) * 1440

    for i in range(len(working_frame)):
        equity_by_bar.append(equity_usd)
        row = working_frame.iloc[i]
        curr_price = float(row["Close"])
        curr_high = float(row["High"])
        curr_low = float(row["Low"])
        curr_time = str(row.name)
        if "Date" in row:
            curr_time = str(row["Date"])

        curr_day = curr_time.split(" ")[0] if " " in curr_time else curr_time
        if curr_day != last_day:
            last_day = curr_day
            daily_trade_count = 0
            consecutive_losses = 0
            cooldown_bars_left = 0
            daily_start_equity = equity_usd
            trading_locked_for_day = False
        elif cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        # 1. Handle Active Trades
        finished_trades_indices = []
        for idx, t in enumerate(active_trades):
            t["bars_held"] += 1
            hit_tp = False
            hit_sl = False
            exit_price = curr_price
            outcome = "NONE"

            if t["side"] == "LONG":
                risk_per_unit = max(t["entry_price"] - t["sl_price_initial"], 1e-8)
                t["mfe_pct"] = max(t["mfe_pct"], ((curr_high - t["entry_price"]) / t["entry_price"]) * 100.0)
                t["mae_pct"] = min(t["mae_pct"], ((curr_low - t["entry_price"]) / t["entry_price"]) * 100.0)
                best_r_multiple = (curr_high - t["entry_price"]) / risk_per_unit
                if best_r_multiple >= break_even_after_r:
                    t["sl_price"] = max(t["sl_price"], t["entry_price"])
                if best_r_multiple >= trail_stop_after_r:
                    trailing_price = curr_high - (risk_per_unit * trail_stop_r_multiple)
                    t["sl_price"] = max(t["sl_price"], trailing_price)
                if curr_high >= t["tp_price"]:
                    hit_tp = True
                    exit_price = t["tp_price"]
                if curr_low <= t["sl_price"]:
                    hit_sl = True
                    exit_price = t["sl_price"]
            else: # SHORT
                risk_per_unit = max(t["sl_price_initial"] - t["entry_price"], 1e-8)
                t["mfe_pct"] = max(t["mfe_pct"], ((t["entry_price"] - curr_low) / t["entry_price"]) * 100.0)
                t["mae_pct"] = min(t["mae_pct"], ((t["entry_price"] - curr_high) / t["entry_price"]) * 100.0)
                best_r_multiple = (t["entry_price"] - curr_low) / risk_per_unit
                if best_r_multiple >= break_even_after_r:
                    t["sl_price"] = min(t["sl_price"], t["entry_price"])
                if best_r_multiple >= trail_stop_after_r:
                    trailing_price = curr_low + (risk_per_unit * trail_stop_r_multiple)
                    t["sl_price"] = min(t["sl_price"], trailing_price)
                if curr_low <= t["tp_price"]:
                    hit_tp = True
                    exit_price = t["tp_price"]
                if curr_high >= t["sl_price"]:
                    hit_sl = True
                    exit_price = t["sl_price"]

            if hit_tp and hit_sl:
                outcome = "FAILED"
                exit_price = t["sl_price"]
            elif hit_tp:
                outcome = "SUCCESS"
            elif hit_sl:
                outcome = "FAILED"
            elif t["bars_held"] >= lookahead_limit:
                outcome = "TIMEOUT"
                exit_price = curr_price

            if outcome != "NONE":
                if t["side"] == "LONG":
                    real_exit_price = exit_price * (1 - slippage_fraction)
                    return_fraction = (real_exit_price - t["entry_price"]) / t["entry_price"]
                    pnl_gross = t["quantity"] * (exit_price - t["raw_entry_price"])
                else:
                    real_exit_price = exit_price * (1 + slippage_fraction)
                    return_fraction = (t["entry_price"] - real_exit_price) / t["entry_price"]
                    pnl_gross = t["quantity"] * (t["raw_entry_price"] - exit_price)

                entry_slip = abs(t["entry_price"] - t["raw_entry_price"]) * t["quantity"]
                exit_slip = abs(real_exit_price - exit_price) * t["quantity"]
                total_slippage = entry_slip + exit_slip

                pnl_before_fees = t["notional_usd"] * return_fraction
                fees_usd = t["notional_usd"] * fee_fraction_per_leg * 2.0
                pnl_net = pnl_before_fees - fees_usd
                equity_usd = max(equity_usd + pnl_net, 0.0)
                realized_outcome = outcome
                if pnl_net > 0:
                    realized_outcome = "SUCCESS"
                elif abs(pnl_net) <= max(fees_usd * 0.25, 1e-8):
                    realized_outcome = "BREAKEVEN"

                if pnl_net < 0:
                    consecutive_losses += 1
                    cooldown_bars_left = max(cooldown_bars_left, cooldown_bars)
                elif pnl_net > 0:
                    consecutive_losses = 0

                trade_records.append(
                    PaperTradeRecord(
                        symbol=symbol,
                        entry_datetime=t["entry_time"],
                        entry_index=t["entry_index"],
                        exit_datetime=curr_time,
                        exit_price=real_exit_price,
                        side=t["side"],
                        entry_price=t["entry_price"],
                        quantity=t["quantity"],
                        notional_usd=t["notional_usd"],
                        take_profit_pct=t["tp_pct"],
                        stop_loss_pct=t["sl_pct"],
                        tp_price=t["tp_price"],
                        sl_price=t["sl_price_initial"],
                        ai_confidence=t["confidence"],
                        ai_qty_ratio=t["qty_ratio"],
                        outcome=realized_outcome,
                        return_fraction=return_fraction,
                        pnl_gross_usd=pnl_gross,
                        slippage_usd=total_slippage,
                        fees_usd=fees_usd,
                        pnl_net_usd=pnl_net,
                        capital_before_usd=t["capital_before"],
                        equity_after_usd=equity_usd,
                        holding_bars=t["bars_held"],
                        holding_duration_mins=t["bars_held"] * interval_mins,
                        mae_pct=float(t["mae_pct"]),
                        mfe_pct=float(t["mfe_pct"]),
                    )
                )
                finished_trades_indices.append(idx)

        # Remove finished trades
        for idx in sorted(finished_trades_indices, reverse=True):
            active_trades.pop(idx)

        if daily_start_equity > 0 and (daily_start_equity - equity_usd) / daily_start_equity >= daily_stop_loss_fraction:
            trading_locked_for_day = True

        # 2. Check for New Signal
        if (
            len(active_trades) < max_slots
            and daily_trade_count < max_daily_trades
            and cooldown_bars_left == 0
            and consecutive_losses < max_consecutive_losses
            and not trading_locked_for_day
        ):
            verdict = int(row["ai_verdict"])
            if verdict != 1:  # 0=BUY, 2=SELL
                side = "LONG" if verdict == 0 else "SHORT"
                
                # RULE: No duplicate side for same symbol
                is_side_active = any(t["symbol"] == symbol and t["side"] == side for t in active_trades)
                if is_side_active:
                    continue

                tp_pct = float(row["ai_take_profit_pct"])
                sl_pct = float(row["ai_stop_loss_pct"])
                directional_edge = float(row.get("ai_directional_edge", 0.0))

                if tp_pct < (sl_pct * config.strategy.MIN_REWARD_RISK_RATIO):
                    continue
                if directional_edge < config.strategy.MIN_DIRECTIONAL_EDGE:
                    continue

                raw_entry_price = curr_price
                side = "LONG" if verdict == 0 else "SHORT"
                entry_price = raw_entry_price * (1 + slippage_fraction if side == "LONG" else 1 - slippage_fraction)
                
                confidence = float(row.get("ai_confidence", 0.0))
                qty_ratio = float(row.get("ai_qty_ratio", 1.0))
                expected_return_pct = float(row.get("ai_expected_return_pct", 0.0))
                if confidence < config.strategy.AI_CONFIDENCE_THRESHOLD:
                    continue
                if expected_return_pct < config.strategy.MIN_EXPECTED_RETURN_PCT:
                    continue
                
                if side == "LONG":
                    tp_price = entry_price * (1 + tp_pct / 100.0)
                    sl_price = entry_price * (1 - sl_pct / 100.0)
                else:
                    tp_price = entry_price * (1 - tp_pct / 100.0)
                    sl_price = entry_price * (1 + sl_pct / 100.0)

                rr_ratio = tp_pct / max(sl_pct, 1e-6)
                confidence_risk_scale = min(max((confidence - config.strategy.AI_CONFIDENCE_THRESHOLD) / 0.20, 0.25), 1.0)
                rr_risk_scale = min(max((rr_ratio - 1.0) / 1.0, 0.50), 1.2)
                adjusted_qty_ratio = min(max(qty_ratio * confidence_risk_scale * rr_risk_scale, 0.10), 1.0)

                quantity, notional_usd = compute_position_size_for_risk_budget(
                    equity_usd=equity_usd,
                    entry_price=entry_price,
                    stop_loss_pct=sl_pct,
                    risk_budget_fraction_of_equity=risk_fraction,
                    max_notional_fraction_of_equity=max_notional_fraction,
                    qty_ratio=adjusted_qty_ratio,
                )

                if quantity > 0:
                    active_trades.append({
                        "symbol": symbol,
                        "side": side,
                        "raw_entry_price": raw_entry_price,
                        "entry_price": entry_price,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "sl_price_initial": sl_price,
                        "tp_pct": tp_pct,
                        "sl_pct": sl_pct,
                        "quantity": quantity,
                        "notional_usd": notional_usd,
                        "entry_time": curr_time,
                        "entry_index": i,
                        "confidence": confidence,
                        "qty_ratio": adjusted_qty_ratio,
                        "capital_before": equity_usd,
                        "bars_held": 0,
                        "mae_pct": 0.0,
                        "mfe_pct": 0.0,
                    })
                    daily_trade_count += 1

    if active_trades and len(working_frame) > 0:
        final_row = working_frame.iloc[-1]
        final_price = float(final_row["Close"])
        final_time = str(final_row.name)
        if "Date" in final_row:
            final_time = str(final_row["Date"])
        for t in active_trades:
            if t["side"] == "LONG":
                real_exit_price = final_price * (1 - slippage_fraction)
                return_fraction = (real_exit_price - t["entry_price"]) / t["entry_price"]
                pnl_gross = t["quantity"] * (final_price - t["raw_entry_price"])
            else:
                real_exit_price = final_price * (1 + slippage_fraction)
                return_fraction = (t["entry_price"] - real_exit_price) / t["entry_price"]
                pnl_gross = t["quantity"] * (t["raw_entry_price"] - final_price)
            fees_usd = t["notional_usd"] * fee_fraction_per_leg * 2.0
            pnl_net = t["notional_usd"] * return_fraction - fees_usd
            equity_usd = max(equity_usd + pnl_net, 0.0)
            trade_records.append(
                PaperTradeRecord(
                    symbol=symbol,
                    entry_datetime=t["entry_time"],
                    entry_index=t["entry_index"],
                    exit_datetime=final_time,
                    exit_price=real_exit_price,
                    side=t["side"],
                    entry_price=t["entry_price"],
                    quantity=t["quantity"],
                    notional_usd=t["notional_usd"],
                    take_profit_pct=t["tp_pct"],
                    stop_loss_pct=t["sl_pct"],
                    tp_price=t["tp_price"],
                    sl_price=t["sl_price_initial"],
                    ai_confidence=t["confidence"],
                    ai_qty_ratio=t["qty_ratio"],
                    outcome="EOD_CLOSE",
                    return_fraction=return_fraction,
                    pnl_gross_usd=pnl_gross,
                    slippage_usd=abs(t["entry_price"] - t["raw_entry_price"]) * t["quantity"],
                    fees_usd=fees_usd,
                    pnl_net_usd=pnl_net,
                    capital_before_usd=t["capital_before"],
                    equity_after_usd=equity_usd,
                    holding_bars=t["bars_held"],
                    holding_duration_mins=t["bars_held"] * interval_mins,
                    mae_pct=float(t["mae_pct"]),
                    mfe_pct=float(t["mfe_pct"]),
                )
            )
        if equity_by_bar:
            equity_by_bar[-1] = equity_usd

    working_frame["paper_equity_curve"] = equity_by_bar
    summary = summarize_trade_feedback(trade_records)
    return working_frame, trade_records, summary

def summarize_trade_feedback(trade_records: List[PaperTradeRecord]) -> dict:
    """Post-trade diagnostics with rich console output."""
    if not trade_records:
        return {
            "trade_count": 0, "win_rate_pct": 0.0, "wins": 0, "losses": 0, "timeouts": 0,
            "final_equity_usd": float(config.strategy.INITIAL_CAPITAL_USD),
            "total_fees_usd": 0.0, "total_pnl_net_usd": 0.0,
            "avg_holding_mins": 0.0, "max_drawdown_usd": 0.0, "max_drawdown_pct": 0.0,
            "profit_factor": 0.0, "avg_win_usd": 0.0, "avg_loss_usd": 0.0
        }

    wins = sum(1 for t in trade_records if t.pnl_net_usd > 0)
    losses = sum(1 for t in trade_records if t.pnl_net_usd < 0)
    resolved = wins + losses
    win_rate = (wins / resolved * 100.0) if resolved > 0 else 0.0

    total_fees = sum(t.fees_usd for t in trade_records)
    total_pnl = sum(t.pnl_net_usd for t in trade_records)
    avg_holding = sum(t.holding_duration_mins for t in trade_records) / len(trade_records)
    gross_profit = sum(t.pnl_net_usd for t in trade_records if t.pnl_net_usd > 0)
    gross_loss = abs(sum(t.pnl_net_usd for t in trade_records if t.pnl_net_usd < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = gross_profit / wins if wins else 0.0
    avg_loss = gross_loss / losses if losses else 0.0

    # Simple MDD calculation from trade sequence
    equities = [t.equity_after_usd for t in trade_records]
    peak = equities[0]
    mdd = 0.0
    mdd_pct = 0.0
    for e in equities:
        if e > peak: peak = e
        dd = peak - e
        if dd > mdd: mdd = dd
        if peak > 0:
            mdd_pct = max(mdd_pct, dd / peak * 100.0)

    return {
        "trade_count": len(trade_records),
        "win_rate_pct": float(win_rate),
        "final_equity_usd": float(trade_records[-1].equity_after_usd),
        "wins": wins,
        "losses": losses,
        "timeouts": sum(1 for t in trade_records if t.outcome == "TIMEOUT"),
        "total_fees_usd": float(total_fees),
        "total_pnl_net_usd": float(total_pnl),
        "avg_holding_mins": float(avg_holding),
        "max_drawdown_usd": float(mdd),
        "max_drawdown_pct": float(mdd_pct),
        "profit_factor": float(profit_factor),
        "avg_win_usd": float(avg_win),
        "avg_loss_usd": float(avg_loss),
        "avg_mae_pct": float(sum(t.mae_pct for t in trade_records) / len(trade_records)),
        "avg_mfe_pct": float(sum(t.mfe_pct for t in trade_records) / len(trade_records)),
    }

def print_rich_summary(strategy_name: str, symbol: str, summary: dict):
    """Prints a professional table summary to the console."""
    from rich.table import Table
    from rich.panel import Panel
    from ui_utils import console

    table = Table(title=f"{strategy_name} - {symbol} PERFORMANCE", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    pnl_color = "green" if summary['total_pnl_net_usd'] > 0 else "red"
    
    table.add_row("Total Trades", str(summary['trade_count']))
    table.add_row("Win Rate", f"{summary['win_rate_pct']:.2f}%")
    table.add_row("Wins / Losses", f"[green]{summary['wins']}[/green] / [red]{summary['losses']}[/red]")
    table.add_row("Net PnL", f"[{pnl_color}]${summary['total_pnl_net_usd']:,.2f}[/{pnl_color}]")
    table.add_row("Profit Factor", f"{summary.get('profit_factor', 0.0):.2f}")
    table.add_row("Avg Win / Loss", f"${summary.get('avg_win_usd', 0.0):,.2f} / ${summary.get('avg_loss_usd', 0.0):,.2f}")
    table.add_row("Total Fees", f"${summary['total_fees_usd']:,.2f}")
    table.add_row("Max Drawdown", f"${summary['max_drawdown_usd']:,.2f} ({summary.get('max_drawdown_pct', 0.0):.2f}%)")
    table.add_row("Avg Holding", f"{summary['avg_holding_mins']:.1f} mins")
    table.add_row("Final Equity", f"[bold]${summary['final_equity_usd']:,.2f}[/bold]")

    console.print(table)
    console.print("\n")


def describe_last_candle_and_model_outputs(
    last_candle_row: pd.Series,
    model_outputs: dict[str, float],
) -> str:
    """Human-readable block for production-style logging."""
    lines = [
        "Last candle (OHLCV):",
        f"  Open={last_candle_row['Open']:.6g}  High={last_candle_row['High']:.6g}  "
        f"Low={last_candle_row['Low']:.6g}  Close={last_candle_row['Close']:.6g}  "
        f"Volume={last_candle_row.get('Volume', float('nan')):.6g}",
        "",
        "Model output (what you would send to execution / risk engine):",
    ]
    for key in sorted(model_outputs.keys()):
        value = model_outputs[key]
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6g}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
