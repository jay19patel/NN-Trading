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
    ai_confidence: float
    ai_qty_ratio: float
    outcome: str
    return_fraction: float
    pnl_before_fees_usd: float
    fees_usd: float
    pnl_net_usd: float
    capital_before_usd: float
    equity_after_usd: float


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
    MAX_DAILY_TRADES: int = 5    # Max 5 trades per symbol per day as requested

    for i in range(len(working_frame)):
        row = working_frame.iloc[i]
        curr_price = float(row["Close"])
        curr_high = float(row["High"])
        curr_low = float(row["Low"])
        curr_time = str(row.name)
        if "Date" in row:
            curr_time = str(row["Date"])

        # 1. Handle Active Trades
        finished_trades_indices = []
        for idx, t in enumerate(active_trades):
            t["bars_held"] += 1
            hit_tp = False
            hit_sl = False
            exit_price = curr_price
            outcome = "NONE"

            if t["side"] == "LONG":
                if curr_high >= t["tp_price"]:
                    hit_tp = True
                    exit_price = t["tp_price"]
                if curr_low <= t["sl_price"]:
                    hit_sl = True
                    exit_price = t["sl_price"]
            else: # SHORT
                if curr_low <= t["tp_price"]:
                    hit_tp = True
                    exit_price = t["tp_price"]
                if curr_high >= t["sl_price"]:
                    hit_sl = True
                    exit_price = t["sl_price"]

            # Same-bar hit logic (conservative: SL first)
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
                # Apply exit slippage (bad for trader)
                if t["side"] == "LONG":
                    real_exit_price = exit_price * (1 - slippage_fraction)
                    return_fraction = (real_exit_price - t["entry_price"]) / t["entry_price"]
                else:
                    real_exit_price = exit_price * (1 + slippage_fraction)
                    return_fraction = (t["entry_price"] - real_exit_price) / t["entry_price"]

                pnl_before_fees = t["notional_usd"] * return_fraction
                fees_usd = t["notional_usd"] * fee_fraction_per_leg * 2.0
                pnl_net = pnl_before_fees - fees_usd
                equity_usd = max(equity_usd + pnl_net, 0.0)

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
                        ai_confidence=t["confidence"],
                        ai_qty_ratio=t["qty_ratio"],
                        outcome=outcome,
                        return_fraction=return_fraction,
                        pnl_before_fees_usd=pnl_before_fees,
                        fees_usd=fees_usd,
                        pnl_net_usd=pnl_net,
                        capital_before_usd=t["capital_before"],
                        equity_after_usd=equity_usd,
                    )
                )
                finished_trades_indices.append(idx)

        # Remove finished trades in reverse to keep indices valid
        for idx in sorted(finished_trades_indices, reverse=True):
            active_trades.pop(idx)

        # 2. Check for New Signal (if we have open slots and daily limit not reached)
        # Handle daily counter reset
        curr_day = curr_time.split(" ")[0] if " " in curr_time else curr_time
        if curr_day != last_day:
            last_day = curr_day
            daily_trade_count = 0

        if len(active_trades) < max_slots and daily_trade_count < MAX_DAILY_TRADES:
            verdict = int(row["ai_verdict"])
            if verdict != 1:  # 0=BUY, 2=SELL
                tp_pct = float(row["ai_take_profit_pct"])
                sl_pct = float(row["ai_stop_loss_pct"])

                # Risk Rule: Enforce minimum 1:1 Reward-to-Risk ratio (Target >= SL)
                if tp_pct < (sl_pct * config.strategy.MIN_REWARD_RISK_RATIO):
                    continue

                raw_entry_price = curr_price
                side = "LONG" if verdict == 0 else "SHORT"
                
                # Apply entry slippage (bad for trader)
                entry_price = raw_entry_price * (1 + slippage_fraction if side == "LONG" else 1 - slippage_fraction)
                
                confidence = float(row.get("ai_confidence", 0.0))
                qty_ratio = float(row.get("ai_qty_ratio", 1.0))
                
                if side == "LONG":
                    tp_price = entry_price * (1 + tp_pct / 100.0)
                    sl_price = entry_price * (1 - sl_pct / 100.0)
                else:
                    tp_price = entry_price * (1 - tp_pct / 100.0)
                    sl_price = entry_price * (1 + sl_pct / 100.0)

                quantity, notional_usd = compute_position_size_for_risk_budget(
                    equity_usd=equity_usd,
                    entry_price=entry_price,
                    stop_loss_pct=sl_pct,
                    risk_budget_fraction_of_equity=risk_fraction,
                    max_notional_fraction_of_equity=max_notional_fraction,
                    qty_ratio=qty_ratio,
                )

                if quantity > 0:
                    active_trades.append({
                        "side": side,
                        "entry_price": entry_price,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "tp_pct": tp_pct,
                        "sl_pct": sl_pct,
                        "quantity": quantity,
                        "notional_usd": notional_usd,
                        "entry_time": curr_time,
                        "entry_index": i,
                        "confidence": confidence,
                        "qty_ratio": qty_ratio,
                        "capital_before": equity_usd,
                        "bars_held": 0
                    })
                    daily_trade_count += 1

        equity_by_bar.append(equity_usd)

    working_frame["paper_equity_curve"] = equity_by_bar
    summary = summarize_trade_feedback(trade_records)
    return working_frame, trade_records, summary


def summarize_trade_feedback(trade_records: List[PaperTradeRecord]) -> dict:
    """
    Post-trade diagnostics: win rate and whether a loss was followed by a win (simple adaptation signal).
    """
    if not trade_records:
        # Always return a complete dict with all keys — prevents KeyError in main.py
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "final_equity_usd": float(config.strategy.INITIAL_CAPITAL_USD),
            "loss_then_win_count": 0,
            "total_fees_usd": 0.0,
            "total_pnl_net_usd": 0.0,
        }

    wins = sum(1 for trade in trade_records if trade.outcome == "SUCCESS")
    losses = sum(1 for trade in trade_records if trade.outcome == "FAILED")
    resolved = wins + losses
    win_rate = (wins / resolved * 100.0) if resolved > 0 else 0.0

    loss_then_win = 0
    for previous, current in zip(trade_records, trade_records[1:]):
        if previous.outcome == "FAILED" and current.outcome == "SUCCESS":
            loss_then_win += 1

    return {
        "trade_count": len(trade_records),
        "win_rate_pct": float(win_rate),
        "final_equity_usd": float(trade_records[-1].equity_after_usd),
        "wins": wins,
        "losses": losses,
        "timeouts": sum(1 for trade in trade_records if trade.outcome == "TIMEOUT"),
        "loss_then_win_count": loss_then_win,
        "total_fees_usd": float(sum(trade.fees_usd for trade in trade_records)),
        "total_pnl_net_usd": float(sum(trade.pnl_net_usd for trade in trade_records)),
    }


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
