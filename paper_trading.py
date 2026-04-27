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
    Walk forward in time: each non-neutral signal opens a trade sized by predicted SL%.

    Expects columns: Close, ai_verdict, ai_outcome, ai_return_fraction, ai_take_profit_pct, ai_stop_loss_pct.
    """
    strategy = config.strategy
    risk_fraction = risk_per_trade_pct_of_equity / 100.0
    max_notional_fraction = max_notional_pct_of_equity
    fee_fraction_per_leg = (round_trip_fee_pct / 100.0) / 2.0

    working_frame = panel.sort_index().copy()
    equity_usd = float(initial_capital_usd)
    trade_records: List[PaperTradeRecord] = []
    equity_by_bar: List[float] = []

    for row_position in range(len(working_frame)):
        row = working_frame.iloc[row_position]
        verdict = int(row["ai_verdict"])
        if verdict != 1:
            outcome = str(row.get("ai_outcome", "NONE"))
            if outcome != "NONE":
                entry_price = float(row["Close"])
                stop_loss_pct = float(row["ai_stop_loss_pct"])
                take_profit_pct = float(row["ai_take_profit_pct"])
                qty_ratio = float(row.get("ai_qty_ratio", 1.0))
                confidence = float(row.get("ai_confidence", 0.0))
                return_fraction = float(row.get("ai_return_fraction", 0.0))
                
                entry_datetime = str(row.name)
                if "Date" in row:
                    entry_datetime = str(row["Date"])

                capital_before = equity_usd

                quantity, notional_usd = compute_position_size_for_risk_budget(
                    equity_usd=equity_usd,
                    entry_price=entry_price,
                    stop_loss_pct=stop_loss_pct,
                    risk_budget_fraction_of_equity=risk_fraction,
                    max_notional_fraction_of_equity=max_notional_fraction,
                    qty_ratio=qty_ratio,
                )
                if quantity > 0 and notional_usd > 0:
                    pnl_before_fees = notional_usd * return_fraction
                    fees_usd = notional_usd * fee_fraction_per_leg * 2.0
                    pnl_net = pnl_before_fees - fees_usd
                    equity_usd = max(equity_usd + pnl_net, 0.0)

                    side = "LONG" if verdict == 0 else "SHORT"
                    trade_records.append(
                        PaperTradeRecord(
                            symbol=symbol,
                            entry_datetime=entry_datetime,
                            entry_index=row_position,
                            side=side,
                            entry_price=entry_price,
                            quantity=quantity,
                            notional_usd=notional_usd,
                            take_profit_pct=take_profit_pct,
                            stop_loss_pct=stop_loss_pct,
                            ai_confidence=confidence,
                            ai_qty_ratio=qty_ratio,
                            outcome=outcome,
                            return_fraction=return_fraction,
                            pnl_before_fees_usd=pnl_before_fees,
                            fees_usd=fees_usd,
                            pnl_net_usd=pnl_net,
                            capital_before_usd=capital_before,
                            equity_after_usd=equity_usd,
                        )
                    )

        equity_by_bar.append(equity_usd)

    working_frame["paper_equity_curve"] = equity_by_bar
    summary = summarize_trade_feedback(trade_records)
    return working_frame, trade_records, summary


def summarize_trade_feedback(trade_records: List[PaperTradeRecord]) -> dict:
    """
    Post-trade diagnostics: win rate and whether a loss was followed by a win (simple adaptation signal).
    """
    if not trade_records:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "final_equity_usd": float(config.strategy.INITIAL_CAPITAL_USD),
            "loss_then_win_count": 0,
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
