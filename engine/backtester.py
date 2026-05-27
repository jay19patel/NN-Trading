# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from config import cfg

def _confidence_to_margin_pct(confidence: float, base_pct: float, max_pct: float) -> float:
    """
    Map model confidence (0-1) to a position margin percentage using sqrt scaling.

    sqrt scaling avoids over-betting on high-confidence signals since models
    can be overconfident.

    Examples (base=2%, max=8%):
      confidence=0.45 → ~3.7%
      confidence=0.60 → ~4.6%
      confidence=0.80 → ~5.4%
      confidence=1.00 → 8.0%
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return base_pct + (max_pct - base_pct) * (confidence ** 0.5)

@dataclass
class PaperTradeRecord:
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
    exit_reason: str = "N/A"
    holding_bars: int = 0
    holding_duration_mins: float = 0.0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    max_price_reached: float = 0.0
    min_price_reached: float = 0.0
    pattern_confidence: float = 0.0
    dynamic_margin_pct: float = 5.0

def compute_position_size_for_margin_budget(
    equity_usd: float,
    entry_price: float,
    margin_budget_fraction_of_equity: float,
    leverage: float,
) -> tuple[float, float]:
    if entry_price <= 0 or equity_usd <= 0:
        return 0.0, 0.0
    margin_usd = equity_usd * margin_budget_fraction_of_equity
    notional_usd = margin_usd * leverage
    quantity = notional_usd / entry_price
    return float(quantity), float(notional_usd)

def run_paper_portfolio_on_signals(
    panel: pd.DataFrame,
    symbol: str,
    initial_capital_usd: float,
    margin_per_trade_pct_of_equity: float,
    leverage: float,
    round_trip_fee_pct: float,
) -> tuple[pd.DataFrame, List[PaperTradeRecord], dict]:
    margin_fraction = margin_per_trade_pct_of_equity / 100.0
    fee_fraction_per_leg = (round_trip_fee_pct / 100.0) / 2.0
    slippage_fraction = cfg.testing.SLIPPAGE_PCT / 100.0
    lookahead_limit = cfg.testing.LOOKAHEAD_BARS
    max_slots = cfg.testing.PARALLEL_SLOTS

    working_frame = panel.sort_index().copy()
    equity_usd = float(initial_capital_usd)
    trade_records: List[PaperTradeRecord] = []
    equity_by_bar: List[float] = []
    active_trades: List[dict] = []

    last_day: str | None = None
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    cooldown_bars_left: int = 0
    daily_start_equity = equity_usd
    trading_locked_for_day = False

    # Load PatternConfidenceEngine for dynamic position sizing (optional — graceful fallback)
    from neural_engine.pattern_confidence import PatternConfidenceEngine, PATTERN_FEATURES
    _pattern_engine = PatternConfidenceEngine.load_or_none()

    for i in range(len(working_frame)):
        equity_by_bar.append(equity_usd)
        row = working_frame.iloc[i]
        curr_price = float(row["Close"])
        curr_high = float(row["High"])
        curr_low = float(row["Low"])
        curr_time = str(row.name)

        curr_day = str(row.name)[:10]
        if curr_day != last_day:
            last_day = curr_day
            daily_trade_count = 0
            daily_start_equity = equity_usd
            trading_locked_for_day = False

        if equity_usd <= daily_start_equity * (1 - cfg.testing.DAILY_STOP_LOSS_PCT / 100.0):
            trading_locked_for_day = True

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        # 1. Handle Active Trades
        finished_trades_indices = []
        for idx, t in enumerate(active_trades):
            t["bars_held"] += 1
            hit_tp = False
            hit_sl = False
            hit_liq = False
            exit_price = curr_price
            outcome = "NONE"

            if t["side"] == "LONG":
                curr_mfe = ((curr_high - t["entry_price"]) / t["entry_price"]) * 100.0
                curr_mae = ((curr_low - t["entry_price"]) / t["entry_price"]) * 100.0
                t["mfe_pct"] = max(t["mfe_pct"], curr_mfe)
                t["mae_pct"] = min(t["mae_pct"], curr_mae)

                if curr_low <= t["liq_price"]:
                    hit_liq, exit_price = True, t["liq_price"]
                elif curr_low <= t["sl_price"]:
                    hit_sl, exit_price = True, t["sl_price"]
                elif curr_high >= t["tp_price"]:
                    hit_tp, exit_price = True, t["tp_price"]
            else: # SHORT
                curr_mfe = ((t["entry_price"] - curr_low) / t["entry_price"]) * 100.0
                curr_mae = ((t["entry_price"] - curr_high) / t["entry_price"]) * 100.0
                t["mfe_pct"] = max(t["mfe_pct"], curr_mfe)
                t["mae_pct"] = min(t["mae_pct"], curr_mae)

                if curr_high >= t["liq_price"]:
                    hit_liq, exit_price = True, t["liq_price"]
                elif curr_low <= t["tp_price"]:
                    hit_tp, exit_price = True, t["tp_price"]
                elif curr_high >= t["sl_price"]:
                    hit_sl, exit_price = True, t["sl_price"]

            if hit_liq:
                outcome, exit_reason = "FAILED", "💀 Liquidated"
                t["mae_pct"] = min(t["mae_pct"], -100.0) # Full margin loss
            elif hit_tp and hit_sl:
                outcome, exit_reason, exit_price = "FAILED", "🛑 SL Hit", t["sl_price"]
                t["mae_pct"] = min(t["mae_pct"], -t["sl_pct"])
            elif hit_tp:
                outcome, exit_reason = "SUCCESS", "🎯 TP Hit"
                t["mfe_pct"] = max(t["mfe_pct"], t["tp_pct"])
            elif hit_sl:
                outcome, exit_reason = "FAILED", "🛑 SL Hit"
                t["mae_pct"] = min(t["mae_pct"], -t["sl_pct"])
            elif t["bars_held"] >= lookahead_limit:
                outcome, exit_reason, exit_price = "TIMEOUT", "⏳ Time Out", curr_price
            else:
                exit_reason = "NONE"

            if outcome != "NONE":
                # --- NEW LOGIC: Compute absolute potential max/min over the full lookahead window ---
                start_idx = t["entry_index"]
                end_idx = min(len(working_frame), start_idx + lookahead_limit + 1)
                future_highs = working_frame['High'].iloc[start_idx:end_idx]
                future_lows = working_frame['Low'].iloc[start_idx:end_idx]
                
                if not future_highs.empty:
                    t["max_price"] = future_highs.max()
                    t["min_price"] = future_lows.min()
                    if t["side"] == "LONG":
                        t["mfe_pct"] = ((t["max_price"] - t["entry_price"]) / t["entry_price"]) * 100.0
                        t["mae_pct"] = ((t["min_price"] - t["entry_price"]) / t["entry_price"]) * 100.0
                    else:
                        t["mfe_pct"] = ((t["entry_price"] - t["min_price"]) / t["entry_price"]) * 100.0
                        t["mae_pct"] = ((t["entry_price"] - t["max_price"]) / t["entry_price"]) * 100.0

                if t["side"] == "LONG":
                    real_exit_price = exit_price * (1 - slippage_fraction)
                    return_fraction = (real_exit_price - t["entry_price"]) / t["entry_price"]
                else:
                    real_exit_price = exit_price * (1 + slippage_fraction)
                    return_fraction = (t["entry_price"] - real_exit_price) / t["entry_price"]

                fees_usd = t["notional_usd"] * fee_fraction_per_leg * 2.0
                pnl_net = t["notional_usd"] * return_fraction - fees_usd
                equity_usd = max(equity_usd + pnl_net, 0.0)
                
                trade_records.append(PaperTradeRecord(
                    symbol=symbol, entry_datetime=t["entry_time"], entry_index=t["entry_index"],
                    exit_datetime=curr_time, exit_price=real_exit_price, side=t["side"],
                    entry_price=t["entry_price"], quantity=t["quantity"], notional_usd=t["notional_usd"],
                    take_profit_pct=t["tp_pct"], stop_loss_pct=t["sl_pct"], tp_price=t["tp_price"],
                    sl_price=t["sl_price_initial"], ai_confidence=t["confidence"], ai_qty_ratio=t["qty_ratio"],
                    outcome=outcome, return_fraction=return_fraction, pnl_gross_usd=pnl_net + fees_usd,
                    slippage_usd=0, fees_usd=fees_usd, pnl_net_usd=pnl_net,
                    capital_before_usd=t["capital_before"], equity_after_usd=equity_usd, 
                    exit_reason=exit_reason, holding_bars=t["bars_held"],
                    mae_pct=t["mae_pct"], mfe_pct=t["mfe_pct"],
                    max_price_reached=t["max_price"], min_price_reached=t["min_price"]
                ))
                finished_trades_indices.append(idx)

                # Update execution guards based on exit
                # FIX: Only apply cooldown after FAILED trades (not after wins)
                if outcome == "FAILED":
                    consecutive_losses += 1
                    cooldown_bars_left = cfg.testing.COOLDOWN_BARS
                elif outcome == "SUCCESS":
                    consecutive_losses = 0
                    # No cooldown after wins - allow taking good opportunities

        for idx in sorted(finished_trades_indices, reverse=True):
            active_trades.pop(idx)

        # 2. Check for New Signal (execute on NEXT bar)
        can_trade = (
            not trading_locked_for_day and
            cooldown_bars_left == 0 and
            consecutive_losses < cfg.testing.MAX_CONSECUTIVE_LOSSES and
            daily_trade_count < cfg.testing.MAX_DAILY_TRADES
        )

        # CRITICAL FIX: Only process signals if we have a NEXT bar to enter on
        # This prevents entering at the same bar's close (impossible in live trading)
        if len(active_trades) < max_slots and can_trade and i < len(working_frame) - 1:
            verdict = int(row["ai_verdict"])
            if verdict != 1:
                side = "LONG" if verdict == 0 else "SHORT"
                confidence = float(row.get("ai_confidence", 0.0))
                # BUG FIX #1: Use nested if instead of continue to avoid skipping daily_trade_count increment
                if confidence >= cfg.testing.AI_CONFIDENCE_THRESHOLD:
                    tp_pct = float(row["ai_take_profit_pct"]) * cfg.testing.AI_TARGET_DISCOUNT_FACTOR
                    sl_pct = float(row["ai_stop_loss_pct"])

                    # CRITICAL FIX: Enter on NEXT bar's open (earliest realistic entry point)
                    # Signal generated at bar i (after close), enter on bar i+1 open
                    next_row = working_frame.iloc[i + 1]
                    next_open = float(next_row["Open"])
                    entry_price = next_open * (1 + slippage_fraction if side == "LONG" else 1 - slippage_fraction)

                    tp_price = entry_price * (1 + tp_pct / 100.0) if side == "LONG" else entry_price * (1 - tp_pct / 100.0)
                    sl_price = entry_price * (1 - sl_pct / 100.0) if side == "LONG" else entry_price * (1 + sl_pct / 100.0)

                    # Liquidation calculation based on leverage
                    liq_price = entry_price * (1 - (1.0 / leverage)) if side == "LONG" else entry_price * (1 + (1.0 / leverage))

                    # BUG FIX #3: Pattern-based confidence for dynamic position sizing
                    pattern_conf = confidence  # fallback: use model confidence
                    if _pattern_engine is not None:
                        # Extract pattern features for current bar
                        live_features = {
                            feat: float(row.get(feat, 0.0))
                            for feat in PATTERN_FEATURES
                            if feat in row.index
                        }
                        if live_features:
                            pattern_conf = _pattern_engine.get_confidence(live_features, side)

                    dynamic_margin_pct = _confidence_to_margin_pct(
                        pattern_conf,
                        base_pct=cfg.testing.BASE_MARGIN_PCT,
                        max_pct=cfg.testing.MAX_MARGIN_PCT,
                    )
                    dynamic_margin_fraction = dynamic_margin_pct / 100.0
                    quantity, notional_usd = compute_position_size_for_margin_budget(
                        equity_usd, entry_price, dynamic_margin_fraction, leverage
                    )

                    if quantity > 0:
                        active_trades.append({
                            "symbol": symbol, "side": side, "entry_price": entry_price, "tp_price": tp_price, "sl_price": sl_price,
                            "liq_price": liq_price, "sl_price_initial": sl_price, "tp_pct": tp_pct, "sl_pct": sl_pct, "quantity": quantity,
                            "notional_usd": notional_usd, "entry_time": str(next_row.name), "entry_index": i + 1,  # Note: entry at i+1
                            "confidence": confidence, "pattern_confidence": pattern_conf, "dynamic_margin_pct": dynamic_margin_pct,
                            "qty_ratio": 1.0, "capital_before": equity_usd, "bars_held": 0,
                            "mae_pct": 0.0, "mfe_pct": 0.0, "max_price": float(next_row["High"]), "min_price": float(next_row["Low"])
                        })
                        daily_trade_count += 1

    working_frame["paper_equity_curve"] = equity_by_bar
    summary = summarize_trade_feedback(trade_records)
    
    # Add time range info
    if not working_frame.empty:
        summary["start_time"] = str(working_frame.index[0])
        summary["end_time"] = str(working_frame.index[-1])
        summary["duration_days"] = (working_frame.index[-1] - working_frame.index[0]).total_seconds() / 86400.0
        summary["trades_per_day"] = len(trade_records) / max(summary["duration_days"], 0.001)
        if len(trade_records) > 1:
            returns = np.array([t.return_fraction for t in trade_records], dtype=np.float64)
            if returns.std() > 0:
                trades_per_year = len(trade_records) / max(summary["duration_days"] / 365.0, 1e-9)
                summary["sharpe_ratio"] = round(float((returns.mean() / returns.std()) * np.sqrt(trades_per_year)), 3)
            else:
                summary["sharpe_ratio"] = 0.0

    # Max Drawdown
    equity_series = pd.Series(equity_by_bar)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    summary["max_drawdown_pct"] = float(drawdown.min() * 100.0)
    
    return working_frame, trade_records, summary

def summarize_trade_feedback(trade_records: List[PaperTradeRecord]) -> dict:
    if not trade_records: return {"trade_count": 0, "win_rate_pct": 0.0}
    
    wins = [t for t in trade_records if t.pnl_net_usd > 0]
    losses = [t for t in trade_records if t.pnl_net_usd <= 0]
    long_trades = [t for t in trade_records if t.side.upper() == "LONG"]
    short_trades = [t for t in trade_records if t.side.upper() == "SHORT"]
    
    total_profit = sum(t.pnl_net_usd for t in wins)
    total_loss = abs(sum(t.pnl_net_usd for t in losses))
    gross_pnl = sum(t.pnl_gross_usd for t in trade_records)
    total_fees = sum(t.fees_usd for t in trade_records)
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_win = total_profit / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0
    expectancy = sum(t.pnl_net_usd for t in trade_records) / len(trade_records)

    def side_stats(side_records: list[PaperTradeRecord]) -> dict:
        side_wins = [t for t in side_records if t.pnl_net_usd > 0]
        side_losses = [t for t in side_records if t.pnl_net_usd <= 0]
        side_profit = sum(t.pnl_net_usd for t in side_wins)
        side_loss = abs(sum(t.pnl_net_usd for t in side_losses))
        return {
            "trades": len(side_records),
            "wins": len(side_wins),
            "losses": len(side_losses),
            "win_rate_pct": (len(side_wins) / len(side_records) * 100.0) if side_records else 0.0,
            "net_pnl_usd": sum(t.pnl_net_usd for t in side_records),
            "profit_factor": side_profit / side_loss if side_loss > 0 else (float("inf") if side_profit > 0 else 0.0),
        }

    max_win_streak = max_loss_streak = current_win_streak = current_loss_streak = 0
    for trade in trade_records:
        if trade.pnl_net_usd > 0:
            current_win_streak += 1
            current_loss_streak = 0
        else:
            current_loss_streak += 1
            current_win_streak = 0
        max_win_streak = max(max_win_streak, current_win_streak)
        max_loss_streak = max(max_loss_streak, current_loss_streak)
    
    return {
        "trade_count": len(trade_records),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": (len(wins) / len(trade_records)) * 100.0,
        "total_pnl_net_usd": sum(t.pnl_net_usd for t in trade_records),
        "total_pnl_gross_usd": gross_pnl,
        "total_fees_usd": total_fees,
        "positive_pnl_usd": total_profit,
        "negative_pnl_usd": -total_loss,
        "profit_factor": profit_factor,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "expectancy_usd": expectancy,
        "avg_holding_bars": sum(t.holding_bars for t in trade_records) / len(trade_records),
        "final_equity_usd": trade_records[-1].equity_after_usd,
        "long": side_stats(long_trades),
        "short": side_stats(short_trades),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }

def print_rich_summary(strategy_name: str, symbol: str, summary: dict):
    print(f"\n--- {strategy_name} | {symbol} SUMMARY ---")
    print(f"Trades: {summary['trade_count']}")
    print(f"Win Rate: {summary.get('win_rate_pct', 0):.2f}%")
    print(f"Net PnL: ${summary.get('total_pnl_net_usd', 0):,.2f}")
    print(f"Final Equity: ${summary.get('final_equity_usd', 0):,.2f}")
    print(f"Sharpe: {summary.get('sharpe_ratio', 0):.3f}")
    print("-" * 30)
