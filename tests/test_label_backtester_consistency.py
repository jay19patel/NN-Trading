# -*- coding: utf-8 -*-
import pandas as pd

from config import config
from engine.backtester import run_paper_portfolio_on_signals
from strategies.oracle import OracleStrategy


def _sample_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=80, freq="15min")
    rows = []
    price = 100.0
    for i in range(len(idx)):
        if i == 5:
            high, low, close = 101.0, 99.0, 100.0
        else:
            high, low, close = price + 0.1, price - 0.1, price
        rows.append({"Open": price, "High": high, "Low": low, "Close": close, "Volume": 1000.0})
        price = close
    return pd.DataFrame(rows, index=idx)


def test_same_bar_tp_sl_ambiguity_is_stop_first():
    df = _sample_frame()
    df["ai_verdict"] = 2
    df["ai_take_profit_pct"] = 0.6
    df["ai_stop_loss_pct"] = 0.3
    df["ai_qty_ratio"] = 1.0
    df["ai_confidence"] = 1.0
    df["ai_directional_edge"] = 1.0
    df["ai_expected_return_pct"] = 1.0

    original_threshold = config.strategy.AI_CONFIDENCE_THRESHOLD
    original_edge = config.strategy.MIN_DIRECTIONAL_EDGE
    original_ev = config.strategy.MIN_EXPECTED_RETURN_PCT
    original_slots = config.strategy.PARALLEL_SLOTS
    try:
        config.strategy.AI_CONFIDENCE_THRESHOLD = 0.0
        config.strategy.MIN_DIRECTIONAL_EDGE = 0.0
        config.strategy.MIN_EXPECTED_RETURN_PCT = 0.0
        config.strategy.PARALLEL_SLOTS = 1
        _, trades, _ = run_paper_portfolio_on_signals(
            panel=df,
            symbol="TEST",
            initial_capital_usd=100.0,
            risk_per_trade_pct_of_equity=1.0,
            max_notional_pct_of_equity=1.0,
            round_trip_fee_pct=0.0,
        )
    finally:
        config.strategy.AI_CONFIDENCE_THRESHOLD = original_threshold
        config.strategy.MIN_DIRECTIONAL_EDGE = original_edge
        config.strategy.MIN_EXPECTED_RETURN_PCT = original_ev
        config.strategy.PARALLEL_SLOTS = original_slots

    assert trades
    assert trades[0].outcome == "FAILED"


def test_oracle_outputs_grid_bounded_labels():
    df = _sample_frame()
    labeled = OracleStrategy().generate_signals(df)
    assert {"ai_verdict", "ai_take_profit_pct", "ai_stop_loss_pct", "ai_expected_return_pct"}.issubset(labeled.columns)
    assert labeled["ai_take_profit_pct"].between(min(config.strategy.TP_GRID_PCT), max(config.strategy.TP_GRID_PCT)).all()
    assert labeled["ai_stop_loss_pct"].between(min(config.strategy.SL_GRID_PCT), max(config.strategy.SL_GRID_PCT)).all()
