# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import os
import json
from config import config
from strategies.base import BaseStrategy
from neural_engine.model import MultiHeadTradingModel
from neural_engine.feature_utils import add_technical_indicators, get_feature_columns

class ShortNNStrategy(BaseStrategy):
    def __init__(self, model_path: str, scaler_mean_path: str, scaler_scale_path: str, threshold_path: str | None = None):
        self.feature_cols = get_feature_columns()
        self.input_dim = len(self.feature_cols)
        self.window_size = config.model.WINDOW_SIZE
        self.long_threshold = config.strategy.LONG_CONFIDENCE_THRESHOLD
        self.short_threshold = config.strategy.SHORT_CONFIDENCE_THRESHOLD
        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
                self.long_threshold = float(thresholds.get("long_probability_threshold", self.long_threshold))
                self.short_threshold = float(thresholds.get("short_probability_threshold", self.short_threshold))
        
        # Load model
        self.model = MultiHeadTradingModel(input_dim=self.input_dim)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            except RuntimeError as exc:
                raise RuntimeError(
                    "Saved model is incompatible with the current feature/model schema. "
                    "Retrain with neural_engine/train.py to regenerate the checkpoint and scaler."
                ) from exc
        self.model.eval()
        
        # Load scaler params
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)

    @property
    def name(self) -> str:
        return "LongShort_Transformer"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Add technical indicators
        df_feats = add_technical_indicators(df)
        
        # 2. Extract features and scale
        X = df_feats[self.feature_cols].values
        X_scaled = (X - self.scaler_mean) / (self.scaler_scale + 1e-9)
        
        # 3. Predict using sliding windows
        n = len(X_scaled)
        short_probs = np.zeros(n)
        confidences = np.zeros(n)
        
        # Initialize outputs
        df['ai_verdict'] = 1 # Neutral
        df['ai_confidence'] = 0.0
        df['ai_take_profit_pct'] = 1.0
        df['ai_stop_loss_pct'] = 0.5
        df['ai_qty_ratio'] = 0.0
        df['ai_directional_edge'] = 0.0
        df['ai_expected_return_pct'] = 0.0

        # We need at least window_size bars to make a prediction
        if n < self.window_size:
            return df

        # Prepare batches for inference to speed up
        batch_size = 256
        X_windows = []
        indices = []
        
        for i in range(self.window_size, n):
            window = X_scaled[i-self.window_size:i]
            X_windows.append(window)
            indices.append(i)
            
            if len(X_windows) == batch_size or i == n - 1:
                X_batch = torch.FloatTensor(np.array(X_windows))
                with torch.no_grad():
                    outputs = self.model(X_batch)
                    logits = outputs["direction"]
                    sizing = outputs["sizing"]
                    
                    # Get probabilities via Softmax
                    probs = torch.softmax(logits, dim=1).numpy()
                    
                    for j, idx in enumerate(indices):
                        # Model classes: 0=long, 1=neutral, 2=short.
                        p_long = probs[j, 0]
                        p_short = probs[j, 2]
                        p_neutral = probs[j, 1]
                        long_edge = p_long - p_neutral
                        short_edge = p_short - p_neutral

                        raw_qty = float(sizing[j, 0].item())
                        atr_source_idx = max(idx - 1, 0)
                        atr_pct = float(df_feats["natr"].iloc[atr_source_idx]) if "natr" in df_feats else 0.0
                        
                        # Model predicts a multiplier of MAX_ATR_TARGET_PCT. 
                        # We want to balance this with a volatility-based floor.
                        raw_tp_pct = float(sizing[j, 1].item()) * config.strategy.MAX_ATR_TARGET_PCT
                        raw_sl_pct = float(sizing[j, 2].item()) * config.strategy.MAX_ATR_STOP_PCT
                        
                        # Use a mix of model prediction and ATR logic
                        tp_pct = float(np.clip(raw_tp_pct, config.strategy.MIN_ATR_TARGET_PCT, config.strategy.MAX_ATR_TARGET_PCT))
                        sl_pct = float(np.clip(raw_sl_pct, config.strategy.MIN_ATR_STOP_PCT, config.strategy.MAX_ATR_STOP_PCT))
                        
                        # Safety: Ensure minimum RR ratio
                        min_rr = config.strategy.MIN_REWARD_RISK_RATIO
                        if tp_pct / max(sl_pct, 1e-9) < min_rr:
                            # If RR is too low, try to increase TP or decrease SL
                            if tp_pct < config.strategy.MIN_ATR_TARGET_PCT:
                                tp_pct = config.strategy.MIN_ATR_TARGET_PCT
                            sl_pct = tp_pct / min_rr

                        trade_cost_pct = config.strategy.ROUND_TRIP_FEE_PCT + 2.0 * config.strategy.SLIPPAGE_PCT
                        long_expected_return_pct = (p_long * tp_pct) - ((1.0 - p_long) * sl_pct) - trade_cost_pct
                        short_expected_return_pct = (p_short * tp_pct) - ((1.0 - p_short) * sl_pct) - trade_cost_pct
                        verdict = 1
                        confidence = max(p_long, p_short)
                        directional_edge = max(long_edge, short_edge)
                        expected_return_pct = max(long_expected_return_pct, short_expected_return_pct)
                        rsi = float(df_feats["rsi"].iloc[idx]) if "rsi" in df_feats else 50.0
                        adx = float(df_feats["adx"].iloc[idx]) if "adx" in df_feats else 0.0
                        trend_sma = float(df_feats["trend_sma_20_50"].iloc[idx]) if "trend_sma_20_50" in df_feats else 0.0
                        regime_enabled = bool(config.strategy.USE_REGIME_FILTER)
                        long_regime_ok = (
                            not regime_enabled
                            or (
                                adx >= config.strategy.REGIME_MIN_ADX
                                and config.strategy.LONG_RSI_MIN <= rsi <= config.strategy.LONG_RSI_MAX
                                and trend_sma >= config.strategy.LONG_MIN_TREND_SMA_20_50
                            )
                        )
                        short_regime_ok = (
                            not regime_enabled
                            or (
                                adx >= config.strategy.REGIME_MIN_ADX
                                and config.strategy.SHORT_RSI_MIN <= rsi <= config.strategy.SHORT_RSI_MAX
                                and trend_sma <= config.strategy.SHORT_MAX_TREND_SMA_20_50
                            )
                        )
                        
                        # Choose the side with higher probability/edge if both qualify
                        is_long_valid = (
                            p_long >= self.long_threshold
                            and long_edge >= config.strategy.MIN_DIRECTIONAL_EDGE
                            and long_expected_return_pct >= config.strategy.MIN_EXPECTED_RETURN_PCT
                            and long_regime_ok
                        )
                        is_short_valid = (
                            p_short >= self.short_threshold
                            and short_edge >= config.strategy.MIN_DIRECTIONAL_EDGE
                            and short_expected_return_pct >= config.strategy.MIN_EXPECTED_RETURN_PCT
                            and short_regime_ok
                        )

                        if is_long_valid and is_short_valid:
                            # If both valid, pick the stronger one
                            if p_long >= p_short:
                                is_short_valid = False
                            else:
                                is_long_valid = False

                        if is_long_valid:
                            verdict = 0
                            confidence = p_long
                            directional_edge = long_edge
                            expected_return_pct = long_expected_return_pct
                        elif is_short_valid:
                            verdict = 2
                            confidence = p_short
                            directional_edge = short_edge
                            expected_return_pct = short_expected_return_pct

                        df.at[df.index[idx], 'ai_verdict'] = verdict
                        
                        df.at[df.index[idx], 'ai_confidence'] = confidence
                        df.at[df.index[idx], 'ai_directional_edge'] = directional_edge
                        df.at[df.index[idx], 'ai_qty_ratio'] = float(np.clip(raw_qty, 0.0, 1.0))
                        df.at[df.index[idx], 'ai_take_profit_pct'] = tp_pct
                        df.at[df.index[idx], 'ai_stop_loss_pct'] = sl_pct
                        df.at[df.index[idx], 'ai_expected_return_pct'] = expected_return_pct
                
                X_windows = []
                indices = []

        return df
