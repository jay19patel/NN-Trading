# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import os
import json
from config import config
from strategies.base import BaseStrategy
from training_short_only.model import MultiHeadTradingModel
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns

class ShortNNStrategy(BaseStrategy):
    def __init__(self, model_path: str, scaler_mean_path: str, scaler_scale_path: str, threshold_path: str | None = None):
        self.feature_cols = get_feature_columns()
        self.input_dim = len(self.feature_cols)
        self.window_size = config.model.WINDOW_SIZE
        self.short_threshold = config.strategy.AI_CONFIDENCE_THRESHOLD
        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                self.short_threshold = float(json.load(f).get("short_probability_threshold", self.short_threshold))
        
        # Load model
        self.model = MultiHeadTradingModel(input_dim=self.input_dim)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            except RuntimeError as exc:
                raise RuntimeError(
                    "Saved model is incompatible with the current feature/model schema. "
                    "Retrain with training_short_only/train.py to regenerate the checkpoint and scaler."
                ) from exc
        self.model.eval()
        
        # Load scaler params
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)

    @property
    def name(self) -> str:
        return "Short_Transformer"

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
                        # Short probability is at index 2
                        p_short = probs[j, 2]
                        p_neutral = probs[j, 1]
                        directional_edge = p_short - p_neutral

                        raw_qty = float(sizing[j, 0].item())
                        raw_tp_pct = float(sizing[j, 1].item()) * config.strategy.ORACLE_MAX_TP_PCT
                        raw_sl_pct = float(sizing[j, 2].item()) * config.strategy.ORACLE_MAX_SL_PCT
                        tp_pct = float(np.clip(raw_tp_pct, config.strategy.ORACLE_MIN_TP_PCT, config.strategy.ORACLE_MAX_TP_PCT))
                        sl_pct = float(np.clip(raw_sl_pct, config.strategy.ORACLE_MIN_SL_PCT, config.strategy.ORACLE_MAX_SL_PCT))
                        if tp_pct / max(sl_pct, 1e-9) < config.strategy.MIN_REWARD_RISK_RATIO:
                            tp_pct = min(config.strategy.ORACLE_MAX_TP_PCT, sl_pct * config.strategy.MIN_REWARD_RISK_RATIO)
                        expected_return_pct = (p_short * tp_pct) - ((1.0 - p_short) * sl_pct) - (
                            config.strategy.ROUND_TRIP_FEE_PCT + 2.0 * config.strategy.SLIPPAGE_PCT
                        )
                        
                        if (
                            p_short >= self.short_threshold
                            and directional_edge >= config.strategy.MIN_DIRECTIONAL_EDGE
                            and expected_return_pct >= config.strategy.MIN_EXPECTED_RETURN_PCT
                            and tp_pct / max(sl_pct, 1e-9) >= config.strategy.MIN_REWARD_RISK_RATIO
                        ):
                            df.at[df.index[idx], 'ai_verdict'] = 2 # Short
                        
                        df.at[df.index[idx], 'ai_confidence'] = p_short
                        df.at[df.index[idx], 'ai_directional_edge'] = directional_edge
                        df.at[df.index[idx], 'ai_qty_ratio'] = float(np.clip(raw_qty, 0.0, 1.0))
                        df.at[df.index[idx], 'ai_take_profit_pct'] = tp_pct
                        df.at[df.index[idx], 'ai_stop_loss_pct'] = sl_pct
                        df.at[df.index[idx], 'ai_expected_return_pct'] = expected_return_pct
                
                X_windows = []
                indices = []

        return df
