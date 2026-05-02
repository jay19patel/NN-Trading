# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import os
from config import config
from strategies.base import BaseStrategy
from training_short_only.model import MultiHeadTradingModel
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns

class ShortNNStrategy(BaseStrategy):
    def __init__(self, model_path: str, scaler_mean_path: str, scaler_scale_path: str):
        self.feature_cols = get_feature_columns()
        self.input_dim = len(self.feature_cols)
        self.window_size = config.model.WINDOW_SIZE
        
        # Load model
        self.model = MultiHeadTradingModel(input_dim=self.input_dim)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
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
                        
                        # Set verdict based on Short probability threshold (e.g. 0.5)
                        if p_short > 0.5:
                            df.at[df.index[idx], 'ai_verdict'] = 2 # Short
                        
                        df.at[df.index[idx], 'ai_confidence'] = p_short
                        df.at[df.index[idx], 'ai_directional_edge'] = p_short - p_neutral
                        df.at[df.index[idx], 'ai_qty_ratio'] = sizing[j, 0].item()
                        
                        # Optional: Use predicted TP/SL if available
                        # df.at[df.index[idx], 'ai_take_profit_pct'] = sizing[j, 1].item() * config.strategy.ORACLE_MAX_TP_PCT
                
                X_windows = []
                indices = []

        return df
