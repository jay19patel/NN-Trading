# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import os
from config import config
from strategies.base import BaseStrategy
from training_short_only.model import SimpleShortNet
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns

class ShortNNStrategy(BaseStrategy):
    def __init__(self, model_path: str, scaler_mean_path: str, scaler_scale_path: str):
        self.feature_cols = get_feature_columns()
        self.input_dim = len(self.feature_cols)
        
        # Load model
        self.model = SimpleShortNet(input_dim=self.input_dim)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        
        # Load scaler params
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)

    @property
    def name(self) -> str:
        return "Short_NN_Model"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Add technical indicators
        df_feats = add_technical_indicators(df)
        
        # 2. Extract features and scale
        X = df_feats[self.feature_cols].values
        X_scaled = (X - self.scaler_mean) / (self.scaler_scale + 1e-9)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # 3. Predict probabilities
        with torch.no_grad():
            probs = self.model(X_tensor).numpy().flatten()
        
        # 4. Map to backtester expected columns
        # Labels: 2 for Short, 1 for Neutral
        df['ai_verdict'] = np.where(probs > 0.5, 2, 1)
        df['ai_confidence'] = probs
        
        # Static TP/SL for this simple strategy
        df['ai_take_profit_pct'] = 1.0
        df['ai_stop_loss_pct'] = 0.5
        df['ai_qty_ratio'] = 1.0
        df['ai_directional_edge'] = probs # Using probability as edge
        
        return df
