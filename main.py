# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Local imports
from data_gathering import fetch_data
from features import create_full_feature_set
from models import MultiHeadTradingModel, RiskAwareLoss

warnings.filterwarnings('ignore')

def main():
    # 0. Hardware Acceleration Setup (Optimized for Mac)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Using device: {device}")

    # 1. Fetch Data
    df = fetch_data(symbol="ADAUSD", total_days=50, interval="15m")
    if df.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Add Features
    df_with_features = create_full_feature_set(df, lookahead=10)

    # 3. Data Preparation
    excluded_columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'upside_pct', 'downside_pct', 'future_drawdown_pct', 'direction_label']
    feature_columns = [col for col in df_with_features.columns if col not in excluded_columns]

    X = df_with_features[feature_columns].dropna()
    y = df_with_features.loc[X.index, ['upside_pct', 'downside_pct', 'future_drawdown_pct']].dropna()
    X = X.loc[y.index]

    print(f"\nFinal Feature Set Shape: {X.shape}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train_df, y_val_df = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

    # 4. Model Initialization & Dummy Test
    input_dim = X_train.shape[1]
    model = MultiHeadTradingModel(input_dim=input_dim).to(device)
    
    # Sample tensors for verification
    X_sample = torch.tensor(X_train[:5], dtype=torch.float32).to(device)
    y_sample_targets = {
        'upside': torch.tensor(y_train_df['upside_pct'].values[:5], dtype=torch.float32).to(device),
        'downside': torch.tensor(y_train_df['downside_pct'].values[:5], dtype=torch.float32).to(device),
        'future_drawdown': torch.tensor(y_train_df['future_drawdown_pct'].values[:5], dtype=torch.float32).to(device)
    }

    # Pass data through model
    model.eval()
    with torch.no_grad():
        outputs = model(X_sample)
    
    print("\n=== SAMPLE MODEL OUTPUT ===")
    for key, value in outputs.items():
        print(f"{key}: {value.cpu().numpy().flatten()}")

    # Calculate Loss
    criterion = RiskAwareLoss().to(device)
    loss_dict = criterion(outputs, y_sample_targets)
    
    print("\n=== SAMPLE LOSS ===")
    for key, value in loss_dict.items():
        val = value.item() if isinstance(value, torch.Tensor) else value
        print(f"{key}: {val}")

if __name__ == "__main__":
    main()
