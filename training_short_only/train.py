# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from ui_utils import console
from engine.data_handler import fetch_data
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns
from training_short_only.model import MultiHeadTradingModel, TradingLoss

class TradingDataset(Dataset):
    def __init__(self, X, y_dir, y_qty, y_tp, y_sl, window_size):
        self.X = torch.FloatTensor(X)
        self.y_dir = torch.LongTensor(y_dir)
        self.y_qty = torch.FloatTensor(y_qty)
        self.y_tp = torch.FloatTensor(y_tp)
        self.y_sl = torch.FloatTensor(y_sl)
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # We want the window leading up to idx + window_size
        actual_idx = idx + self.window_size
        return (
            self.X[actual_idx - self.window_size : actual_idx],
            {
                "direction": self.y_dir[actual_idx],
                "qty_ratio": self.y_qty[actual_idx],
                "take_profit_pct": self.y_tp[actual_idx],
                "stop_loss_pct": self.y_sl[actual_idx],
                "actual_pnl_pct": torch.tensor(0.0) # Placeholder if real PnL not calculated
            }
        )

def train_short_model():
    symbol = config.data.SYMBOLS[0]
    days = config.data.TOTAL_DAYS
    interval = config.data.INTERVAL
    window_size = config.model.WINDOW_SIZE
    
    # 1. Fetch and Prepare Data
    df = fetch_data(symbol, days, interval)
    if df.empty:
        console.print("[error]No data fetched.[/error]")
        return
        
    # Generate Oracle Labels for training
    console.print(f"[info]Generating Oracle labels for {len(df)} bars...[/info]")
    from strategies.oracle import OracleStrategy
    oracle = OracleStrategy()
    df = oracle.generate_signals(df)
    
    # Map Oracle output to expected label names
    df['direction_label'] = df['ai_verdict']
    df['qty_ratio'] = df['ai_qty_ratio']
    df['take_profit_pct'] = df['ai_take_profit_pct']
    df['stop_loss_pct'] = df['ai_stop_loss_pct']
    
    df = add_technical_indicators(df)
    feature_cols = get_feature_columns()
    
    X = df[feature_cols].values
    # Short-Only Mapping: Map Long(0) to Neutral(1), keep Short(2) as is
    y_dir = df['direction_label'].values
    y_dir = np.where(y_dir == 0, 1, y_dir)
    
    y_qty = df['qty_ratio'].values
    y_tp = df['take_profit_pct'].values
    y_sl = df['stop_loss_pct'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Dataset and Loader
    dataset = TradingDataset(X_scaled, y_dir, y_qty, y_tp, y_sl, window_size)
    loader = DataLoader(dataset, batch_size=config.training.BATCH_SIZE, shuffle=True)
    
    # 3. Initialize Model and Loss
    device = config.DEVICE
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    criterion = TradingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.LR)
    
    # 4. Training Loop
    epochs = config.training.EPOCHS
    console.print(f"[info]Starting Transformer training for {symbol} Short-Only...[/info]")
    
    from ui_utils import get_progress
    with get_progress() as progress:
        task = progress.add_task("[cyan]Training Transformer...", total=epochs)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_targets in loader:
                # Move targets to device
                batch_X = batch_X.to(device)
                for k in batch_targets:
                    batch_targets[k] = batch_targets[k].to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss_dict = criterion(predictions, batch_targets)
                loss = loss_dict["total"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                console.print(f"  [dim]↳ Epoch {epoch+1:02d}: Loss {avg_loss:.6f}[/dim]")
            
    # 5. Save Model and Scaler
    model_dir = "results/models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "short_model_eth.pth"))
    np.save(os.path.join(model_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(model_dir, "scaler_scale.npy"), scaler.scale_)
    
    console.print("[success]✅ Transformer Model saved to results/models[/success]")

if __name__ == "__main__":
    train_short_model()
