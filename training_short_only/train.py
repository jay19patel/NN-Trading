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

from config import config, bars_per_day
from ui_utils import console
import time
from engine.data_handler import fetch_data
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns
from training_short_only.model import MultiHeadTradingModel, TradingLoss
from training_short_only.backtest_short import run_short_backtest

class TradingDataset(Dataset):
    def __init__(self, X_seq, y_dir, y_qty, y_tp, y_sl, device="cpu"):
        # Pre-load entire dataset to GPU (MPS) to eliminate transfer overhead
        self.X = torch.from_numpy(X_seq).float().to(device)
        self.y_dir = torch.from_numpy(y_dir).long().to(device)
        self.y_qty = torch.from_numpy(y_qty).float().to(device)
        self.y_tp = torch.from_numpy(y_tp).float().to(device)
        self.y_sl = torch.from_numpy(y_sl).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            {
                "direction": self.y_dir[idx],
                "qty_ratio": self.y_qty[idx],
                "take_profit_pct": self.y_tp[idx],
                "stop_loss_pct": self.y_sl[idx],
                "actual_pnl_pct": torch.tensor(0.0, device=self.X.device)
            }
        )

def train_short_model():
    symbols = config.data.SYMBOLS
    train_days = config.training.TRAINING_DATA_DAYS
    test_days = config.data.TOTAL_DAYS
    total_days_to_fetch = train_days + test_days
    interval = config.data.INTERVAL
    window_size = config.model.WINDOW_SIZE
    
    X_all, y_dir_all, y_qty_all, y_tp_all, y_sl_all = [], [], [], [], []
    feature_cols = get_feature_columns()
    
    for symbol in symbols:
        console.print(f"[info]Processing data for {symbol}...[/info]")
        cache_path = f"data/processed_train_{symbol}_{interval}.parquet"
        
        if os.path.exists(cache_path):
            df_full = pd.read_parquet(cache_path)
        else:
            df = fetch_data(symbol, total_days_to_fetch, interval)
            if df.empty: continue
            
            from strategies.oracle import OracleStrategy
            oracle = OracleStrategy()
            df = oracle.generate_signals(df)
            df['direction_label'] = df['ai_verdict']
            df['qty_ratio'] = df['ai_qty_ratio']
            df['take_profit_pct'] = df['ai_take_profit_pct']
            df['stop_loss_pct'] = df['ai_stop_loss_pct']
            df = add_technical_indicators(df)
            os.makedirs("data", exist_ok=True)
            df.to_parquet(cache_path)
            df_full = df

        # Data Split
        bars_day = bars_per_day(interval)
        test_bars = int(test_days * bars_day)
        train_df = df_full.iloc[:-test_bars].copy()
        
        # Scale and Window per symbol
        X = train_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        y_dir = train_df['direction_label'].values
        y_dir = np.where(y_dir == 0, 1, y_dir) # Short-only mapping
        
        from numpy.lib.stride_tricks import sliding_window_view
        X_seq = sliding_window_view(X_scaled[:-1], (window_size, X_scaled.shape[1])).squeeze(1)
        
        X_all.append(X_seq)
        y_dir_all.append(y_dir[window_size:])
        y_qty_all.append(train_df['qty_ratio'].values[window_size:])
        y_tp_all.append(train_df['take_profit_pct'].values[window_size:])
        y_sl_all.append(train_df['stop_loss_pct'].values[window_size:])

    if not X_all:
        console.print("[error]No data collected for any symbol.[/error]")
        return

    # Combine all symbols
    X_final = np.concatenate(X_all, axis=0)
    y_dir_final = np.concatenate(y_dir_all, axis=0)
    y_qty_final = np.concatenate(y_qty_all, axis=0)
    y_tp_final = np.concatenate(y_tp_all, axis=0)
    y_sl_final = np.concatenate(y_sl_all, axis=0)
    
    console.print(f"[success]Combined Dataset: {len(X_final)} total samples from {len(symbols)} symbols.[/success]")
    
    # 2. Dataset and Loader
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = TradingDataset(X_final, y_dir_final, y_qty_final, y_tp_final, y_sl_final, device=device)
    # Increased Batch Size for maximum GPU saturation
    batch_size = 1024 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate Class Weights for Focal Loss (Inverse Frequency)
    y_dir_int = y_dir.astype(np.int64)
    class_counts = np.bincount(y_dir_int, minlength=3)
    total_samples = len(y_dir)
    # Handle zero counts safely
    class_weights = total_samples / (len(class_counts) * (class_counts + 1e-6))
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    console.print(f"[info]Class Weights: Neutral={class_weights[1]:.2f}, Short={class_weights[2]:.2f}[/info]")

    # 3. Initialize Model and Loss
    device = config.DEVICE
    console.print(f"[info]Using Device: [bold cyan]{device.upper()}[/bold cyan][/info]")
    
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    criterion = TradingLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. Training Loop
    epochs = config.training.EPOCHS
    console.print(f"[info]Starting Optimized Transformer training for {symbol}...[/info]")
    
    best_loss = float('inf')
    patience = 7
    no_improve_epochs = 0
    
    from ui_utils import get_progress
    with get_progress() as progress:
        task = progress.add_task("[cyan]Training Transformer...", total=epochs * len(loader))
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_targets in loader:
                # Data is already on device, no transfer needed!
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss_dict = criterion(predictions, batch_targets)
                loss = loss_dict["total"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                console.print(f"  [dim]↳ Epoch {epoch+1:02d}: Loss {avg_loss:.6f}[/dim]")
            
            # Early Stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_epochs = 0
                # Save best model
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), os.path.join("models", "short_model_eth_best.pth"))
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    console.print(f"[warning]Early stopping at epoch {epoch+1}[/warning]")
                    break
            
    # 5. Save Final Model and Scaler
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "short_model_eth.pth"))
    np.save(os.path.join(model_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(model_dir, "scaler_scale.npy"), scaler.scale_)
    
    console.print("[success]✅ Optimized Transformer Model saved to models/[/success]")
    
    # 6. Automatic Backtest on Unseen Data
    console.print("\n[info]🚀 Triggering Automatic Backtest on Unseen Data...[/info]")
    run_short_backtest()

if __name__ == "__main__":
    train_short_model()
