import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.data_handler import fetch_data
from engine.features import add_oracle_target_labels
from config import config
from ui_utils import console
from training_short_only.model import SimpleShortNet
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns

def train_short_model():
    symbol = config.data.SYMBOLS[0]
    days = config.data.TOTAL_DAYS
    interval = config.data.INTERVAL
    
    # 1. Fetch and Prepare Data
    df = fetch_data(symbol, days, interval)
    if df.empty:
        console.print("[error]No data found for training.[/error]")
        return

    # Add indicators and labels
    df = add_technical_indicators(df)
    df = add_oracle_target_labels(df)
    
    feature_cols = get_feature_columns()
    
    # 2. Prepare Training Samples (Short Only Filter)
    # Class 1: Short (label 2)
    # Class 0: Everything else (label 0, 1)
    X = df[feature_cols].values
    y = (df['direction_label'] == 2).astype(float).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors (using .copy() to avoid non-writable warning)
    X_tensor = torch.FloatTensor(X_scaled.copy())
    y_tensor = torch.FloatTensor(y.copy()).view(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. Initialize Model
    model = SimpleShortNet(input_dim=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 50
    console.print(f"[info]Starting training for {symbol} Short-Only Model...[/info]")
    
    from ui_utils import get_progress
    with get_progress() as progress:
        task = progress.add_task("[cyan]Training Neural Net...", total=epochs)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                console.print(f"  [dim]↳ Epoch {epoch+1:02d}: Loss {avg_loss:.6f}[/dim]")
            
    # 5. Save Model and Scaler
    model_dir = "results/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "short_model_eth.pth"))
    
    # Save scaler parameters (simplified for this demo)
    np.save(os.path.join(model_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(model_dir, "scaler_scale.npy"), scaler.scale_)
    
    console.print(f"[success]✅ Model saved to {model_dir}[/success]")
    return model, scaler

if __name__ == "__main__":
    train_short_model()
