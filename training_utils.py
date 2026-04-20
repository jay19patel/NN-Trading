# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
from models import MultiHeadTradingModel, RiskAwareLoss

class TradingDataset(Dataset):
    """
    Custom Dataset for trading data.
    FIXED: BUG #1 - Now supports 3D sequence tensors [Batch, Seq, Features].
    """
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        self.X = torch.FloatTensor(X)
        self.y_upside = torch.FloatTensor(y['upside'])
        self.y_downside = torch.FloatTensor(y['downside'])
        self.y_drawdown = torch.FloatTensor(y['drawdown'])
        self.y_direction = torch.LongTensor(y['direction'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {
            'upside': self.y_upside[idx],
            'downside': self.y_downside[idx],
            'future_drawdown': self.y_drawdown[idx],
            'direction': self.y_direction[idx]
        }

def create_sequences(X: np.ndarray, y: Dict[str, np.ndarray], seq_len: int = 32) -> Tuple[np.ndarray, Dict]:
    """
    Transforms 2D data into 3D temporal sequences.
    Shape: [N, Features] -> [N - seq_len, seq_len, Features]
    """
    X_seq = []
    y_seq = {k: [] for k in y.keys()}
    
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        for k in y.keys():
            y_seq[k].append(y[k][i-1]) # Target is the label of the LAST bar in sequence
            
    return np.array(X_seq), {k: np.array(v) for k, v in y_seq.items()}

def prepare_data(df: pd.DataFrame, test_days: int = 10) -> Tuple:
    """Prepare and scale data for training. Splits last N days for testing."""
    # 1. Identify feature columns (exclude targets and non-numeric)
    # FIXED: BUG #4 - Included future-derived ratios in target_cols to prevent leakage.
    target_cols = [
        'upside_pct', 'downside_pct', 'future_drawdown_pct', 
        'reward_risk_ratio', 'edge_ratio', 'pain_ratio', # <-- Leakage sources added
        'direction_label', 'time', 'label', 'index'
    ]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in target_cols]
    
    # 2. Handle NaNs/Infs
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    
    # 3. Split into Train and Test (Last 10 Days)
    bars_per_day = (24 * 60) // 15
    test_size = test_days * bars_per_day
    
    train_df = df_clean.iloc[:-test_size]
    test_df = df_clean.iloc[-test_size:]
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    
    # 5. Prepare Targets
    def get_targets(data_df):
        return {
            'upside': data_df['upside_pct'].values,
            'downside': data_df['downside_pct'].values,
            'drawdown': data_df['future_drawdown_pct'].values,
            'direction': data_df['direction_label'].values
        }
    
    y_train = get_targets(train_df)
    y_test = get_targets(test_df)
    
    return X_train, y_train, X_test, y_test, feature_cols, scaler

def count_parameters(model: nn.Module) -> int:
    """Calculate total trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(X_train: np.ndarray, y_train: Dict, device: torch.device, input_dim: int, 
                X_val: np.ndarray = None, y_val: Dict = None, epochs: int = 50):
    """Handle the training loop with Balanced Sampling and Overfitting Diagnostics"""
    model = MultiHeadTradingModel(input_dim=input_dim).to(device)
    param_count = count_parameters(model)
    
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"Features (Inputs): {input_dim}")
    print(f"Total Trainable Parameters: {param_count:,}")
    
    criterion = RiskAwareLoss().to(device)
    # FIXED: BUG #10 - Reduced weight_decay for better pattern stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # ⚖️ BALANCED SAMPLING LOGIC
    # FIXED: BUG #6 - Corrected dtype and tensor conversion for sampling
    labels = y_train['direction'].astype(np.int64) # Force int64
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = torch.FloatTensor(class_weights[labels])
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    dataset = TradingDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    print(f"\n🧠 AI Pattern Learning (Device: {device})...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_direction = 0
        total_samples = 0
        
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = {k: v.to(device) for k, v in batch_y.items()}
            
            optimizer.zero_grad()
            preds = model(batch_X)
            loss_dict = criterion(preds, batch_y)
            loss = loss_dict['total']
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pred_dir = torch.argmax(preds['direction'], dim=1)
            correct_direction += (pred_dir == batch_y['direction']).sum().item()
            total_samples += batch_X.size(0)
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
            
        if (epoch + 1) % 10 == 0:
            train_acc = (correct_direction / total_samples) * 100
            
            val_acc_str = ""
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(device)
                    v_preds = model(X_v)
                    v_dir = torch.argmax(v_preds['direction'], dim=1)
                    v_act = torch.LongTensor(y_val['direction']).to(device)
                    val_acc = (v_dir == v_act).sum().item() / len(X_val) * 100
                    val_acc_str = f" | Val-Acc: {val_acc:.1f}%"
            
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Train-Acc: {train_acc:.1f}%{val_acc_str}")
            
    return model

@torch.no_grad()
def get_ai_predictions(model: nn.Module, X: np.ndarray, device: torch.device) -> pd.DataFrame:
    """Run inference and generate verdicts using the new Direction Head"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    preds = model(X_tensor)
    
    # Use the direction head probabilities
    probs = torch.softmax(preds['direction'], dim=1)
    verdicts = torch.argmax(probs, dim=1).cpu().numpy()
    
    upside = preds['upside'].cpu().numpy().flatten()
    confidence = preds['confidence'].cpu().numpy().flatten()
    
    res = pd.DataFrame({
        'ai_upside': upside,
        'ai_confidence': confidence,
        'ai_verdict': verdicts
    })
    
    # Analyze raw signals before filtering
    raw_buy = (res['ai_verdict'] == 0).sum()
    raw_sell = (res['ai_verdict'] == 2).sum()
    print(f"📉 Raw AI Signals Found: Buy={raw_buy}, Sell={raw_sell}")

    # 🏁 SAFETY OVERRIDE:
    # Lowered threshold to 0.3 to allow more signals while still filtering low-confidence noise
    safe_threshold = 0.3
    safe_mask = (res['ai_confidence'] < safe_threshold)
    overrides = res.loc[safe_mask & (res['ai_verdict'] != 1), 'ai_verdict'].count()
    
    if overrides > 0:
        print(f"🛡️ Safety Filter: Suppressed {overrides} low-confidence signals (Conf < {safe_threshold})")
    
    res.loc[safe_mask, 'ai_verdict'] = 1 # Force Neutral if low confidence
    
    return res
