# -*- coding: utf-8 -*-
import json
import os
import sys
import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import bars_per_day, config
from engine.data_handler import fetch_data
from neural_engine.labeler import OracleLabeler
from neural_engine.backtest_short import run_backtest
from neural_engine.feature_utils import add_technical_indicators, get_feature_columns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TradingDataset(Dataset):
    def __init__(self, X_seq, y_dir, y_qty, y_tp, y_sl, y_pnl, y_mag, y_time, device="cpu"):
        self.X = torch.from_numpy(X_seq).float().to(device)
        self.y_dir = torch.from_numpy(y_dir).long().to(device)
        self.y_qty = torch.from_numpy(y_qty).float().to(device)
        self.y_tp = torch.from_numpy(y_tp).float().to(device)
        self.y_sl = torch.from_numpy(y_sl).float().to(device)
        self.y_pnl = torch.from_numpy(y_pnl).float().to(device)
        self.y_mag = torch.from_numpy(y_mag).float().to(device)
        self.y_time = torch.from_numpy(y_time).float().to(device)

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
                "actual_pnl_pct": self.y_pnl[idx],
                "magnitude": self.y_mag[idx],
                "time": self.y_time[idx],
            },
        )

def _prepare_symbol_frame(symbol: str, total_days_to_fetch: int, interval: str, feature_cols: list[str]) -> pd.DataFrame:
    lookahead = int(config.features.LOOKAHEAD_BARS)
    cache_path = f"data/processed_train_{symbol}_{interval}_L{lookahead}.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = fetch_data(symbol, total_days_to_fetch, interval)
        if df.empty:
            return df

    required_cols = set(["Open", "High", "Low", "Close", "Volume"])
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{symbol} cache is missing OHLCV columns: {required_cols - set(df.columns)}")

    labeler = OracleLabeler()
    df = labeler.generate_labels(df.copy())
    df = add_technical_indicators(df)

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature generation missing columns: {missing_features}")

    os.makedirs("data", exist_ok=True)
    df.to_parquet(cache_path)
    return df.sort_index()

def _ordered_split(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars_day = bars_per_day(interval)
    val_bars = int(config.training.VALIDATION_DATA_DAYS * bars_day)
    test_bars = int(config.training.TEST_DATA_DAYS * bars_day)
    purge_bars = int(config.features.PURGE_BARS)
    min_train_bars = config.model.WINDOW_SIZE + 10

    if len(df) <= min_train_bars + val_bars + test_bars:
        train_end = max(int(len(df) * 0.70), min_train_bars)
        val_end = max(train_end + 1, int(len(df) * 0.85))
        return (
            df.iloc[:max(train_end - purge_bars, min_train_bars)].copy(),
            df.iloc[min(train_end + purge_bars, val_end):val_end].copy(),
            df.iloc[min(val_end + purge_bars, len(df)):].copy(),
        )

    test_start = len(df) - test_bars
    val_start = test_start - val_bars
    train_end = max(val_start - purge_bars, min_train_bars)
    val_begin = min(val_start + purge_bars, test_start)
    val_end = max(test_start - purge_bars, val_begin)
    test_begin = min(test_start + purge_bars, len(df))
    return df.iloc[:train_end].copy(), df.iloc[val_begin:val_end].copy(), df.iloc[test_begin:].copy()

def _make_sequences(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> tuple[np.ndarray, ...]:
    if len(df) <= config.model.WINDOW_SIZE:
        feature_count = len(feature_cols)
        empty_x = np.empty((0, config.model.WINDOW_SIZE, feature_count), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_x, empty_y.astype(np.int64), empty_y, empty_y, empty_y, empty_y

    X_scaled = scaler.transform(df[feature_cols].values)
    from numpy.lib.stride_tricks import sliding_window_view

    X_seq = sliding_window_view(X_scaled[:-1], (config.model.WINDOW_SIZE, X_scaled.shape[1])).squeeze(1)
    raw_dir = df["direction_label"].values[config.model.WINDOW_SIZE:]
    y_dir = raw_dir.astype(np.int64)

    y_tp = df["take_profit_pct"].values[config.model.WINDOW_SIZE:].astype(np.float32)
    y_sl = df["stop_loss_pct"].values[config.model.WINDOW_SIZE:].astype(np.float32)
    y_qty = (y_dir != 1).astype(np.float32)
    y_pnl = df.get("expected_return_pct", pd.Series(0.0, index=df.index)).values[config.model.WINDOW_SIZE:].astype(np.float32)
    
    # New targets normalized
    y_mag = df["magnitude_label"].values[config.model.WINDOW_SIZE:].astype(np.float32) / config.strategy.MAX_ATR_TARGET_PCT
    y_time = df["time_to_target"].values[config.model.WINDOW_SIZE:].astype(np.float32) / config.features.LOOKAHEAD_BARS
    
    return X_seq.astype(np.float32), y_dir, y_qty, y_tp, y_sl, y_pnl, y_mag, y_time

def _concat_parts(parts: list[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    non_empty = [p for p in parts if len(p[0])]
    if non_empty:
        return tuple(np.concatenate([p[i] for p in non_empty], axis=0) for i in range(8))
    feature_count = len(get_feature_columns())
    empty_x = np.empty((0, config.model.WINDOW_SIZE, feature_count), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.float32)
    return empty_x, empty_y.astype(np.int64), empty_y, empty_y, empty_y, empty_y, empty_y, empty_y

@torch.no_grad()
def _predict(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int = 2048) -> dict[str, np.ndarray]:
    model.eval()
    if len(X) == 0:
        return {
            "probs": np.empty((0, 3), dtype=np.float32), 
            "sizing": np.empty((0, 3), dtype=np.float32),
            "magnitude": np.empty((0, 1), dtype=np.float32),
            "time": np.empty((0, 1), dtype=np.float32)
        }
    probs, sizing, magnitude, time = [], [], [], []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        outputs = model(batch)
        probs.append(torch.softmax(outputs["direction"], dim=1).cpu().numpy())
        sizing.append(outputs["sizing"].cpu().numpy())
        magnitude.append(outputs["magnitude"].cpu().numpy())
        time.append(outputs["time"].cpu().numpy())
    return {
        "probs": np.concatenate(probs, axis=0), 
        "sizing": np.concatenate(sizing, axis=0),
        "magnitude": np.concatenate(magnitude, axis=0),
        "time": np.concatenate(time, axis=0)
    }

def _classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    long_precision, long_recall, long_f1, _ = precision_recall_fscore_support(
        y_true == 0, pred == 0, average="binary", zero_division=0
    )
    short_precision, short_recall, short_f1, _ = precision_recall_fscore_support(
        y_true == 2, pred == 2, average="binary", zero_division=0
    )
    return {
        "accuracy": float((pred == y_true).mean()) if len(y_true) else 0.0,
        "long_f1": float(long_f1),
        "short_f1": float(short_f1),
        "macro_f1": float((long_f1 + short_f1) / 2.0),
    }

def _threshold_search(y_true: np.ndarray, y_tp: np.ndarray, y_sl: np.ndarray, probs: np.ndarray) -> dict:
    # Simplified search for best probability threshold
    best_score = -1e9
    best_thresholds = {"long": 0.4, "short": 0.4}
    
    for thr in np.arange(0.3, 0.6, 0.05):
        selected_l = probs[:, 0] >= thr
        selected_s = probs[:, 2] >= thr
        
        # Simple score: (sum of tp on correct) - (sum of sl on incorrect)
        l_score = np.sum(y_tp[selected_l & (y_true == 0)]) - np.sum(y_sl[selected_l & (y_true != 0)])
        s_score = np.sum(y_tp[selected_s & (y_true == 2)]) - np.sum(y_sl[selected_s & (y_true != 2)])
        
        if (l_score + s_score) > best_score:
            best_score = l_score + s_score
            best_thresholds = {"long": float(thr), "short": float(thr)}
            
    return {"best": best_thresholds, "score": float(best_score)}

def train_short_model():
    symbols = config.data.SYMBOLS
    interval = config.data.INTERVAL
    total_days = config.training.TRAINING_DATA_DAYS + config.training.VALIDATION_DATA_DAYS + config.training.TEST_DATA_DAYS
    feature_cols = get_feature_columns()

    raw_splits: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for symbol in symbols:
        logger.info(f"Preparing data for {symbol}...")
        df_full = _prepare_symbol_frame(symbol, total_days, interval, feature_cols)
        if df_full.empty: continue
        
        train_df, val_df, test_df = _ordered_split(df_full, interval)
        raw_splits["train"].append(train_df)
        raw_splits["val"].append(val_df)
        raw_splits["test"].append(test_df)

    if not raw_splits["train"]:
        logger.error("No training data found.")
        return

    scaler = StandardScaler()
    scaler.fit(np.vstack([df[feature_cols].values for df in raw_splits["train"]]))

    X_train, y_train, y_qty_train, y_tp_train, y_sl_train, y_pnl_train, y_mag_train, y_time_train = _concat_parts([_make_sequences(df, feature_cols, scaler) for df in raw_splits["train"]])
    X_val, y_val, y_qty_val, y_tp_val, y_sl_val, y_pnl_val, y_mag_val, y_time_val = _concat_parts([_make_sequences(df, feature_cols, scaler) for df in raw_splits["val"]])
    
    device = config.DEVICE
    train_loader = DataLoader(TradingDataset(X_train, y_train, y_qty_train, y_tp_train, y_sl_train, y_pnl_train, y_mag_train, y_time_train, device=device), batch_size=config.training.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TradingDataset(X_val, y_val, y_qty_val, y_tp_val, y_sl_val, y_pnl_val, y_mag_val, y_time_val, device=device), batch_size=config.training.BATCH_SIZE, shuffle=False)

    from neural_engine.model import MultiHeadTradingModel, TradingLoss
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    
    class_counts = np.bincount(y_train.astype(np.int64), minlength=3)
    class_weights = torch.FloatTensor(len(y_train) / (3 * (class_counts + 1e-6))).to(device)
    criterion = TradingLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.LR, weight_decay=1e-4)

    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "short_model_eth.pth")
    best_val_loss = float("inf")

    logger.info(f"Starting training on {device}...")
    for epoch in range(config.training.EPOCHS):
        model.train()
        total_loss, total_sig, total_mag, total_time = 0.0, 0.0, 0.0, 0.0
        for batch_X, batch_targets in train_loader:
            optimizer.zero_grad()
            losses = criterion(model(batch_X), batch_targets)
            loss = losses["total"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sig += losses["signal_loss"].item()
            total_mag += losses["magnitude_loss"].item()
            total_time += losses["time_loss"].item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_targets in val_loader:
                val_loss += criterion(model(batch_X), batch_targets)["total"].item()
        
        avg_val_loss = val_loss / len(val_loader)
        n_batches = len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{config.training.EPOCHS} | "
            f"Loss: {total_loss/n_batches:.4f} | "
            f"Sig: {total_sig/n_batches:.3f} | "
            f"Mag: {total_mag/n_batches:.3f} | "
            f"Time: {total_time/n_batches:.3f} | "
            f"Val: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    # Save scaler and thresholds
    np.save(os.path.join("models", "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join("models", "scaler_scale.npy"), scaler.scale_)
    
    # Final threshold search
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    val_pred = _predict(model, X_val, device)
    best_thr = _threshold_search(y_val, y_tp_val, y_sl_val, val_pred["probs"])["best"]
    
    with open(os.path.join("models", "short_thresholds.json"), "w") as f:
        json.dump({
            "long_probability_threshold": best_thr["long"],
            "short_probability_threshold": best_thr["short"]
        }, f)

    logger.info("Training complete. Model and artifacts saved in models/")
    run_backtest()

if __name__ == "__main__":
    train_short_model()
