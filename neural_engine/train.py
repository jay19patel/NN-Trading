# -*- coding: utf-8 -*-
"""
Training Pipeline
==================
Prepares data, trains the MultiHeadTradingModel, and saves model artifacts.

Pipeline order:
  1. Fetch OHLCV data (or load from parquet cache)
  2. Oracle labeling (future-peek ground truth)
  3. Technical indicator generation (40 features)
  4. Chronological train/val/test split with purge gap
  5. StandardScaler fitted on train only
  6. Sliding-window sequence construction
  7. Weighted sampler for class balance
  8. Training loop with:
     - FocalLoss (direction) + MSE (sizing/magnitude/time)
     - AdamW + ReduceLROnPlateau
     - Early stopping
     - Validation metrics every epoch
  9. Fine-grained margin-based threshold search on validation set
 10. Save model weights, scaler, and thresholds
 11. Auto-run backtest
"""
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import cfg, bars_per_day
from engine.data_handler import fetch_data
from neural_engine.labeler import OracleLabeler
from neural_engine.feature_utils import add_technical_indicators, get_feature_columns
from neural_engine.model import MultiHeadTradingModel, TradingLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Bars to drop at the start of each dataframe before training.
# Covers daily multi-timeframe indicator warmup before training rows are used.
INDICATOR_WARMUP_BARS = 35 * 24


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TradingDataset(Dataset):
    """PyTorch dataset wrapping pre-computed sliding-window sequences."""

    def __init__(
        self,
        X_seq: np.ndarray,
        y_dir: np.ndarray,
        y_qty: np.ndarray,
        y_tp: np.ndarray,
        y_sl: np.ndarray,
        y_pnl: np.ndarray,
        y_mag: np.ndarray,
        y_time: np.ndarray,
        device: str = "cpu",
    ):
        self.X = torch.from_numpy(X_seq).float()
        self.y_dir = torch.from_numpy(y_dir).long()
        self.y_qty = torch.from_numpy(y_qty).float()
        self.y_tp = torch.from_numpy(y_tp).float()
        self.y_sl = torch.from_numpy(y_sl).float()
        self.y_pnl = torch.from_numpy(y_pnl).float()
        self.y_mag = torch.from_numpy(y_mag).float()
        self.y_time = torch.from_numpy(y_time).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], {
            "direction": self.y_dir[idx],
            "qty_ratio": self.y_qty[idx],
            "take_profit_pct": self.y_tp[idx],
            "stop_loss_pct": self.y_sl[idx],
            "actual_pnl_pct": self.y_pnl[idx],
            "magnitude": self.y_mag[idx],
            "time": self.y_time[idx],
        }


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _cache_path(symbol: str, interval: str, lookahead: int, version: int) -> str:
    """Versioned parquet cache path — changing version forces a rebuild."""
    return f"data/processed_train_{symbol}_{interval}_L{lookahead}_v{version}.parquet"


def _prepare_symbol_frame(
    symbol: str,
    total_days: int,
    interval: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Load (or build) the fully-processed dataframe for a symbol.

    Processing order:
      1. Fetch raw OHLCV
      2. Oracle labeling
      3. Technical indicator generation
      4. Drop indicator warmup bars
      5. Cache to parquet
    """
    lookahead = cfg.training.LOOKAHEAD_BARS
    version = cfg.training.FEATURE_CACHE_VERSION
    cache = _cache_path(symbol, interval, lookahead, version)

    if os.path.exists(cache):
        logger.info(f"Loading cached features: {cache}")
        df = pd.read_parquet(cache)
    else:
        logger.info(f"Building features for {symbol} (cache miss)...")
        df = fetch_data(symbol, total_days, interval)
        if df.empty:
            return df

        required_ohlcv = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_ohlcv - set(df.columns)
        if missing:
            raise ValueError(f"{symbol}: missing OHLCV columns {missing}")

        df = add_technical_indicators(df)
        labeler = OracleLabeler()
        df = labeler.generate_labels(df.copy())

        # Drop the warmup window to avoid NaN-polluted rows from slow indicators
        df = df.iloc[INDICATOR_WARMUP_BARS:]

        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Feature generation missing columns: {missing_features}")

        os.makedirs("data", exist_ok=True)
        df.to_parquet(cache)

    return df.sort_index()


def _detect_and_log_gaps(df: pd.DataFrame, interval: str, symbol: str) -> None:
    """Log large time gaps that can contaminate rolling windows."""
    if len(df) < 2:
        return
    expected_delta = pd.Timedelta(interval)
    time_diffs = df.index.to_series().diff().dropna()
    large_gaps = time_diffs[time_diffs > expected_delta * 5]
    if len(large_gaps) > 0:
        logger.warning(
            f"{symbol}: found {len(large_gaps)} data gap(s). "
            f"First gaps: {large_gaps.head(5).to_dict()}"
        )


def _ordered_split(
    df: pd.DataFrame,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/val/test split with purge gaps to prevent data leakage.

    Layout (oldest → newest):
      [  TRAIN  ] [PURGE] [ VAL ] [PURGE] [ TEST ]
    """
    bars_day = bars_per_day(interval)
    val_bars = int(cfg.training.VALIDATION_DATA_DAYS * bars_day)
    test_bars = int(cfg.training.TEST_DATA_DAYS * bars_day)
    purge = int(cfg.training.PURGE_BARS)
    min_train = cfg.model.WINDOW_SIZE + 10

    if len(df) <= min_train + val_bars + test_bars:
        # Fallback for small datasets
        train_end = max(int(len(df) * 0.70), min_train)
        val_end = max(train_end + 1, int(len(df) * 0.85))
        return (
            df.iloc[: max(train_end - purge, min_train)].copy(),
            df.iloc[min(train_end + purge, val_end) : val_end].copy(),
            df.iloc[min(val_end + purge, len(df)) :].copy(),
        )

    test_start = len(df) - test_bars
    val_start = test_start - val_bars
    train_end = max(val_start - purge, min_train)
    val_begin = min(val_start + purge, test_start)
    val_end = max(test_start - purge, val_begin)
    test_begin = min(test_start + purge, len(df))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[val_begin:val_end].copy(),
        df.iloc[test_begin:].copy(),
    )


def _make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> tuple[np.ndarray, ...]:
    """
    Build (N, WINDOW_SIZE, num_features) sliding-window sequences.

    Returns 8-tuple: X_seq, y_dir, y_qty, y_tp, y_sl, y_pnl, y_mag, y_time
    """
    window = cfg.model.WINDOW_SIZE
    max_tp = cfg.testing.MAX_ATR_TARGET_PCT
    lookahead = cfg.training.LOOKAHEAD_BARS

    if len(df) <= window:
        feat_count = len(feature_cols)
        empty_x = np.empty((0, window, feat_count), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_x, empty_y.astype(np.int64), empty_y, empty_y, empty_y, empty_y, empty_y, empty_y

    X_scaled = scaler.transform(df[feature_cols].values)

    from numpy.lib.stride_tricks import sliding_window_view
    X_seq = sliding_window_view(X_scaled[:-1], (window, X_scaled.shape[1])).squeeze(1)

    y_dir = df["direction_label"].values[window:].astype(np.int64)
    y_tp = df["take_profit_pct"].values[window:].astype(np.float32)
    y_sl = df["stop_loss_pct"].values[window:].astype(np.float32)
    y_qty = (y_dir != 1).astype(np.float32)
    y_pnl = df.get("expected_return_pct", pd.Series(0.0, index=df.index)).values[window:].astype(np.float32)
    y_mag = df["magnitude_label"].values[window:].astype(np.float32) / max_tp
    y_time = df["time_to_target"].values[window:].astype(np.float32) / lookahead

    return X_seq.astype(np.float32), y_dir, y_qty, y_tp, y_sl, y_pnl, y_mag, y_time


def _concat_parts(parts: list[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    """Concatenate sequence tuples from multiple symbols."""
    non_empty = [p for p in parts if len(p[0])]
    if non_empty:
        return tuple(np.concatenate([p[i] for p in non_empty], axis=0) for i in range(8))
    feat_count = len(get_feature_columns())
    window = cfg.model.WINDOW_SIZE
    empty_x = np.empty((0, window, feat_count), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.float32)
    return empty_x, empty_y.astype(np.int64), empty_y, empty_y, empty_y, empty_y, empty_y, empty_y


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Run model inference in batches and return all output heads."""
    model.eval()
    if len(X) == 0:
        return {
            "probs": np.empty((0, 3), dtype=np.float32),
            "sizing": np.empty((0, 3), dtype=np.float32),
            "magnitude": np.empty((0, 1), dtype=np.float32),
            "time": np.empty((0, 1), dtype=np.float32),
        }
    probs, sizing, magnitude, time_out = [], [], [], []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
        out = model(batch)
        probs.append(torch.softmax(out["direction"], dim=1).cpu().numpy())
        sizing.append(out["sizing"].cpu().numpy())
        magnitude.append(out["magnitude"].cpu().numpy())
        time_out.append(out["time"].cpu().numpy())
    return {
        "probs": np.concatenate(probs, axis=0),
        "sizing": np.concatenate(sizing, axis=0),
        "magnitude": np.concatenate(magnitude, axis=0),
        "time": np.concatenate(time_out, axis=0),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute accuracy, per-class F1, and macro F1."""
    pred = np.argmax(probs, axis=1)
    _, _, long_f1, _ = precision_recall_fscore_support(y_true == 0, pred == 0, average="binary", zero_division=0)
    _, _, short_f1, _ = precision_recall_fscore_support(y_true == 2, pred == 2, average="binary", zero_division=0)
    metrics = {
        "accuracy": float((pred == y_true).mean()) if len(y_true) else 0.0,
        "long_f1": float(long_f1),
        "short_f1": float(short_f1),
        "macro_f1": float((long_f1 + short_f1) / 2.0),
    }
    try:
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        metrics["roc_auc_ovr"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Threshold / margin search
# ---------------------------------------------------------------------------

def _threshold_search(
    y_true: np.ndarray,
    y_tp: np.ndarray,
    y_sl: np.ndarray,
    probs: np.ndarray,
) -> dict:
    """
    Find the optimal signal margin threshold on the validation set.

    NEW APPROACH — margin-based instead of absolute probability:
      A LONG signal fires when:   prob_long - prob_neutral >= margin_thr
      A SHORT signal fires when:  prob_short - prob_neutral >= margin_thr

    This is more robust because the model calibrates around 33% per class
    (three-class softmax) rather than 50%.  A margin of 8% means the leading
    class has meaningfully more confidence than the neutral baseline.

    Score = sum(TP on correct trades) - sum(SL on incorrect trades)
    """
    best_score = -1e9
    best_margin_long = 0.08
    best_margin_short = 0.08

    long_margins = probs[:, 0] - probs[:, 1]   # prob_long - prob_neutral
    short_margins = probs[:, 2] - probs[:, 1]  # prob_short - prob_neutral

    for l_margin in np.arange(cfg.testing.SIGNAL_MARGIN_THRESHOLD, 0.35, 0.02):
        for s_margin in np.arange(cfg.testing.SIGNAL_MARGIN_THRESHOLD, 0.35, 0.02):
            selected_l = long_margins >= l_margin
            selected_s = short_margins >= s_margin

            l_score = (
                np.sum(y_tp[selected_l & (y_true == 0)])
                - np.sum(y_sl[selected_l & (y_true != 0)])
            )
            s_score = (
                np.sum(y_tp[selected_s & (y_true == 2)])
                - np.sum(y_sl[selected_s & (y_true != 2)])
            )

            n_trades = selected_l.sum() + selected_s.sum()
            if n_trades < cfg.training.THRESHOLD_MIN_TRADES:
                continue

            total_score = l_score + s_score
            if total_score > best_score and total_score > 0:
                best_score = total_score
                best_margin_long = float(l_margin)
                best_margin_short = float(s_margin)

    if best_score <= 0:
        logger.warning("Failed to find profitable threshold. Defaulting to 0.10 margin.")
        best_margin_long = 0.10
        best_margin_short = 0.10

    logger.info(
        f"Threshold search: long_margin={best_margin_long:.2f}, "
        f"short_margin={best_margin_short:.2f}, score={best_score:.2f}"
    )
    return {
        "best": {"long_margin": best_margin_long, "short_margin": best_margin_short},
        "score": float(best_score),
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_trading_model() -> None:
    """
    Full training pipeline.  Reads from `cfg`, saves artifacts to models/.
    After training automatically runs the backtest.
    """
    from neural_engine.backtest_engine import run_backtest

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    symbols = cfg.model.SYMBOLS
    interval = cfg.model.INTERVAL
    total_days = (
        cfg.training.TRAINING_DATA_DAYS
        + cfg.training.VALIDATION_DATA_DAYS
        + cfg.training.TEST_DATA_DAYS
    )
    feature_cols = get_feature_columns()

    # ── Step 1: Load / build per-symbol dataframes ────────────────────────
    raw_splits: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for symbol in symbols:
        logger.info(f"Preparing data for {symbol}...")
        df_full = _prepare_symbol_frame(symbol, total_days, interval, feature_cols)
        if df_full.empty:
            logger.warning(f"No data for {symbol}, skipping.")
            continue
        _detect_and_log_gaps(df_full, interval, symbol)
        train_df, val_df, test_df = _ordered_split(df_full, interval)
        raw_splits["train"].append(train_df)
        raw_splits["val"].append(val_df)
        raw_splits["test"].append(test_df)

    if not raw_splits["train"]:
        logger.error("No training data found. Aborting.")
        return

    # ── Step 2: Fit scaler on training data only ──────────────────────────
    scaler = StandardScaler()
    scaler.fit(np.vstack([df[feature_cols].values for df in raw_splits["train"]]))

    # ── Step 3: Build sequences ───────────────────────────────────────────
    X_train, y_train, y_qty_tr, y_tp_tr, y_sl_tr, y_pnl_tr, y_mag_tr, y_time_tr = _concat_parts(
        [_make_sequences(df, feature_cols, scaler) for df in raw_splits["train"]]
    )
    X_val, y_val, y_qty_v, y_tp_v, y_sl_v, y_pnl_v, y_mag_v, y_time_v = _concat_parts(
        [_make_sequences(df, feature_cols, scaler) for df in raw_splits["val"]]
    )

    # ── Step 4: Log class distribution ───────────────────────────────────
    class_counts = np.bincount(y_train, minlength=3)
    total = len(y_train)
    logger.info(
        f"Class distribution — "
        f"LONG: {class_counts[0]} ({class_counts[0]/total*100:.1f}%) | "
        f"NEUTRAL: {class_counts[1]} ({class_counts[1]/total*100:.1f}%) | "
        f"SHORT: {class_counts[2]} ({class_counts[2]/total*100:.1f}%)"
    )

    # ── Step 5: Build data loaders ────────────────────────────────────────
    device = cfg.DEVICE
    train_dataset = TradingDataset(X_train, y_train, y_qty_tr, y_tp_tr, y_sl_tr, y_pnl_tr, y_mag_tr, y_time_tr, device)
    val_dataset = TradingDataset(X_val, y_val, y_qty_v, y_tp_v, y_sl_v, y_pnl_v, y_mag_v, y_time_v, device)

    # Weighted sampler: inverse-frequency weighting forces the model to see
    # equal numbers of LONG, NEUTRAL, and SHORT samples per epoch
    sample_weights = 1.0 / (class_counts[y_train.astype(np.int64)].astype(np.float64) + 1e-6)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(y_train),
        replacement=True,
    )
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=2,
    )

    # ── Step 6: Model, loss, optimizer, scheduler ─────────────────────────
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)

    # Note: We rely on WeightedRandomSampler to balance batches. We do NOT
    # pass class weights to the loss function to avoid double-weighting.
    criterion = TradingLoss(label_smoothing=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    # ── Step 7: Training loop ─────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "trading_model.pth")
    best_val_loss = float("inf")
    patience_counter = 0
    val_pred = None

    def move_batch(batch_X: torch.Tensor, batch_targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_X = batch_X.to(device, non_blocking=pin_memory)
        batch_targets = {key: value.to(device, non_blocking=pin_memory) for key, value in batch_targets.items()}
        return batch_X, batch_targets

    logger.info(f"Starting training on {device}...")
    logger.info(
        f"Train: {len(X_train)} seqs | Val: {len(X_val)} seqs | "
        f"Features: {X_train.shape[2]} | Window: {X_train.shape[1]}"
    )

    for epoch in range(cfg.training.EPOCHS):
        # ── Train ──
        model.train()
        total_loss = total_sig = total_mag = total_time = 0.0
        for batch_X, batch_targets in train_loader:
            batch_X, batch_targets = move_batch(batch_X, batch_targets)
            optimizer.zero_grad()
            losses = criterion(model(batch_X), batch_targets)
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_sig += losses["signal_loss"].item()
            total_mag += losses["magnitude_loss"].item()
            total_time += losses["time_loss"].item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_targets in val_loader:
                batch_X, batch_targets = move_batch(batch_X, batch_targets)
                val_loss += criterion(model(batch_X), batch_targets)["total"].item()

        avg_val_loss = val_loss / len(val_loader)
        n_batches = len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log train loss every epoch
        logger.info(
            f"Epoch {epoch+1}/{cfg.training.EPOCHS} | "
            f"Loss: {total_loss/n_batches:.4f} | "
            f"Sig: {total_sig/n_batches:.3f} | "
            f"Mag: {total_mag/n_batches:.3f} | "
            f"Time: {total_time/n_batches:.3f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        val_pred = _predict(model, X_val, device)
        metrics = _classification_metrics(y_val, val_pred["probs"])

        # ── Detailed val metrics every 5 epochs ──
        if (epoch + 1) % 5 == 0:
            pred_labels = np.argmax(val_pred["probs"], axis=1)
            cm = confusion_matrix(y_val, pred_labels, labels=[0, 1, 2])
            logger.info(
                f"  Val Metrics | Acc: {metrics['accuracy']:.3f} | "
                f"Long F1: {metrics['long_f1']:.3f} | Short F1: {metrics['short_f1']:.3f} | "
                f"Macro F1: {metrics['macro_f1']:.3f} | ROC-AUC: {metrics['roc_auc_ovr']:.3f}"
            )
            logger.info("  Confusion Matrix (rows=true, cols=pred) [L/N/S]:")
            logger.info(f"    LONG:    {cm[0]}")
            logger.info(f"    NEUTRAL: {cm[1]}")
            logger.info(f"    SHORT:   {cm[2]}")

        # ── LR scheduler ──
        scheduler.step(avg_val_loss)

        # ── Early stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ✅ New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  ⚠️ No improvement ({patience_counter}/{cfg.training.EARLY_STOP_PATIENCE})")
            if patience_counter >= cfg.training.EARLY_STOP_PATIENCE:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break

    # ── Step 8: Save scaler ────────────────────────────────────────────────
    np.save(os.path.join("models", "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join("models", "scaler_scale.npy"), scaler.scale_)

    # Threshold search results are now purely for logging; we use config.py for execution
    if val_pred is None:
        val_pred = _predict(model, X_val, device)
    best_thr = _threshold_search(y_val, y_tp_v, y_sl_v, val_pred["probs"])["best"]
    logger.info(
        f"Recommended Thresholds (Manual override): "
        f"Long: {best_thr['long_margin']:.2f}, Short: {best_thr['short_margin']:.2f}"
    )

    logger.info("Training complete. Model and artifacts saved in models/")

    # ── Step 10: Auto-run backtest ─────────────────────────────────────────
    run_backtest(symbol=cfg.model.SYMBOLS[0])



if __name__ == "__main__":
    train_trading_model()
