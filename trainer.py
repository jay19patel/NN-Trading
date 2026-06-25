# -*- coding: utf-8 -*-
"""
Train the max-return predictor.

Features: 14 basic candle/indicator features computed fresh from OHLCV.
Target:   max_return (% signed) for the next LOOKAHEAD_BARS candles.

Run:
    uv run python trainer.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import cfg, LOOKAHEAD_BARS, MAX_RETURN_PCT
from evaluator import evaluate_model, print_evaluation, print_data_stats, plot_predictions
from horizon_labeler import HorizonLabeler
from model import ReturnPredictorModel

CSV_PATH   = "data/labeled_BTCUSD_15m.csv"
MODEL_DIR  = "models"
MODEL_PATH = f"{MODEL_DIR}/direction_nn.pt"
METRICS_PATH = f"{MODEL_DIR}/nn_metrics.json"

console = Console()


# ── Feature engineering ───────────────────────────────────────────────────────

FEATURE_NAMES = [
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "natr",
    "candle_vs_atr",
    "log_return_1",
    "roc_6",
    "dist_ema_21_pct",
    "ema_9_21_spread_pct",
    "rsi_14",
    "bb_position",
    "volume_ratio_20",
    "hour_sin",
    "hour_cos",
]


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 14 clean candle features from OHLCV only."""
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df["Volume"]

    candle_range = (high - low).replace(0.0, np.nan)

    feats = pd.DataFrame(index=df.index)

    # ── Candle structure ─────────────────────────────────────────────────────
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    feats["body_pct"]         = ((close - open_).abs() / candle_range).fillna(0.0).clip(0.0, 1.0)
    feats["upper_shadow_pct"] = (upper_wick / candle_range).fillna(0.0).clip(0.0, 1.0)
    feats["lower_shadow_pct"] = (lower_wick / candle_range).fillna(0.0).clip(0.0, 1.0)

    # ── Volatility ───────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    feats["natr"]          = (atr14 / close * 100.0).fillna(0.0)
    feats["candle_vs_atr"] = (candle_range / atr14.replace(0.0, np.nan)).fillna(1.0)

    # ── Returns / momentum ───────────────────────────────────────────────────
    feats["log_return_1"] = np.log(close / close.shift(1)).fillna(0.0)
    feats["roc_6"]        = (close.pct_change(6) * 100.0).fillna(0.0)

    # ── Trend ────────────────────────────────────────────────────────────────
    ema9  = close.ewm(span=9,  adjust=False, min_periods=9).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    feats["dist_ema_21_pct"]     = ((close - ema21) / ema21 * 100.0).fillna(0.0)
    feats["ema_9_21_spread_pct"] = ((ema9 - ema21) / ema21 * 100.0).fillna(0.0)

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta    = close.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    feats["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    # ── Bollinger Band position ───────────────────────────────────────────────
    bb_mid   = close.rolling(20, min_periods=20).mean()
    bb_std   = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    feats["bb_position"] = (
        (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    ).fillna(0.5).clip(0.0, 1.0)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_mean20 = volume.rolling(20, min_periods=1).mean()
    feats["volume_ratio_20"] = (volume / (vol_mean20 + 1e-9)).fillna(1.0)

    # ── Time encoding ────────────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour + df.index.minute / 60.0
        feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    else:
        feats["hour_sin"] = 0.0
        feats["hour_cos"] = 0.0

    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_bounds(n: int) -> tuple[int, int]:
    train_end = int(n * cfg.ml_training.TRAIN_FRAC)
    val_end   = int(n * (cfg.ml_training.TRAIN_FRAC + cfg.ml_training.VAL_FRAC))
    return train_end, val_end


# ── Dataset ───────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(
        self,
        features:  torch.Tensor,
        y_return:  torch.Tensor,
        positions: np.ndarray,
        window:    int,
    ) -> None:
        self.features  = features
        self.y_return  = y_return
        self.positions = positions
        self.window    = window

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        t   = int(self.positions[i])
        seq = self.features[t - self.window + 1 : t + 1]   # (W, F)
        return seq, self.y_return[t]


# ── Data preparation ──────────────────────────────────────────────────────────

def _build():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH!r}. Run `uv run python app.py` first.")

    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

    # Compute labels (UpsideMaxReturn / DownsideMaxReturn / MaxReturn)
    labeler    = HorizonLabeler(lookahead_bars=LOOKAHEAD_BARS)
    df_labeled = labeler.generate(df_raw)
    df_valid   = df_labeled[df_labeled["horizon_label_valid"]].copy()

    # Compute features from OHLCV only
    feats = _compute_features(df_valid)
    X     = feats[FEATURE_NAMES].to_numpy(dtype=np.float32)

    # Target: max_return in %, clipped to ±MAX_RETURN_PCT
    y = np.clip(
        df_valid["max_return"].to_numpy(dtype=np.float32),
        -MAX_RETURN_PCT, MAX_RETURN_PCT,
    )

    train_end, val_end = split_bounds(len(df_valid))
    scaler = StandardScaler().fit(X[:train_end])
    Xs     = np.nan_to_num(scaler.transform(X).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    W         = cfg.nn.WINDOW_SIZE
    all_pos   = np.arange(W - 1, len(df_valid))
    train_pos = all_pos[all_pos < train_end]
    val_pos   = all_pos[(all_pos >= train_end) & (all_pos < val_end)]
    test_pos  = all_pos[all_pos >= val_end]

    console.print(
        f"  [dim]Sequences — Train: {len(train_pos):,} | Val: {len(val_pos):,} | "
        f"Test: {len(test_pos):,} | Features: {len(FEATURE_NAMES)}[/dim]"
    )
    return df_valid, scaler, Xs, y, (train_pos, val_pos, test_pos)


def _loader(Xs_t, y_t, pos, window, shuffle, *, pin_memory=False):
    ds = WindowDataset(Xs_t, y_t, pos, window)
    return DataLoader(
        ds,
        batch_size=cfg.nn.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=pin_memory,
    )


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_split(train_pos, val_pos, test_pos, y):
    t = Table(
        title=f"[bold cyan]Dataset Split[/bold cyan]  (window={cfg.nn.WINDOW_SIZE}, lookahead={LOOKAHEAD_BARS})",
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Split",     style="bold")
    t.add_column("Windows",   justify="right")
    t.add_column("Mean target %", justify="right")
    t.add_column("Upside bars",   justify="right")
    t.add_column("Downside bars", justify="right")
    for name, pos in (("Train", train_pos), ("Validation", val_pos), ("Test", test_pos)):
        yp = y[pos]
        t.add_row(
            name, str(len(pos)),
            f"{yp.mean():+.3f}",
            str(int((yp > 0).sum())),
            str(int((yp < 0).sum())),
        )
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Max Return Predictor — Training[/bold blue]")
    torch.manual_seed(cfg.nn.RANDOM_STATE)
    device = _device()
    console.print(f"\n[bold]Device:[/bold] {device}")

    df_valid, scaler, Xs, y, (train_pos, val_pos, test_pos) = _build()

    _print_split(train_pos, val_pos, test_pos, y)

    Xs_t = torch.from_numpy(Xs)
    y_t  = torch.from_numpy(y).float()

    pin = device.type == "cuda"
    train_loader = _loader(Xs_t, y_t, train_pos, cfg.nn.WINDOW_SIZE, True,  pin_memory=pin)
    val_loader   = _loader(Xs_t, y_t, val_pos,   cfg.nn.WINDOW_SIZE, False, pin_memory=pin)

    model = ReturnPredictorModel(input_dim=len(FEATURE_NAMES)).to(device)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    _raw = getattr(model, "_orig_mod", model)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.nn.LR, weight_decay=cfg.nn.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=10, min_lr=5e-6
    )

    _mps_ok       = device.type == "mps" and tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 1)
    _use_autocast  = device.type == "cuda" or _mps_ok
    _autocast_dtype = torch.bfloat16 if _use_autocast else torch.float32

    best_val_mae, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(1, cfg.nn.EPOCHS + 1):
        model.train()
        running_loss, n_samples = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype, enabled=_use_autocast):
                pred = model(xb)
                loss = F.huber_loss(pred, yb, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running_loss += loss.item() * len(xb)
            n_samples    += len(xb)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device, non_blocking=True)).cpu().numpy()
                val_preds.append(pred)
                val_ys.append(yb.numpy())
        val_pred_np = np.concatenate(val_preds)
        val_y_np    = np.concatenate(val_ys)
        val_mae     = float(np.abs(val_pred_np - val_y_np).mean())
        sched.step(val_mae)
        lr = optim.param_groups[0]["lr"]

        console.print(
            f"  Epoch {epoch:3d}/{cfg.nn.EPOCHS} | "
            f"Loss: {running_loss/n_samples:.4f} | "
            f"Val MAE: {val_mae:.4f}% | LR: {lr:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae, patience_cnt = val_mae, 0
            best_state = {k: v.cpu().clone() for k, v in _raw.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.nn.EARLY_STOP_PATIENCE:
                console.print(f"  [dim]Early stop (best val MAE={best_val_mae:.4f}%)[/dim]")
                break

    if best_state is not None:
        _raw.load_state_dict(best_state)
    model.to(device)

    # ── Evaluate on val + test ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    for split_name, pos in (("val", val_pos), ("test", test_pos)):
        X_seq = np.stack([Xs[t - cfg.nn.WINDOW_SIZE + 1 : t + 1] for t in pos])
        metrics = evaluate_model(model, X_seq, y[pos], device)
        saveable = {k: v for k, v in metrics.items() if not k.startswith("_")}
        with open(f"{MODEL_DIR}/eval_{split_name}.json", "w") as fh:
            json.dump(saveable, fh, indent=2)
        if split_name == "test":
            console.print(f"\n  [bold]Test Evaluation[/bold]")
            print_data_stats(metrics["_y_true"], metrics["_y_pred"])
            print_evaluation(metrics)
            plot_predictions(metrics["_y_true"], metrics["_y_pred"])

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict":    _raw.state_dict(),
        "features":      FEATURE_NAMES,
        "window":        cfg.nn.WINDOW_SIZE,
        "input_dim":     len(FEATURE_NAMES),
        "scaler_mean":   scaler.mean_.astype(np.float32),
        "scaler_scale":  scaler.scale_.astype(np.float32),
        "lookahead_bars": LOOKAHEAD_BARS,
        "max_return_pct": MAX_RETURN_PCT,
        "hidden_dim":    cfg.nn.HIDDEN_DIM,
        "num_layers":    cfg.nn.NUM_LAYERS,
        "num_heads":     cfg.nn.NUM_HEADS,
        "dropout":       cfg.nn.DROPOUT,
    }, MODEL_PATH)
    console.print(f"\n  [dim]Model → {MODEL_PATH}[/dim]")

    with open(METRICS_PATH, "w") as fh:
        json.dump({"best_val_mae": best_val_mae}, fh, indent=2)
    console.print(f"  [dim]Metrics → {METRICS_PATH}[/dim]")
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
