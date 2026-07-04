# -*- coding: utf-8 -*-
"""
Train the direction + magnitude predictor.

Features: 18 causal candle/indicator features computed fresh from OHLCV.
Targets (next LOOKAHEAD_BARS candles):
  y_up  — max upside move %   (always >= 0)
  y_dn  — max downside move % (always >= 0)
  y_dir — 1 if upside dominates, 0 if downside dominates

Run:
    uv run python trainer.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import cfg, LOOKAHEAD_BARS, MAX_RETURN_PCT
from evaluator import evaluate_model, print_evaluation, plot_predictions
from horizon_labeler import HorizonLabeler
from model import DirectionMagnitudeModel

CSV_PATH     = "data/labeled_BTCUSD_15m.csv"
MODEL_DIR    = "models"
MODEL_PATH   = f"{MODEL_DIR}/direction_nn.pt"
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
    "dist_ema_50_pct",
    "macd_hist_pct",
    "rsi_14",
    "bb_position",
    "volume_ratio_20",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 18 clean causal features from OHLCV only."""
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
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    feats["dist_ema_21_pct"]     = ((close - ema21) / ema21 * 100.0).fillna(0.0)
    feats["ema_9_21_spread_pct"] = ((ema9 - ema21) / ema21 * 100.0).fillna(0.0)
    feats["dist_ema_50_pct"]     = ((close - ema50) / ema50 * 100.0).fillna(0.0)

    # ── MACD histogram (12, 26, 9), normalized by price ──────────────────────
    ema12     = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26     = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    feats["macd_hist_pct"] = ((macd_line - macd_sig) / close * 100.0).fillna(0.0)

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
        dow  = df.index.dayofweek
        feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        feats["dow_sin"]  = np.sin(2.0 * np.pi * dow / 7.0)
        feats["dow_cos"]  = np.cos(2.0 * np.pi * dow / 7.0)
    else:
        feats["hour_sin"] = 0.0
        feats["hour_cos"] = 0.0
        feats["dow_sin"]  = 0.0
        feats["dow_cos"]  = 0.0

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
        targets:   torch.Tensor,   # (N, 3): [y_up, y_dn, y_dir]
        positions: np.ndarray,
        window:    int,
    ) -> None:
        self.features  = features
        self.targets   = targets
        self.positions = positions
        self.window    = window

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        t   = int(self.positions[i])
        seq = self.features[t - self.window + 1 : t + 1]   # (W, F)
        return seq, self.targets[t]


# ── Data preparation ──────────────────────────────────────────────────────────

def build_dataset():
    """Load CSV, compute labels + features, return everything needed for train/backtest."""
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

    # Targets
    y_up  = np.clip(df_valid["upside_max_return"].to_numpy(dtype=np.float32),   0.0, MAX_RETURN_PCT)
    y_dn  = np.clip(df_valid["downside_max_return"].to_numpy(dtype=np.float32), 0.0, MAX_RETURN_PCT)
    y_dir = (y_up >= y_dn).astype(np.float32)   # 1 = upside dominates

    train_end, val_end = split_bounds(len(df_valid))
    scaler = StandardScaler().fit(X[:train_end])
    Xs     = np.nan_to_num(scaler.transform(X).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Purged split: drop bars whose label window crosses a split boundary so that
    # training labels never peek into validation/test price data.
    W       = cfg.nn.WINDOW_SIZE
    all_pos = np.arange(W - 1, len(df_valid))
    train_pos = all_pos[all_pos < train_end - LOOKAHEAD_BARS]
    val_pos   = all_pos[(all_pos >= train_end) & (all_pos < val_end - LOOKAHEAD_BARS)]
    test_pos  = all_pos[all_pos >= val_end]

    console.print(
        f"  [dim]Sequences — Train: {len(train_pos):,} | Val: {len(val_pos):,} | "
        f"Test: {len(test_pos):,} | Features: {len(FEATURE_NAMES)} | "
        f"Embargo: {LOOKAHEAD_BARS} bars[/dim]"
    )
    return df_valid, scaler, Xs, y_up, y_dn, y_dir, (train_pos, val_pos, test_pos)


def make_windows(Xs: np.ndarray, positions: np.ndarray, window: int) -> np.ndarray:
    """Vectorized (N, W, F) window extraction for evaluation/backtest."""
    idx = positions[:, None] - np.arange(window - 1, -1, -1)[None, :]
    return Xs[idx]


def _loader(Xs_t, targets_t, pos, window, shuffle, *, pin_memory=False):
    ds = WindowDataset(Xs_t, targets_t, pos, window)
    return DataLoader(
        ds,
        batch_size=cfg.nn.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=pin_memory,
    )


# ── Loss ──────────────────────────────────────────────────────────────────────

def multitask_loss(dir_logit, up_pred, dn_pred, targets):
    """
    Direction BCE (edge-weighted) + magnitude Huber (move-weighted).

    Edge weight: bars where |upside − downside| is large are the clearest
    directional examples — weight them more so the classifier focuses on
    tradeable moves instead of 50/50 chop.
    """
    y_up, y_dn, y_dir = targets[:, 0], targets[:, 1], targets[:, 2]

    edge     = (y_up - y_dn).abs().clamp(max=3.0)
    dir_w    = (1.0 + edge).detach()
    loss_dir = (F.binary_cross_entropy_with_logits(dir_logit, y_dir, reduction="none") * dir_w).mean()

    mag_w_up = (1.0 + y_up).detach()
    mag_w_dn = (1.0 + y_dn).detach()
    loss_up  = (F.huber_loss(up_pred, y_up, delta=1.5, reduction="none") * mag_w_up).mean()
    loss_dn  = (F.huber_loss(dn_pred, y_dn, delta=1.5, reduction="none") * mag_w_dn).mean()

    return (
        cfg.nn.DIR_LOSS_WEIGHT * loss_dir
        + cfg.nn.MAG_LOSS_WEIGHT * (loss_up + loss_dn)
    )


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_split(train_pos, val_pos, test_pos, y_up, y_dn, y_dir):
    mag = np.maximum(y_up, y_dn)
    t = Table(
        title=f"[bold cyan]Dataset Split[/bold cyan]  (window={cfg.nn.WINDOW_SIZE}, lookahead={LOOKAHEAD_BARS})",
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Split",      style="bold")
    t.add_column("Windows",    justify="right")
    t.add_column("UP bars",    justify="right")
    t.add_column("DOWN bars",  justify="right")
    t.add_column("Mean move %", justify="right")
    t.add_column(">1.0% bars", justify="right")
    for name, pos in (("Train", train_pos), ("Validation", val_pos), ("Test", test_pos)):
        n_up = int(y_dir[pos].sum())
        t.add_row(
            name, f"{len(pos):,}",
            f"{n_up:,} ({n_up/len(pos)*100:.0f}%)",
            f"{len(pos)-n_up:,} ({(len(pos)-n_up)/len(pos)*100:.0f}%)",
            f"{mag[pos].mean():.3f}",
            f"{int((mag[pos] > 1.0).sum()):,}",
        )
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Direction + Magnitude Predictor — Training[/bold blue]")
    torch.manual_seed(cfg.nn.RANDOM_STATE)
    device = _device()
    console.print(f"\n[bold]Device:[/bold] {device}")

    df_valid, scaler, Xs, y_up, y_dn, y_dir, (train_pos, val_pos, test_pos) = build_dataset()

    _print_split(train_pos, val_pos, test_pos, y_up, y_dn, y_dir)

    Xs_t      = torch.from_numpy(Xs)
    targets_t = torch.from_numpy(np.stack([y_up, y_dn, y_dir], axis=1)).float()

    pin = device.type == "cuda"
    train_loader = _loader(Xs_t, targets_t, train_pos, cfg.nn.WINDOW_SIZE, True,  pin_memory=pin)
    val_loader   = _loader(Xs_t, targets_t, val_pos,   cfg.nn.WINDOW_SIZE, False, pin_memory=pin)

    model = DirectionMagnitudeModel(input_dim=len(FEATURE_NAMES)).to(device)
    if hasattr(torch, "compile") and device.type in ("cuda", "mps"):
        try:
            import logging as _log
            _log.getLogger("torch._inductor").setLevel(_log.ERROR)
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    _raw = getattr(model, "_orig_mod", model)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.nn.LR, weight_decay=cfg.nn.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=4, min_lr=5e-6
    )

    _mps_ok        = device.type == "mps" and tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 1)
    _use_autocast  = device.type == "cuda" or _mps_ok
    _autocast_dtype = torch.bfloat16 if _use_autocast else torch.float32

    best_val_loss, best_state, patience_cnt = float("inf"), None, 0
    best_val_acc = 0.0

    for epoch in range(1, cfg.nn.EPOCHS + 1):
        model.train()
        running_loss, n_samples = 0.0, 0

        for xb, tb in train_loader:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype, enabled=_use_autocast):
                dir_logit, up_pred, dn_pred = model(xb)
                loss = multitask_loss(dir_logit, up_pred, dn_pred, tb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running_loss += loss.item() * len(xb)
            n_samples    += len(xb)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        v_logits, v_ups, v_dns, v_targets = [], [], [], []
        with torch.no_grad():
            for xb, tb in val_loader:
                dl, up, dn = model(xb.to(device, non_blocking=True))
                v_logits.append(dl.cpu())
                v_ups.append(up.cpu())
                v_dns.append(dn.cpu())
                v_targets.append(tb)
        v_logit  = torch.cat(v_logits)
        v_up     = torch.cat(v_ups)
        v_dn     = torch.cat(v_dns)
        v_target = torch.cat(v_targets)

        val_loss = float(multitask_loss(v_logit, v_up, v_dn, v_target))
        val_acc  = float(((v_logit >= 0) == (v_target[:, 2] >= 0.5)).float().mean())
        val_mae  = float(
            (torch.abs(v_up - v_target[:, 0]) + torch.abs(v_dn - v_target[:, 1])).mean() / 2
        )
        sched.step(val_loss)
        lr = optim.param_groups[0]["lr"]

        console.print(
            f"  Epoch {epoch:3d}/{cfg.nn.EPOCHS} | "
            f"Loss: {running_loss/n_samples:.4f} | Val loss: {val_loss:.4f} | "
            f"Val dir acc: {val_acc*100:.2f}% | Val mag MAE: {val_mae:.4f}% | LR: {lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, patience_cnt = val_loss, val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in _raw.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.nn.EARLY_STOP_PATIENCE:
                console.print(f"  [dim]Early stop (best val loss={best_val_loss:.4f})[/dim]")
                break

    if best_state is not None:
        _raw.load_state_dict(best_state)
    model.to(device)

    # ── Evaluate on val + test ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    W = cfg.nn.WINDOW_SIZE
    for split_name, pos in (("val", val_pos), ("test", test_pos)):
        X_seq   = make_windows(Xs, pos, W)
        metrics = evaluate_model(model, X_seq, y_up[pos], y_dn[pos], device)
        saveable = {k: v for k, v in metrics.items() if not k.startswith("_")}
        with open(f"{MODEL_DIR}/eval_{split_name}.json", "w") as fh:
            json.dump(saveable, fh, indent=2)
        if split_name == "test":
            print_evaluation(metrics, title="TEST SET (unseen data)")
            plot_predictions(metrics)

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
        json.dump({"best_val_loss": best_val_loss, "best_val_dir_acc": best_val_acc}, fh, indent=2)
    console.print(f"  [dim]Metrics → {METRICS_PATH}[/dim]")
    console.rule("[bold green]Done — ab backtest chalao:  uv run python backtest.py[/bold green]")


if __name__ == "__main__":
    main()
