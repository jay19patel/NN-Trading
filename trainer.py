# -*- coding: utf-8 -*-
"""
Train the Quantile Transformer (direction + move quantiles).

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
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import cfg
from evaluator import evaluate_model, print_evaluation
from horizon_labeler import HorizonLabeler, MAX_MFE_PCT
from model import QuantileTradingModel
from set_label import (
    LABEL_NAMES, LONG, NEUTRAL, SHORT, get_feature_columns,
    _add_moving_average_status_columns,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "data/labeled_BTCUSD_15m.csv"
MODEL_DIR  = "models"
MODEL_PATH = f"{MODEL_DIR}/direction_nn.pt"
METRICS_PATH = f"{MODEL_DIR}/nn_metrics.json"

CLASSES      = (LONG, NEUTRAL, SHORT)
CLASS_LABELS = tuple(LABEL_NAMES[c] for c in CLASSES)

console = Console()


# ── Shared data helpers ───────────────────────────────────────────────────────

def split_bounds(n: int) -> tuple[int, int]:
    train_end = int(n * cfg.ml_training.TRAIN_FRAC)
    val_end   = int(n * (cfg.ml_training.TRAIN_FRAC + cfg.ml_training.VAL_FRAC))
    return train_end, val_end


def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(
        self,
        features:  torch.Tensor,
        y_dir:     torch.Tensor,
        y_mfe:     torch.Tensor,
        y_valid:   torch.Tensor,
        positions: np.ndarray,
        window:    int,
    ) -> None:
        self.features  = features
        self.y_dir     = y_dir
        self.y_mfe     = y_mfe
        self.y_valid   = y_valid
        self.positions = positions
        self.window    = window

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        t   = int(self.positions[i])
        seq = self.features[t - self.window + 1 : t + 1]
        return seq, {
            "direction":   self.y_dir[t],
            "mfe_pct":     self.y_mfe[t],
            "label_valid": self.y_valid[t],
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

class TwoTaskLoss(nn.Module):
    """
    Focal-CE(direction, class-weighted) + Pinball(move_q). Masked by label_valid.

    Focal loss (gamma > 0) down-weights easy examples so the model focuses on
    hard LONG / SHORT bars instead of collapsing to NEUTRAL predictions.
    Class weights correct for the residual class imbalance in the training split.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.10,
        w_dir: float = 1.0,
        w_move: float = 0.3,
        focal_gamma: float = 1.5,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.w_dir       = w_dir
        self.w_move      = w_move
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, pred: dict, tgt: dict) -> dict:
        device = pred["direction"].device
        y_dir  = tgt["direction"].long().to(device)
        valid  = tgt["label_valid"].float().to(device)
        n_val  = valid.sum().clamp(min=1.0)

        weights = self.class_weights.to(device) if self.class_weights is not None else None
        ce = F.cross_entropy(
            pred["direction"], y_dir,
            weight=weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Focal modulation: p_t must come from raw (unweighted) CE so the
        # focal weight reflects the actual predicted probability, not the
        # class-weight-inflated loss. Using weighted CE here causes NEUTRAL
        # (weight≈0.31) to appear "easy" (p_t→1, focal→0) and get almost
        # zero gradient, collapsing the model to LONG/SHORT-only predictions.
        raw_ce = F.cross_entropy(
            pred["direction"], y_dir,
            reduction="none",
        )
        p_t   = torch.exp(-raw_ce.detach())
        focal = (1.0 - p_t) ** self.focal_gamma * ce
        dir_loss  = (focal * valid).sum() / n_val
        move_loss = self._pinball(pred["move_q"], tgt["mfe_pct"].float().to(device), valid, n_val)

        total = self.w_dir * dir_loss + self.w_move * move_loss
        return {"total": total, "dir_loss": dir_loss, "move_loss": move_loss}

    @staticmethod
    def _pinball(pred_q, target, valid, n_val):
        qs  = torch.tensor([0.10, 0.50, 0.90], device=pred_q.device)
        e   = target.unsqueeze(1) - pred_q
        l   = torch.max(qs * e, (qs - 1.0) * e)
        return (l.mean(dim=1) * valid).sum() / n_val


# ── Data preparation ──────────────────────────────────────────────────────────

def _load_and_label() -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH!r}. Run `uv run python app.py` first.")
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    # Recompute MA columns so dist_*_pct features are always present,
    # even if the CSV was generated before this feature was added.
    df_raw = _add_moving_average_status_columns(df_raw)

    labeler = HorizonLabeler(
        lookahead_bars    = cfg.training.LOOKAHEAD_BARS,
        tp_atr_multiplier = cfg.training.FIXED_TP_ATR_MULTIPLIER,
        sl_atr_multiplier = cfg.training.FIXED_SL_ATR_MULTIPLIER,
    )
    df_labeled = labeler.generate(df_raw)
    df_valid   = df_labeled[df_labeled["horizon_label_valid"]].copy()

    exclude = {
        "mfe_pct", "mfe_up_pct", "mae_down_pct", "bars_to_peak",
        "horizon_direction_label", "horizon_label_valid",
    }
    features = [f for f in get_feature_columns(df_valid) if f not in exclude]
    return df_valid, features


def _build():
    df, features = _load_and_label()
    X = feature_matrix(df, features).fillna(0.0).to_numpy(dtype=np.float32)

    y_dir   = df["horizon_direction_label"].astype(int).to_numpy()
    y_mfe   = np.clip(df["mfe_pct"].to_numpy(dtype=np.float32) / MAX_MFE_PCT, 0.0, 1.0)
    y_valid = np.ones(len(df), dtype=np.float32)

    train_end, val_end = split_bounds(len(df))
    scaler = StandardScaler().fit(X[:train_end])
    Xs     = np.nan_to_num(scaler.transform(X).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    W         = cfg.nn.WINDOW_SIZE
    all_pos   = np.arange(W - 1, len(df))
    train_pos = all_pos[all_pos < train_end]
    val_pos   = all_pos[(all_pos >= train_end) & (all_pos < val_end)]
    test_pos  = all_pos[all_pos >= val_end]

    console.print(
        f"  [dim]Sequences — Train: {len(train_pos):,} | Val: {len(val_pos):,} | "
        f"Test: {len(test_pos):,} | Features: {len(features)}[/dim]"
    )
    return df, features, scaler, Xs, y_dir, y_mfe, y_valid, (train_pos, val_pos, test_pos)


def _loader(Xs_t, y_dir_t, y_mfe_t, y_valid_t, pos, window, shuffle, *, pin_memory=False):
    ds = WindowDataset(Xs_t, y_dir_t, y_mfe_t, y_valid_t, pos, window)
    return DataLoader(
        ds,
        batch_size=cfg.nn.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def _predict_dir(model, loader, device):
    model.eval()
    all_probs, all_ys = [], []
    for xb, tgt in loader:
        probs = model.calibrated_direction_probs(xb.to(device, non_blocking=True))
        all_probs.append(probs.cpu().numpy())
        all_ys.append(tgt["direction"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_ys)


def _print_split(train_pos, val_pos, test_pos, y_dir, n_features):
    t = Table(
        title=f"[bold cyan]Dataset Split[/bold cyan]  (window={cfg.nn.WINDOW_SIZE})",
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Split", style="bold")
    t.add_column("Windows", justify="right")
    for lbl in CLASS_LABELS:
        t.add_column(lbl, justify="right")
    for name, pos in (("Train", train_pos), ("Validation", val_pos), ("Test", test_pos)):
        counts = [int((y_dir[pos] == c).sum()) for c in CLASSES]
        t.add_row(name, str(len(pos)), *map(str, counts))
    t.add_section()
    t.add_row("Features", str(n_features), *[""] * len(CLASSES))
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Quantile Transformer — Training[/bold blue]")
    torch.manual_seed(cfg.nn.RANDOM_STATE)
    device = _device()
    console.print(f"\n[bold]Device:[/bold] {device}")

    (df, features, scaler, Xs,
     y_dir, y_mfe, y_valid,
     (train_pos, val_pos, test_pos)) = _build()

    _print_split(train_pos, val_pos, test_pos, y_dir, len(features))

    Xs_t      = torch.from_numpy(Xs)
    y_dir_t   = torch.from_numpy(y_dir).long()
    y_mfe_t   = torch.from_numpy(y_mfe).float()
    y_valid_t = torch.from_numpy(y_valid).float()

    pin = device.type == "cuda"
    train_loader = _loader(Xs_t, y_dir_t, y_mfe_t, y_valid_t, train_pos, cfg.nn.WINDOW_SIZE, True,  pin_memory=pin)
    val_loader   = _loader(Xs_t, y_dir_t, y_mfe_t, y_valid_t, val_pos,   cfg.nn.WINDOW_SIZE, False, pin_memory=pin)

    # Class weights from training labels — inverse-frequency, normalised so mean=1.
    train_labels = y_dir[train_pos]
    counts       = np.bincount(train_labels, minlength=3).astype(np.float32)
    cw_np        = len(train_labels) / (3.0 * counts)            # inv-freq
    cw_tensor    = torch.tensor(cw_np / cw_np.mean(), dtype=torch.float32).to(device)
    console.print(
        f"  [dim]Class weights — LONG: {cw_tensor[0]:.3f}  "
        f"NEUTRAL: {cw_tensor[1]:.3f}  SHORT: {cw_tensor[2]:.3f}[/dim]"
    )

    model = QuantileTradingModel(input_dim=len(features)).to(device)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            console.print("  [dim]torch.compile: enabled[/dim]")
        except Exception:
            console.print("  [dim]torch.compile: skipped[/dim]")
    # Always use the underlying (uncompiled) module for state_dict / fit_temperature
    # so checkpoint keys never get an _orig_mod. prefix from torch.compile.
    _raw = getattr(model, "_orig_mod", model)

    criterion = TwoTaskLoss(
        class_weights=cw_tensor,
        label_smoothing=cfg.nn.LABEL_SMOOTHING,
        focal_gamma=1.0,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.nn.LR, weight_decay=cfg.nn.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=12, min_lr=5e-6
    )

    best_f1, best_state, patience = -1.0, None, 0
    # autocast: CUDA always OK; MPS only on PyTorch ≥ 2.1 (added in that release)
    _mps_ok       = device.type == "mps" and tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 1)
    _use_autocast  = device.type == "cuda" or _mps_ok
    _autocast_dtype = torch.bfloat16 if _use_autocast else torch.float32
    console.print(f"  [dim]autocast: {'bfloat16' if _use_autocast else 'disabled (fp32)'}[/dim]")

    for epoch in range(1, cfg.nn.EPOCHS + 1):
        model.train()
        running  = {"total": 0.0, "dir_loss": 0.0, "move_loss": 0.0}
        n_samples = 0

        for xb, tgt_b in train_loader:
            xb    = xb.to(device, non_blocking=True)
            tgt_b = {k: v.to(device, non_blocking=True) for k, v in tgt_b.items()}
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype, enabled=_use_autocast):
                losses = criterion(model(xb), tgt_b)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            bs = len(xb)
            for k in running:
                running[k] += losses[k].item() * bs
            n_samples += bs

        val_prob, val_y = _predict_dir(model, val_loader, device)
        val_pred = val_prob.argmax(1)
        macro    = f1_score(val_y, val_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        sched.step(macro)
        lr = optim.param_groups[0]["lr"]

        console.print(
            f"  Epoch {epoch:2d}/{cfg.nn.EPOCHS} | "
            f"Loss: {running['total']/n_samples:.4f} | "
            f"Dir: {running['dir_loss']/n_samples:.3f} | "
            f"Move: {running['move_loss']/n_samples:.3f} | "
            f"Val macroF1: {macro:.3f} | LR: {lr:.6f}"
        )

        if macro > best_f1:
            best_f1, patience = macro, 0
            best_state = {k: v.cpu().clone() for k, v in _raw.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.nn.EARLY_STOP_PATIENCE:
                console.print(f"  [dim]Early stop (best macroF1={best_f1:.3f})[/dim]")
                break

    if best_state is not None:
        _raw.load_state_dict(best_state)
    model.to(device)

    # ── Temperature calibration ───────────────────────────────────────────────
    console.print("\n  Fitting temperature...")
    logits_list = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits_list.append(model(xb.to(device, non_blocking=True))["direction"].cpu())
    T = _raw.fit_temperature(torch.cat(logits_list).to(device), y_dir_t[val_pos].to(device))
    console.print(f"  [dim]T = {T:.3f}  ({'softens' if T > 1 else 'sharpens'})[/dim]")

    # ── Evaluate on val + test ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    for split_name, pos in (("val", val_pos), ("test", test_pos)):
        X_seq = np.stack([Xs[t - cfg.nn.WINDOW_SIZE + 1 : t + 1] for t in pos])
        metrics = evaluate_model(
            model, X_seq, y_dir[pos], y_mfe[pos] * MAX_MFE_PCT, device, MAX_MFE_PCT
        )
        with open(f"{MODEL_DIR}/eval_{split_name}.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        if split_name == "val":
            # Compact one-liner for validation (already used for model selection above)
            console.print(
                f"  [dim]Val   — dir_acc {metrics['dir_acc']*100:.1f}%  "
                f"ECE {metrics['ece']:.4f}  "
                f"signals {metrics['n_fired']:,}/{metrics['n_total']:,}[/dim]"
            )
        else:
            console.print(f"\n  [bold]Test Evaluation[/bold]")
            print_evaluation(metrics)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict":     _raw.state_dict(),
        "features":       features,
        "classes":        list(CLASSES),
        "window":         cfg.nn.WINDOW_SIZE,
        "input_dim":      len(features),
        "scaler_mean":    scaler.mean_.astype(np.float32),
        "scaler_scale":   scaler.scale_.astype(np.float32),
        "temperature":    T,
        "lookahead_bars": cfg.training.LOOKAHEAD_BARS,
        "max_mfe_pct":    float(MAX_MFE_PCT),
    }, MODEL_PATH)
    console.print(f"\n  [dim]Model → {MODEL_PATH}[/dim]")

    with open(METRICS_PATH, "w") as fh:
        json.dump({"best_val_macro_f1": best_f1, "temperature": T}, fh, indent=2)
    console.print(f"  [dim]Metrics → {METRICS_PATH}[/dim]")
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
