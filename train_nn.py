# -*- coding: utf-8 -*-
"""
Train the causal Transformer direction model on the labeled 15m data.

Reuses TB-v1's existing data pipeline (load_valid / split_bounds /
get_feature_columns) so the held-out TEST window matches model_backtest /
nn_backtest exactly. Differences from the sklearn trainer:

  - Input is a WINDOW of the last cfg.nn.WINDOW_SIZE bars (not a single bar).
  - Features are standardized (StandardScaler fit on TRAIN only).
  - A causal Transformer (nn_model.DirectionTransformer) learns the sequence.
  - Class imbalance handled with inverse-frequency CrossEntropy weights.
  - Early stopping on validation macro-F1.

Saves a bundle (weights + scaler + feature list + window) so nn_backtest.py
can reconstruct identical inputs.

Usage:
    uv run python train_nn.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import cfg
from nn_model import DirectionTransformer
from set_label import LABEL_NAMES, LONG, NEUTRAL, SHORT, get_feature_columns
from train_model import CSV_PATH, feature_matrix, load_valid, split_bounds

MODEL_DIR = "models"
NN_MODEL_PATH = f"{MODEL_DIR}/direction_nn.pt"
NN_METRICS_PATH = f"{MODEL_DIR}/nn_metrics.json"

CLASSES = (LONG, NEUTRAL, SHORT)
CLASS_LABELS = tuple(LABEL_NAMES[c] for c in CLASSES)

console = Console()


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WindowDataset(Dataset):
    """Lazily slices windows out of a shared (n, F) feature tensor — avoids
    materializing every overlapping window in memory."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 positions: np.ndarray, window: int):
        self.features = features
        self.labels = labels
        self.positions = positions
        self.window = window

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        t = int(self.positions[i])
        seq = self.features[t - self.window + 1 : t + 1]  # (W, F)
        return seq, self.labels[t]


def _build():
    """Load data, scale, and split bar positions into train/val/test."""
    df = load_valid(CSV_PATH)
    features = get_feature_columns(df)
    X = feature_matrix(df, features).fillna(0.0).to_numpy(dtype=np.float32)
    y = df["direction_label"].astype(int).to_numpy()

    train_end, val_end = split_bounds(len(df))

    scaler = StandardScaler().fit(X[:train_end])
    Xs = scaler.transform(X).astype(np.float32)

    W = cfg.nn.WINDOW_SIZE
    all_pos = np.arange(W - 1, len(df))  # bars with a full window of history
    train_pos = all_pos[all_pos < train_end]
    val_pos = all_pos[(all_pos >= train_end) & (all_pos < val_end)]
    test_pos = all_pos[all_pos >= val_end]

    return df, features, scaler, Xs, y, (train_pos, val_pos, test_pos)


def _loader(Xs_t, y_t, pos, window, shuffle):
    ds = WindowDataset(Xs_t, y_t, pos, window)
    return DataLoader(ds, batch_size=cfg.nn.BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        p = F.softmax(model(xb.to(device)), dim=1)
        probs.append(p.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def _print_split(train_pos, val_pos, test_pos, y, n_features):
    t = Table(title=f"[bold cyan]NN Dataset Split[/bold cyan]  (window={cfg.nn.WINDOW_SIZE}, chronological)",
              box=box.ROUNDED, show_lines=True)
    t.add_column("Split", style="bold")
    t.add_column("Windows", justify="right")
    for label in CLASS_LABELS:
        t.add_column(label, justify="right")
    for name, pos in (("Train", train_pos), ("Validation", val_pos), ("Test", test_pos)):
        counts = [int((y[pos] == c).sum()) for c in CLASSES]
        t.add_row(name, str(len(pos)), *map(str, counts))
    t.add_section()
    t.add_row("Features", str(n_features), *[""] * len(CLASSES))
    console.print(t)


def main() -> None:
    console.rule("[bold blue]Hudu — Transformer Direction Model[/bold blue]")
    torch.manual_seed(cfg.nn.RANDOM_STATE)
    device = _device()
    console.print(f"\n[bold]Device:[/bold] {device}  |  loading {CSV_PATH}")

    df, features, scaler, Xs, y, (train_pos, val_pos, test_pos) = _build()
    _print_split(train_pos, val_pos, test_pos, y, len(features))

    Xs_t = torch.from_numpy(Xs)
    y_t = torch.from_numpy(y).long()
    train_loader = _loader(Xs_t, y_t, train_pos, cfg.nn.WINDOW_SIZE, shuffle=True)
    val_loader = _loader(Xs_t, y_t, val_pos, cfg.nn.WINDOW_SIZE, shuffle=False)

    model = DirectionTransformer(input_dim=len(features), num_classes=len(CLASSES)).to(device)

    # Inverse-frequency class weights (handle the ~80% NEUTRAL imbalance).
    counts = np.array([(y[train_pos] == c).sum() for c in CLASSES], dtype=np.float64)
    weights = (counts.sum() / (len(CLASSES) * np.clip(counts, 1, None))).astype(np.float32)
    weight_t = torch.tensor(weights, device=device)
    console.print(f"  [dim]class weights {dict(zip(CLASS_LABELS, weights.round(2)))}[/dim]")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.nn.LR, weight_decay=cfg.nn.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.nn.EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=weight_t,
                                   label_smoothing=cfg.nn.LABEL_SMOOTHING)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item() * len(xb)

        val_prob, val_y = _predict(model, val_loader, device)
        val_pred = val_prob.argmax(1)
        macro = f1_score(val_y, val_pred, labels=list(range(len(CLASSES))),
                         average="macro", zero_division=0)
        sched.step(macro)
        console.print(f"  epoch {epoch:2d}  train_loss={total/len(train_pos):.4f}  "
                      f"val_macroF1={macro:.3f}  val_balAcc={balanced_accuracy_score(val_y, val_pred):.3f}")

        if macro > best_f1:
            best_f1, patience = macro, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.nn.EARLY_STOP_PATIENCE:
                console.print(f"  [dim]early stop at epoch {epoch} (best val macroF1={best_f1:.3f})[/dim]")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test evaluation ─────────────────────────────────────────────────────
    test_loader = _loader(Xs_t, y_t, test_pos, cfg.nn.WINDOW_SIZE, shuffle=False)
    test_prob, test_y = _predict(model, test_loader, device)
    test_pred = test_prob.argmax(1)
    report = classification_report(test_y, test_pred, labels=list(CLASSES),
                                   target_names=list(CLASS_LABELS), output_dict=True, zero_division=0)

    rt = Table(title="[bold cyan]Transformer Test Performance[/bold cyan]", box=box.ROUNDED, show_lines=True)
    for col in ("Class", "Precision", "Recall", "F1", "Support"):
        rt.add_column(col, justify=("left" if col == "Class" else "right"), style=("bold" if col == "Class" else ""))
    for label in CLASS_LABELS:
        r = report[label]
        rt.add_row(label, f"{r['precision']*100:.1f}%", f"{r['recall']*100:.1f}%",
                   f"{r['f1-score']:.3f}", str(int(r["support"])))
    console.print(rt)
    console.print(f"  test balanced_acc={balanced_accuracy_score(test_y, test_pred):.3f}  "
                  f"macroF1={f1_score(test_y, test_pred, average='macro', zero_division=0):.3f}")

    # ── Save bundle ─────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "features": features,
        "classes": list(CLASSES),
        "window": cfg.nn.WINDOW_SIZE,
        "input_dim": len(features),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }, NN_MODEL_PATH)
    console.print(f"  [dim]Transformer saved → {NN_MODEL_PATH}[/dim]")

    with open(NN_METRICS_PATH, "w") as fh:
        json.dump({"best_val_macro_f1": best_f1,
                   "test_balanced_acc": float(balanced_accuracy_score(test_y, test_pred)),
                   "test_macro_f1": float(f1_score(test_y, test_pred, average="macro", zero_division=0)),
                   "report": report}, fh, indent=2)
    console.print(f"  [dim]Metrics → {NN_METRICS_PATH}[/dim]")
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
