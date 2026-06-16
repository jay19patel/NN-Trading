# -*- coding: utf-8 -*-
"""
Direction classifier on oracle-labeled 15m OHLCV data.

What this does
--------------
  - Loads the labeled CSV produced by app.py
  - Uses the leakage-safe causal indicators (rsi, macd, adx, bb, all the
    `condition_*` signal columns, MA-status, mtf features ...) as input
    features via `get_feature_columns()` — NO future-derived column is used.
  - Target = `direction_label`  (0=LONG, 1=NEUTRAL, 2=SHORT)
  - Splits the data CHRONOLOGICALLY into train / validation / test
    (no shuffle — shuffling a time series leaks the future into the past).
  - Trains two candidate models with balanced class weights, compares them
    on the validation set, and evaluates the winner on the held-out test set.
  - Reports accuracy, balanced-accuracy, macro-F1, per-class precision/
    recall/F1, the confusion matrix, and the most important features.
  - Saves the trained model (+ feature list + split config), the test
    predictions, and a metrics JSON.

TASK_MODE
---------
  "binary"     → LONG vs SHORT only. NEUTRAL bars are dropped for training so
                 the model focuses on direction. This is the default because
                 the 79% NEUTRAL majority otherwise swamps the signal. The
                 model exposes probabilities, so model_backtest.py can stay
                 flat on low-confidence bars (a confidence threshold replaces
                 the dropped NEUTRAL class).
  "multiclass" → LONG / NEUTRAL / SHORT (the model itself decides when to
                 stay out via the NEUTRAL class).

Why these metrics (and not just "accuracy")
--------------------------------------------
The classes are heavily imbalanced. A model that always predicts the majority
class scores high "accuracy" while being useless for trading. Balanced
accuracy, macro-F1 and the per-class LONG/SHORT scores tell the real story.

Usage
-----
    uv run python train_model.py
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import cfg
from set_label import LABEL_NAMES, LONG, NEUTRAL, SHORT, get_feature_columns

# ── Config ────────────────────────────────────────────────────────────────────
# Tunable knobs live in config.py (cfg.ml_training). Paths stay here.
CSV_PATH: str = "data/labeled_BTCUSD_15m.csv"
MODEL_DIR: str = "models"
MODEL_PATH: str = f"{MODEL_DIR}/direction_clf.joblib"
PRED_PATH: str = "data/test_predictions.csv"
METRICS_PATH: str = f"{MODEL_DIR}/metrics.json"

TASK_MODE: str = cfg.ml_training.TASK_MODE          # "binary" | "multiclass"
TRAIN_FRAC: float = cfg.ml_training.TRAIN_FRAC
VAL_FRAC: float = cfg.ml_training.VAL_FRAC          # test = 1 - TRAIN - VAL
RANDOM_STATE: int = cfg.ml_training.RANDOM_STATE

TRADE_CLASSES: tuple[int, ...] = (LONG, SHORT)  # classes that actually trade
CLASSES: tuple[int, ...] = TRADE_CLASSES if TASK_MODE == "binary" else (LONG, NEUTRAL, SHORT)
CLASS_LABELS: tuple[str, ...] = tuple(LABEL_NAMES[c] for c in CLASSES)

console = Console()


# ── Shared data helpers (also imported by model_backtest.py) ──────────────────

def load_valid(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the labeled CSV and keep only the contiguous, valid-label rows.

    Both training and the model backtest call this so they agree exactly on
    which rows exist and therefore on the chronological split boundaries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labeled CSV not found at {path!r}. Run `uv run python app.py` first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "direction_label" not in df.columns:
        raise ValueError("CSV is missing the 'direction_label' target column.")
    if "label_valid" in df.columns:
        df = df[df["label_valid"].fillna(False).astype(bool)].copy()
    return df


def split_bounds(n: int) -> tuple[int, int]:
    """Return (train_end, val_end) index positions for an n-row frame."""
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Coerce the chosen feature columns to a clean numeric matrix."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


# ── Data preparation ──────────────────────────────────────────────────────────

@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    features: list[str]
    test_index: pd.Index


def _mask_trades(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """In binary mode keep only LONG/SHORT rows; otherwise pass through."""
    if TASK_MODE != "binary":
        return X, y
    keep = y.isin(TRADE_CLASSES)
    return X[keep], y[keep]


def select_features(X_train: pd.DataFrame, y_train: pd.Series, features: list[str]) -> list[str]:
    """Filter the feature set per cfg.ml_training.FEATURE_SELECTION.

    Fit ONLY on the training split (no leakage from val/test). The
    "importance_top_k" method ranks features with a quick RandomForest and
    keeps the strongest TOP_K_FEATURES.
    """
    method = cfg.ml_training.FEATURE_SELECTION
    if method == "all":
        return features

    if method == "importance_top_k":
        k = min(int(cfg.ml_training.TOP_K_FEATURES), len(features))
        ranker = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        )
        ranker.fit(X_train[features].fillna(0.0), y_train)
        order = np.argsort(ranker.feature_importances_)[::-1][:k]
        return [features[i] for i in order]

    raise ValueError(f"Unknown FEATURE_SELECTION: {method!r}")


def prepare(df: pd.DataFrame) -> Dataset:
    """Build leakage-safe X/y and split chronologically into train/val/test.

    The split boundaries are computed on the full contiguous frame so they
    line up with model_backtest.py. In binary mode each split is then filtered
    down to decisive (LONG/SHORT) bars for fitting and scoring. Features are
    filtered (config-driven) using the training split only.
    """
    all_features = get_feature_columns(df)
    X = feature_matrix(df, all_features)
    y = df["direction_label"].astype(int)

    train_end, val_end = split_bounds(len(df))

    X_tr, y_tr = _mask_trades(X.iloc[:train_end], y.iloc[:train_end])
    X_va, y_va = _mask_trades(X.iloc[train_end:val_end], y.iloc[train_end:val_end])
    X_te, y_te = _mask_trades(X.iloc[val_end:], y.iloc[val_end:])

    features = select_features(X_tr, y_tr, all_features)
    console.print(
        f"  [dim]Feature filter[/dim] [bold]{cfg.ml_training.FEATURE_SELECTION}[/bold]: "
        f"{len(all_features)} → [bold]{len(features)}[/bold] features"
    )

    return Dataset(
        X_train=X_tr[features], y_train=y_tr,
        X_val=X_va[features], y_val=y_va,
        X_test=X_te[features], y_test=y_te,
        features=features,
        test_index=X_te.index,
    )


# ── Models ──────────────────────────────────────────────────────────────────

def build_candidates() -> dict[str, object]:
    """Two leakage-robust, imbalance-aware tabular classifiers to compare."""
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(model: object, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute the full metric bundle for a fitted model on (X, y)."""
    y_pred = model.predict(X)
    report = classification_report(
        y, y_pred, labels=list(CLASSES), target_names=list(CLASS_LABELS),
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=list(CLASSES))

    # Signal precision: of the bars the model flagged LONG/SHORT, how many were
    # truly that direction — the practically useful number for trading.
    y_true = y.to_numpy()
    signal_mask = np.isin(y_pred, TRADE_CLASSES)
    signal_total = int(signal_mask.sum())
    signal_correct = int((y_pred[signal_mask] == y_true[signal_mask]).sum())

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", labels=list(CLASSES), zero_division=0)),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "signal_precision": (signal_correct / signal_total) if signal_total else 0.0,
        "signal_trades": signal_total,
        "y_pred": y_pred,
    }


# ── Console printers ──────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def print_split_summary(d: Dataset) -> None:
    t = Table(
        title=f"[bold cyan]Dataset Split[/bold cyan]  (chronological, no shuffle, mode={TASK_MODE})",
        box=box.ROUNDED, show_lines=True,
    )
    t.add_column("Split", style="bold")
    t.add_column("Rows", justify="right")
    for label in CLASS_LABELS:
        t.add_column(label, justify="right")
    for name, y in (("Train", d.y_train), ("Validation", d.y_val), ("Test", d.y_test)):
        vc = y.value_counts()
        counts = [str(int(vc.get(c, 0))) for c in CLASSES]
        t.add_row(name, str(len(y)), *counts)
    t.add_section()
    t.add_row("Features", str(len(d.features)), *[""] * len(CLASSES))
    console.print(t)


def print_comparison(val_metrics: dict[str, dict], winner: str) -> None:
    t = Table(title="[bold cyan]Model Comparison[/bold cyan]  (validation set)",
              box=box.ROUNDED, show_lines=True)
    t.add_column("Model", style="bold")
    t.add_column("Accuracy", justify="right")
    t.add_column("Balanced Acc", justify="right")
    t.add_column("Macro F1", justify="right")
    t.add_column("Signal Prec", justify="right")
    for name, m in val_metrics.items():
        star = " [green]★[/green]" if name == winner else ""
        t.add_row(name + star, _pct(m["accuracy"]), _pct(m["balanced_accuracy"]),
                  f"{m['macro_f1']:.3f}", _pct(m["signal_precision"]))
    console.print(t)


def print_test_report(name: str, m: dict) -> None:
    t = Table(title=f"[bold cyan]Test Performance[/bold cyan]  —  {name}",
              box=box.ROUNDED, show_lines=True)
    t.add_column("Metric", style="bold", min_width=24)
    t.add_column("Value", justify="right", min_width=14)
    t.add_row("Accuracy", _pct(m["accuracy"]))
    t.add_row("Balanced accuracy", _pct(m["balanced_accuracy"]))
    t.add_row("Macro F1", f"{m['macro_f1']:.3f}")
    t.add_section()
    t.add_row("Signal precision (LONG/SHORT)", _pct(m["signal_precision"]))
    t.add_row("Signals fired", str(m["signal_trades"]))
    console.print(t)

    pc = Table(title="[bold]Per-Class Report (test)[/bold]", box=box.ROUNDED, show_lines=True)
    pc.add_column("Class", style="bold")
    pc.add_column("Precision", justify="right")
    pc.add_column("Recall", justify="right")
    pc.add_column("F1", justify="right")
    pc.add_column("Support", justify="right")
    for label in CLASS_LABELS:
        r = m["report"][label]
        pc.add_row(label, _pct(r["precision"]), _pct(r["recall"]),
                   f"{r['f1-score']:.3f}", str(int(r["support"])))
    console.print(pc)

    cm = m["confusion_matrix"]
    cmt = Table(title="[bold]Confusion Matrix (test)[/bold]  rows=actual, cols=predicted",
                box=box.ROUNDED, show_lines=True)
    cmt.add_column("actual \\ pred", style="bold")
    for label in CLASS_LABELS:
        cmt.add_column(label, justify="right")
    for i, label in enumerate(CLASS_LABELS):
        cmt.add_row(label, *[str(cm[i][j]) for j in range(len(CLASS_LABELS))])
    console.print(cmt)


def print_top_features(names: list[str], importances: np.ndarray, top_n: int = 20) -> None:
    order = np.argsort(importances)[::-1][:top_n]
    t = Table(title=f"[bold cyan]Top {top_n} Features[/bold cyan]  (permutation importance)",
              box=box.ROUNDED, show_lines=False)
    t.add_column("#", style="dim", justify="right")
    t.add_column("Feature", style="bold")
    t.add_column("Importance", justify="right")
    for rank, idx in enumerate(order, 1):
        t.add_row(str(rank), names[idx], f"{importances[idx]:.4f}")
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Hudu — Direction Model Training[/bold blue]")

    console.print(f"\n[bold]Loading[/bold] {CSV_PATH}  (task={TASK_MODE})")
    df = load_valid(CSV_PATH)
    data = prepare(df)
    print_split_summary(data)

    # ── Train candidates & compare on validation ────────────────────────────
    candidates = build_candidates()
    fitted: dict[str, object] = {}
    val_metrics: dict[str, dict] = {}
    for name, model in candidates.items():
        with console.status(f"Training {name}..."):
            model.fit(data.X_train, data.y_train)
        fitted[name] = model
        val_metrics[name] = evaluate(model, data.X_val, data.y_val)

    # Winner = best validation macro-F1 (robust to class imbalance).
    winner = max(val_metrics, key=lambda k: val_metrics[k]["macro_f1"])
    print_comparison(val_metrics, winner)

    # ── Evaluate winner on the held-out test set ────────────────────────────
    best_model = fitted[winner]
    test_metrics = evaluate(best_model, data.X_test, data.y_test)
    print_test_report(winner, test_metrics)

    # ── Feature importance (permutation, on validation to avoid test peeking) ─
    with console.status("Computing feature importances..."):
        perm = permutation_importance(
            best_model, data.X_val, data.y_val,
            n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1,
        )
    print_top_features(data.features, perm.importances_mean)

    # ── Persist model, predictions, metrics ─────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": data.features,
            "classes": list(CLASSES),
            "task_mode": TASK_MODE,
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
        },
        MODEL_PATH,
    )
    console.print(f"  [dim]Model saved        → {MODEL_PATH}[/dim]")

    pred_df = pd.DataFrame(
        {
            "timestamp": data.test_index,
            "actual": data.y_test.to_numpy(),
            "actual_name": [LABEL_NAMES[c] for c in data.y_test.to_numpy()],
            "predicted": test_metrics["y_pred"],
            "predicted_name": [LABEL_NAMES[c] for c in test_metrics["y_pred"]],
        }
    )
    pred_df.to_csv(PRED_PATH, index=False)
    console.print(f"  [dim]Test predictions   → {PRED_PATH}[/dim]")

    metrics_out = {
        "task_mode": TASK_MODE,
        "winner": winner,
        "validation": {k: _strip(v) for k, v in val_metrics.items()},
        "test": _strip(test_metrics),
        "n_train": len(data.y_train),
        "n_val": len(data.y_val),
        "n_test": len(data.y_test),
        "n_features": len(data.features),
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    console.print(f"  [dim]Metrics JSON       → {METRICS_PATH}[/dim]")

    console.rule("[bold green]Done[/bold green]")


def _strip(m: dict) -> dict:
    """Drop the bulky raw prediction array before serializing to JSON."""
    return {k: v for k, v in m.items() if k != "y_pred"}


if __name__ == "__main__":
    main()
