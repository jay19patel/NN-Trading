# -*- coding: utf-8 -*-
"""
Metrics for time-series trading models: regression + classification without relying on
raw "accuracy" as the sole measure (class imbalance and market noise dominate).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def bars_per_day_from_interval_minutes(interval_minutes: int = 15) -> int:
    return (24 * 60) // interval_minutes


def sequence_count_for_split(raw_bar_count: int, sequence_length: int) -> int:
    """Sequences produced by create_sequences: one sample per end index from seq_len .. len-1."""
    if raw_bar_count <= sequence_length:
        return 0
    return raw_bar_count - sequence_length


def regression_metrics_numpy(
    predicted_qty: np.ndarray,
    predicted_tp: np.ndarray,
    predicted_sl: np.ndarray,
    true_qty: np.ndarray,
    true_tp: np.ndarray,
    true_sl: np.ndarray,
) -> Dict[str, float]:
    return {
        "mae_qty_ratio": float(mean_absolute_error(true_qty, predicted_qty)),
        "mae_take_profit_pct": float(mean_absolute_error(true_tp, predicted_tp)),
        "mae_stop_loss_pct": float(mean_absolute_error(true_sl, predicted_sl)),
    }


def directional_accuracy_numpy(
    predicted_direction: np.ndarray,
    true_direction: np.ndarray,
    neutral_label: int = 1,
) -> Dict[str, float]:
    """Share of bars where predicted direction matches label (including neutral)."""
    return {"direction_accuracy_all": float(accuracy_score(true_direction, predicted_direction))}


def directional_accuracy_non_neutral_numpy(
    predicted_direction: np.ndarray,
    true_direction: np.ndarray,
    neutral_label: int = 1,
) -> Dict[str, float]:
    """Hit rate when restricting to actual non-neutral moves (more informative than overall acc)."""
    mask = true_direction != neutral_label
    if mask.sum() == 0:
        return {"direction_accuracy_excl_neutral": float("nan")}
    return {
        "direction_accuracy_excl_neutral": float(
            accuracy_score(true_direction[mask], predicted_direction[mask])
        )
    }


def classification_metrics_numpy(
    predicted_direction: np.ndarray,
    true_direction: np.ndarray,
    label_names: Tuple[str, str, str] = ("Buy", "Neutral", "Sell"),
) -> Dict[str, float | str]:
    macro_f1 = f1_score(true_direction, predicted_direction, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_direction, predicted_direction, average="weighted", zero_division=0)
    report = classification_report(
        true_direction,
        predicted_direction,
        target_names=list(label_names),
        labels=[0, 1, 2],
        zero_division=0,
    )
    confusion = confusion_matrix(true_direction, predicted_direction, labels=[0, 1, 2])
    return {
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "classification_report_text": report,
        "confusion_matrix_3x3": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_model_on_split(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float | str]:
    """Run model on numpy arrays and aggregate sklearn / numpy metrics."""
    model.eval()
    batch_tensor = torch.FloatTensor(features).to(device)
    predictions = model(batch_tensor)
    pred_dir = torch.argmax(predictions["direction"], dim=1).cpu().numpy()
    targets_direction = targets["direction"]

    # Decode sizing head [qty_ratio, tp_raw, sl_raw]
    sizing = predictions["sizing"].cpu().numpy()
    pred_qty = sizing[:, 0]
    
    # Needs config for scaling, so we'll import it
    from config import config
    pred_tp = sizing[:, 1] * config.strategy.LABEL_TP_PCT_MAX
    pred_sl = sizing[:, 2] * config.strategy.LABEL_SL_PCT_MAX
    
    true_qty = targets["qty_ratio"]
    true_tp = targets["take_profit_pct"]
    true_sl = targets["stop_loss_pct"]

    metrics: Dict[str, float | str] = {}
    metrics.update(
        regression_metrics_numpy(pred_qty, pred_tp, pred_sl, true_qty, true_tp, true_sl)
    )
    metrics.update(directional_accuracy_numpy(pred_dir, targets_direction))
    metrics.update(directional_accuracy_non_neutral_numpy(pred_dir, targets_direction))
    metrics.update(classification_metrics_numpy(pred_dir, targets_direction))

    return metrics
