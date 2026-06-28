# -*- coding: utf-8 -*-
"""
Volatility magnitude evaluation.

Model predicts: how much will price move in next LOOKAHEAD_BARS candles (% unsigned).
Direction is evaluated separately using a simple momentum rule (sign of roc_6).

Key metrics:
  - MAE / RMSE on magnitude
  - Threshold precision/recall: when model says >X%, what % actually moved >X%?
  - Direction cross-analysis: on TP bars (model signalled big move AND it happened),
    how well does momentum rule predict direction?
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


THRESHOLDS = [0.50, 1.00, 1.50, 2.00]


def evaluate_model(
    model,
    X_test:       np.ndarray,   # (N, W, features) — already scaled
    y_true:       np.ndarray,   # (N,) actual magnitude in % (always >= 0)
    y_dir:        np.ndarray,   # (N,) actual direction: +1=UP, -1=DOWN, 0=equal
    momentum_dir: np.ndarray,   # (N,) simple rule direction: sign(roc_6)
    device:       torch.device,
    batch_size:   int = 1024,
) -> dict:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(X_test[i : i + batch_size]).float().to(device)
            preds.append(model(batch).cpu().numpy())
    y_pred = np.concatenate(preds)   # (N,) predicted magnitudes, always >= 0

    mae  = float(np.abs(y_pred - y_true).mean())
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # ── Threshold precision / recall ──────────────────────────────────────────
    threshold_stats = []
    for t in THRESHOLDS:
        actual_above = y_true >= t
        pred_above   = y_pred >= t
        n_actual = int(actual_above.sum())
        n_pred   = int(pred_above.sum())
        n_tp     = int((actual_above & pred_above).sum())
        precision = n_tp / n_pred   if n_pred   > 0 else 0.0
        recall    = n_tp / n_actual if n_actual > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        threshold_stats.append({
            "threshold": t,
            "n_actual":  n_actual,
            "n_pred":    n_pred,
            "n_tp":      n_tp,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })

    # ── Direction cross-analysis at each threshold ────────────────────────────
    dir_stats: dict[float, dict] = {}
    for t in THRESHOLDS:
        tp_mask = (y_pred >= t) & (y_true >= t)   # model signalled AND actually moved
        n_hit   = int(tp_mask.sum())
        if n_hit == 0:
            dir_stats[t] = {"n_hit": 0}
            continue

        hit_dir = y_dir[tp_mask]
        hit_mom = momentum_dir[tp_mask]

        actual_up   = hit_dir > 0
        actual_down = hit_dir < 0
        n_up   = int(actual_up.sum())
        n_down = int(actual_down.sum())

        mom_correct      = hit_mom == hit_dir
        mom_correct_up   = int((actual_up   & mom_correct).sum())
        mom_correct_down = int((actual_down & mom_correct).sum())
        mom_wrong_up     = int((actual_up   & ~mom_correct).sum())
        mom_wrong_down   = int((actual_down & ~mom_correct).sum())

        dir_stats[t] = {
            "n_hit":             n_hit,
            "n_up":              n_up,
            "n_down":            n_down,
            "mom_correct":       int(mom_correct.sum()),
            "mom_correct_up":    mom_correct_up,
            "mom_correct_down":  mom_correct_down,
            "mom_wrong_up":      mom_wrong_up,
            "mom_wrong_down":    mom_wrong_down,
        }

    return {
        "mae":                   mae,
        "rmse":                  rmse,
        "mean_actual_magnitude": float(y_true.mean()),
        "mean_pred_magnitude":   float(y_pred.mean()),
        "n_total":               len(y_true),
        "threshold_stats":       threshold_stats,
        "direction_stats":       {str(k): v for k, v in dir_stats.items()},
        "_y_pred":               y_pred,
        "_y_true":               y_true,
        "_y_dir":                y_dir,
        "_momentum_dir":         momentum_dir,
    }


def print_data_stats(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    pass  # merged into print_evaluation


def print_evaluation(metrics: dict) -> None:
    n    = metrics["n_total"]
    y_t  = metrics["_y_true"]
    y_p  = metrics["_y_pred"]
    sep  = "─" * 64
    h    = "═" * 64

    def _p(num, den):
        return f"{num/den*100:.1f}%" if den > 0 else "  N/A"

    print(f"\n{h}")
    print(f"  VOLATILITY PREDICTOR — TEST RESULTS  ({n:,} bars)")
    print(h)

    # ── Data snapshot ─────────────────────────────────────────────────────────
    print(f"\n  DATA SNAPSHOT")
    print(f"  {'Actual':12}  avg {y_t.mean():.3f}%   median {np.median(y_t):.3f}%   max {y_t.max():.2f}%")
    print(f"  {'Predicted':12}  avg {y_p.mean():.3f}%   median {np.median(y_p):.3f}%   max {y_p.max():.2f}%")

    # ── Regression error ──────────────────────────────────────────────────────
    print(f"\n  ACCURACY")
    print(f"  MAE  {metrics['mae']:.4f}%    RMSE  {metrics['rmse']:.4f}%")

    # ── Signal quality table ──────────────────────────────────────────────────
    print(f"\n  SIGNAL QUALITY  (how often model correctly flags big moves)")
    print(f"  Move >  │  Actual │ Flagged │ Precision │  Recall │    F1")
    print(f"  {sep}")
    for s in metrics["threshold_stats"]:
        t = s["threshold"]
        print(
            f"  {t:5.2f}%  │ {s['n_actual']:>7,} │ {s['n_pred']:>7,} │ "
            f"  {s['precision']*100:>5.1f}%   │  {s['recall']*100:>5.1f}% │ {s['f1']*100:>5.1f}%"
        )

    # ── Direction table ────────────────────────────────────────────────────────
    print(f"\n  DIRECTION  (on confirmed big-move bars, momentum rule = sign of 6-bar ROC)")
    print(f"  Threshold │  TP bars │    UP  │  DOWN  │ Mom accuracy")
    print(f"  {sep}")
    for t in THRESHOLDS:
        ds    = metrics["direction_stats"].get(str(t), {})
        n_hit = ds.get("n_hit", 0)
        if n_hit == 0:
            print(f"  >{t:.2f}%   │        — │     —  │    —   │     —")
            continue
        n_up   = ds["n_up"]
        n_down = ds["n_down"]
        m_corr = ds["mom_correct"]
        print(
            f"  >{t:.2f}%   │  {n_hit:>6,} │ "
            f"{_p(n_up, n_hit):>5}  │ {_p(n_down, n_hit):>5}  │  {_p(m_corr, n_hit):>5}"
        )

    print(f"\n{h}\n")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "models/prediction_analysis.png",
) -> None:
    """
    4-panel figure:
      1. Scatter: actual magnitude vs predicted magnitude
      2. Distribution: histogram of actual vs predicted magnitudes
      3. Time series: actual and predicted magnitudes (first 800 bars)
      4. Precision curve: precision at each threshold
    """
    fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
    fig.suptitle("Volatility Magnitude — Actual vs Predicted", fontsize=16, color="white", fontweight="bold", y=0.98)

    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    DARK   = "#0e1117"
    PANEL  = "#1a1f2e"
    GREEN  = "#00d26a"
    RED    = "#ff4757"
    BLUE   = "#3d84ff"
    ORANGE = "#ffa502"
    GRAY   = "#8a8f9e"

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    # ── Panel 1: Scatter ──────────────────────────────────────────────────────
    err     = np.abs(y_pred - y_true)
    err_pct = np.percentile(err, 60)
    close   = err <= err_pct
    ax1.scatter(y_true[close],  y_pred[close],  c=GREEN, alpha=0.35, s=5, label="Close pred")
    ax1.scatter(y_true[~close], y_pred[~close], c=RED,   alpha=0.20, s=5, label="Off pred")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax1.plot([0, lim], [0, lim], color=ORANGE, linewidth=0.8, linestyle=":", label="Perfect fit")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Actual Magnitude %", color=GRAY, fontsize=10)
    ax1.set_ylabel("Predicted Magnitude %", color=GRAY, fontsize=10)
    ax1.set_title("Scatter: Actual vs Predicted Magnitude", color="white", fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 2: Histogram ────────────────────────────────────────────────────
    bins = np.linspace(0, max(y_true.max(), y_pred.max()), 60)
    ax2.hist(y_true, bins=bins, color=BLUE,   alpha=0.6, label="Actual",    density=True)
    ax2.hist(y_pred, bins=bins, color=ORANGE, alpha=0.6, label="Predicted", density=True)
    ax2.set_xlabel("Magnitude %", color=GRAY, fontsize=10)
    ax2.set_ylabel("Density",     color=GRAY, fontsize=10)
    ax2.set_title("Distribution: Actual vs Predicted", color="white", fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 3: Time series ──────────────────────────────────────────────────
    n_show = min(800, len(y_true))
    idx    = np.arange(n_show)
    ax3.plot(idx, y_true[:n_show], color=BLUE,   linewidth=0.8, alpha=0.85, label="Actual")
    ax3.plot(idx, y_pred[:n_show], color=ORANGE, linewidth=0.8, alpha=0.85, label="Predicted")
    ax3.fill_between(idx, y_true[:n_show], 0, alpha=0.12, color=BLUE)
    ax3.set_xlabel("Bar Index",  color=GRAY, fontsize=10)
    ax3.set_ylabel("Magnitude %", color=GRAY, fontsize=10)
    ax3.set_title(f"Time Series (first {n_show} bars)", color="white", fontsize=11, pad=8)
    ax3.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 4: Precision curve ──────────────────────────────────────────────
    thresholds = np.linspace(0.1, min(y_true.max(), 5.0), 50)
    precisions, recalls = [], []
    for t in thresholds:
        actual_above = y_true >= t
        pred_above   = y_pred >= t
        n_pred = pred_above.sum()
        n_act  = actual_above.sum()
        tp     = (actual_above & pred_above).sum()
        precisions.append(tp / n_pred   if n_pred > 0 else 0.0)
        recalls.append(   tp / n_act    if n_act  > 0 else 0.0)

    ax4.plot(thresholds, precisions, color=GREEN,  linewidth=1.5, label="Precision")
    ax4.plot(thresholds, recalls,    color=ORANGE, linewidth=1.5, label="Recall")
    ax4.axhline(0.5, color=GRAY, linewidth=0.6, linestyle="--", alpha=0.5)
    for t in THRESHOLDS:
        ax4.axvline(t, color=RED, linewidth=0.5, linestyle=":", alpha=0.5)
    ax4.set_xlabel("Threshold %", color=GRAY, fontsize=10)
    ax4.set_ylabel("Score",       color=GRAY, fontsize=10)
    ax4.set_title("Precision & Recall vs Threshold", color="white", fontsize=11, pad=8)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"\n  Graph saved → {save_path}\n")
