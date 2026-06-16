# -*- coding: utf-8 -*-
"""
Model evaluation: direction accuracy + confidence calibration + move accuracy.

Two evaluation tasks:
  A. Direction accuracy and confidence calibration
     "Model ne LONG bola 87% confidence se, kya price upar gaya?"
     Metric: per-bin reliability diagram + ECE

  B. Move magnitude accuracy
     "Model ne 1% bola, actual 0.5% hua — kitna sahi tha?"
     Metric: move_ratio = actual_mfe / predicted_q50 + quantile coverage

Usage:
    from evaluator import evaluate_model, print_evaluation

    metrics = evaluate_model(model, X_test, y_true_dir, y_true_mfe, device)
    print_evaluation(metrics)
"""
from __future__ import annotations

import json

import numpy as np
import torch

LONG    = 0
NEUTRAL = 1
SHORT   = 2


def evaluate_model(
    model,
    X_test:      np.ndarray,    # (N, W, features)
    y_true_dir:  np.ndarray,    # (N,) int direction labels
    y_true_mfe:  np.ndarray,    # (N,) actual MFE % (de-normalised)
    device:      torch.device,
    max_mfe_pct: float = 3.0,
    batch_size:  int   = 1024,
) -> dict:
    """
    Full evaluation of direction accuracy, confidence calibration, and move prediction.

    Parameters
    ----------
    model       : trained QuantileTradingModel with calibrated temperature
    X_test      : numpy array (N, W, features), already scaled
    y_true_dir  : (N,) integer direction labels (0=LONG, 1=NEUTRAL, 2=SHORT)
    y_true_mfe  : (N,) actual max favorable move in % (direction-aware, de-normalised)
    device      : torch device
    max_mfe_pct : de-normalisation scalar (move_q * max_mfe_pct = predicted %)
    batch_size  : inference batch size

    Returns
    -------
    dict with keys: n_total, n_fired, fire_rate, dir_acc, long_acc, short_acc,
                    ece, calib_bins, move (sub-dict)
    """
    model.eval()
    all_probs:  list[np.ndarray] = []
    all_move_q: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch  = torch.from_numpy(X_test[i : i + batch_size]).float().to(device)
            probs  = model.calibrated_direction_probs(batch).cpu().numpy()
            out    = model(batch)
            mq     = out["move_q"].cpu().numpy() * max_mfe_pct   # de-normalise to %
            all_probs.append(probs)
            all_move_q.append(mq)

    probs  = np.concatenate(all_probs)    # (N, 3)
    move_q = np.concatenate(all_move_q)  # (N, 3) in %

    pred_dir   = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    # ── TASK A: Direction accuracy ────────────────────────────────────────────
    fired   = pred_dir != NEUTRAL
    correct = (pred_dir == y_true_dir) & fired
    dir_acc = float(correct[fired].mean())  if fired.sum() > 0 else 0.0

    long_fired  = fired & (pred_dir == LONG)
    short_fired = fired & (pred_dir == SHORT)
    long_acc    = float(correct[long_fired].mean())  if long_fired.sum()  > 0 else 0.0
    short_acc   = float(correct[short_fired].mean()) if short_fired.sum() > 0 else 0.0

    # Calibration bins (10 bins from 0.30 to 1.00)
    bins       = np.arange(0.30, 1.01, 0.10)
    calib_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi) & fired
        if mask.sum() == 0:
            continue
        calib_rows.append({
            "conf_lo":    round(float(lo), 2),
            "conf_hi":    round(float(hi), 2),
            "n":          int(mask.sum()),
            "mean_conf":  float(confidence[mask].mean()),
            "actual_acc": float(correct[mask].mean()),
            "gap":        float(abs(confidence[mask].mean() - correct[mask].mean())),
        })

    n_total = int(fired.sum())
    ece = (
        sum(r["n"] / n_total * r["gap"] for r in calib_rows)
        if n_total > 0 else 0.0
    )

    # ── TASK B: Move accuracy (only on correct-direction predictions) ─────────
    move_metrics: dict = {}
    if correct.sum() > 0:
        actual   = y_true_mfe[correct]
        q10_pred = move_q[correct, 0]
        q50_pred = move_q[correct, 1]
        q90_pred = move_q[correct, 2]

        ratio = np.clip(actual / (q50_pred + 1e-6), 0.0, 2.0)

        move_metrics = {
            "n_correct_dir":      int(correct.sum()),
            "mean_move_ratio":    float(ratio.mean()),
            "median_move_ratio":  float(np.median(ratio)),
            "pct_within_50pct":   float(((ratio >= 0.5) & (ratio <= 1.5)).mean()),
            "pct_overestimated":  float((ratio < 0.5).mean()),
            "pct_underestimated": float((ratio > 1.5).mean()),
            "q10_coverage":       float((actual >= q10_pred).mean()),
            "q50_coverage":       float((actual >= q50_pred).mean()),
            "q90_coverage":       float((actual <= q90_pred).mean()),
            "mean_actual_mfe":    float(actual.mean()),
            "mean_predicted_q50": float(q50_pred.mean()),
        }

    return {
        "n_total":   int(len(y_true_dir)),
        "n_fired":   int(fired.sum()),
        "fire_rate": float(fired.mean()),
        "dir_acc":   float(dir_acc),
        "long_acc":  float(long_acc),
        "short_acc": float(short_acc),
        "ece":       float(ece),
        "calib_bins": calib_rows,
        "move":       move_metrics,
    }


def print_evaluation(metrics: dict) -> None:
    """Print a clean formatted evaluation report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("MODEL EVALUATION REPORT")
    print(sep)

    print(f"\nSIGNAL SUMMARY:")
    print(f"  Total bars:     {metrics['n_total']:,}")
    print(f"  Signals fired:  {metrics['n_fired']:,} ({metrics['fire_rate']*100:.1f}%)")

    print(f"\nDIRECTION ACCURACY:")
    print(f"  Overall:  {metrics['dir_acc']*100:.1f}%")
    print(f"  LONG:     {metrics['long_acc']*100:.1f}%")
    print(f"  SHORT:    {metrics['short_acc']*100:.1f}%")
    ece_tag = (
        "excellent"   if metrics['ece'] < 0.05
        else "acceptable" if metrics['ece'] < 0.10
        else "NEEDS REFIT"
    )
    print(f"  ECE:      {metrics['ece']:.4f}  ({ece_tag})")

    print(f"\nCONFIDENCE CALIBRATION:")
    hdr = f"  {'Conf':^12} {'N':>6} {'Mean Conf':>10} {'Actual Acc':>11} {'Gap':>6} {'Status':>8}"
    print(hdr)
    print(f"  {'-'*56}")
    for r in metrics["calib_bins"]:
        status = "✓" if r["gap"] < 0.05 else ("~" if r["gap"] < 0.10 else "✗")
        print(
            f"  {r['conf_lo']:.1f}-{r['conf_hi']:.1f}    "
            f"{r['n']:>6}   {r['mean_conf']:>9.3f}   {r['actual_acc']:>10.3f}   "
            f"{r['gap']:>5.3f}  {status:>8}"
        )

    m = metrics.get("move", {})
    if m:
        print(f"\nMOVE PREDICTION ACCURACY  (on {m['n_correct_dir']} correct-direction bars):")
        print(f"  Mean actual MFE:      {m['mean_actual_mfe']:.3f}%")
        print(f"  Mean predicted q50:   {m['mean_predicted_q50']:.3f}%")
        ratio_tag = (
            "≈ perfect" if 0.9 <= m['mean_move_ratio'] <= 1.1
            else "overestimating" if m['mean_move_ratio'] < 0.9
            else "underestimating"
        )
        print(f"  Mean move ratio:      {m['mean_move_ratio']:.3f}  (1.0=perfect — {ratio_tag})")
        print(f"  Within 50% of pred:   {m['pct_within_50pct']*100:.1f}%")
        print(f"  Model overestimated:  {m['pct_overestimated']*100:.1f}%  (ratio < 0.5)")
        print(f"  Model underestimated: {m['pct_underestimated']*100:.1f}%  (ratio > 1.5)")

        def _cov_tag(actual_cov, target):
            return "OK" if abs(actual_cov - target) < 0.10 else "ADJUST"

        print(f"\n  QUANTILE COVERAGE  (targets: q10≈90%, q50≈50%, q90≈90%):")
        print(f"  q10 coverage: {m['q10_coverage']*100:.1f}%  ({_cov_tag(m['q10_coverage'], 0.90)})")
        print(f"  q50 coverage: {m['q50_coverage']*100:.1f}%  ({_cov_tag(m['q50_coverage'], 0.50)})")
        print(f"  q90 coverage: {m['q90_coverage']*100:.1f}%  ({_cov_tag(m['q90_coverage'], 0.90)})")

    print(f"\n{sep}")
