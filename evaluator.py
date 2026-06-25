# -*- coding: utf-8 -*-
"""
Achievement-based evaluation for max-return prediction.

The model predicts max_return for the next LOOKAHEAD_BARS candles:
  +value = model expects upside move
  -value = model expects downside move

Achievement tiers (correct direction + magnitude match):
  25%  — correct direction AND |predicted| ≥ 0.25 × |actual|
  50%  — correct direction AND |predicted| ≥ 0.50 × |actual|
  75%  — correct direction AND |predicted| ≥ 0.75 × |actual|
  100% — correct direction AND |predicted| ≥ 1.00 × |actual|

Metrics are broken down overall, and separately for UPSIDE and DOWNSIDE bars.
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _tier_counts(mask_correct: np.ndarray, ratio: np.ndarray) -> dict:
    """Return achievement counts for a subset (e.g. upside or downside bars)."""
    n = len(mask_correct)
    at_25  = mask_correct & (ratio >= 0.25)
    at_50  = mask_correct & (ratio >= 0.50)
    at_75  = mask_correct & (ratio >= 0.75)
    at_100 = mask_correct & (ratio >= 1.00)
    return {
        "n":          n,
        "n_correct":  int(mask_correct.sum()),
        "n_25":       int(at_25.sum()),
        "n_50":       int(at_50.sum()),
        "n_75":       int(at_75.sum()),
        "n_100":      int(at_100.sum()),
    }


def evaluate_model(
    model,
    X_test:     np.ndarray,   # (N, W, features) — already scaled
    y_true:     np.ndarray,   # (N,) actual max_return in %
    device:     torch.device,
    batch_size: int = 1024,
) -> dict:
    model.eval()
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(X_test[i : i + batch_size]).float().to(device)
            preds.append(model(batch).cpu().numpy())

    y_pred = np.concatenate(preds)   # (N,)

    # ── Masks ─────────────────────────────────────────────────────────────────
    meaningful  = np.abs(y_true) >= 0.01          # exclude near-zero actual moves
    actual_up   = meaningful & (y_true > 0)        # truly upside bars
    actual_down = meaningful & (y_true < 0)        # truly downside bars

    same_sign   = (np.sign(y_pred) == np.sign(y_true)) & meaningful

    # ── Achievement ratio (only valid where direction is correct) ──────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            same_sign,
            np.abs(y_pred) / (np.abs(y_true) + 1e-9),
            0.0,
        )

    # ── Overall tiers ─────────────────────────────────────────────────────────
    n            = len(y_true)
    n_meaningful = int(meaningful.sum())
    n_correct    = int(same_sign.sum())

    at_25  = same_sign & (ratio >= 0.25)
    at_50  = same_sign & (ratio >= 0.50)
    at_75  = same_sign & (ratio >= 0.75)
    at_100 = same_sign & (ratio >= 1.00)

    # ── Upside breakdown ──────────────────────────────────────────────────────
    # Among actual-upside bars: how many did model predict UP (correct) vs DOWN (missed)
    up_correct  = same_sign & actual_up        # predicted UP, actual UP
    up_missed   = actual_up & ~up_correct      # predicted DOWN, actual UP (missed)
    up_ratio    = np.where(up_correct, ratio, 0.0)
    up_stats    = _tier_counts(up_correct, up_ratio)
    up_stats["n_missed"] = int(up_missed.sum())

    # ── Downside breakdown ────────────────────────────────────────────────────
    # Among actual-downside bars: how many did model predict DOWN (correct) vs UP (missed)
    down_correct = same_sign & actual_down     # predicted DOWN, actual DOWN
    down_missed  = actual_down & ~down_correct # predicted UP, actual DOWN (missed)
    down_ratio   = np.where(down_correct, ratio, 0.0)
    down_stats   = _tier_counts(down_correct, down_ratio)
    down_stats["n_missed"] = int(down_missed.sum())

    return {
        # ── Summary ───────────────────────────────────────────────────────────
        "n_total":                n,
        "n_meaningful":           n_meaningful,
        "n_actual_up":            int(actual_up.sum()),
        "n_actual_down":          int(actual_down.sum()),
        "direction_accuracy":     float(same_sign[meaningful].mean()) if n_meaningful > 0 else 0.0,
        "mae":                    float(np.abs(y_pred - y_true).mean()),
        "rmse":                   float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mean_actual_max_return": float(np.mean(y_true)),
        "mean_pred_max_return":   float(np.mean(y_pred)),

        # ── Overall achievement ────────────────────────────────────────────────
        "n_correct":   n_correct,
        "n_25":        int(at_25.sum()),
        "n_50":        int(at_50.sum()),
        "n_75":        int(at_75.sum()),
        "n_100":       int(at_100.sum()),

        # ── Per-direction breakdown ────────────────────────────────────────────
        "upside":   up_stats,
        "downside": down_stats,

        # raw arrays for downstream analysis
        "_y_pred": y_pred,
        "_y_true": y_true,
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "models/prediction_analysis.png") -> None:
    """
    4-panel figure:
      1. Scatter: actual vs predicted (coloured by correct/wrong direction)
      2. Distribution: histogram overlay of actual vs predicted
      3. Time series: actual and predicted over bar index (first 500 bars)
      4. Direction confusion bar chart
    """
    correct_dir = np.sign(y_pred) == np.sign(y_true)

    fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
    fig.suptitle("Actual vs Predicted — Max Return Analysis", fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])   # full-width bottom

    DARK   = "#0e1117"
    PANEL  = "#1a1f2e"
    GREEN  = "#00d26a"
    RED    = "#ff4757"
    BLUE   = "#3d84ff"
    ORANGE = "#ffa502"
    GRAY   = "#8a8f9e"

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    # ── Panel 1: Scatter actual vs predicted ─────────────────────────────────
    ax1.scatter(y_true[correct_dir],  y_pred[correct_dir],  c=GREEN, alpha=0.35, s=6, label="Correct dir")
    ax1.scatter(y_true[~correct_dir], y_pred[~correct_dir], c=RED,   alpha=0.25, s=6, label="Wrong dir")
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max()) * 1.05
    ax1.axhline(0, color=GRAY, linewidth=0.6, linestyle="--")
    ax1.axvline(0, color=GRAY, linewidth=0.6, linestyle="--")
    ax1.plot([-lim, lim], [-lim, lim], color=ORANGE, linewidth=0.8, linestyle=":", label="Perfect fit")
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("Actual Return %", color=GRAY, fontsize=10)
    ax1.set_ylabel("Predicted Return %", color=GRAY, fontsize=10)
    ax1.set_title("Scatter: Actual vs Predicted", color="white", fontsize=11, pad=8)
    leg1 = ax1.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 2: Distribution histogram ──────────────────────────────────────
    bins = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 80)
    ax2.hist(y_true, bins=bins, color=BLUE,   alpha=0.6, label="Actual",    density=True)
    ax2.hist(y_pred, bins=bins, color=ORANGE, alpha=0.6, label="Predicted", density=True)
    ax2.axvline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Return %", color=GRAY, fontsize=10)
    ax2.set_ylabel("Density", color=GRAY, fontsize=10)
    ax2.set_title("Distribution: Actual vs Predicted", color="white", fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 3: Time series (first 800 bars) ─────────────────────────────────
    n_show = min(800, len(y_true))
    idx    = np.arange(n_show)
    ax3.plot(idx, y_true[:n_show], color=BLUE,   linewidth=0.8, alpha=0.85, label="Actual")
    ax3.plot(idx, y_pred[:n_show], color=ORANGE, linewidth=0.8, alpha=0.85, label="Predicted")
    ax3.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax3.fill_between(idx, y_true[:n_show], 0, where=(y_true[:n_show] > 0), alpha=0.12, color=GREEN)
    ax3.fill_between(idx, y_true[:n_show], 0, where=(y_true[:n_show] < 0), alpha=0.12, color=RED)
    ax3.set_xlabel("Bar Index", color=GRAY, fontsize=10)
    ax3.set_ylabel("Return %", color=GRAY, fontsize=10)
    ax3.set_title(f"Time Series: Actual vs Predicted  (first {n_show} bars)", color="white", fontsize=11, pad=8)
    ax3.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Stats annotation ──────────────────────────────────────────────────────
    n_up   = int((y_true > 0).sum())
    n_down = int((y_true < 0).sum())
    n      = len(y_true)
    pu     = int(((y_true > 0) & (y_pred > 0)).sum())
    pd_    = int(((y_true < 0) & (y_pred < 0)).sum())
    stats_txt = (
        f"Total: {n:,}   Actual↑: {n_up:,} ({n_up/n*100:.1f}%)   Actual↓: {n_down:,} ({n_down/n*100:.1f}%)\n"
        f"Dir accuracy — Upside: {pu/max(n_up,1)*100:.1f}%   Downside: {pd_/max(n_down,1)*100:.1f}%"
    )
    fig.text(0.5, 0.005, stats_txt, ha="center", fontsize=9, color=GRAY, style="italic")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"\n  Graph saved → {save_path}\n")


def print_data_stats(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print detailed statistics about the actual dataset and model prediction distribution.
    Helps understand what the real data looks like and where model is biased.
    """
    sep  = "=" * 70
    sep2 = "-" * 70

    def _pct(num, den):
        return f"{num/den*100:.1f}%" if den > 0 else "  N/A "

    # ── Split actual data ──────────────────────────────────────────────────────
    up_mask   = y_true > 0
    down_mask = y_true < 0
    zero_mask = y_true == 0

    y_up   = y_true[up_mask]
    y_down = y_true[down_mask]

    n       = len(y_true)
    n_up    = int(up_mask.sum())
    n_down  = int(down_mask.sum())
    n_zero  = int(zero_mask.sum())

    # ── Move buckets ──────────────────────────────────────────────────────────
    # Buckets for absolute move size
    BUCKETS = [
        ("0.00 – 0.10%",  0.00, 0.10),
        ("0.10 – 0.25%",  0.10, 0.25),
        ("0.25 – 0.50%",  0.25, 0.50),
        ("0.50 – 1.00%",  0.50, 1.00),
        ("1.00 – 2.00%",  1.00, 2.00),
        ("2.00 – 5.00%",  2.00, 5.00),
        ("> 5.00%",        5.00, 9999),
    ]

    def _bucket_counts(arr):
        abs_arr = np.abs(arr)
        rows = []
        for label, lo, hi in BUCKETS:
            cnt = int(((abs_arr >= lo) & (abs_arr < hi)).sum())
            rows.append((label, cnt))
        return rows

    up_buckets   = _bucket_counts(y_up)
    down_buckets = _bucket_counts(y_down)

    # ── Model prediction distribution ─────────────────────────────────────────
    pred_up   = int((y_pred > 0).sum())
    pred_down = int((y_pred < 0).sum())
    pred_zero = int((y_pred == 0).sum())

    print(f"\n{sep}")
    print("  DATASET STATISTICS — Actual Data Distribution")
    print(sep)

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n  OVERALL")
    print(f"  {sep2}")
    print(f"  {'Total bars':<28} {n:>7,}")
    print(f"  {'Upside   (actual > 0)':<28} {n_up:>7,}  ({_pct(n_up, n)})")
    print(f"  {'Downside (actual < 0)':<28} {n_down:>7,}  ({_pct(n_down, n)})")
    print(f"  {'Zero     (actual = 0)':<28} {n_zero:>7,}  ({_pct(n_zero, n)})")
    print(f"  {'Mean actual return':<28} {y_true.mean():>+8.4f}%")
    print(f"  {'Median actual return':<28} {float(np.median(y_true)):>+8.4f}%")
    print(f"  {'Std deviation':<28} {y_true.std():>8.4f}%")
    print(f"  {'Max upside move':<28} {y_up.max() if n_up else 0:>+8.4f}%")
    print(f"  {'Max downside move':<28} {y_down.min() if n_down else 0:>+8.4f}%")

    # ── Move size distribution ─────────────────────────────────────────────────
    print(f"\n  MOVE SIZE DISTRIBUTION  (|actual return| bucketed)")
    print(f"  {'Bucket':<16} {'Upside':>10}  {'%Up':>7}  {'Downside':>10}  {'%Down':>7}")
    print(f"  {'-'*58}")
    for i, (label, lo, hi) in enumerate(BUCKETS):
        u_cnt = up_buckets[i][1]
        d_cnt = down_buckets[i][1]
        print(
            f"  {label:<16} {u_cnt:>10,}  {_pct(u_cnt, n_up):>7}  {d_cnt:>10,}  {_pct(d_cnt, n_down):>7}"
        )

    # ── Upside stats ──────────────────────────────────────────────────────────
    if n_up > 0:
        print(f"\n  UPSIDE BARS DETAIL  ({n_up:,} bars, {_pct(n_up, n)} of total)")
        print(f"  {sep2}")
        print(f"  {'Min upside move':<28} {y_up.min():>+8.4f}%")
        print(f"  {'Max upside move':<28} {y_up.max():>+8.4f}%")
        print(f"  {'Mean upside move':<28} {y_up.mean():>+8.4f}%")
        print(f"  {'Median upside move':<28} {float(np.median(y_up)):>+8.4f}%")
        print(f"  {'Std deviation':<28} {y_up.std():>8.4f}%")
        p25, p75, p95 = np.percentile(y_up, [25, 75, 95])
        print(f"  {'25th percentile':<28} {p25:>+8.4f}%")
        print(f"  {'75th percentile':<28} {p75:>+8.4f}%")
        print(f"  {'95th percentile':<28} {p95:>+8.4f}%")

    # ── Downside stats ────────────────────────────────────────────────────────
    if n_down > 0:
        print(f"\n  DOWNSIDE BARS DETAIL  ({n_down:,} bars, {_pct(n_down, n)} of total)")
        print(f"  {sep2}")
        print(f"  {'Min downside move (smallest)':<28} {y_down.max():>+8.4f}%")
        print(f"  {'Max downside move (largest)':<28} {y_down.min():>+8.4f}%")
        print(f"  {'Mean downside move':<28} {y_down.mean():>+8.4f}%")
        print(f"  {'Median downside move':<28} {float(np.median(y_down)):>+8.4f}%")
        print(f"  {'Std deviation':<28} {y_down.std():>8.4f}%")
        p25, p75, p95 = np.percentile(np.abs(y_down), [25, 75, 95])
        print(f"  {'25th percentile (abs)':<28} {p25:>8.4f}%")
        print(f"  {'75th percentile (abs)':<28} {p75:>8.4f}%")
        print(f"  {'95th percentile (abs)':<28} {p95:>8.4f}%")

    # ── Model prediction bias ─────────────────────────────────────────────────
    print(f"\n  MODEL PREDICTION BIAS")
    print(f"  {sep2}")
    print(f"  {'Model predicted UP   (pred > 0)':<32} {pred_up:>6,}  ({_pct(pred_up, n)})")
    print(f"  {'Model predicted DOWN (pred < 0)':<32} {pred_down:>6,}  ({_pct(pred_down, n)})")
    if pred_zero > 0:
        print(f"  {'Model predicted ZERO (pred = 0)':<32} {pred_zero:>6,}  ({_pct(pred_zero, n)})")
    print(f"  {'Mean predicted return':<32} {y_pred.mean():>+8.4f}%")
    print(f"  {'Median predicted return':<32} {float(np.median(y_pred)):>+8.4f}%")

    # ── Confusion at direction level ───────────────────────────────────────────
    print(f"\n  DIRECTION CONFUSION MATRIX  (actual → model)")
    print(f"  {sep2}")
    TP_up   = int(((y_true > 0) & (y_pred > 0)).sum())   # actual UP,   pred UP   ✓
    FN_up   = int(((y_true > 0) & (y_pred <= 0)).sum())  # actual UP,   pred DOWN ✗
    TP_down = int(((y_true < 0) & (y_pred < 0)).sum())   # actual DOWN, pred DOWN ✓
    FN_down = int(((y_true < 0) & (y_pred >= 0)).sum())  # actual DOWN, pred UP   ✗
    print(f"  {'':20} {'Pred UP':>10}  {'Pred DOWN':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Actual UP':<20} {TP_up:>10,}  {FN_up:>10,}   (hit: {_pct(TP_up, n_up)})")
    print(f"  {'Actual DOWN':<20} {FN_down:>10,}  {TP_down:>10,}   (hit: {_pct(TP_down, n_down)})")

    print(f"\n{sep}\n")


def print_evaluation(metrics: dict) -> None:
    sep  = "=" * 70
    sep2 = "-" * 70
    n    = metrics["n_total"]
    nm   = metrics["n_meaningful"]
    n_up = metrics["n_actual_up"]
    n_dn = metrics["n_actual_down"]

    def _pct(num, den):
        return f"{num/den*100:.1f}%" if den > 0 else "  N/A "

    print(f"\n{sep}")
    print("  MODEL EVALUATION — Max Return Prediction")
    print(sep)

    # ── Sample summary ────────────────────────────────────────────────────────
    print(f"\n  SAMPLE SUMMARY")
    print(f"  {'Total bars':<30} {n:>7,}")
    print(f"  {'Meaningful bars (|actual|≥0.01%)':<30} {nm:>7,}  ({_pct(nm, n)})")
    print(f"  {'  ↑ Actual UPSIDE bars':<30} {n_up:>7,}  ({_pct(n_up, nm)} of meaningful)")
    print(f"  {'  ↓ Actual DOWNSIDE bars':<30} {n_dn:>7,}  ({_pct(n_dn, nm)} of meaningful)")
    print(f"  {'Mean actual max_return':<30} {metrics['mean_actual_max_return']:>+8.3f}%")
    print(f"  {'Mean predicted return':<30} {metrics['mean_pred_max_return']:>+8.3f}%")

    # ── Regression error ──────────────────────────────────────────────────────
    print(f"\n  REGRESSION ERROR")
    print(f"  {'MAE':<30} {metrics['mae']:>8.4f}%")
    print(f"  {'RMSE':<30} {metrics['rmse']:>8.4f}%")

    # ── Overall direction accuracy ────────────────────────────────────────────
    da      = metrics["direction_accuracy"]
    n_corr  = metrics["n_correct"]
    print(f"\n  DIRECTION ACCURACY  (on meaningful bars)")
    print(f"  {da*100:.1f}%  —  {n_corr:,} correct out of {nm:,} bars")

    # ── Helper: print one direction block ─────────────────────────────────────
    def _print_side(label: str, stats: dict, n_actual: int):
        n_corr_s  = stats["n_correct"]
        n_miss_s  = stats["n_missed"]
        print(f"\n  {label}  (actual {label.lower()} bars: {n_actual:,})")
        print(f"  {sep2}")
        print(f"  {'Model correct direction':<32} {n_corr_s:>6,}  ({_pct(n_corr_s, n_actual)})")
        print(f"  {'Model wrong  direction (missed)':<32} {n_miss_s:>6,}  ({_pct(n_miss_s, n_actual)})")
        print()
        print(f"  {'Tier':<8} {'Signals':>8}  {'% of actual':>12}  {'% of correct':>13}")
        print(f"  {'-'*48}")
        for tier, key in (("≥ 25%", "n_25"), ("≥ 50%", "n_50"), ("≥ 75%", "n_75"), ("≥ 100%", "n_100")):
            cnt = stats[key]
            print(
                f"  {tier:<8} {cnt:>8,}  {_pct(cnt, n_actual):>12}  {_pct(cnt, n_corr_s):>13}"
            )

    # ── Upside block ──────────────────────────────────────────────────────────
    _print_side("UPSIDE ↑", metrics["upside"], n_up)

    # ── Downside block ────────────────────────────────────────────────────────
    _print_side("DOWNSIDE ↓", metrics["downside"], n_dn)

    # ── Overall achievement ───────────────────────────────────────────────────
    print(f"\n  OVERALL ACHIEVEMENT TIERS  (both directions combined)")
    print(f"  {sep2}")
    print(f"  {'Tier':<8} {'Signals':>8}  {'% of meaningful':>16}  {'% of correct':>13}")
    print(f"  {'-'*52}")
    for tier, key in (("≥ 25%", "n_25"), ("≥ 50%", "n_50"), ("≥ 75%", "n_75"), ("≥ 100%", "n_100")):
        cnt = metrics[key]
        print(
            f"  {tier:<8} {cnt:>8,}  {_pct(cnt, nm):>16}  {_pct(cnt, n_corr):>13}"
        )

    print(f"\n{sep}\n")
