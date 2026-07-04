# -*- coding: utf-8 -*-
"""
Direction + magnitude evaluation.

Answers, in plain terms:
  1. Model ne UP bola to kitni baar actually UP gaya? (direction accuracy)
  2. Confidence jitna zyada, accuracy utni zyada hai ya nahi? (calibration)
  3. Model ne "1% jayega" bola to actually kitna gaya? (magnitude achievement)
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

CONFIDENCE_BUCKETS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
MAGNITUDE_BUCKETS  = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, np.inf)]


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_batches(
    model,
    X: np.ndarray,            # (N, W, F) — already scaled
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over windows, return (p_up, up_mag, down_mag)."""
    model.eval()
    p_ups, ups, dns = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            dir_logit, up_mag, dn_mag = model(batch)
            p_ups.append(torch.sigmoid(dir_logit).cpu().numpy())
            ups.append(up_mag.cpu().numpy())
            dns.append(dn_mag.cpu().numpy())
    return np.concatenate(p_ups), np.concatenate(ups), np.concatenate(dns)


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_seq: np.ndarray,   # (N, W, F) — already scaled
    y_up:  np.ndarray,   # (N,) actual max upside move %
    y_dn:  np.ndarray,   # (N,) actual max downside move %
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    p_up, up_pred, dn_pred = predict_batches(model, X_seq, device, batch_size)

    actual_up  = y_up >= y_dn                 # actual dominant direction
    pred_up    = p_up >= 0.5
    correct    = pred_up == actual_up
    confidence = np.maximum(p_up, 1.0 - p_up)

    n = len(y_up)

    # ── Direction scorecard ───────────────────────────────────────────────────
    n_pred_up   = int(pred_up.sum())
    n_pred_down = n - n_pred_up
    direction = {
        "n_total":            n,
        "accuracy":           float(correct.mean()),
        "base_rate_up":       float(actual_up.mean()),
        "n_pred_up":          n_pred_up,
        "n_pred_up_correct":  int((pred_up & actual_up).sum()),
        "n_pred_down":        n_pred_down,
        "n_pred_down_correct": int((~pred_up & ~actual_up).sum()),
    }

    # ── Accuracy by confidence bucket ─────────────────────────────────────────
    conf_stats = []
    for lo, hi in CONFIDENCE_BUCKETS:
        mask = (confidence >= lo) & (confidence < hi)
        cnt  = int(mask.sum())
        conf_stats.append({
            "bucket":   f"{lo:.2f}–{min(hi, 1.0):.2f}",
            "n":        cnt,
            "pct_bars": cnt / n,
            "accuracy": float(correct[mask].mean()) if cnt > 0 else None,
        })

    # ── Calibration: predicted P(up) vs actual UP frequency ──────────────────
    calib = []
    bins  = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_up >= lo) & (p_up < hi) if hi < 1.0 else (p_up >= lo)
        cnt  = int(mask.sum())
        if cnt < 20:
            continue
        calib.append({
            "bucket":         f"{lo:.1f}–{hi:.1f}",
            "n":              cnt,
            "mean_p_up":      float(p_up[mask].mean()),
            "actual_up_rate": float(actual_up[mask].mean()),
        })

    # ── Magnitude achievement in the PREDICTED direction ─────────────────────
    # "Model ne X% bola tha, actual kitna mila?"
    pred_mag     = np.where(pred_up, up_pred, dn_pred)
    achieved_mag = np.where(pred_up, y_up, y_dn)
    mag_stats = []
    for lo, hi in MAGNITUDE_BUCKETS:
        mask = (pred_mag >= lo) & (pred_mag < hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            mag_stats.append({"bucket": f"{lo:.1f}–{hi if np.isfinite(hi) else '+'}", "n": 0})
            continue
        pm, am = pred_mag[mask], achieved_mag[mask]
        mag_stats.append({
            "bucket":          f"{lo:.1f}–{f'{hi:.1f}' if np.isfinite(hi) else '∞'}%",
            "n":               cnt,
            "mean_predicted":  float(pm.mean()),
            "median_achieved": float(np.median(am)),
            "mean_achieved":   float(am.mean()),
            "hit_full_pct":    float((am >= pm).mean()),          # achieved >= predicted
            "hit_80_pct":      float((am >= 0.8 * pm).mean()),    # achieved >= 80% of predicted
        })

    return {
        "direction":     direction,
        "confidence":    conf_stats,
        "calibration":   calib,
        "magnitude":     mag_stats,
        "mae_up":        float(np.abs(up_pred - y_up).mean()),
        "mae_down":      float(np.abs(dn_pred - y_dn).mean()),
        "_p_up":         p_up,
        "_up_pred":      up_pred,
        "_dn_pred":      dn_pred,
        "_y_up":         y_up,
        "_y_dn":         y_dn,
        "_confidence":   confidence,
        "_correct":      correct,
        "_pred_mag":     pred_mag,
        "_achieved_mag": achieved_mag,
        "_pred_up":      pred_up,
        "_actual_up":    actual_up,
    }


# ── Console report ────────────────────────────────────────────────────────────

def _pct(x) -> str:
    return f"{x*100:.1f}%" if x is not None else "—"


def print_evaluation(metrics: dict, title: str = "EVALUATION") -> None:
    d = metrics["direction"]

    console.rule(f"[bold cyan]{title}  ({d['n_total']:,} bars)[/bold cyan]")

    # ── 1. Direction scorecard ────────────────────────────────────────────────
    t = Table(title="[bold]1) DIRECTION — model ne bola vs actual kya hua[/bold]",
              box=box.ROUNDED)
    t.add_column("Model ne bola", style="bold")
    t.add_column("Kitni baar bola", justify="right")
    t.add_column("Kitni baar sahi", justify="right")
    t.add_column("Accuracy", justify="right", style="bold")
    t.add_row("UP ⬆",
              f"{d['n_pred_up']:,}",
              f"{d['n_pred_up_correct']:,}",
              _pct(d['n_pred_up_correct'] / d['n_pred_up']) if d['n_pred_up'] else "—")
    t.add_row("DOWN ⬇",
              f"{d['n_pred_down']:,}",
              f"{d['n_pred_down_correct']:,}",
              _pct(d['n_pred_down_correct'] / d['n_pred_down']) if d['n_pred_down'] else "—")
    t.add_row("[bold]OVERALL[/bold]", f"{d['n_total']:,}",
              f"{d['n_pred_up_correct'] + d['n_pred_down_correct']:,}",
              f"[bold]{_pct(d['accuracy'])}[/bold]")
    console.print(t)
    console.print(
        f"  [dim]Base rate (hamesha UP bolne wala dummy): {_pct(max(d['base_rate_up'], 1-d['base_rate_up']))} "
        f"— model isse upar hai to hi useful hai.[/dim]\n"
    )

    # ── 2. Confidence buckets ────────────────────────────────────────────────
    t = Table(title="[bold]2) CONFIDENCE — jitna sure, utna sahi?[/bold]", box=box.ROUNDED)
    t.add_column("Confidence", style="bold")
    t.add_column("Bars", justify="right")
    t.add_column("% of all", justify="right")
    t.add_column("Direction accuracy", justify="right", style="bold")
    for s in metrics["confidence"]:
        t.add_row(s["bucket"], f"{s['n']:,}", _pct(s["pct_bars"]), _pct(s["accuracy"]))
    console.print(t)
    console.print("  [dim]High-confidence bucket me accuracy zyada honi chahiye — tabhi confidence trustable hai.[/dim]\n")

    # ── 3. Calibration ────────────────────────────────────────────────────────
    t = Table(title="[bold]3) CALIBRATION — P(up) bola vs actually UP hua[/bold]", box=box.ROUNDED)
    t.add_column("P(up) range", style="bold")
    t.add_column("Bars", justify="right")
    t.add_column("Avg P(up) predicted", justify="right")
    t.add_column("Actually UP gaya", justify="right", style="bold")
    for s in metrics["calibration"]:
        t.add_row(s["bucket"], f"{s['n']:,}", _pct(s["mean_p_up"]), _pct(s["actual_up_rate"]))
    console.print(t)
    console.print("  [dim]Dono columns match karein = perfectly calibrated. (60% bola to 60% baar UP jana chahiye)[/dim]\n")

    # ── 4. Magnitude achievement ──────────────────────────────────────────────
    t = Table(title="[bold]4) MAGNITUDE — kitna % move bola vs kitna mila (predicted direction me)[/bold]",
              box=box.ROUNDED)
    t.add_column("Predicted move", style="bold")
    t.add_column("Bars", justify="right")
    t.add_column("Avg predicted", justify="right")
    t.add_column("Median achieved", justify="right")
    t.add_column("Full target hit", justify="right")
    t.add_column("80% target hit", justify="right", style="bold")
    for s in metrics["magnitude"]:
        if s["n"] == 0:
            t.add_row(s["bucket"], "0", "—", "—", "—", "—")
            continue
        t.add_row(
            s["bucket"], f"{s['n']:,}",
            f"{s['mean_predicted']:.2f}%",
            f"{s['median_achieved']:.2f}%",
            _pct(s["hit_full_pct"]),
            _pct(s["hit_80_pct"]),
        )
    console.print(t)
    console.print(
        f"  [dim]Magnitude MAE — upside: {metrics['mae_up']:.3f}%  |  downside: {metrics['mae_down']:.3f}%[/dim]\n"
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_predictions(metrics: dict, save_path: str = "models/prediction_analysis.png") -> None:
    """
    4-panel figure:
      1. Calibration curve (predicted P(up) vs actual UP rate)
      2. Direction accuracy by confidence bucket
      3. Scatter: predicted vs achieved magnitude (predicted direction)
      4. Distribution of achieved move: correct vs wrong direction calls
    """
    DARK, PANEL = "#0e1117", "#1a1f2e"
    GREEN, RED, BLUE, ORANGE, GRAY = "#00d26a", "#ff4757", "#3d84ff", "#ffa502", "#8a8f9e"

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle("Direction + Magnitude — Model Quality", fontsize=16, color="white",
                 fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    ax1, ax2, ax3, ax4 = axes

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3e")

    # ── Panel 1: Calibration ──────────────────────────────────────────────────
    calib = metrics["calibration"]
    if calib:
        xs = [c["mean_p_up"] for c in calib]
        ys = [c["actual_up_rate"] for c in calib]
        ax1.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=1, label="Perfect")
        ax1.plot(xs, ys, color=GREEN, marker="o", linewidth=1.5, label="Model")
    ax1.set_xlabel("Predicted P(up)", color=GRAY, fontsize=10)
    ax1.set_ylabel("Actual UP rate", color=GRAY, fontsize=10)
    ax1.set_title("Calibration", color="white", fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 2: Accuracy by confidence ───────────────────────────────────────
    conf   = [c for c in metrics["confidence"] if c["accuracy"] is not None]
    labels = [c["bucket"] for c in conf]
    accs   = [c["accuracy"] * 100 for c in conf]
    bars   = ax2.bar(labels, accs, color=BLUE, alpha=0.85)
    for b, a in zip(bars, accs):
        ax2.text(b.get_x() + b.get_width() / 2, a + 0.5, f"{a:.1f}%",
                 ha="center", color="white", fontsize=9)
    ax2.axhline(50, color=RED, linewidth=0.8, linestyle="--", label="Coin flip (50%)")
    ax2.set_ylabel("Direction accuracy %", color=GRAY, fontsize=10)
    ax2.set_xlabel("Confidence bucket", color=GRAY, fontsize=10)
    ax2.set_title("Accuracy vs Confidence", color="white", fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 3: Predicted vs achieved magnitude ─────────────────────────────
    pm, am, ok = metrics["_pred_mag"], metrics["_achieved_mag"], metrics["_correct"]
    ax3.scatter(pm[ok],  am[ok],  c=GREEN, alpha=0.30, s=5, label="Direction correct")
    ax3.scatter(pm[~ok], am[~ok], c=RED,   alpha=0.20, s=5, label="Direction wrong")
    lim = max(pm.max(), np.percentile(am, 99)) * 1.05
    ax3.plot([0, lim], [0, lim], color=ORANGE, linewidth=0.8, linestyle=":", label="Achieved = Predicted")
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.set_xlabel("Predicted move %", color=GRAY, fontsize=10)
    ax3.set_ylabel("Achieved move % (predicted direction)", color=GRAY, fontsize=10)
    ax3.set_title("Magnitude: Predicted vs Achieved", color="white", fontsize=11, pad=8)
    ax3.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    # ── Panel 4: Achieved move distribution ──────────────────────────────────
    bins = np.linspace(0, np.percentile(am, 99), 50)
    ax4.hist(am[ok],  bins=bins, color=GREEN, alpha=0.6, label="Correct calls",  density=True)
    ax4.hist(am[~ok], bins=bins, color=RED,   alpha=0.6, label="Wrong calls", density=True)
    ax4.set_xlabel("Achieved move % (predicted direction)", color=GRAY, fontsize=10)
    ax4.set_ylabel("Density", color=GRAY, fontsize=10)
    ax4.set_title("Move Distribution: Correct vs Wrong Direction", color="white", fontsize=11, pad=8)
    ax4.legend(fontsize=8, facecolor="#2a2f3e", edgecolor="none", labelcolor="white")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    console.print(f"  [dim]Graph → {save_path}[/dim]\n")
