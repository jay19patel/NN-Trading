# -*- coding: utf-8 -*-
"""Static PNG diagnostics for training, validation, and paper trading results."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BG = (18, 24, 33)
PANEL = (27, 36, 48)
GRID = (61, 73, 88)
TEXT = (232, 238, 245)
MUTED = (148, 163, 184)
GREEN = (45, 212, 146)
RED = (248, 113, 113)
AMBER = (251, 191, 36)
BLUE = (96, 165, 250)
PURPLE = (167, 139, 250)


def _font(size: int = 18, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((32, 24), title, fill=TEXT, font=_font(28, bold=True))
    return image, draw


def _save(image: Image.Image, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path


def _safe_range(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < 1e-9:
        pad = max(abs(hi) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _plot_line(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    color: tuple[int, int, int],
    label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=PANEL, outline=GRID)
    if len(values) < 2:
        draw.text((x0 + 16, y0 + 16), f"{label}: insufficient points", fill=MUTED, font=_font(16))
        return
    if y_min is None or y_max is None:
        y_min, y_max = _safe_range(values)
    for i in range(5):
        yy = y0 + int((y1 - y0) * i / 4)
        draw.line((x0, yy, x1, yy), fill=GRID)
    points = []
    for i, value in enumerate(values):
        x = x0 + int((x1 - x0) * i / max(len(values) - 1, 1))
        ratio = (float(value) - y_min) / max(y_max - y_min, 1e-9)
        y = y1 - int((y1 - y0) * ratio)
        points.append((x, y))
    draw.line(points, fill=color, width=3)
    draw.text((x0 + 14, y0 + 10), label, fill=color, font=_font(17, bold=True))
    draw.text((x0 + 14, y1 - 28), f"min {min(values):.4f}  max {max(values):.4f}", fill=MUTED, font=_font(14))


def save_training_history_chart(history: list[dict[str, Any]], output_dir: str) -> str | None:
    if not history:
        return None
    image, draw = _canvas(1200, 780, "Training History")
    epochs = [int(row.get("epoch", i + 1)) for i, row in enumerate(history)]
    train_loss = [float(row.get("train_loss", np.nan)) for row in history]
    val_loss = [float(row.get("val_loss", np.nan)) for row in history]
    val_acc = [float(row.get("val_direction_accuracy_pct", np.nan)) for row in history]
    loss_min, loss_max = _safe_range(train_loss + val_loss)
    _plot_line(draw, (70, 100, 1130, 390), train_loss, BLUE, "Train loss", loss_min, loss_max)
    _plot_line(draw, (70, 100, 1130, 390), val_loss, AMBER, "Val loss", loss_min, loss_max)
    _plot_line(draw, (70, 460, 1130, 710), val_acc, GREEN, "Validation direction accuracy (%)", 0.0, 100.0)
    draw.text((70, 730), f"Epochs completed: {epochs[-1]}", fill=MUTED, font=_font(16))
    return _save(image, output_dir, "training_history.png")


def save_confusion_matrix_chart(matrix: list[list[int]], output_dir: str) -> str | None:
    if not matrix:
        return None
    arr = np.asarray(matrix, dtype=float)
    labels = ["Buy", "Neutral", "Sell"]
    image, draw = _canvas(920, 760, "Held-out Test Confusion Matrix")
    x0, y0 = 220, 150
    cell = 150
    max_val = max(float(arr.max()), 1.0)
    for i, true_label in enumerate(labels):
        draw.text((80, y0 + i * cell + 58), true_label, fill=TEXT, font=_font(19, bold=True))
    for j, pred_label in enumerate(labels):
        draw.text((x0 + j * cell + 38, 105), pred_label, fill=TEXT, font=_font(19, bold=True))
    draw.text((32, 410), "Actual", fill=MUTED, font=_font(18, bold=True))
    draw.text((480, 70), "Predicted", fill=MUTED, font=_font(18, bold=True))
    for i in range(3):
        row_total = max(arr[i].sum(), 1.0)
        for j in range(3):
            intensity = int(45 + 170 * (arr[i, j] / max_val))
            color = (30, min(90 + intensity // 2, 210), min(110 + intensity, 255))
            box = (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell)
            draw.rectangle(box, fill=color, outline=BG, width=3)
            pct = arr[i, j] / row_total * 100.0
            draw.text((box[0] + 42, box[1] + 46), f"{int(arr[i, j])}", fill=TEXT, font=_font(26, bold=True))
            draw.text((box[0] + 43, box[1] + 82), f"{pct:.1f}%", fill=TEXT, font=_font(16))
    return _save(image, output_dir, "confusion_matrix.png")


def save_metrics_bar_chart(metrics: dict[str, Any], output_dir: str) -> str:
    keys = [
        ("direction_accuracy_all", "Accuracy"),
        ("balanced_accuracy", "Balanced acc"),
        ("f1_macro", "F1 macro"),
        ("f1_weighted", "F1 weighted"),
        ("majority_class_baseline_accuracy", "Majority baseline"),
    ]
    image, draw = _canvas(1100, 680, "Held-out Test Metrics")
    x0, y0, width, row_h = 360, 140, 620, 72
    for idx, (key, label) in enumerate(keys):
        value = float(metrics.get(key, 0.0) or 0.0)
        y = y0 + idx * row_h
        draw.text((70, y + 12), label, fill=TEXT, font=_font(20, bold=True))
        draw.rectangle((x0, y + 14, x0 + width, y + 44), fill=PANEL, outline=GRID)
        bar_w = int(width * max(0.0, min(value, 1.0)))
        color = GREEN if key != "majority_class_baseline_accuracy" else PURPLE
        draw.rectangle((x0, y + 14, x0 + bar_w, y + 44), fill=color)
        draw.text((x0 + width + 18, y + 10), f"{value * 100:.1f}%", fill=TEXT, font=_font(19, bold=True))
    draw.text((70, 570), "If model bars do not beat the majority baseline, direction accuracy is not yet a tradable edge.", fill=MUTED, font=_font(18))
    return _save(image, output_dir, "test_metrics.png")


def save_signal_distribution_chart(predictions: pd.DataFrame, targets: dict[str, np.ndarray], output_dir: str) -> str:
    image, draw = _canvas(1100, 720, "Signal Distribution and Confidence")
    labels = ["Buy", "Neutral", "Sell"]
    actual = pd.Series(targets["direction"]).value_counts().reindex([0, 1, 2], fill_value=0).values
    raw_pred = predictions["ai_raw_verdict"].value_counts().reindex([0, 1, 2], fill_value=0).values if "ai_raw_verdict" in predictions else predictions["ai_verdict"].value_counts().reindex([0, 1, 2], fill_value=0).values
    final_pred = predictions["ai_verdict"].value_counts().reindex([0, 1, 2], fill_value=0).values
    max_count = max(int(max(actual.max(), raw_pred.max(), final_pred.max())), 1)
    groups = [("Actual labels", actual, BLUE), ("Raw model", raw_pred, AMBER), ("After risk filters", final_pred, GREEN)]
    for g, (name, values, color) in enumerate(groups):
        base_y = 135 + g * 160
        draw.text((70, base_y), name, fill=TEXT, font=_font(20, bold=True))
        for i, label in enumerate(labels):
            y = base_y + 42 + i * 34
            draw.text((90, y - 4), label, fill=MUTED, font=_font(16))
            draw.rectangle((210, y, 970, y + 22), fill=PANEL, outline=GRID)
            w = int(760 * int(values[i]) / max_count)
            draw.rectangle((210, y, 210 + w, y + 22), fill=color)
            draw.text((985, y - 4), str(int(values[i])), fill=TEXT, font=_font(16, bold=True))
    conf = predictions["ai_confidence"].astype(float)
    draw.text((70, 640), f"Confidence: mean={conf.mean():.3f}, max={conf.max():.3f}. Low confidence is why strict trading can block all entries.", fill=MUTED, font=_font(17))
    return _save(image, output_dir, "signal_distribution.png")


def save_equity_curve_chart(equity_curves: dict[str, dict[str, Any]], output_dir: str, filename: str = "equity_curve.png") -> str:
    image, draw = _canvas(1200, 720, "Paper Trading Equity Curve")
    box = (80, 130, 1130, 620)
    all_values: list[float] = []
    for payload in equity_curves.values():
        all_values.extend([float(v) for v in payload.get("equity", [])])
    y_min, y_max = _safe_range(all_values or [1000.0])
    colors = [BLUE, GREEN, AMBER, PURPLE, RED]
    for idx, (symbol, payload) in enumerate(equity_curves.items()):
        values = [float(v) for v in payload.get("equity", [])]
        _plot_line(draw, box, values, colors[idx % len(colors)], symbol, y_min, y_max)
        draw.text((90 + idx * 190, 645), f"{symbol}: ${values[-1]:.2f}" if values else symbol, fill=colors[idx % len(colors)], font=_font(17, bold=True))
    return _save(image, output_dir, filename)


def save_trade_pnl_chart(trades: list[Any], output_dir: str, filename: str = "trade_pnl.png") -> str:
    image, draw = _canvas(1200, 700, "Trade PnL Diagnostics")
    pnls = [float(getattr(t, "pnl_net_usd", 0.0)) for t in trades]
    if not pnls:
        draw.rectangle((70, 140, 1130, 560), fill=PANEL, outline=GRID)
        draw.text((120, 315), "No trades passed the selected filters.", fill=TEXT, font=_font(28, bold=True))
        draw.text((120, 360), "This is a safety result: validation did not prove positive expectancy.", fill=MUTED, font=_font(20))
        return _save(image, output_dir, filename)
    lo, hi = _safe_range(pnls)
    zero_y = 560 - int((560 - 140) * ((0 - lo) / max(hi - lo, 1e-9)))
    draw.rectangle((70, 140, 1130, 560), fill=PANEL, outline=GRID)
    draw.line((70, zero_y, 1130, zero_y), fill=MUTED, width=2)
    bar_w = max(8, int(1000 / max(len(pnls), 1)))
    for i, pnl in enumerate(pnls):
        x = 90 + i * bar_w
        y = 560 - int((560 - 140) * ((pnl - lo) / max(hi - lo, 1e-9)))
        color = GREEN if pnl >= 0 else RED
        draw.rectangle((x, min(y, zero_y), x + max(bar_w - 2, 2), max(y, zero_y)), fill=color)
    draw.text((80, 610), f"Trades={len(pnls)}  Net=${sum(pnls):.2f}  Avg=${np.mean(pnls):.2f}", fill=TEXT, font=_font(20, bold=True))
    return _save(image, output_dir, filename)


def save_feature_relevance_chart(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str,
    top_n: int = 25,
) -> tuple[str | None, str | None]:
    if "direction_label" not in dataframe.columns or not feature_columns:
        return None, None
    relevance_rows = []
    target = dataframe["direction_label"].astype(float)
    for column in feature_columns:
        if column not in dataframe.columns:
            continue
        series = dataframe[column].replace([np.inf, -np.inf], np.nan)
        valid = series.notna() & target.notna()
        if valid.sum() < 50 or series[valid].std() == 0:
            continue
        corr = series[valid].corr(target[valid], method="spearman")
        if pd.isna(corr):
            continue
        relevance_rows.append(
            {
                "feature": column,
                "spearman_abs": abs(float(corr)),
                "spearman_signed": float(corr),
            }
        )
    if not relevance_rows:
        return None, None

    relevance = pd.DataFrame(relevance_rows).sort_values("spearman_abs", ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "feature_relevance.csv")
    relevance.to_csv(csv_path, index=False)

    top = relevance.head(top_n).iloc[::-1]
    image, draw = _canvas(1200, 880, "Curated Feature Relevance")
    x0, y0, x1 = 390, 120, 1080
    row_h = 26
    max_score = max(float(top["spearman_abs"].max()), 1e-6)
    draw.text((70, 75), f"Top {len(top)} by absolute Spearman correlation vs direction label", fill=MUTED, font=_font(17))
    for idx, row in enumerate(top.itertuples(index=False)):
        y = y0 + idx * row_h
        feature = str(row.feature)
        score = float(row.spearman_abs)
        signed = float(row.spearman_signed)
        color = GREEN if signed >= 0 else RED
        draw.text((70, y - 2), feature[:30], fill=TEXT, font=_font(15))
        draw.rectangle((x0, y, x1, y + 16), fill=PANEL, outline=GRID)
        bar_w = int((x1 - x0) * score / max_score)
        draw.rectangle((x0, y, x0 + bar_w, y + 16), fill=color)
        draw.text((x1 + 15, y - 3), f"{score:.4f}", fill=TEXT, font=_font(14, bold=True))
    draw.text((70, 820), "Low values mean the OHLCV indicators have weak standalone relation to future labels.", fill=MUTED, font=_font(17))
    image_path = _save(image, output_dir, "feature_relevance.png")
    return image_path, csv_path


def save_diagnostic_summary(payload: dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "diagnostic_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
