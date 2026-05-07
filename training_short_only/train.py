# -*- coding: utf-8 -*-
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import bars_per_day, config
from engine.data_handler import fetch_data
from strategies.oracle import OracleStrategy
from training_short_only.backtest_short import run_short_backtest
from training_short_only.feature_utils import add_technical_indicators, get_feature_columns
from training_short_only.model import MultiHeadTradingModel, TradingLoss
from ui_utils import console


class TradingDataset(Dataset):
    def __init__(self, X_seq, y_dir, y_qty, y_tp, y_sl, y_pnl, device="cpu"):
        self.X = torch.from_numpy(X_seq).float().to(device)
        self.y_dir = torch.from_numpy(y_dir).long().to(device)
        self.y_qty = torch.from_numpy(y_qty).float().to(device)
        self.y_tp = torch.from_numpy(y_tp).float().to(device)
        self.y_sl = torch.from_numpy(y_sl).float().to(device)
        self.y_pnl = torch.from_numpy(y_pnl).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            {
                "direction": self.y_dir[idx],
                "qty_ratio": self.y_qty[idx],
                "take_profit_pct": self.y_tp[idx],
                "stop_loss_pct": self.y_sl[idx],
                "actual_pnl_pct": self.y_pnl[idx],
            },
        )


def _prepare_symbol_frame(symbol: str, total_days_to_fetch: int, interval: str, feature_cols: list[str]) -> pd.DataFrame:
    cache_path = f"data/processed_train_{symbol}_{interval}.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = fetch_data(symbol, total_days_to_fetch, interval)
        if df.empty:
            return df

    required_cols = set(["Open", "High", "Low", "Close", "Volume"])
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{symbol} cache is missing OHLCV columns: {required_cols - set(df.columns)}")

    oracle = OracleStrategy()
    df = oracle.generate_signals(df.copy())
    df["direction_label"] = df["ai_verdict"]
    df["qty_ratio"] = df["ai_qty_ratio"]
    df["take_profit_pct"] = df["ai_take_profit_pct"]
    df["stop_loss_pct"] = df["ai_stop_loss_pct"]
    df["expected_return_pct"] = df.get("ai_expected_return_pct", 0.0)
    df = add_technical_indicators(df)

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature generation missing columns: {missing_features}")

    os.makedirs("data", exist_ok=True)
    df.to_parquet(cache_path)
    return df.sort_index()


def _ordered_split(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars_day = bars_per_day(interval)
    val_bars = int(config.training.VALIDATION_DATA_DAYS * bars_day)
    test_bars = int(config.training.TEST_DATA_DAYS * bars_day)
    purge_bars = int(config.features.PURGE_BARS)
    min_train_bars = config.model.WINDOW_SIZE + 10

    if len(df) <= min_train_bars + val_bars + test_bars:
        train_end = max(int(len(df) * 0.70), min_train_bars)
        val_end = max(train_end + 1, int(len(df) * 0.85))
        return (
            df.iloc[:max(train_end - purge_bars, min_train_bars)].copy(),
            df.iloc[min(train_end + purge_bars, val_end):val_end].copy(),
            df.iloc[min(val_end + purge_bars, len(df)):].copy(),
        )

    test_start = len(df) - test_bars
    val_start = test_start - val_bars
    train_end = max(val_start - purge_bars, min_train_bars)
    val_begin = min(val_start + purge_bars, test_start)
    val_end = max(test_start - purge_bars, val_begin)
    test_begin = min(test_start + purge_bars, len(df))
    return df.iloc[:train_end].copy(), df.iloc[val_begin:val_end].copy(), df.iloc[test_begin:].copy()


def _walk_forward_splits(df: pd.DataFrame, interval: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    bars_day = bars_per_day(interval)
    val_bars = int(config.training.WALK_FORWARD_VAL_DAYS * bars_day)
    purge_bars = int(config.features.PURGE_BARS)
    folds = max(int(config.training.WALK_FORWARD_FOLDS), 1)
    min_train_bars = config.model.WINDOW_SIZE + 10
    test_bars = int(config.training.TEST_DATA_DAYS * bars_day)
    usable_end = max(len(df) - test_bars, min_train_bars)
    if usable_end <= min_train_bars + purge_bars + val_bars:
        train_df, val_df, _ = _ordered_split(df, interval)
        return [(train_df, val_df)]

    fold_ends = np.linspace(min_train_bars + purge_bars + val_bars, usable_end, folds, dtype=int)
    out = []
    for fold_end in fold_ends:
        val_end = int(fold_end)
        val_start = max(val_end - val_bars, min_train_bars + purge_bars)
        train_end = max(val_start - purge_bars, min_train_bars)
        val_begin = min(val_start + purge_bars, val_end)
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_begin:val_end].copy()
        if len(train_df) > config.model.WINDOW_SIZE and len(val_df) > config.model.WINDOW_SIZE:
            out.append((train_df, val_df))
    return out


def _make_sequences(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> tuple[np.ndarray, ...]:
    if len(df) <= config.model.WINDOW_SIZE:
        empty_x = np.empty((0, config.model.WINDOW_SIZE, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_x, empty_y.astype(np.int64), empty_y, empty_y, empty_y, empty_y

    X_scaled = scaler.transform(df[feature_cols].values)
    from numpy.lib.stride_tricks import sliding_window_view

    X_seq = sliding_window_view(X_scaled[:-1], (config.model.WINDOW_SIZE, X_scaled.shape[1])).squeeze(1)
    raw_dir = df["direction_label"].values[config.model.WINDOW_SIZE:]
    y_dir = raw_dir.astype(np.int64)

    y_tp = df["take_profit_pct"].values[config.model.WINDOW_SIZE:].astype(np.float32)
    y_sl = df["stop_loss_pct"].values[config.model.WINDOW_SIZE:].astype(np.float32)
    y_qty = (y_dir != 1).astype(np.float32)
    y_pnl = df.get("expected_return_pct", pd.Series(0.0, index=df.index)).values[config.model.WINDOW_SIZE:].astype(np.float32)
    return X_seq.astype(np.float32), y_dir, y_qty, y_tp, y_sl, y_pnl


def _concat_parts(parts: list[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    non_empty = [p for p in parts if len(p[0])]
    if non_empty:
        return tuple(np.concatenate([p[i] for p in non_empty], axis=0) for i in range(6))
    feature_count = len(get_feature_columns())
    empty_x = np.empty((0, config.model.WINDOW_SIZE, feature_count), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.float32)
    return empty_x, empty_y.astype(np.int64), empty_y, empty_y, empty_y, empty_y


def _label_diagnostics(df_by_symbol: dict[str, pd.DataFrame]) -> dict:
    out: dict[str, dict] = {}
    label_names = {0: "long", 1: "neutral", 2: "short"}
    for symbol, df in df_by_symbol.items():
        counts = df["direction_label"].value_counts().sort_index()
        out[symbol] = {
            "class_distribution": {label_names.get(int(k), str(k)): int(v) for k, v in counts.items()},
            "avg_expected_return_pct_by_class": {
                label_names.get(int(k), str(k)): float(v)
                for k, v in df.groupby("direction_label")["expected_return_pct"].mean().items()
            },
            "tp_sl_bucket_counts": {
                f"tp={tp:.2f}|sl={sl:.2f}": int(count)
                for (tp, sl), count in df[df["direction_label"] != 1]
                .groupby(["take_profit_pct", "stop_loss_pct"])
                .size()
                .items()
            },
        }
        if isinstance(df.index, pd.DatetimeIndex):
            monthly = df.groupby([df.index.to_period("M"), "direction_label"]).size().unstack(fill_value=0)
            out[symbol]["monthly_class_distribution"] = {
                str(idx): {label_names.get(int(k), str(k)): int(v) for k, v in row.items()}
                for idx, row in monthly.iterrows()
            }
    return out


@torch.no_grad()
def _predict(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int = 2048) -> dict[str, np.ndarray]:
    model.eval()
    if len(X) == 0:
        return {"probs": np.empty((0, 3), dtype=np.float32), "sizing": np.empty((0, 3), dtype=np.float32)}
    probs, sizing = [], []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        outputs = model(batch)
        probs.append(torch.softmax(outputs["direction"], dim=1).cpu().numpy())
        sizing.append(outputs["sizing"].cpu().numpy())
    return {"probs": np.concatenate(probs, axis=0), "sizing": np.concatenate(sizing, axis=0)}


def _classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    long_precision, long_recall, long_f1, _ = precision_recall_fscore_support(
        y_true == 0, pred == 0, average="binary", zero_division=0
    )
    short_precision, short_recall, short_f1, _ = precision_recall_fscore_support(
        y_true == 2, pred == 2, average="binary", zero_division=0
    )
    return {
        "accuracy": float((pred == y_true).mean()) if len(y_true) else 0.0,
        "long_precision": float(long_precision),
        "long_recall": float(long_recall),
        "long_f1": float(long_f1),
        "short_precision": float(short_precision),
        "short_recall": float(short_recall),
        "short_f1": float(short_f1),
        "macro_f1_trading_sides": float((long_f1 + short_f1) / 2.0),
        "confusion_matrix_labels_0_long_1_neutral_2_short": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
    }


def _threshold_search_for_side(y_true: np.ndarray, y_tp: np.ndarray, y_sl: np.ndarray, probs: np.ndarray, side_label: int) -> dict:
    best = {"score": -1e9, "threshold": config.strategy.AI_CONFIDENCE_THRESHOLD}
    rows = []
    min_trades = int(config.training.THRESHOLD_MIN_TRADES)
    for threshold in np.arange(0.35, 0.56, 0.025):
        p_side = probs[:, side_label]
        edge = p_side - probs[:, 1]
        selected = (p_side >= threshold) & (edge >= config.strategy.MIN_DIRECTIONAL_EDGE)
        if not selected.any():
            rows.append({"threshold": float(threshold), "trade_count": 0})
            continue

        wins = selected & (y_true == side_label)
        losses = selected & (y_true != side_label)
        gross_win = float(np.sum(y_tp[wins]))
        gross_loss = float(np.sum(y_sl[losses]))
        fee_drag = float(selected.sum() * (config.strategy.ROUND_TRIP_FEE_PCT + 2.0 * config.strategy.SLIPPAGE_PCT))
        net_expectancy_pct = (gross_win - gross_loss - fee_drag) / max(int(selected.sum()), 1)
        profit_factor = gross_win / max(gross_loss + fee_drag, 1e-9)
        win_rate = float(wins.sum() / selected.sum())
        row = {
            "threshold": float(threshold),
            "trade_count": int(selected.sum()),
            "win_rate": win_rate,
            "profit_factor_proxy": float(profit_factor),
            "net_expectancy_pct_proxy": float(net_expectancy_pct),
        }
        rows.append(row)
        trade_count = int(selected.sum())
        sample_penalty = min(trade_count / max(min_trades, 1), 1.0)
        score = net_expectancy_pct * sample_penalty
        if trade_count < min_trades:
            score -= 1.0
        if score > best["score"]:
            best = {**row, "score": float(score)}
    return {"best": best, "grid": rows}


def _threshold_search(y_true: np.ndarray, y_tp: np.ndarray, y_sl: np.ndarray, probs: np.ndarray) -> dict:
    long_search = _threshold_search_for_side(y_true, y_tp, y_sl, probs, side_label=0)
    short_search = _threshold_search_for_side(y_true, y_tp, y_sl, probs, side_label=2)
    combined_score = (
        long_search["best"].get("score", -1e9)
        + short_search["best"].get("score", -1e9)
    ) / 2.0
    return {
        "long": long_search,
        "short": short_search,
        "best": {
            "long_threshold": long_search["best"].get("threshold", config.strategy.LONG_CONFIDENCE_THRESHOLD),
            "short_threshold": short_search["best"].get("threshold", config.strategy.SHORT_CONFIDENCE_THRESHOLD),
            "score": float(combined_score),
        },
    }


def _model_selection_score(val_thresholds: dict, val_metrics: dict, avg_val_loss: float) -> float:
    threshold_score = float(val_thresholds["best"].get("score", -1e9))
    if threshold_score > -1e8:
        return threshold_score

    # Early epochs can be too uncertain to pass the production threshold grid.
    # Keep checkpointing those epochs by falling back to classification quality.
    return float(val_metrics["macro_f1_trading_sides"] - avg_val_loss)


def train_short_model():
    symbols = config.data.SYMBOLS
    interval = config.data.INTERVAL
    total_days_to_fetch = (
        config.training.TRAINING_DATA_DAYS
        + config.training.VALIDATION_DATA_DAYS
        + config.training.TEST_DATA_DAYS
    )
    feature_cols = get_feature_columns()

    frames: dict[str, pd.DataFrame] = {}
    split_parts: Dict[str, list[tuple[np.ndarray, ...]]] = {"train": [], "val": [], "test": []}
    raw_splits: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for symbol in symbols:
        console.print(f"[info]Processing data for {symbol}...[/info]")
        df_full = _prepare_symbol_frame(symbol, total_days_to_fetch, interval, feature_cols)
        if df_full.empty:
            continue
        frames[symbol] = df_full
        train_df, val_df, test_df = _ordered_split(df_full, interval)
        raw_splits["train"].append(train_df)
        raw_splits["val"].append(val_df)
        raw_splits["test"].append(test_df)

    if not raw_splits["train"]:
        console.print("[error]No data collected for any symbol.[/error]")
        return

    scaler = StandardScaler()
    scaler.fit(np.vstack([df[feature_cols].values for df in raw_splits["train"]]))

    for split_name, dfs in raw_splits.items():
        for df in dfs:
            split_parts[split_name].append(_make_sequences(df, feature_cols, scaler))

    X_train, y_train, y_qty_train, y_tp_train, y_sl_train, y_pnl_train = _concat_parts(split_parts["train"])
    X_val, y_val, y_qty_val, y_tp_val, y_sl_val, y_pnl_val = _concat_parts(split_parts["val"])
    X_test, y_test, y_qty_test, y_tp_test, y_sl_test, y_pnl_test = _concat_parts(split_parts["test"])

    console.print(
        f"[success]Dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
        f"features={len(feature_cols)}[/success]"
    )

    device = config.DEVICE
    train_dataset = TradingDataset(X_train, y_train, y_qty_train, y_tp_train, y_sl_train, y_pnl_train, device=device)
    val_dataset = TradingDataset(X_val, y_val, y_qty_val, y_tp_val, y_sl_val, y_pnl_val, device=device)
    train_loader = DataLoader(train_dataset, batch_size=config.training.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.BATCH_SIZE, shuffle=False)

    class_counts = np.bincount(y_train.astype(np.int64), minlength=3)
    class_weights = len(y_train) / (len(class_counts) * (class_counts + 1e-6))
    class_weights = torch.FloatTensor(class_weights).to(device)
    console.print(
        f"[info]3-class objective. Class counts: long={class_counts[0]}, neutral={class_counts[1]}, short={class_counts[2]}[/info]"
    )

    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    criterion = TradingLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_score = -float("inf")
    no_improve_epochs = 0
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    best_model_path = os.path.join("models", "short_model_eth_best.pth")

    from ui_utils import get_progress

    with get_progress() as progress:
        task = progress.add_task("[cyan]Training long/neutral/short Transformer...", total=config.training.EPOCHS * len(train_loader))
        for epoch in range(config.training.EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_X, batch_targets in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_targets)["total"]
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress.update(task, advance=1, description=f"Epoch {epoch + 1}/{config.training.EPOCHS} | Loss {loss.item():.4f}")

            avg_train_loss = train_loss / max(len(train_loader), 1)
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch_X, batch_targets in val_loader:
                    val_loss += criterion(model(batch_X), batch_targets)["total"].item()
            avg_val_loss = val_loss / max(len(val_loader), 1)
            scheduler.step(avg_val_loss)

            val_pred = _predict(model, X_val, device)
            val_metrics = _classification_metrics(y_val, val_pred["probs"])
            val_thresholds = _threshold_search(y_val, y_tp_val, y_sl_val, val_pred["probs"])
            threshold_score = float(val_thresholds["best"].get("score", -1e9))
            val_score = _model_selection_score(val_thresholds, val_metrics, avg_val_loss)

            if (epoch + 1) % 5 == 0:
                console.print(
                    f"[dim]Epoch {epoch + 1:02d}: train_loss={avg_train_loss:.5f}, "
                    f"val_loss={avg_val_loss:.5f}, val_long_f1={val_metrics['long_f1']:.3f}, "
                    f"val_short_f1={val_metrics['short_f1']:.3f}, threshold_score={threshold_score:.4f}, "
                    f"selection_score={val_score:.4f}[/dim]"
                )

            if val_score > best_val_score:
                best_val_score = val_score
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= config.training.EARLY_STOP_PATIENCE:
                    console.print(f"[warning]Early stopping at epoch {epoch + 1}[/warning]")
                    break

    if not os.path.exists(best_model_path):
        console.print("[warning]No best checkpoint was produced; saving final epoch weights instead.[/warning]")
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    torch.save(model.state_dict(), os.path.join("models", "short_model_eth.pth"))
    np.save(os.path.join("models", "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join("models", "scaler_scale.npy"), scaler.scale_)

    val_pred = _predict(model, X_val, device)
    test_pred = _predict(model, X_test, device)
    diagnostics = {
        "objective": config.training.SHORT_OBJECTIVE,
        "split_samples": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
        "validation_scheme": {
            "type": "purged_time_ordered_holdout",
            "purge_bars": int(config.features.PURGE_BARS),
            "walk_forward_folds_available": {
                symbol: len(_walk_forward_splits(frame, interval)) for symbol, frame in frames.items()
            },
        },
        "label_diagnostics": _label_diagnostics(frames),
        "validation_metrics": _classification_metrics(y_val, val_pred["probs"]),
        "validation_threshold_search": _threshold_search(y_val, y_tp_val, y_sl_val, val_pred["probs"]),
        "test_metrics_once": _classification_metrics(y_test, test_pred["probs"]),
        "test_threshold_proxy_once": _threshold_search(y_test, y_tp_test, y_sl_test, test_pred["probs"]),
    }
    with open(os.path.join("reports", "model_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    threshold_info = diagnostics["validation_threshold_search"]["best"]
    with open(os.path.join("models", "short_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "long_probability_threshold": threshold_info.get("long_threshold", config.strategy.LONG_CONFIDENCE_THRESHOLD),
                "short_probability_threshold": threshold_info.get("short_threshold", config.strategy.SHORT_CONFIDENCE_THRESHOLD),
            },
            f,
            indent=2,
        )

    console.print("[success]Model, train-only scaler, thresholds, and reports/model_diagnostics.json saved.[/success]")
    run_short_backtest()


if __name__ == "__main__":
    train_short_model()
