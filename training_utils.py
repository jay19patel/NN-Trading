# -*- coding: utf-8 -*-
import contextlib
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from config import bars_per_day, config
from ui_utils import console
from rich.live import Live
from rich.table import Table

from models import MultiHeadTradingModel, TradingLoss, direction_logits_to_probabilities
from evaluation_metrics import evaluate_model_on_split


def _configure_backends_for_device(device: torch.device) -> None:
    """Best-effort MatMul / cuDNN settings for faster training (no label changes)."""
    if device.type == "cuda" and torch.cuda.is_available():
        if getattr(config.training, "CUDA_MATMUL_TF32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if getattr(config.training, "CUDNN_BENCHMARK", True):
            torch.backends.cudnn.benchmark = True


def _dataloader_worker_count(device: torch.device) -> int:
    configured = getattr(config.training, "DATALOADER_NUM_WORKERS", -1)
    if configured is not None and configured >= 0:
        return int(configured)
    if device.type == "mps" or sys.platform == "darwin":
        return 0
    if device.type == "cuda":
        return min(8, max(2, (os.cpu_count() or 4) // 2))
    return 0


def _dataloader_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


@contextlib.contextmanager
def _training_autocast(device: torch.device):
    """
    Mixed precision where it is safe.

    MPS + autocast(fp16) mixes fp16 logits with fp32 loss buffers (class weights,
    CE internals) and triggers MetalGraph failures:
      mps.subtract(f16, f32) — same element type required.
    So we run full FP32 on MPS; CUDA still uses fp16 autocast.
    """
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def _maybe_torch_compile(model: nn.Module, device: torch.device) -> nn.Module:
    if not getattr(config.training, "USE_TORCH_COMPILE", False):
        return model
    if device.type == "mps":
        logger.info("torch.compile skipped on MPS (often unsupported or slower)")
        return model
    try:
        return torch.compile(model, mode="default")  # type: ignore[assignment]
    except Exception as exc:
        logger.warning("torch.compile failed (%s); continuing without compile", exc)
        return model


class TradingDataset(torch.utils.data.Dataset):
    """Sequence tensors [N, seq, feat] with multi-task targets."""

    direction_targets: torch.Tensor
    float_target_tensors: dict[str, torch.Tensor]

    def __init__(self, feature_windows: np.ndarray, task_targets: dict[str, np.ndarray]):
        # ? ZERO-COPY FROM NUMPY WHEN CONTIGUOUS FLOAT32 — HUGE VS FloatTensor(COPY)
        feature_contiguous = np.ascontiguousarray(feature_windows, dtype=np.float32)
        self.feature_windows = torch.from_numpy(feature_contiguous)
        direction_arr = np.ascontiguousarray(task_targets["direction"].astype(np.int64))
        self.direction_targets = torch.from_numpy(direction_arr)
        self.float_target_tensors = {
            target_name: torch.from_numpy(
                np.ascontiguousarray(task_targets[target_name], dtype=np.float32)
            )
            for target_name in task_targets.keys()
            if target_name != "direction"
        }

    def __len__(self) -> int:
        return len(self.feature_windows)

    def __getitem__(self, index: int):
        batch_targets: dict[str, torch.Tensor] = {}
        for tensor_name, tensor in self.float_target_tensors.items():
            loss_key = "future_drawdown" if tensor_name == "drawdown" else tensor_name
            batch_targets[loss_key] = tensor[index]
        batch_targets["direction"] = self.direction_targets[index]
        return self.feature_windows[index], batch_targets


def create_sequences(
    feature_matrix: np.ndarray,
    task_targets: Dict[str, np.ndarray],
    sequence_length: int = 32,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build sliding windows efficiently using NumPy stride tricks.
    
    VECTORIZED via numpy.lib.stride_tricks.sliding_window_view — O(1) memory view
    creation instead of a Python loop over ~70k rows.
    """
    n_rows, n_feat = feature_matrix.shape
    if n_rows <= sequence_length:
        empty_x = np.empty((0, sequence_length, n_feat), dtype=np.float32)
        empty_y: Dict[str, np.ndarray] = {}
        for key in task_targets.keys():
            dtype = np.int64 if key == "direction" else np.float32
            empty_y[key] = np.empty((0,), dtype=dtype)
        return empty_x, empty_y

    # Sliding window view: [N-W+1, W, n_feat]
    # We use ascontiguousarray to ensure stride tricks work optimally
    windows = np.lib.stride_tricks.sliding_window_view(
        np.ascontiguousarray(feature_matrix, dtype=np.float32), 
        (sequence_length, n_feat)
    ).squeeze(1)
    
    # The original loop logic:
    # for end_index in range(sequence_length, len(feature_matrix)):
    #   window = feature_matrix[end_index - sequence_length : end_index]
    #   target = task_targets[end_index - 1]
    #
    # This means for range(48, 1000):
    # - first window: 0:48, target: 47
    # - last window: 951:999, target: 998
    # Total windows: 1000 - 48 = 952.
    # sliding_window_view(1000) with window 48 gives 1000 - 48 + 1 = 953 windows.
    # So we drop the last window to match legacy behavior.
    
    stacked_windows = windows[:-1]
    target_slice = slice(sequence_length - 1, n_rows - 1)
    
    stacked_targets: Dict[str, np.ndarray] = {}
    for key in task_targets.keys():
        dtype = np.int64 if key == "direction" else np.float32
        stacked_targets[key] = np.asarray(task_targets[key][target_slice], dtype=dtype)
        
    return stacked_windows, stacked_targets


def get_feature_cols(dataframe: pd.DataFrame) -> List[str]:
    """Columns safe to use as inputs (excludes targets and forward-looking label helpers)."""
    excluded_targets = {
        # Forward-looking features (future data — never use as model inputs)
        "upside_pct",
        "downside_pct",
        "future_drawdown_pct",
        "reward_risk_ratio",
        "edge_ratio",
        "pain_ratio",
        # Oracle label-only columns (derived from future price — leakage if included)
        "oracle_tp_pct",
        "oracle_sl_pct",
        "oracle_rr_ratio",
        "oracle_capacity_score",
        # Training targets (labels — never use as model inputs)
        "direction_label",
        "label_take_profit_pct",
        "label_stop_loss_pct",
        "actual_pnl_pct",
        "label_qty_ratio",
        # Absolute Price Columns (scale-dependent, bad for multi-symbol global models)
        "Open", "High", "Low", "Close", "VWAP",
        "BB_upper", "BB_lower", "BB_middle",
        "KC_upper", "KC_lower",
        "supertrend",
        "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100", "EMA_200",
        "SMA_20", "SMA_50", "SMA_100", "SMA_200",
        "OBV", "AD", "VPT",
        # Index / meta columns
        "time",
        "label",
        "index",
        "feature_cache_version",
    }
    safe_numeric_columns = [
        c for c in dataframe.select_dtypes(include=[np.number]).columns if c not in excluded_targets
    ]
    if config.features.USE_CURATED_FEATURES:
        curated_columns = [
            column
            for column in config.features.CURATED_FEATURES
            if column in safe_numeric_columns
        ]
        missing_columns = [
            column
            for column in config.features.CURATED_FEATURES
            if column not in safe_numeric_columns
        ]
        if missing_columns:
            console.print(
                f"[warning]Curated feature columns missing and skipped: "
                f"{', '.join(missing_columns[:8])}[/warning]"
            )
        return curated_columns
    return safe_numeric_columns


def sanitize_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean a split without backward-filling future values into earlier rows."""
    return frame.copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def drop_lookahead_boundary(frame: pd.DataFrame, lookahead_bars: int) -> pd.DataFrame:
    """
    Remove the last `lookahead_bars` rows of a partition.

    ? CRITICAL LEAKAGE FIX:
    ? add_risk_reward_features and add_oracle_target_labels look up to LOOKAHEAD_BARS bars
    ? into the FUTURE of every row to compute upside_pct/downside_pct/direction_label.
    ? If we slice train|val|test on the raw frame WITHOUT trimming, the last `lookahead_bars`
    ? rows of the train slice have labels computed from val data, and the last `lookahead_bars`
    ? rows of val have labels computed from test data. That is a direct future-data leak
    ? into supervised training.
    ? This helper trims the boundary and is now applied per partition before scaling.
    """
    if lookahead_bars <= 0 or len(frame) <= lookahead_bars:
        return frame.copy()
    return frame.iloc[: -lookahead_bars].copy()


def build_target_arrays(dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
    required_label_columns = (
        "upside_pct",
        "downside_pct",
        "future_drawdown_pct",
        "label_take_profit_pct",
        "label_stop_loss_pct",
        "direction_label",
        "actual_pnl_pct",
        "label_qty_ratio"
    )
    missing_columns = [column for column in required_label_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Feature frame is missing label columns "
            f"{missing_columns}. Delete stale `features_cache_*.csv` files and rebuild."
        )
    return {
        "upside": dataframe["upside_pct"].values,
        "downside": dataframe["downside_pct"].values,
        "drawdown": dataframe["future_drawdown_pct"].values,
        "take_profit_pct": dataframe["label_take_profit_pct"].values,
        "stop_loss_pct": dataframe["label_stop_loss_pct"].values,
        "direction": dataframe["direction_label"].values,
        "actual_pnl_pct": dataframe["actual_pnl_pct"].values,
        "qty_ratio": dataframe["label_qty_ratio"].values,
    }


def prepare_data(
    dataframe: pd.DataFrame, test_days: int = 10
) -> Tuple[np.ndarray, Dict, np.ndarray, Dict, List[str], StandardScaler]:
    """Single-symbol train/test prep (backward-compatible wrapper)."""
    feature_columns = get_feature_cols(dataframe)
    cleaned = sanitize_partition(dataframe)
    interval_bars_per_day = bars_per_day(config.data.INTERVAL)
    test_row_count = test_days * interval_bars_per_day
    train_frame = cleaned.iloc[:-test_row_count]
    test_frame = cleaned.iloc[-test_row_count:]
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_frame[feature_columns])
    test_features = scaler.transform(test_frame[feature_columns])
    return (
        train_features,
        build_target_arrays(train_frame),
        test_features,
        build_target_arrays(test_frame),
        feature_columns,
        scaler,
    )


def prepare_multi_symbol_data(
    symbol_dataframes: Dict[str, pd.DataFrame],
    test_days: int = 10,
    val_days: int = 2,
) -> Tuple[np.ndarray, Dict, np.ndarray, Dict, np.ndarray, Dict, List[str], Dict[str, StandardScaler]]:
    """
    Chronological split per symbol: train | validation | test (most recent = test).

    Scaler is fit only on the training portion (pooled across symbols), then applied to val/test.
    Sequences are built inside each symbol to avoid mixing bars across assets or gaps.
    """
    first_frame = next(iter(symbol_dataframes.values()))
    feature_columns = get_feature_cols(first_frame)
    sequence_length = config.model.SEQ_LEN
    interval_bars_per_day = bars_per_day(config.data.INTERVAL)
    test_row_count = test_days * interval_bars_per_day
    val_row_count = val_days * interval_bars_per_day

    train_frames: Dict[str, pd.DataFrame] = {}
    validation_frames: Dict[str, pd.DataFrame] = {}
    test_frames: Dict[str, pd.DataFrame] = {}

    # ? LEAKAGE-SAFE BOUNDARY TRIMMING:
    # ? THE LAST `LOOKAHEAD_BARS` ROWS OF TRAIN AND VAL ARE LABELED USING FUTURE VALIDATION/TEST
    # ? PRICES BY add_oracle_target_labels. WE DROP THEM TO ELIMINATE BOUNDARY LEAKAGE.
    lookahead_bars = config.features.LOOKAHEAD_BARS

    train_feature_pool: List[pd.DataFrame] = []
    for symbol, raw_frame in symbol_dataframes.items():
        train_slice = raw_frame.iloc[: -(val_row_count + test_row_count)]
        val_slice = raw_frame.iloc[-(val_row_count + test_row_count) : -test_row_count]
        test_slice = raw_frame.iloc[-test_row_count:]
        # ? TRIM TRAIN AND VAL BOUNDARY (TEST DOES NOT NEED TRIMMING — WE ONLY EVALUATE,
        # ? AND THE LABELS OF THE LAST FEW ROWS ARE NEUTRAL DUE TO OUT-OF-RANGE LOOKAHEAD).
        train_slice = drop_lookahead_boundary(train_slice, lookahead_bars)
        val_slice = drop_lookahead_boundary(val_slice, lookahead_bars)
        train_frames[symbol] = sanitize_partition(train_slice)
        validation_frames[symbol] = sanitize_partition(val_slice)
        test_frames[symbol] = sanitize_partition(test_slice)
        train_feature_pool.append(train_frames[symbol][feature_columns])

    console.print(
        f"[info]Boundary leakage fix: dropped last {lookahead_bars} rows from each train and val partition[/info]"
    )

    global_train_feature_frame = pd.concat(train_feature_pool, axis=0)
    scaler = StandardScaler()
    scaler.fit(global_train_feature_frame)

    # ---- AUTO-PRUNE FEATURES: variance + correlation filters ----
    train_scaled_check = scaler.transform(global_train_feature_frame)
    variances = np.var(train_scaled_check, axis=0)
    var_mask = variances > config.features.VARIANCE_FLOOR
    kept_cols = [c for c, keep in zip(feature_columns, var_mask) if keep]
    dropped_var = len(feature_columns) - len(kept_cols)
    if dropped_var > 0:
        console.print(f"[warning]Feature pruning: dropped {dropped_var} low-variance features[/warning]")

    # Correlation filter on survivors
    if len(kept_cols) > 2:
        kept_idx = [feature_columns.index(c) for c in kept_cols]
        corr_data = train_scaled_check[:, kept_idx]
        corr_matrix = np.corrcoef(corr_data.T)
        to_drop_corr = set()
        for ci in range(len(kept_cols)):
            if ci in to_drop_corr:
                continue
            for cj in range(ci + 1, len(kept_cols)):
                if cj in to_drop_corr:
                    continue
                if abs(corr_matrix[ci, cj]) > config.features.CORRELATION_CEILING:
                    to_drop_corr.add(cj)
        if to_drop_corr:
            kept_cols = [c for i, c in enumerate(kept_cols) if i not in to_drop_corr]
            console.print(f"[warning]Feature pruning: dropped {len(to_drop_corr)} highly correlated features[/warning]")

    console.print(f"[info]Features after pruning: {len(kept_cols)} (was {len(feature_columns)})[/info]")
    feature_columns = kept_cols

    # Re-fit scaler on pruned columns only
    pruned_train_pool = [tf[feature_columns] for tf in [train_frames[s] for s in symbol_dataframes]]
    global_train_pruned = pd.concat(pruned_train_pool, axis=0)
    scaler = StandardScaler()
    scaler.fit(global_train_pruned)

    train_windows: List[np.ndarray] = []
    validation_windows: List[np.ndarray] = []
    test_windows: List[np.ndarray] = []
    symbol_scalers: Dict[str, StandardScaler] = {}

    train_target_parts: Dict[str, List[np.ndarray]] = {name: [] for name in build_target_arrays(first_frame)}
    val_target_parts: Dict[str, List[np.ndarray]] = {name: [] for name in train_target_parts}
    test_target_parts: Dict[str, List[np.ndarray]] = {name: [] for name in train_target_parts}

    for symbol in symbol_dataframes:
        # Fit scaler on THIS symbol's training data only
        symbol_scaler = StandardScaler()
        symbol_scaler.fit(train_frames[symbol][feature_columns])
        symbol_scalers[symbol] = symbol_scaler
        
        train_features = symbol_scaler.transform(train_frames[symbol][feature_columns])
        validation_features = symbol_scaler.transform(validation_frames[symbol][feature_columns])
        test_features = symbol_scaler.transform(test_frames[symbol][feature_columns])

        train_targets = build_target_arrays(train_frames[symbol])
        validation_targets = build_target_arrays(validation_frames[symbol])
        test_targets = build_target_arrays(test_frames[symbol])

        train_X, train_y = create_sequences(train_features, train_targets, sequence_length=sequence_length)
        val_X, val_y = create_sequences(validation_features, validation_targets, sequence_length=sequence_length)
        test_X, test_y = create_sequences(test_features, test_targets, sequence_length=sequence_length)

        train_windows.append(train_X)
        validation_windows.append(val_X)
        test_windows.append(test_X)
        for key in train_target_parts:
            train_target_parts[key].append(train_y[key])
            val_target_parts[key].append(val_y[key])
            test_target_parts[key].append(test_y[key])

    combined_train_X = np.concatenate(train_windows, axis=0)
    combined_val_X = np.concatenate(validation_windows, axis=0)
    combined_test_X = np.concatenate(test_windows, axis=0)
    combined_train_y = {key: np.concatenate(parts, axis=0) for key, parts in train_target_parts.items()}
    combined_val_y = {key: np.concatenate(parts, axis=0) for key, parts in val_target_parts.items()}
    combined_test_y = {key: np.concatenate(parts, axis=0) for key, parts in test_target_parts.items()}

    return (
        combined_train_X,
        combined_train_y,
        combined_val_X,
        combined_val_y,
        combined_test_X,
        combined_test_y,
        feature_columns,
        symbol_scalers,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(
    train_features: np.ndarray,
    train_targets: Dict,
    device: torch.device,
    input_dim: int,
    val_features: Optional[np.ndarray] = None,
    val_targets: Optional[Dict] = None,
    test_features: Optional[np.ndarray] = None,
    test_targets: Optional[Dict] = None,
    epochs: int = 50,
) -> Tuple[MultiHeadTradingModel, Dict[str, float]]:
    """Train with optional validation early stopping; report held-out test metrics once at the end."""
    _configure_backends_for_device(device)

    model = MultiHeadTradingModel(input_dim=input_dim).to(device)
    model = _maybe_torch_compile(model, device)
    parameter_count = count_trainable_parameters(model)
    console.print(
        f"[info]Model: inputs={input_dim}, parameters={parameter_count:,}, device={device}[/info]"
    )

    eval_chunk = getattr(config.training, "EVAL_CHUNK_SIZE", 4096)
    num_workers = _dataloader_worker_count(device)
    pin_memory = _dataloader_pin_memory(device)
    console.print(
        f"[info]Throughput: dataloader_workers={num_workers}, pin_memory={pin_memory}, "
        f"eval_chunk={eval_chunk}, autocast={'fp16 (cuda only)' if device.type == 'cuda' else 'off (fp32 on mps/cpu)'}[/info]"
    )

    val_tensors_device: Optional[Dict[str, torch.Tensor]] = None
    if val_features is not None and val_targets is not None:
        # ? HOIST VAL DATA TO GPU ONCE — AVOID RE-COPYING EVERY EPOCH (100× SPEEDUP ON THAT PATH)
        val_tensors_device = {
            "X": torch.as_tensor(val_features, dtype=torch.float32, device=device),
            "direction": torch.as_tensor(val_targets["direction"].astype(np.int64), dtype=torch.long, device=device),
            "upside": torch.as_tensor(val_targets["upside"], dtype=torch.float32, device=device),
            "downside": torch.as_tensor(val_targets["downside"], dtype=torch.float32, device=device),
            "drawdown": torch.as_tensor(val_targets["drawdown"], dtype=torch.float32, device=device),
            "take_profit_pct": torch.as_tensor(val_targets["take_profit_pct"], dtype=torch.float32, device=device),
            "stop_loss_pct": torch.as_tensor(val_targets["stop_loss_pct"], dtype=torch.float32, device=device),
            "actual_pnl_pct": torch.as_tensor(val_targets["actual_pnl_pct"], dtype=torch.float32, device=device),
            "qty_ratio": torch.as_tensor(val_targets["qty_ratio"], dtype=torch.float32, device=device),
        }

    # Compute class weights from actual label distribution instead of WeightedSampler.
    # WeightedSampler causes train/val distribution mismatch (val < train loss at epoch 1)
    # because it artificially balances training, but validation has real natural distribution.
    # Passing weights to CrossEntropyLoss is mathematically equivalent but avoids this mismatch.
    direction_labels = train_targets["direction"].astype(np.int64)
    class_counts = np.bincount(direction_labels, minlength=3).astype(np.float32)
    inverse_frequency = class_counts.sum() / (3.0 * class_counts + 1e-6)
    class_weights_np = np.power(inverse_frequency, config.training.CLASS_WEIGHT_POWER)
    class_weights_np = np.clip(
        class_weights_np,
        config.training.MIN_CLASS_WEIGHT,
        config.training.MAX_CLASS_WEIGHT,
    )
    class_weights_np[0] *= config.training.CLASS_WEIGHT_DIRECTIONAL_SCALE
    class_weights_np[2] *= config.training.CLASS_WEIGHT_DIRECTIONAL_SCALE
    class_weights_np[1] *= config.training.CLASS_WEIGHT_NEUTRAL_SCALE
    class_weights_np = np.clip(
        class_weights_np,
        config.training.MIN_CLASS_WEIGHT,
        config.training.MAX_CLASS_WEIGHT,
    )
    class_weight_tensor = torch.FloatTensor(class_weights_np).to(device)
    console.print(
        f"[info]Class weights (precision-tuned): LONG={class_weights_np[0]:.2f}, "
        f"NEUTRAL={class_weights_np[1]:.2f}, SHORT={class_weights_np[2]:.2f}[/info]"
    )

    # Phase 1: Supervised Pre-training
    loss_fn = TradingLoss(
        alpha=config.training.LOSS_ALPHA,
        beta=config.training.LOSS_BETA,
        gamma=0.0,                        # No PnL penalty in Phase 1
        class_weights=class_weight_tensor,
        focal_gamma=config.training.FOCAL_GAMMA,
        use_focal=config.training.USE_FOCAL_LOSS,
        consistency_weight=config.training.LOSS_CONSISTENCY,
    ).to(device)
    console.print(f"[info]Loss: {'FocalLoss(γ=' + str(config.training.FOCAL_GAMMA) + ')' if config.training.USE_FOCAL_LOSS else 'CrossEntropy'}[/info]")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.LEARNING_RATE,
        weight_decay=config.training.WEIGHT_DECAY,
    )

    train_dataset = TradingDataset(train_features, train_targets)
    loader_kwargs: Dict[str, object] = {
        "batch_size": config.training.BATCH_SIZE,
        "shuffle": True,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and getattr(config.training, "DATALOADER_PERSISTENT_WORKERS", True):
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # OneCycleLR: linear warmup (10%) + cosine annealing. Much more stable than ReduceLROnPlateau.
    # ReduceLROnPlateau was reducing LR too late to prevent the val loss spike at epoch 2.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,          # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,        # Start LR = max_lr / 10
        final_div_factor=100.0, # End LR = max_lr / 1000
    )

    best_val_loss = float("inf")
    best_composite_score = float("-inf")
    best_state: Optional[Dict] = None
    patience_left = config.training.EARLY_STOP_PATIENCE
    plateau_strikes = 0
    plateau_threshold = 5
    training_history: List[Dict[str, float | int | str]] = []

    metrics_table = Table(show_header=True, header_style="bold magenta", border_style="bright_black")
    metrics_table.add_column("Epoch", justify="center")
    metrics_table.add_column("Train loss", justify="right")
    metrics_table.add_column("Val loss", justify="right")
    metrics_table.add_column("Val dir acc", justify="right")
    metrics_table.add_column("PnL Effect", justify="right")
    metrics_table.add_column("Status", justify="center")

    with Live(metrics_table, console=console, refresh_per_second=4):
        for epoch in range(epochs):
            model.train()
            # Accumulate loss as GPU tensor — avoid .item() per batch (each .item() = GPU-CPU sync)
            running_loss_tensor = torch.tensor(0.0, device=device)
            batch_count = 0
            non_blocking = pin_memory
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device, non_blocking=non_blocking)
                batch_targets = {
                    name: tensor.to(device, non_blocking=non_blocking)
                    for name, tensor in batch_targets.items()
                }
                optimizer.zero_grad(set_to_none=True)
                
                with _training_autocast(device):
                    outputs = model(batch_features)
                    loss_dict = loss_fn(outputs, batch_targets)
                    loss = loss_dict["total"]
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss_tensor = running_loss_tensor + loss.detach()
                batch_count += 1

            # Single GPU-CPU sync per epoch (was 478+ syncs before)
            average_train_loss = (running_loss_tensor / max(batch_count, 1)).item()

            val_loss_value = float("nan")
            val_direction_accuracy = float("nan")
            if val_tensors_device is not None:
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                correct_direction = 0
                total_direction = 0
                val_tensor = val_tensors_device["X"]
                direction_actual = val_tensors_device["direction"]
                with torch.inference_mode():
                    for start in range(0, val_tensor.size(0), eval_chunk):
                        end = min(start + eval_chunk, val_tensor.size(0))
                        batch_X = val_tensor[start:end]
                        batch_y_dir = direction_actual[start:end]
                        with _training_autocast(device):
                            val_outputs = model(batch_X)
                        val_chunk_targets = {
                            "upside": val_tensors_device["upside"][start:end],
                            "downside": val_tensors_device["downside"][start:end],
                            "future_drawdown": val_tensors_device["drawdown"][start:end],
                            "take_profit_pct": val_tensors_device["take_profit_pct"][start:end],
                            "stop_loss_pct": val_tensors_device["stop_loss_pct"][start:end],
                            "direction": batch_y_dir,
                            "actual_pnl_pct": val_tensors_device["actual_pnl_pct"][start:end],
                            "qty_ratio": val_tensors_device["qty_ratio"][start:end],
                        }
                        batch_loss = loss_fn(val_outputs, val_chunk_targets)["total"]
                        val_loss_sum += batch_loss.item()
                        val_batches += 1
                        predicted_direction = torch.argmax(val_outputs["direction"], dim=1)
                        correct_direction += (predicted_direction == batch_y_dir).sum().item()
                        total_direction += batch_X.size(0)
                val_loss_value = val_loss_sum / max(val_batches, 1)
                val_direction_accuracy = (correct_direction / max(total_direction, 1)) * 100.0
                # OneCycleLR is step-based — do NOT call scheduler.step() here

                # ? COMPOSITE EARLY-STOP SCORE — REWARDS BOTH LOWER LOSS AND HIGHER DIR ACC.
                # ? PURE VAL_LOSS PLATEAUS BEFORE DIRECTIONAL ACCURACY DOES (THE TWO ARE
                # ? RELATED BUT NOT IDENTICAL UNDER FOCAL LOSS WITH DIRECTIONAL PENALTY).
                composite_score = -val_loss_value + 0.01 * val_direction_accuracy
                improved_loss = val_loss_value < best_val_loss - 1e-6
                improved_composite = composite_score > best_composite_score + 1e-6

                if improved_loss or improved_composite:
                    if improved_loss:
                        best_val_loss = val_loss_value
                    if improved_composite:
                        best_composite_score = composite_score
                    best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
                    patience_left = config.training.EARLY_STOP_PATIENCE
                    plateau_strikes = 0
                else:
                    patience_left -= 1
                    plateau_strikes += 1

            # ? AUTOMATED LR DECAY ON PLATEAU — IF VAL LOSS HASN'T IMPROVED FOR
            # ? `plateau_threshold` EPOCHS, MANUALLY DAMPEN THE OPTIMIZER LR TO 0.5X.
            # ? ONECYCLELR ALONE CAN'T REACT TO PLATEAUS; THIS GIVES IT A SECOND CHANCE.
            if plateau_strikes >= plateau_threshold and val_tensors_device is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, config.training.LEARNING_RATE * 0.01)
                console.print(
                    f"[warning]Plateau detected — LR dampened to "
                    f"{optimizer.param_groups[0]['lr']:.2e}[/warning]"
                )
                plateau_strikes = 0

            status = "Phase 1"
            if patience_left <= 0 and val_tensors_device is not None:
                status = "early-stop"
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(average_train_loss),
                    "val_loss": float(val_loss_value),
                    "val_direction_accuracy_pct": float(val_direction_accuracy),
                    "status": status,
                }
            )
            metrics_table.add_row(
                f"{epoch + 1}/{epochs}",
                f"{average_train_loss:.5f}",
                f"{val_loss_value:.5f}" if val_tensors_device is not None else "—",
                f"{val_direction_accuracy:.1f}%" if val_tensors_device is not None else "—",
                "—",  # PnL Effect is Phase 2 only
                status,
            )

            if val_tensors_device is not None and patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    rl_enabled = config.training.RL_FINE_TUNE_EPOCHS > 0 and config.training.LOSS_GAMMA > 0
    if rl_enabled:
        console.print("\n[highlight]Phase 2: Reinforcement Learning (PnL Consequence Fine-Tuning)[/highlight]")
        console.print(f"[info]RL gamma={config.training.LOSS_GAMMA} | LR={config.training.RL_LEARNING_RATE} | epochs={config.training.RL_FINE_TUNE_EPOCHS}[/info]")

        for param in model.input_projection.parameters():
            param.requires_grad = False
        for param in model.input_norm.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.positional_encoding.parameters():
            param.requires_grad = False
        trainable_rl = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"[info]RL trainable params: {trainable_rl:,} (backbone frozen)[/info]")

        rl_loss_fn = TradingLoss(
            alpha=config.training.LOSS_ALPHA,
            beta=config.training.LOSS_BETA,
            gamma=config.training.LOSS_GAMMA,
            class_weights=class_weight_tensor,
            focal_gamma=config.training.FOCAL_GAMMA,
            use_focal=config.training.USE_FOCAL_LOSS,
            consistency_weight=config.training.LOSS_CONSISTENCY,
        ).to(device)

        rl_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.training.RL_LEARNING_RATE,
            weight_decay=config.training.WEIGHT_DECAY,
        )

        rl_best_val_loss = float("inf")
        rl_best_state: dict | None = None
        patience_left = config.training.EARLY_STOP_PATIENCE

        with Live(metrics_table, console=console, refresh_per_second=4):
            rl_epochs = config.training.RL_FINE_TUNE_EPOCHS
            for epoch in range(rl_epochs):
                model.train()
                running_loss_tensor = torch.tensor(0.0, device=device)
                running_pnl_tensor = torch.tensor(0.0, device=device)
                batch_count = 0
                non_blocking = pin_memory
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(device, non_blocking=non_blocking)
                    batch_targets = {
                        name: tensor.to(device, non_blocking=non_blocking)
                        for name, tensor in batch_targets.items()
                    }
                    rl_optimizer.zero_grad(set_to_none=True)

                    with _training_autocast(device):
                        outputs = model(batch_features)
                        loss_dict = rl_loss_fn(outputs, batch_targets)
                        loss = loss_dict["total"]

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    rl_optimizer.step()
                    running_loss_tensor = running_loss_tensor + loss.detach()
                    running_pnl_tensor = running_pnl_tensor + loss_dict["pnl_effect"].detach()
                    batch_count += 1

                average_train_loss = (running_loss_tensor / max(batch_count, 1)).item()
                average_pnl_effect = (running_pnl_tensor / max(batch_count, 1)).item()

                val_loss_value = float("nan")
                val_direction_accuracy = float("nan")
                if val_tensors_device is not None:
                    model.eval()
                    val_loss_sum = 0.0
                    val_batches = 0
                    correct_direction = 0
                    total_direction = 0
                    val_tensor = val_tensors_device["X"]
                    direction_actual = val_tensors_device["direction"]
                    with torch.inference_mode():
                        for start in range(0, val_tensor.size(0), eval_chunk):
                            end = min(start + eval_chunk, val_tensor.size(0))
                            batch_X = val_tensor[start:end]
                            batch_y_dir = direction_actual[start:end]
                            with _training_autocast(device):
                                val_outputs = model(batch_X)
                            val_chunk_targets = {
                                "upside": val_tensors_device["upside"][start:end],
                                "downside": val_tensors_device["downside"][start:end],
                                "future_drawdown": val_tensors_device["drawdown"][start:end],
                                "take_profit_pct": val_tensors_device["take_profit_pct"][start:end],
                                "stop_loss_pct": val_tensors_device["stop_loss_pct"][start:end],
                                "direction": batch_y_dir,
                                "actual_pnl_pct": val_tensors_device["actual_pnl_pct"][start:end],
                                "qty_ratio": val_tensors_device["qty_ratio"][start:end],
                            }
                            batch_loss = rl_loss_fn(val_outputs, val_chunk_targets)["total"]
                            val_loss_sum += batch_loss.item()
                            val_batches += 1
                            predicted_direction = torch.argmax(val_outputs["direction"], dim=1)
                            correct_direction += (predicted_direction == batch_y_dir).sum().item()
                            total_direction += batch_X.size(0)
                    val_loss_value = val_loss_sum / max(val_batches, 1)
                    val_direction_accuracy = (correct_direction / max(total_direction, 1)) * 100.0

                    if val_loss_value < rl_best_val_loss - 1e-6:
                        rl_best_val_loss = val_loss_value
                        rl_best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
                        patience_left = config.training.EARLY_STOP_PATIENCE
                    else:
                        patience_left -= 1

                status = "Phase 2 (RL)"
                if patience_left <= 0 and val_tensors_device is not None:
                    status = "early-stop ✓"
                metrics_table.add_row(
                    f"RL {epoch + 1}/{rl_epochs}",
                    f"{average_train_loss:.5f}",
                    f"{val_loss_value:.5f}" if val_tensors_device is not None else "—",
                    f"{val_direction_accuracy:.1f}%" if val_tensors_device is not None else "—",
                    f"{average_pnl_effect:.4f}",
                    status,
                )

                if val_tensors_device is not None and patience_left <= 0:
                    break

        if rl_best_state is not None:
            model.load_state_dict(rl_best_state)
            console.print("[success]✅ Phase 2 checkpoint restored (RL improved val loss)[/success]")
        elif best_state is not None:
            model.load_state_dict(best_state)
            console.print("[warning]⚠️  Phase 2 did not improve val loss. Using Phase 1 best checkpoint.[/warning]")

        for param in model.parameters():
            param.requires_grad = True
    else:
        console.print(
            "\n[warning]Phase 2 skipped: RL fine-tuning is disabled because the current objective is supervised and not production-safe as an RL surrogate.[/warning]"
        )

    # ---- Per-class confidence calibration on val set ----
    calibrated_thresholds = {0: 1.01, 2: 1.01}
    if val_tensors_device is not None and val_targets is not None:
        model.eval()
        prob_parts: List[torch.Tensor] = []
        sizing_parts: List[torch.Tensor] = []
        with torch.inference_mode():
            val_cal_x = val_tensors_device["X"]
            for start in range(0, val_cal_x.size(0), eval_chunk):
                end = min(start + eval_chunk, val_cal_x.size(0))
                with _training_autocast(device):
                    chunk_out = model(val_cal_x[start:end])
                prob_parts.append(direction_logits_to_probabilities(chunk_out["direction"]).float().cpu())
                sizing_parts.append(chunk_out["sizing"].float().cpu())
        val_probs = torch.cat(prob_parts, dim=0).numpy()
        val_sizing = torch.cat(sizing_parts, dim=0).numpy()
        val_true = val_targets["direction"].astype(int)
        minimum_support = max(20, int(len(val_true) * 0.01))
        predicted_tp = val_sizing[:, 1] * config.strategy.LABEL_TP_PCT_MAX
        predicted_sl = val_sizing[:, 2] * config.strategy.LABEL_SL_PCT_MAX
        directional_edge = np.maximum(val_probs[:, 0], val_probs[:, 2]) - val_probs[:, 1]
        sorted_val_probs = np.sort(val_probs, axis=1)
        softmax_margin = sorted_val_probs[:, -1] - sorted_val_probs[:, -2]
        margin_floor = float(config.strategy.MIN_SOFTMAX_MARGIN)
        estimated_round_trip_cost_pct = (
            config.strategy.ROUND_TRIP_FEE_PCT
            + (2.0 * config.strategy.SLIPPAGE_PCT)
            + config.strategy.COST_BUFFER_PCT
        )
        for cls, cls_name in [(0, "BUY"), (2, "SELL")]:
            best_thresh = 1.01
            best_expectancy = float("-inf")
            best_prec = 0.0
            best_support = 0
            edge_floor = config.strategy.MIN_DIRECTIONAL_EDGE
            for thresh in np.arange(config.strategy.AI_CONFIDENCE_THRESHOLD, 0.86, 0.02):
                pred_mask = (
                    (np.argmax(val_probs, axis=1) == cls)
                    & (val_probs[:, cls] >= thresh)
                    & (directional_edge >= edge_floor)
                    & (softmax_margin >= margin_floor)
                )
                total_pred = pred_mask.sum()
                if total_pred < minimum_support:
                    continue
                correct = (val_true[pred_mask] == cls).sum()
                precision = correct / total_pred
                avg_tp = float(np.mean(predicted_tp[pred_mask])) if total_pred > 0 else config.strategy.TARGET_PROFIT_PCT
                avg_sl = float(np.mean(predicted_sl[pred_mask])) if total_pred > 0 else config.strategy.STOP_LOSS_PCT
                expectancy = precision * avg_tp - (1.0 - precision) * avg_sl - estimated_round_trip_cost_pct
                if (
                    precision < config.strategy.MIN_VALIDATION_PRECISION
                    or expectancy < config.strategy.MIN_VALIDATION_EXPECTANCY_PCT
                ):
                    continue
                score = expectancy * np.log1p(total_pred)
                if score > best_expectancy:
                    best_expectancy = score
                    best_prec = precision
                    best_thresh = float(thresh)
                    best_support = int(total_pred)
            if best_support == 0:
                best_expectancy = 0.0
                # ? NEVER LEAVE best_thresh AT 1.01 — SOFTMAX CONFIDENCE IS ALWAYS ≤ 1.0,
                # ? SO ai_confidence < 1.01 FOR EVERY ROW AND ALL BUY/SELL SIGNALS BECOME NEUTRAL.
                # ? FALL BACK TO BASE CONFIG THRESHOLD SO BACKTESTS STILL RUN (WITH CLEAR WARNING).
                calibrated_thresholds[cls] = float(config.strategy.AI_CONFIDENCE_THRESHOLD)
                console.print(
                    f"[warning]Calibrated {cls_name} threshold: BLOCKED "
                    f"(no validation slice met precision/expectancy gates). "
                    f"Falling back to AI_CONFIDENCE_THRESHOLD={calibrated_thresholds[cls]:.2f} "
                    f"for inference — review gates or model quality.[/warning]"
                )
                continue
            calibrated_thresholds[cls] = best_thresh
            console.print(
                f"[info]Calibrated {cls_name} threshold: {best_thresh:.2f} "
                f"(precision={best_prec:.1%}, support={best_support}, net_expectancy={best_expectancy:.3f}%)[/info]"
            )

    final_report: Dict[str, float] = {}
    if test_features is not None and test_targets is not None:
        final_report = evaluate_model_on_split(model, test_features, test_targets, device)
    final_report["calibrated_thresholds"] = calibrated_thresholds
    final_report["training_history"] = training_history
    return model, final_report


def run_inference_with_confidence_filter(
    model: nn.Module, feature_windows: np.ndarray, device: torch.device,
    calibrated_thresholds: Dict[int, float] | None = None,
    min_directional_edge: float | None = None,
    min_reward_risk_ratio: float | None = None,
    min_softmax_margin: float | None = None,
    use_breakeven_confidence_gate: bool | None = None,
) -> pd.DataFrame:
    """Direction softmax, directional-edge gate, and volatility-aware TP/SL percentage heads."""
    model.eval()
    input_tensor = torch.FloatTensor(feature_windows).to(device)
    with torch.no_grad():
        raw_outputs = model(input_tensor)
    
    direction_probabilities = direction_logits_to_probabilities(raw_outputs["direction"])
    verdict_indices = torch.argmax(direction_probabilities, dim=1).cpu().numpy()
    
    # Sigmoid sizing output decoding
    sizing = raw_outputs["sizing"].cpu().numpy()
    qty_ratios = sizing[:, 0]
    take_profit_pct = sizing[:, 1] * config.strategy.LABEL_TP_PCT_MAX
    stop_loss_pct = sizing[:, 2] * config.strategy.LABEL_SL_PCT_MAX
    
    buy_probability = direction_probabilities[:, 0].cpu().numpy()
    neutral_probability = direction_probabilities[:, 1].cpu().numpy()
    sell_probability = direction_probabilities[:, 2].cpu().numpy()
    
    # Re-calculate confidence based on max probability
    confidence_scores = np.max(direction_probabilities.cpu().numpy(), axis=1)

    result_frame = pd.DataFrame(
        {
            "ai_confidence": confidence_scores,
            "ai_raw_verdict": verdict_indices,
            "ai_qty_ratio": qty_ratios,
            "ai_verdict": verdict_indices,
            "ai_take_profit_pct": take_profit_pct,
            "ai_stop_loss_pct": stop_loss_pct,
            "ai_prob_buy": buy_probability,
            "ai_prob_neutral": neutral_probability,
            "ai_prob_sell": sell_probability,
            "ai_directional_edge": np.maximum(buy_probability, sell_probability) - neutral_probability,
        }
    )

    raw_buy_signals = int((result_frame["ai_verdict"] == 0).sum())
    raw_sell_signals = int((result_frame["ai_verdict"] == 2).sum())
    console.print(
        f"[info]Raw direction signals: Buy={raw_buy_signals}, Sell={raw_sell_signals}[/info]"
    )

    # Per-class confidence filtering (calibrated thresholds)
    if calibrated_thresholds is None:
        calibrated_thresholds = {0: config.strategy.AI_CONFIDENCE_THRESHOLD,
                                 2: config.strategy.AI_CONFIDENCE_THRESHOLD}
    
    suppressed_count = 0
    for cls, thresh in calibrated_thresholds.items():
        cls_mask = (result_frame["ai_verdict"] == cls) & (result_frame["ai_confidence"] < thresh)
        suppressed_count += int(cls_mask.sum())
        result_frame.loc[cls_mask, "ai_verdict"] = 1
    if suppressed_count > 0:
        console.print(
            f"[warning]Risk Rule 1: forced neutral on {suppressed_count} rows "
            f"(predicted-class prob < threshold Buy={calibrated_thresholds[0]:.2f}, "
            f"Sell={calibrated_thresholds[2]:.2f}; softmax uses inference temperature)[/warning]"
        )

    directional_edge_floor = (
        config.strategy.MIN_DIRECTIONAL_EDGE
        if min_directional_edge is None
        else min_directional_edge
    )
    edge_mask = result_frame["ai_directional_edge"] < directional_edge_floor
    suppressed_edge = int((edge_mask & (result_frame["ai_verdict"] != 1)).sum())
    if suppressed_edge > 0:
        console.print(
            f"[warning]Risk Rule 1B: forced neutral on {suppressed_edge} weak-edge rows (edge < {directional_edge_floor:.2f})[/warning]"
        )
    result_frame.loc[edge_mask, "ai_verdict"] = 1
    
    # R:R Rule
    rr_ratios = result_frame["ai_take_profit_pct"] / (result_frame["ai_stop_loss_pct"] + 1e-6)
    min_rr = config.strategy.MIN_REWARD_RISK_RATIO if min_reward_risk_ratio is None else min_reward_risk_ratio
    poor_rr_mask = rr_ratios < min_rr
    suppressed_rr = int((poor_rr_mask & (result_frame["ai_verdict"] != 1)).sum())
    if suppressed_rr > 0:
        console.print(f"[warning]Risk Rule 2: forced neutral on {suppressed_rr} poor R:R rows (< {min_rr})[/warning]")
    result_frame.loc[poor_rr_mask, "ai_verdict"] = 1

    # ? SOFTMAX MARGIN — TOP CLASS MUST CLEARLY BEAT RUNNER-UP (NOT A TIE-BREAK COIN FLIP).
    margin_floor = (
        config.strategy.MIN_SOFTMAX_MARGIN
        if min_softmax_margin is None
        else float(min_softmax_margin)
    )
    stacked_probs = np.stack(
        [
            result_frame["ai_prob_buy"].values,
            result_frame["ai_prob_neutral"].values,
            result_frame["ai_prob_sell"].values,
        ],
        axis=1,
    )
    sorted_row_probs = np.sort(stacked_probs, axis=1)
    softmax_margin_arr = sorted_row_probs[:, -1] - sorted_row_probs[:, -2]
    result_frame["ai_softmax_margin"] = softmax_margin_arr
    margin_mask = softmax_margin_arr < margin_floor
    suppressed_margin = int((margin_mask & (result_frame["ai_verdict"] != 1)).sum())
    if suppressed_margin > 0:
        console.print(
            f"[warning]Risk Rule 3: forced neutral on {suppressed_margin} rows "
            f"(softmax margin < {margin_floor:.2f})[/warning]"
        )
    result_frame.loc[margin_mask, "ai_verdict"] = 1

    # ? PER-ROW BREAKEVEN WIN RATE VS ROUND-TRIP COST (FEES + SLIPPAGE ONLY FOR ENTRY GATE).
    gate_on = (
        config.strategy.BREAKEVEN_CONFIDENCE_GATE
        if use_breakeven_confidence_gate is None
        else bool(use_breakeven_confidence_gate)
    )
    if gate_on:
        tp_arr = result_frame["ai_take_profit_pct"].values.astype(np.float64)
        sl_arr = result_frame["ai_stop_loss_pct"].values.astype(np.float64)
        leg_cost_pct = (
            config.strategy.ROUND_TRIP_FEE_PCT + 2.0 * config.strategy.SLIPPAGE_PCT
        )
        denom = tp_arr + sl_arr + 1e-8
        breakeven_p = (sl_arr + leg_cost_pct) / denom
        buffer = float(config.strategy.BREAKEVEN_CONFIDENCE_BUFFER)
        conf_arr = result_frame["ai_confidence"].values
        below_be = conf_arr < (breakeven_p + buffer)
        active_side = result_frame["ai_verdict"].values != 1
        breakeven_mask = active_side & below_be
        suppressed_be = int(breakeven_mask.sum())
        if suppressed_be > 0:
            console.print(
                f"[warning]Risk Rule 4: forced neutral on {suppressed_be} rows "
                f"(confidence below TP/SL implied breakeven + {buffer:.2f})[/warning]"
            )
        result_frame.loc[breakeven_mask, "ai_verdict"] = 1

    return result_frame


def predict_model_outputs_for_single_window(
    model: nn.Module,
    window_features: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run a single [seq_len, n_features] window and return production-style scalars (CPU floats).
    """
    model.eval()
    batch = torch.FloatTensor(window_features).unsqueeze(0).to(device)
    verdict_names = ("BUY", "NEUTRAL", "SELL")
    with torch.no_grad():
        outputs = model(batch)
    direction_probabilities = direction_logits_to_probabilities(outputs["direction"])[0]
    verdict_index = int(torch.argmax(direction_probabilities).item())
    
    sizing = outputs["sizing"][0]
    qty_ratio = float(sizing[0].item())
    tp_pct = float(sizing[1].item() * config.strategy.LABEL_TP_PCT_MAX)
    sl_pct = float(sizing[2].item() * config.strategy.LABEL_SL_PCT_MAX)
    
    confidence = float(torch.max(direction_probabilities).item())
    directional_edge = float(max(direction_probabilities[0].item(), direction_probabilities[2].item()) - direction_probabilities[1].item())
    probs_np = direction_probabilities.detach().float().cpu().numpy()
    sorted_p = np.sort(probs_np)
    softmax_margin = float(sorted_p[-1] - sorted_p[-2])

    leg_cost_pct = config.strategy.ROUND_TRIP_FEE_PCT + 2.0 * config.strategy.SLIPPAGE_PCT
    breakeven_p = (sl_pct + leg_cost_pct) / (tp_pct + sl_pct + 1e-8)
    below_breakeven = confidence < (breakeven_p + config.strategy.BREAKEVEN_CONFIDENCE_BUFFER)

    # Apply rules (match run_inference_with_confidence_filter; per-class calibrator not available here)
    if (
        confidence < config.strategy.AI_CONFIDENCE_THRESHOLD
        or directional_edge < config.strategy.MIN_DIRECTIONAL_EDGE
        or (tp_pct / (sl_pct + 1e-6)) < config.strategy.MIN_REWARD_RISK_RATIO
        or softmax_margin < config.strategy.MIN_SOFTMAX_MARGIN
        or (config.strategy.BREAKEVEN_CONFIDENCE_GATE and below_breakeven)
    ):
        verdict_index = 1
    
    return {
        "direction_verdict_code": verdict_index,
        "direction_verdict_label": verdict_names[verdict_index],
        "prob_buy": float(direction_probabilities[0].item()),
        "prob_neutral": float(direction_probabilities[1].item()),
        "prob_sell": float(direction_probabilities[2].item()),
        "confidence": confidence,
        "directional_edge": directional_edge,
        "qty_ratio": qty_ratio,
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
    }
