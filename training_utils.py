# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from config import config
from ui_utils import console
from rich.live import Live
from rich.table import Table

from models import MultiHeadTradingModel, TradingLoss
from evaluation_metrics import evaluate_model_on_split


class TradingDataset(torch.utils.data.Dataset):
    """Sequence tensors [N, seq, feat] with multi-task targets."""

    direction_targets: torch.Tensor
    float_target_tensors: dict[str, torch.Tensor]

    def __init__(self, feature_windows: np.ndarray, task_targets: dict[str, np.ndarray]):
        self.feature_windows = torch.FloatTensor(feature_windows)
        self.direction_targets = torch.LongTensor(task_targets["direction"])
        self.float_target_tensors = {
            target_name: torch.FloatTensor(task_targets[target_name])
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
    Build sliding windows; label at index i-1 is the supervised target for window ending at i-1.

    For each i in [sequence_length, len), appends X[i-sequence_length:i] and y[i-1].
    """
    window_list: List[np.ndarray] = []
    target_sequences: Dict[str, List] = {key: [] for key in task_targets.keys()}

    for end_index in range(sequence_length, len(feature_matrix)):
        window_list.append(feature_matrix[end_index - sequence_length : end_index])
        for target_name in task_targets.keys():
            target_sequences[target_name].append(task_targets[target_name][end_index - 1])

    stacked_windows = np.stack(window_list, axis=0)
    stacked_targets = {key: np.asarray(values) for key, values in target_sequences.items()}
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
        # Index / meta columns
        "time",
        "label",
        "index",
    }
    return [c for c in dataframe.select_dtypes(include=[np.number]).columns if c not in excluded_targets]


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
    cleaned = dataframe.copy().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    bars_per_day = (24 * 60) // 15
    test_row_count = test_days * bars_per_day
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
) -> Tuple[np.ndarray, Dict, np.ndarray, Dict, np.ndarray, Dict, List[str], StandardScaler]:
    """
    Chronological split per symbol: train | validation | test (most recent = test).

    Scaler is fit only on the training portion (pooled across symbols), then applied to val/test.
    Sequences are built inside each symbol to avoid mixing bars across assets or gaps.
    """
    first_frame = next(iter(symbol_dataframes.values()))
    feature_columns = get_feature_cols(first_frame)
    sequence_length = config.model.SEQ_LEN
    bars_per_day = (24 * 60) // 15
    test_row_count = test_days * bars_per_day
    val_row_count = val_days * bars_per_day

    train_frames: Dict[str, pd.DataFrame] = {}
    validation_frames: Dict[str, pd.DataFrame] = {}
    test_frames: Dict[str, pd.DataFrame] = {}

    train_feature_pool: List[pd.DataFrame] = []
    for symbol, raw_frame in symbol_dataframes.items():
        cleaned_frame = raw_frame.copy().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        train_frames[symbol] = cleaned_frame.iloc[: -(val_row_count + test_row_count)]
        validation_frames[symbol] = cleaned_frame.iloc[
            -(val_row_count + test_row_count) : -test_row_count
        ]
        test_frames[symbol] = cleaned_frame.iloc[-test_row_count:]
        train_feature_pool.append(train_frames[symbol][feature_columns])

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

    train_target_parts: Dict[str, List[np.ndarray]] = {name: [] for name in build_target_arrays(first_frame)}
    val_target_parts: Dict[str, List[np.ndarray]] = {name: [] for name in train_target_parts}
    test_target_parts: Dict[str, List[np.ndarray]] = {name: [] for name in train_target_parts}

    for symbol in symbol_dataframes:
        train_features = scaler.transform(train_frames[symbol][feature_columns])
        validation_features = scaler.transform(validation_frames[symbol][feature_columns])
        test_features = scaler.transform(test_frames[symbol][feature_columns])

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
        scaler,
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
    model = MultiHeadTradingModel(input_dim=input_dim).to(device)
    parameter_count = count_trainable_parameters(model)
    console.print(
        f"[info]Model: inputs={input_dim}, parameters={parameter_count:,}, device={device}[/info]"
    )

    # Compute class weights from actual label distribution instead of WeightedSampler.
    # WeightedSampler causes train/val distribution mismatch (val < train loss at epoch 1)
    # because it artificially balances training, but validation has real natural distribution.
    # Passing weights to CrossEntropyLoss is mathematically equivalent but avoids this mismatch.
    direction_labels = train_targets["direction"].astype(np.int64)
    class_counts = np.bincount(direction_labels, minlength=3).astype(np.float32)
    class_weights_np = (class_counts.sum() / (3.0 * class_counts + 1e-6))
    class_weight_tensor = torch.FloatTensor(class_weights_np).to(device)
    console.print(
        f"[info]Class weights: LONG={class_weights_np[0]:.2f}, "
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
    ).to(device)
    console.print(f"[info]Loss: {'FocalLoss(γ=' + str(config.training.FOCAL_GAMMA) + ')' if config.training.USE_FOCAL_LOSS else 'CrossEntropy'}[/info]")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.LEARNING_RATE,
        weight_decay=config.training.WEIGHT_DECAY,
    )

    train_dataset = TradingDataset(train_features, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        # num_workers=0 is correct for MPS — macOS multiprocessing with GPU is unreliable
        # pin_memory=False for MPS — only helps CUDA, wastes memory on Apple Silicon
        num_workers=0,
        pin_memory=False,
    )

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
    best_state: Optional[Dict] = None
    patience_left = config.training.EARLY_STOP_PATIENCE

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
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = {name: tensor.to(device) for name, tensor in batch_targets.items()}
                optimizer.zero_grad(set_to_none=True)
                
                # MPS Autocast for M4 speed boost (Mixed Precision)
                with torch.autocast(device_type="mps" if "mps" in str(device) else "cpu", enabled=True):
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
            if val_features is not None and val_targets is not None:
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                correct_direction = 0
                total_direction = 0
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(val_features).to(device)
                    direction_actual = torch.LongTensor(val_targets["direction"]).to(device)
                    chunk_size = 2048
                    for start in range(0, val_tensor.size(0), chunk_size):
                        end = min(start + chunk_size, val_tensor.size(0))
                        batch_X = val_tensor[start:end]
                        batch_y_dir = direction_actual[start:end]
                        val_outputs = model(batch_X)
                        val_chunk_targets = {
                            "upside": torch.FloatTensor(val_targets["upside"][start:end]).to(device),
                            "downside": torch.FloatTensor(val_targets["downside"][start:end]).to(device),
                            "future_drawdown": torch.FloatTensor(
                                val_targets["drawdown"][start:end]
                            ).to(device),
                            "take_profit_pct": torch.FloatTensor(
                                val_targets["take_profit_pct"][start:end]
                            ).to(device),
                            "stop_loss_pct": torch.FloatTensor(
                                val_targets["stop_loss_pct"][start:end]
                            ).to(device),
                            "direction": batch_y_dir,
                            "actual_pnl_pct": torch.FloatTensor(val_targets["actual_pnl_pct"][start:end]).to(device),
                            "qty_ratio": torch.FloatTensor(val_targets["qty_ratio"][start:end]).to(device),
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

                if val_loss_value < best_val_loss - 1e-6:
                    best_val_loss = val_loss_value
                    best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
                    patience_left = config.training.EARLY_STOP_PATIENCE
                else:
                    patience_left -= 1

            status = "Phase 1"
            if patience_left <= 0 and val_features is not None:
                status = "early-stop"
            metrics_table.add_row(
                f"{epoch + 1}/{epochs}",
                f"{average_train_loss:.5f}",
                f"{val_loss_value:.5f}" if val_features is not None else "—",
                f"{val_direction_accuracy:.1f}%" if val_features is not None else "—",
                "—",  # PnL Effect is Phase 2 only
                status,
            )

            if val_features is not None and patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    console.print("\n[highlight]Phase 2: Reinforcement Learning (PnL Consequence Fine-Tuning)[/highlight]")
    console.print(f"[info]RL gamma={config.training.LOSS_GAMMA} | LR={config.training.RL_LEARNING_RATE} | epochs={config.training.RL_FINE_TUNE_EPOCHS}[/info]")

    # Phase 2: RL Fine-Tuning with PnL Consequence Loss
    # FREEZE encoder backbone — only fine-tune heads to prevent catastrophic forgetting
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

    # Cache Phase 1 predictions for KL divergence regularization
    model.eval()
    with torch.no_grad():
        phase1_val_logits = None
        if val_features is not None:
            val_tensor_p1 = torch.FloatTensor(val_features).to(device)
            phase1_val_logits = model(val_tensor_p1)["direction"]
            phase1_val_probs = F.softmax(phase1_val_logits, dim=1)

    rl_loss_fn = TradingLoss(
        alpha=config.training.LOSS_ALPHA,
        beta=config.training.LOSS_BETA,
        gamma=config.training.LOSS_GAMMA,  # PnL penalty now active
        class_weights=class_weight_tensor,
        focal_gamma=config.training.FOCAL_GAMMA,
        use_focal=config.training.USE_FOCAL_LOSS,
    ).to(device)

    rl_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.RL_LEARNING_RATE,
        weight_decay=config.training.WEIGHT_DECAY,
    )

    # FIXED: Independent tracking for Phase 2
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
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = {name: tensor.to(device) for name, tensor in batch_targets.items()}
                rl_optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="mps" if "mps" in str(device) else "cpu", enabled=True):
                    outputs = model(batch_features)
                    loss_dict = rl_loss_fn(outputs, batch_targets)
                    loss = loss_dict["total"]
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                rl_optimizer.step()
                running_loss_tensor = running_loss_tensor + loss.detach()
                running_pnl_tensor = running_pnl_tensor + loss_dict["pnl_effect"].detach()
                batch_count += 1

            # Single GPU-CPU sync per epoch
            average_train_loss = (running_loss_tensor / max(batch_count, 1)).item()
            average_pnl_effect = (running_pnl_tensor / max(batch_count, 1)).item()

            val_loss_value = float("nan")
            val_direction_accuracy = float("nan")
            if val_features is not None and val_targets is not None:
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                correct_direction = 0
                total_direction = 0
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(val_features).to(device)
                    direction_actual = torch.LongTensor(val_targets["direction"]).to(device)
                    chunk_size = 2048
                    for start in range(0, val_tensor.size(0), chunk_size):
                        end = min(start + chunk_size, val_tensor.size(0))
                        batch_X = val_tensor[start:end]
                        batch_y_dir = direction_actual[start:end]
                        val_outputs = model(batch_X)
                        val_chunk_targets = {
                            "upside": torch.FloatTensor(val_targets["upside"][start:end]).to(device),
                            "downside": torch.FloatTensor(val_targets["downside"][start:end]).to(device),
                            "future_drawdown": torch.FloatTensor(val_targets["drawdown"][start:end]).to(device),
                            "take_profit_pct": torch.FloatTensor(val_targets["take_profit_pct"][start:end]).to(device),
                            "stop_loss_pct": torch.FloatTensor(val_targets["stop_loss_pct"][start:end]).to(device),
                            "direction": batch_y_dir,
                            "actual_pnl_pct": torch.FloatTensor(val_targets["actual_pnl_pct"][start:end]).to(device),
                            "qty_ratio": torch.FloatTensor(val_targets["qty_ratio"][start:end]).to(device),
                        }
                        batch_loss = rl_loss_fn(val_outputs, val_chunk_targets)["total"]
                        val_loss_sum += batch_loss.item()
                        val_batches += 1
                        predicted_direction = torch.argmax(val_outputs["direction"], dim=1)
                        correct_direction += (predicted_direction == batch_y_dir).sum().item()
                        total_direction += batch_X.size(0)
                val_loss_value = val_loss_sum / max(val_batches, 1)
                val_direction_accuracy = (correct_direction / max(total_direction, 1)) * 100.0

                # FIXED: Use separate rl_best_val_loss and rl_best_state for Phase 2
                if val_loss_value < rl_best_val_loss - 1e-6:
                    rl_best_val_loss = val_loss_value
                    rl_best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
                    patience_left = config.training.EARLY_STOP_PATIENCE
                else:
                    patience_left -= 1

            status = "Phase 2 (RL)"
            if patience_left <= 0 and val_features is not None:
                status = "early-stop ✓"
            metrics_table.add_row(
                f"RL {epoch + 1}/{rl_epochs}",
                f"{average_train_loss:.5f}",
                f"{val_loss_value:.5f}" if val_features is not None else "—",
                f"{val_direction_accuracy:.1f}%" if val_features is not None else "—",
                f"{average_pnl_effect:.4f}",  # Monitor RL health — should stay near 0
                status,
            )

            if val_features is not None and patience_left <= 0:
                break

    # FIXED: Restore the best Phase 2 checkpoint if it improved on Phase 1
    if rl_best_state is not None:
        model.load_state_dict(rl_best_state)
        console.print("[success]✅ Phase 2 checkpoint restored (RL improved val loss)[/success]")
    elif best_state is not None:
        # Phase 2 didn't improve — keep Phase 1's best
        model.load_state_dict(best_state)
        console.print("[warning]⚠️  Phase 2 did not improve val loss. Using Phase 1 best checkpoint.[/warning]")

    # Unfreeze all params for inference
    for param in model.parameters():
        param.requires_grad = True

    # ---- Per-class confidence calibration on val set ----
    calibrated_thresholds = {0: config.strategy.AI_CONFIDENCE_THRESHOLD,
                             2: config.strategy.AI_CONFIDENCE_THRESHOLD}
    if val_features is not None and val_targets is not None:
        model.eval()
        with torch.no_grad():
            val_tensor_cal = torch.FloatTensor(val_features).to(device)
            val_out = model(val_tensor_cal)
            val_probs = torch.softmax(val_out["direction"], dim=1).cpu().numpy()
        val_true = val_targets["direction"].astype(int)
        for cls, cls_name in [(0, "BUY"), (2, "SELL")]:
            # Find threshold that maximizes F1-like score (balance precision + recall)
            best_thresh = config.strategy.AI_CONFIDENCE_THRESHOLD
            best_f1 = 0.0
            best_prec = 0.0
            # Force threshold to be at least 0.44 (0.33 is random, anything below 0.44 is too noisy)
            for thresh in np.arange(0.44, 0.65, 0.01):
                pred_mask = (np.argmax(val_probs, axis=1) == cls) & (val_probs[:, cls] >= thresh)
                total_pred = pred_mask.sum()
                if total_pred < 5:  # lowered sample requirement since we are looking at higher thresholds
                    continue
                correct = (val_true[pred_mask] == cls).sum()
                precision = correct / total_pred
                # Recall: of all true cls, how many did we catch at this threshold?
                true_cls_mask = val_true == cls
                recall = correct / max(true_cls_mask.sum(), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-6)
                if f1 > best_f1 and precision >= 0.45:
                    best_f1 = f1
                    best_prec = precision
                    best_thresh = float(thresh)
            calibrated_thresholds[cls] = best_thresh
            console.print(f"[info]Calibrated {cls_name} threshold: {best_thresh:.2f} (precision={best_prec:.1%}, f1={best_f1:.3f})[/info]")

    final_report: Dict[str, float] = {}
    if test_features is not None and test_targets is not None:
        final_report = evaluate_model_on_split(model, test_features, test_targets, device)
    final_report["calibrated_thresholds"] = calibrated_thresholds
    return model, final_report


def run_inference_with_confidence_filter(
    model: nn.Module, feature_windows: np.ndarray, device: torch.device,
    calibrated_thresholds: Dict[int, float] | None = None,
) -> pd.DataFrame:
    """Direction softmax, per-class confidence gate, and volatility-aware TP/SL percentage heads."""
    model.eval()
    input_tensor = torch.FloatTensor(feature_windows).to(device)
    with torch.no_grad():
        raw_outputs = model(input_tensor)
    
    direction_probabilities = torch.softmax(raw_outputs["direction"], dim=1)
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
            "ai_qty_ratio": qty_ratios,
            "ai_verdict": verdict_indices,
            "ai_take_profit_pct": take_profit_pct,
            "ai_stop_loss_pct": stop_loss_pct,
            "ai_prob_buy": buy_probability,
            "ai_prob_neutral": neutral_probability,
            "ai_prob_sell": sell_probability,
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
        console.print(f"[warning]Risk Rule 1: forced neutral on {suppressed_count} low-confidence rows (per-class calibrated)[/warning]")
    
    # R:R Rule
    rr_ratios = result_frame["ai_take_profit_pct"] / (result_frame["ai_stop_loss_pct"] + 1e-6)
    min_rr = config.strategy.MIN_REWARD_RISK_RATIO
    poor_rr_mask = rr_ratios < min_rr
    suppressed_rr = int((poor_rr_mask & (result_frame["ai_verdict"] != 1)).sum())
    if suppressed_rr > 0:
        console.print(f"[warning]Risk Rule 2: forced neutral on {suppressed_rr} poor R:R rows (< {min_rr})[/warning]")
    result_frame.loc[poor_rr_mask, "ai_verdict"] = 1

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
    direction_probabilities = torch.softmax(outputs["direction"], dim=1)[0]
    verdict_index = int(torch.argmax(direction_probabilities).item())
    
    sizing = outputs["sizing"][0]
    qty_ratio = float(sizing[0].item())
    tp_pct = float(sizing[1].item() * config.strategy.LABEL_TP_PCT_MAX)
    sl_pct = float(sizing[2].item() * config.strategy.LABEL_SL_PCT_MAX)
    
    confidence = float(torch.max(direction_probabilities).item())
    
    # Apply rules
    if confidence < config.strategy.AI_CONFIDENCE_THRESHOLD or (tp_pct / (sl_pct + 1e-6)) < config.strategy.MIN_REWARD_RISK_RATIO:
        verdict_index = 1
    
    return {
        "direction_verdict_code": verdict_index,
        "direction_verdict_label": verdict_names[verdict_index],
        "prob_buy": float(direction_probabilities[0].item()),
        "prob_neutral": float(direction_probabilities[1].item()),
        "prob_sell": float(direction_probabilities[2].item()),
        "confidence": confidence,
        "qty_ratio": qty_ratio,
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
    }
