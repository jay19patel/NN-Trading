# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

from config import config
from ui_utils import console
from rich.live import Live
from rich.table import Table

from models import MultiHeadTradingModel, RiskAwareLoss
from evaluation_metrics import evaluate_model_on_split


class TradingDataset(torch.utils.data.Dataset):
    """Sequence tensors [N, seq, feat] with multi-task targets."""

    def __init__(self, feature_windows: np.ndarray, task_targets: Dict[str, np.ndarray]):
        self.feature_windows = torch.FloatTensor(feature_windows)
        self.upside_targets = torch.FloatTensor(task_targets["upside"])
        self.downside_targets = torch.FloatTensor(task_targets["downside"])
        self.drawdown_targets = torch.FloatTensor(task_targets["drawdown"])
        self.direction_targets = torch.LongTensor(task_targets["direction"])

    def __len__(self) -> int:
        return len(self.feature_windows)

    def __getitem__(self, index: int):
        return self.feature_windows[index], {
            "upside": self.upside_targets[index],
            "downside": self.downside_targets[index],
            "future_drawdown": self.drawdown_targets[index],
            "direction": self.direction_targets[index],
        }


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
        "upside_pct",
        "downside_pct",
        "future_drawdown_pct",
        "reward_risk_ratio",
        "edge_ratio",
        "pain_ratio",
        "direction_label",
        "time",
        "label",
        "index",
    }
    return [c for c in dataframe.select_dtypes(include=[np.number]).columns if c not in excluded_targets]


def build_target_arrays(dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "upside": dataframe["upside_pct"].values,
        "downside": dataframe["downside_pct"].values,
        "drawdown": dataframe["future_drawdown_pct"].values,
        "direction": dataframe["direction_label"].values,
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

    loss_fn = RiskAwareLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.LEARNING_RATE,
        weight_decay=config.training.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    train_dataset = TradingDataset(train_features, train_targets)
    sampler: Optional[WeightedRandomSampler] = None
    if config.training.USE_WEIGHTED_SAMPLER:
        direction_labels = train_targets["direction"].astype(np.int64)
        class_counts = np.bincount(direction_labels, minlength=3)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = torch.DoubleTensor(class_weights[direction_labels])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    patience_left = config.training.EARLY_STOP_PATIENCE

    metrics_table = Table(show_header=True, header_style="bold magenta", border_style="bright_black")
    metrics_table.add_column("Epoch", justify="center")
    metrics_table.add_column("Train loss", justify="right")
    metrics_table.add_column("Val loss", justify="right")
    metrics_table.add_column("Val dir acc", justify="right")
    metrics_table.add_column("Status", justify="center")

    with Live(metrics_table, console=console, refresh_per_second=4):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = {name: tensor.to(device) for name, tensor in batch_targets.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_features)
                loss_dict = loss_fn(outputs, batch_targets)
                loss = loss_dict["total"]
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1

            average_train_loss = running_loss / max(batch_count, 1)

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
                        batch_loss = loss_fn(val_outputs, {
                            "upside": torch.FloatTensor(val_targets["upside"][start:end]).to(device),
                            "downside": torch.FloatTensor(val_targets["downside"][start:end]).to(device),
                            "future_drawdown": torch.FloatTensor(
                                val_targets["drawdown"][start:end]
                            ).to(device),
                            "direction": batch_y_dir,
                        })["total"]
                        val_loss_sum += batch_loss.item()
                        val_batches += 1
                        predicted_direction = torch.argmax(val_outputs["direction"], dim=1)
                        correct_direction += (predicted_direction == batch_y_dir).sum().item()
                        total_direction += batch_X.size(0)
                val_loss_value = val_loss_sum / max(val_batches, 1)
                val_direction_accuracy = (correct_direction / max(total_direction, 1)) * 100.0
                scheduler.step(val_loss_value)

                if val_loss_value < best_val_loss - 1e-6:
                    best_val_loss = val_loss_value
                    best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
                    patience_left = config.training.EARLY_STOP_PATIENCE
                else:
                    patience_left -= 1

            status = "🚀"
            if patience_left <= 0 and val_features is not None:
                status = "early-stop"
            metrics_table.add_row(
                f"{epoch + 1}/{epochs}",
                f"{average_train_loss:.5f}",
                f"{val_loss_value:.5f}" if val_features is not None else "—",
                f"{val_direction_accuracy:.1f}%" if val_features is not None else "—",
                status,
            )

            if val_features is not None and patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_report: Dict[str, float] = {}
    if test_features is not None and test_targets is not None:
        final_report = evaluate_model_on_split(
            model,
            test_features,
            test_targets["direction"],
            test_targets["upside"],
            test_targets["downside"],
            device,
        )
    return model, final_report


def run_inference_with_confidence_filter(
    model: nn.Module, feature_windows: np.ndarray, device: torch.device
) -> pd.DataFrame:
    """Direction softmax + confidence threshold; mirrors legacy column names for downstream tables."""
    model.eval()
    input_tensor = torch.FloatTensor(feature_windows).to(device)
    with torch.no_grad():
        raw_outputs = model(input_tensor)
    direction_probabilities = torch.softmax(raw_outputs["direction"], dim=1)
    verdict_indices = torch.argmax(direction_probabilities, dim=1).cpu().numpy()
    predicted_upside = raw_outputs["upside"].cpu().numpy().reshape(-1)
    confidence_scores = raw_outputs["confidence"].cpu().numpy().reshape(-1)

    result_frame = pd.DataFrame(
        {
            "ai_upside": predicted_upside,
            "ai_confidence": confidence_scores,
            "ai_verdict": verdict_indices,
        }
    )

    raw_buy_signals = int((result_frame["ai_verdict"] == 0).sum())
    raw_sell_signals = int((result_frame["ai_verdict"] == 2).sum())
    console.print(
        f"[info]Raw direction signals: Buy={raw_buy_signals}, Sell={raw_sell_signals}[/info]"
    )

    confidence_floor = config.strategy.AI_CONFIDENCE_THRESHOLD
    low_confidence_mask = result_frame["ai_confidence"] < confidence_floor
    suppressed = int((low_confidence_mask & (result_frame["ai_verdict"] != 1)).sum())
    if suppressed > 0:
        console.print(
            f"[warning]Confidence filter: forced neutral on {suppressed} low-confidence rows (< {confidence_floor})[/warning]"
        )
    result_frame.loc[low_confidence_mask, "ai_verdict"] = 1
    return result_frame
