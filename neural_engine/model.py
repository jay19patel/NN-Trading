# -*- coding: utf-8 -*-
import math
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional

from config import config

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Sinusoidal positions so the model can distinguish order within the window."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, token_sequence: torch.Tensor) -> torch.Tensor:
        sequence_length = token_sequence.size(1)
        return token_sequence + self.pe[:, :sequence_length, :]

class AttentionPool(nn.Module):
    """
    Multi-statistic attention pooling.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.fuse = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # ATTENTION-WEIGHTED SUM
        scores = self.attention_weights(sequence)
        weights = functional.softmax(scores, dim=1)
        attn_pooled = (weights * sequence).sum(dim=1)

        # MEAN POOLING
        mean_pooled = sequence.mean(dim=1)

        # LAST TIMESTEP
        last_pooled = sequence[:, -1, :]

        fused = torch.cat([attn_pooled, mean_pooled, last_pooled], dim=-1)
        return self.fuse(fused)

class MultiHeadTradingModel(nn.Module):
    """
    Causal Transformer with attention pooling for multi-task trading prediction.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        max_sequence_len: int = 512,
    ):
        super().__init__()
        hidden_dim = hidden_dim or config.model.HIDDEN_DIM
        num_heads = num_heads or config.model.NUM_HEADS
        num_layers = num_layers or config.model.NUM_LAYERS
        # Increased dropout for better regularization (calibration)
        dropout = dropout if dropout is not None else 0.2
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_sequence_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._sequence_max_len = max_sequence_len

        self.attention_pool = AttentionPool(hidden_dim)

        self.shared_linear = nn.Linear(hidden_dim, hidden_dim)
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.shared_activation = nn.GELU()
        self.shared_dropout = nn.Dropout(dropout)

        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4), # High dropout for calibration
            nn.Linear(64, 3),
        )

        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, window_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if window_features.size(1) > self._sequence_max_len:
            raise ValueError(f"Sequence length {window_features.size(1)} exceeds positional encoding max")

        projected = self.input_projection(window_features)
        projected = self.input_norm(projected)
        projected = self.input_dropout(projected)
        encoded_positions = self.positional_encoding(projected)

        sequence_length = encoded_positions.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sequence_length, device=encoded_positions.device
        )
        encoded_sequence = self.encoder(encoded_positions, mask=causal_mask)

        pooled = self.attention_pool(encoded_sequence)

        shared_out = self.shared_linear(pooled)
        shared_out = self.shared_norm(shared_out)
        shared_out = self.shared_activation(shared_out)
        shared_out = self.shared_dropout(shared_out)
        shared = pooled + shared_out

        return {
            "direction": self.signal_head(shared),
            "sizing": self.sizing_head(shared),
        }

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probabilities = functional.softmax(logits, dim=1)
        
        # Explicit label smoothing: mix 1-hot with uniform distribution
        smoothing = 0.1
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
            
        target_probs = (probabilities * true_dist).sum(dim=1)
        ce_loss = functional.cross_entropy(logits, targets, reduction="none")
        
        focal_weight = (1.0 - target_probs).clamp(min=1e-6) ** self.gamma

        pred_labels = torch.argmax(logits, dim=1)
        is_opposite = ((targets == 0) & (pred_labels == 2)) | ((targets == 2) & (pred_labels == 0))
        directional_multiplier = torch.ones_like(focal_weight)
        directional_multiplier[is_opposite] = 2.0

        opposite_prob = torch.zeros_like(target_probs)
        long_mask = (targets == 0)
        short_mask = (targets == 2)
        if long_mask.any(): opposite_prob[long_mask] = probabilities[long_mask, 2]
        if short_mask.any(): opposite_prob[short_mask] = probabilities[short_mask, 0]
        directional_multiplier = directional_multiplier * (1.0 + 2.0 * opposite_prob)

        false_trade_signal_on_neutral = (targets == 1) & (pred_labels != 1)
        neutral_violation_scale = float(config.training.FOCAL_NEUTRAL_VIOLATION_SCALE)
        if neutral_violation_scale > 1.0:
            directional_multiplier = torch.where(false_trade_signal_on_neutral, directional_multiplier * neutral_violation_scale, directional_multiplier)

        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_weight

        focal_loss = focal_weight * ce_loss * directional_multiplier
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum() if self.reduction == "sum" else focal_loss

class TradingLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.3, gamma: float = 0.05, class_weights: torch.Tensor | None = None, focal_gamma: float = 2.0, consistency_weight: float = 0.10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.consistency_weight = consistency_weight
        self.direction_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.mse = nn.MSELoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        signal_logits = predictions["direction"]
        sizing_out = predictions["sizing"]
        true_signal = targets["direction"].long()

        # Simplified sizing loss
        true_qty = targets.get("qty_ratio", torch.zeros_like(sizing_out[:, 0]))
        true_tp = targets.get("take_profit_pct", torch.zeros_like(sizing_out[:, 1])) / config.strategy.MAX_ATR_TARGET_PCT
        true_sl = targets.get("stop_loss_pct", torch.zeros_like(sizing_out[:, 2])) / config.strategy.MAX_ATR_STOP_PCT
        true_sizing = torch.stack([true_qty, true_tp, true_sl], dim=1)

        signal_loss = self.direction_loss(signal_logits, true_signal)
        sizing_loss = self.mse(sizing_out, true_sizing)

        # actual_pnl_pct is the expected return from oracle.
        # We want to weight the signal loss by the magnitude of the potential gain/loss.
        actual_pnl = targets.get("actual_pnl_pct", torch.zeros_like(true_qty))
        pnl_weight = (1.0 + actual_pnl.abs()).detach()
        weighted_signal_loss = (self.direction_loss(signal_logits, true_signal) * pnl_weight.mean())

        # Consistency loss: sizing should generally follow the signal confidence
        with_softmax = functional.softmax(signal_logits, dim=1)
        directional_edge = torch.maximum(with_softmax[:, 0], with_softmax[:, 2]) - with_softmax[:, 1]
        target_qty_from_signal = directional_edge.clamp(0.0, 1.0)
        consistency_loss = self.mse(sizing_out[:, 0], target_qty_from_signal.detach())

        total = (self.alpha * weighted_signal_loss + self.beta * sizing_loss + self.consistency_weight * consistency_loss)
        return {"total": total, "signal_loss": signal_loss, "sizing_loss": sizing_loss, "consistency_loss": consistency_loss}
