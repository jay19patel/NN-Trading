# -*- coding: utf-8 -*-
"""
Neural Network Model Architecture
===================================
Causal Transformer Encoder with multi-task output heads.

Outputs:
  direction  — 3-class logits (LONG=0, NEUTRAL=1, SHORT=2)
  sizing     — 3 values: qty_ratio, TP%, SL% (normalized 0-1)
  magnitude  — expected move size (0-1 normalized)
  time       — estimated bars to target (0-1 normalized by lookahead)
"""
import math
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding so the model knows candle order within the window."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, token_sequence: torch.Tensor) -> torch.Tensor:
        seq_len = token_sequence.size(1)
        return token_sequence + self.pe[:, :seq_len, :]


class SequencePool(nn.Module):
    """
    Extracts the last timestep from the sequence.
    For a causal time-series model, the last timestep already contains
    the aggregated causal context of the entire window.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence shape: (B, T, H)
        last_pooled = sequence[:, -1, :]  # (B, H)
        return self.norm(last_pooled)


class MultiHeadTradingModel(nn.Module):
    """
    Causal Transformer with multi-task output heads for trading signal generation.

    Architecture:
      Linear projection → LayerNorm → Dropout → Positional Encoding
      → TransformerEncoder (pre-norm, causal mask) × NUM_LAYERS
      → SequencePool (last timestep) → Shared trunk → 4 task heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        max_sequence_len: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or cfg.model.HIDDEN_DIM
        num_heads = num_heads or cfg.model.NUM_HEADS
        num_layers = num_layers or cfg.model.NUM_LAYERS
        dropout = dropout if dropout is not None else cfg.model.DROPOUT
        max_sequence_len = max_sequence_len or cfg.model.MAX_SEQ_LEN

        # ── Input projection ──────────────────────────────────────────────
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_sequence_len)
        self._sequence_max_len = max_sequence_len

        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm: better gradient flow and training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Pooling + shared trunk ────────────────────────────────────────
        self.sequence_pool = SequencePool(hidden_dim)

        self.shared_linear = nn.Linear(hidden_dim, hidden_dim)
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.shared_activation = nn.GELU()
        self.shared_dropout = nn.Dropout(dropout)

        # ── Task heads ────────────────────────────────────────────────────
        # Direction head: 3-class classifier (LONG / NEUTRAL / SHORT)
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Sizing head: qty_ratio, normalized TP%, normalized SL%  (all 0-1)
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        # Magnitude head: expected move size (0-1, normalized by MAX_ATR_TARGET_PCT)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Time head: bars-to-target estimate (0-1, normalized by LOOKAHEAD_BARS)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Temperature parameter for confidence calibration (post-training scaling)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialisation for linear layers (helps with deep transformers)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, window_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            window_features: (batch, window_size, input_dim)

        Returns:
            dict with keys: direction, sizing, magnitude, time
        """
        if window_features.size(1) > self._sequence_max_len:
            raise ValueError(
                f"Sequence length {window_features.size(1)} exceeds "
                f"positional encoding max {self._sequence_max_len}"
            )

        # Input projection
        x = self.input_projection(window_features)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = self.positional_encoding(x)

        # Causal mask: each timestep can only attend to itself and the past
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        x = self.encoder(x, mask=causal_mask)

        # Extract the final timestep
        pooled = self.sequence_pool(x)

        # Shared trunk with residual connection
        trunk = self.shared_linear(pooled)
        trunk = self.shared_norm(trunk)
        trunk = self.shared_activation(trunk)
        trunk = self.shared_dropout(trunk)
        shared = pooled + trunk

        return {
            "direction": self.signal_head(shared),
            "sizing": self.sizing_head(shared),
            "magnitude": self.magnitude_head(shared),
            "time": self.time_head(shared),
        }

    def calibrated_direction_probs(self, window_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling for better confidence calibration.
        Temperature > 1 softens probabilities; < 1 sharpens them.

        Returns: softmax probabilities (batch, 3)
        """
        logits = self.forward(window_features)["direction"]
        return F.softmax(logits / self.temperature.clamp(min=0.1), dim=1)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TradingLoss(nn.Module):
    """
    Multi-task loss combining:
      1. CrossEntropy on direction (primary, weight=1.0)
      2. MSE on sizing outputs — qty, TP%, SL% (weight=0.3)
      3. MSE on magnitude (weight=0.3)
      4. MSE on time-to-target (weight=0.1)

    Simplified loss provides stable gradients compared to custom focal loss
    with discontinuous manual multipliers.
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        signal_logits = predictions["direction"]
        sizing_out = predictions["sizing"]
        true_signal = targets["direction"].long()

        # ── Direction loss ──────────────────────────────────────────────
        # Use standard CrossEntropy with label smoothing.
        # Note: We do NOT pass class weights here because the DataLoader
        # already uses WeightedRandomSampler. Doing both is mathematically flawed
        # and double-weights minority classes, leading to model collapse.
        signal_loss = F.cross_entropy(
            signal_logits, 
            true_signal, 
            label_smoothing=self.label_smoothing
        )

        # ── Sizing loss ─────────────────────────────────────────────────
        max_tp = cfg.testing.MAX_ATR_TARGET_PCT
        max_sl = cfg.testing.MAX_ATR_STOP_PCT
        true_qty = targets.get("qty_ratio", torch.zeros_like(sizing_out[:, 0]))
        true_tp = targets.get("take_profit_pct", torch.zeros_like(sizing_out[:, 1])) / max_tp
        true_sl = targets.get("stop_loss_pct", torch.zeros_like(sizing_out[:, 2])) / max_sl
        true_sizing = torch.stack([true_qty, true_tp.clamp(0, 1), true_sl.clamp(0, 1)], dim=1)
        sizing_loss = self.mse(sizing_out, true_sizing)

        # ── Magnitude + Time losses ──────────────────────────────────────
        true_magnitude = targets.get(
            "magnitude", torch.zeros_like(sizing_out[:, 0])
        ).unsqueeze(1)
        true_time = targets.get(
            "time", torch.zeros_like(sizing_out[:, 0])
        ).unsqueeze(1)
        magnitude_loss = self.mse(predictions["magnitude"], true_magnitude)
        time_loss = self.mse(predictions["time"], true_time)

        # ── Total (signal dominates) ─────────────────────────────────────
        total = (
            1.0 * signal_loss
            + 0.3 * sizing_loss
            + 0.3 * magnitude_loss
            + 0.1 * time_loss
        )
        return {
            "total": total,
            "signal_loss": signal_loss,
            "sizing_loss": sizing_loss,
            "magnitude_loss": magnitude_loss,
            "time_loss": time_loss,
        }
