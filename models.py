# -*- coding: utf-8 -*-
"""
Transformer encoder for causal sequence modeling on scaled OHLCV-derived features.

Uses PyTorch's nn.TransformerEncoder with a causal attention mask so each timestep
only attends to past and current bars (no lookahead inside the window).
"""
import math
from typing import Dict

import torch
import torch.nn as nn

from config import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positions so the model can distinguish order within the window."""

    def __init__(self, d_model: int, max_len: int = 256):
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


class MultiHeadTradingModel(nn.Module):
    """
    Causal Transformer over a fixed-length window, last timestep pooled for heads.

    Outputs: regression (upside/downside %), risk, confidence, and direction logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        max_sequence_len: int = 256,
    ):
        super().__init__()
        hidden_dim = hidden_dim or config.model.HIDDEN_DIM
        num_heads = num_heads or config.model.NUM_HEADS
        num_layers = num_layers or config.model.NUM_LAYERS
        dropout = dropout if dropout is not None else config.model.DROPOUT

        self.input_projection = nn.Linear(input_dim, hidden_dim)
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

        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.upside_head = nn.Linear(hidden_dim, 1)
        self.downside_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 3)

    def forward(self, window_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # window_features: [batch, seq_len, features]
        if window_features.size(1) > self._sequence_max_len:
            raise ValueError(
                f"Sequence length {window_features.size(1)} exceeds positional encoding max {self._sequence_max_len}"
            )
        projected = self.input_projection(window_features)
        encoded_positions = self.positional_encoding(projected)
        # Causal attention: upper triangle masked so timestep i cannot attend to j > i.
        sequence_length = encoded_positions.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sequence_length, device=encoded_positions.device
        )
        encoded_sequence = self.encoder(encoded_positions, mask=causal_mask)
        last_step = encoded_sequence[:, -1, :]
        shared = self.shared_layer(last_step)
        return {
            "confidence": torch.sigmoid(self.confidence_head(shared)),
            "upside": self.upside_head(shared),
            "downside": self.downside_head(shared),
            "risk": torch.sigmoid(self.risk_head(shared)),
            "direction": self.direction_head(shared),
        }


class RiskAwareLoss(nn.Module):
    """
    Multi-task loss: direction (CE) plus regression on upside, downside, risk, confidence.

    Confidence targets are derived only from ground-truth targets (no feedback from predicted
    confidence into regression loss — that would let the model game the loss).
    """

    def __init__(
        self,
        confidence_weight: float = 0.5,
        upside_weight: float = 1.5,
        downside_weight: float = 1.5,
        risk_weight: float = 0.5,
        direction_weight: float = 1.0,
    ):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
        self.risk_weight = risk_weight
        self.direction_weight = direction_weight
        self.regression_loss = nn.SmoothL1Loss(reduction="mean")
        self.confidence_loss = nn.MSELoss(reduction="mean")
        self.direction_loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        upside = predictions["upside"].squeeze(-1)
        downside = predictions["downside"].squeeze(-1)
        confidence = predictions["confidence"].squeeze(-1)

        upside_targets = targets["upside"]
        downside_targets = targets["downside"]

        upside_term = self.regression_loss(upside, upside_targets)
        downside_term = self.regression_loss(downside, downside_targets)

        direction_term = self.direction_loss_fn(predictions["direction"], targets["direction"].long())

        significant_move = torch.maximum(
            upside_targets, torch.abs(downside_targets)
        ) > 1.0
        strong_bull = upside_targets > torch.abs(downside_targets) * 1.5
        strong_bear = torch.abs(downside_targets) > upside_targets * 1.5
        confidence_target = (significant_move & (strong_bull | strong_bear)).float()
        confidence_term = self.confidence_loss(confidence, confidence_target)

        risk_target = torch.clamp(torch.abs(targets["future_drawdown"]) / 20.0, 0.0, 1.0)
        risk_term = self.regression_loss(predictions["risk"].squeeze(-1), risk_target)

        total_loss = (
            self.confidence_weight * confidence_term
            + self.upside_weight * upside_term
            + self.downside_weight * downside_term
            + self.risk_weight * risk_term
            + self.direction_weight * direction_term
        )
        return {
            "total": total_loss,
            "confidence": confidence_term,
            "upside": upside_term,
            "downside": downside_term,
            "risk": risk_term,
            "direction": direction_term,
        }
