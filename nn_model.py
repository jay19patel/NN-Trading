# -*- coding: utf-8 -*-
"""
Causal Transformer for direction classification.

Ported from the `basic` branch's MultiHeadTradingModel, trimmed to a single
direction head (LONG / NEUTRAL / SHORT). The model reads a window of the last
N bars of features and attends causally (each timestep sees only itself and
the past), so it can pick up temporal/sequence structure that the single-bar
sklearn model cannot.

Architecture:
    Linear projection → LayerNorm → Dropout → Positional Encoding
    → TransformerEncoder (pre-norm, causal mask) × NUM_LAYERS
    → last-timestep pool → shared trunk (residual) → 3-class head
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding so the model knows candle order."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DirectionTransformer(nn.Module):
    """Causal Transformer encoder with a 3-class direction head."""

    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        hidden = cfg.nn.HIDDEN_DIM
        heads = cfg.nn.NUM_HEADS
        layers = cfg.nn.NUM_LAYERS
        dropout = cfg.nn.DROPOUT
        self._max_len = cfg.nn.MAX_SEQ_LEN

        self.input_projection = nn.Linear(input_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(hidden, max_len=self._max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: stabler deep-transformer training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.pool_norm = nn.LayerNorm(hidden)

        self.shared = nn.Linear(hidden, hidden)
        self.shared_norm = nn.LayerNorm(hidden)
        self.shared_dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (batch, seq_len, input_dim) → logits (batch, num_classes)."""
        if window.size(1) > self._max_len:
            raise ValueError(f"seq_len {window.size(1)} > MAX_SEQ_LEN {self._max_len}")

        x = self.input_projection(window)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = self.pos_encoding(x)

        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.encoder(x, mask=mask)

        pooled = self.pool_norm(x[:, -1, :])           # last timestep = causal summary
        trunk = self.shared_dropout(F.gelu(self.shared_norm(self.shared(pooled))))
        return self.head(pooled + trunk)               # residual

    @torch.no_grad()
    def predict_proba(self, window: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(window), dim=1)
