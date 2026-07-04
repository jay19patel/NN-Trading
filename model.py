# -*- coding: utf-8 -*-
"""
Causal Transformer — direction + magnitude prediction.

Three outputs for the next LOOKAHEAD_BARS candles:
  dir_logit — logit of P(upside move dominates downside move)
  up_mag    — predicted max upside move  (%, range [0, MAX_RETURN_PCT])
  down_mag  — predicted max downside move (%, range [0, MAX_RETURN_PCT])

At inference:
  p_up       = sigmoid(dir_logit)          → direction probability
  confidence = max(p_up, 1 - p_up)         → how sure the model is
  magnitude  = up_mag if p_up >= 0.5 else down_mag
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg, MAX_RETURN_PCT


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DirectionMagnitudeModel(nn.Module):
    """
    Causal Transformer with a shared encoder and three heads.

    Input  : (B, T, F) — T past candle feature windows
    Output : (dir_logit, up_mag, down_mag) — each (B,)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden   = cfg.nn.HIDDEN_DIM
        heads    = cfg.nn.NUM_HEADS
        layers   = cfg.nn.NUM_LAYERS
        dropout  = cfg.nn.DROPOUT
        self._max_len    = cfg.nn.MAX_SEQ_LEN
        self._max_return = MAX_RETURN_PCT

        # ── Shared encoder ────────────────────────────────────────────────────
        self.input_projection = nn.Linear(input_dim, hidden)
        self.input_norm       = nn.LayerNorm(hidden)
        self.input_dropout    = nn.Dropout(dropout)
        self.pos_encoding     = PositionalEncoding(hidden, max_len=self._max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.pool_norm   = nn.LayerNorm(hidden)
        self.shared      = nn.Linear(hidden, hidden)
        self.shared_norm = nn.LayerNorm(hidden)
        self.shared_drop = nn.Dropout(dropout)

        # ── Heads ─────────────────────────────────────────────────────────────
        def _mag_head() -> nn.Sequential:
            # Sigmoid output → scaled to [0, MAX_RETURN_PCT]
            return nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
                nn.Sigmoid(),
            )

        self.dir_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.up_head   = _mag_head()
        self.down_head = _mag_head()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, window: torch.Tensor) -> torch.Tensor:
        """Shared causal encoder. window: (B, T, F) → pooled (B, hidden)."""
        if window.size(1) > self._max_len:
            raise ValueError(f"seq_len {window.size(1)} > MAX_SEQ_LEN {self._max_len}")
        x = self.input_projection(window)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = self.pos_encoding(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        x      = self.encoder(x, mask=causal_mask, is_causal=True)
        pooled = self.pool_norm(x[:, -1, :])
        trunk  = self.shared_drop(F.gelu(self.shared_norm(self.shared(pooled))))
        return pooled + trunk

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        window : (B, T, F)

        Returns
        -------
        dir_logit : (B,) — logit of P(up)
        up_mag    : (B,) — predicted upside move %  [0, MAX_RETURN_PCT]
        down_mag  : (B,) — predicted downside move % [0, MAX_RETURN_PCT]
        """
        z = self._encode(window)
        dir_logit = self.dir_head(z).squeeze(-1)
        up_mag    = self.up_head(z).squeeze(-1)   * self._max_return
        down_mag  = self.down_head(z).squeeze(-1) * self._max_return
        return dir_logit, up_mag, down_mag

    @torch.no_grad()
    def predict(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference helper returning probabilities and magnitudes."""
        dir_logit, up_mag, down_mag = self.forward(window)
        p_up = torch.sigmoid(dir_logit)
        return {
            "p_up":       p_up,
            "confidence": torch.maximum(p_up, 1.0 - p_up),
            "up_mag":     up_mag,
            "down_mag":   down_mag,
        }
