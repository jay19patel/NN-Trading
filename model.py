# -*- coding: utf-8 -*-
"""
Causal Transformer v2 — 2-output architecture.

Output heads (exactly 2):
  direction   : (B, 3) raw logits for LONG=0 / NEUTRAL=1 / SHORT=2
  move_q      : (B, 3) sorted quantiles [q10, q50, q90] of max favorable
                move %, normalised to [0, 1].  De-normalise by × MAX_MFE_PCT.

Removed from v1:
  mae_q head, time head (these are not model predictions — they are risk
  management decisions handled outside the model).

Temperature calibration:
  self.temperature — fitted post-training via fit_temperature(val_logits, val_labels).
  Use calibrated_direction_probs() for inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


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


class QuantileTradingModel(nn.Module):
    """
    Causal Transformer with direction classification and move quantile regression.

    Monotone quantile ordering is enforced by torch.sort on the move_head output —
    simpler and more stable than cumulative softplus during early training.
    """

    def __init__(self, input_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        hidden   = cfg.nn.HIDDEN_DIM
        heads    = cfg.nn.NUM_HEADS
        layers   = cfg.nn.NUM_LAYERS
        dropout  = cfg.nn.DROPOUT
        self._max_len = cfg.nn.MAX_SEQ_LEN

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
        self.pool_norm = nn.LayerNorm(hidden)
        self.shared      = nn.Linear(hidden, hidden)
        self.shared_norm = nn.LayerNorm(hidden)
        self.shared_drop = nn.Dropout(dropout)

        # ── Head 1: direction classification (3 classes) ──────────────────────
        self.direction_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

        # ── Head 2: move quantile regression ─────────────────────────────────
        # Outputs 3 raw sigmoid values in [0, 1], then sorted to enforce
        # q10 ≤ q50 ≤ q90 analytically.
        self.move_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )

        # ── Temperature (fitted post-training, not trained via SGD) ───────────
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

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
        x    = self.input_projection(window)
        x    = self.input_norm(x)
        x    = self.input_dropout(x)
        x    = self.pos_encoding(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        pooled = self.pool_norm(x[:, -1, :])
        trunk  = self.shared_drop(F.gelu(self.shared_norm(self.shared(pooled))))
        return pooled + trunk                        # residual

    def forward(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        {
          "direction": (B, 3)  raw logits (not softmax)
          "move_q":    (B, 3)  sorted [q10, q50, q90] in [0, 1] normalised space
        }
        """
        z = self._encode(window)
        direction = self.direction_head(z)
        move_raw  = self.move_head(z)                # (B, 3), values in (0, 1)
        move_q, _ = torch.sort(move_raw, dim=1)      # enforce q10 ≤ q50 ≤ q90
        return {"direction": direction, "move_q": move_q}

    def calibrated_direction_probs(self, window: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled softmax probabilities, shape (B, 3)."""
        logits = self.forward(window)["direction"]
        return F.softmax(logits / self.temperature.clamp(min=0.1), dim=1)

    def fit_temperature(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter:   int = 200,
    ) -> float:
        """
        Post-training temperature scaling on held-out validation logits.

        Minimises NLL w.r.t. temperature using L-BFGS.
        Updates self.temperature in-place; returns fitted scalar.
        """
        T = nn.Parameter(torch.ones(1, device=val_logits.device))
        opt = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)
        labels = val_labels.long()

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(val_logits / T.clamp(min=0.05), labels)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.temperature.copy_(T.clamp(min=0.3, max=10.0))
        return float(self.temperature.item())
