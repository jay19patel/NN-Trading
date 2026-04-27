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
import torch.nn.functional as functional

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
        # HEAD 1: Signal prediction (Buy=1, Sell=2, Hold=0)
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
        
        # HEAD 2: Position sizing (qty, target, sl)
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

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
        
        signal_logits = self.signal_head(shared)
        sizing = self.sizing_head(shared)
        
        # Output bounds mapping applied at inference/loss time
        return {
            "direction": signal_logits,
            "sizing": sizing, # [qty_ratio, target_pct_raw, sl_pct_raw]
        }


class TradingLoss(nn.Module):
    """
    Combined loss that learns from:
    1. Signal correctness (was direction right?)
    2. Sizing accuracy (was qty/target/sl good?)
    3. Actual PnL (what did the trade actually do?)
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha     # Signal loss weight
        self.beta = beta       # Sizing loss weight
        self.gamma = gamma     # PnL penalty weight
        
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        signal_logits = predictions["direction"]
        sizing_out = predictions["sizing"]
        
        true_signal = targets["direction"].long()
        
        # Build true sizing tensor [qty, target_scaled, sl_scaled]
        # target and sl need to be scaled to 0-1 for MSE against Sigmoid output
        strategy = config.strategy
        max_tp = strategy.LABEL_TP_PCT_MAX
        max_sl = strategy.LABEL_SL_PCT_MAX
        
        true_qty = targets["qty_ratio"]
        true_tp = targets["take_profit_pct"] / max_tp
        true_sl = targets["stop_loss_pct"] / max_sl
        
        true_sizing = torch.stack([true_qty, true_tp, true_sl], dim=1)
        
        # Loss 1: Signal correctness
        signal_loss = self.ce(signal_logits, true_signal)
        
        # Loss 2: Sizing accuracy
        sizing_loss = self.mse(sizing_out, true_sizing)
        
        # Loss 3: PnL-based consequence learning
        # We use the actual_pnl_pct from targets to penalize/reward
        actual_pnl = targets["actual_pnl_pct"]
        pred_qty = sizing_out[:, 0]
        
        is_loss = (actual_pnl < 0).float()
        is_profit = (actual_pnl > 0).float()
        
        # Penalty: lose money + high qty = big penalty
        # Reward: make money = incentive to be confident
        pnl_effect = (
            is_loss * actual_pnl.abs() * pred_qty
            - is_profit * actual_pnl.abs() * (1 - pred_qty)
        ).mean()
        
        total = (
            self.alpha * signal_loss 
            + self.beta * sizing_loss 
            + self.gamma * pnl_effect
        )
        
        return {
            "total": total,
            "signal_loss": signal_loss,
            "sizing_loss": sizing_loss,
            "pnl_effect": pnl_effect
        }
