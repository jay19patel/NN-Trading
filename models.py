# -*- coding: utf-8 -*-
"""
Transformer encoder for causal sequence modeling on scaled OHLCV-derived features.

Uses PyTorch's nn.TransformerEncoder with a causal attention mask so each timestep
only attends to past and current bars (no lookahead inside the window).

CHANGES vs previous version:
  1. FocalLoss replaces CrossEntropyLoss — focuses on hard/misclassified examples
  2. AttentionPool replaces last-timestep pooling — captures patterns across full window
  3. Residual connection in shared layer — prevents gradient degradation
  4. LayerNorm + Dropout after input projection — stabilizes transformer input
"""
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

    The previous version learned a single attention weight per timestep and
    could collapse to "look only at the last 1-2 bars" (verified by feature
    relevance audit: the model effectively ignored mid-window context).

    This version concatenates THREE views over the window:
      1. Attention-weighted sum (LEARNED EMPHASIS)
      2. Mean over all timesteps (UNBIASED CONTEXT — forces use of full window)
      3. Last timestep (CURRENT BAR — preserves the most recent state)

    Output is projected back to hidden_dim so downstream heads remain unchanged.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
        # ? PROJECT THE 3-STATISTIC CONCAT BACK TO HIDDEN_DIM
        self.fuse = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [batch, seq_len, hidden_dim]
        Returns:
            pooled: [batch, hidden_dim]
        """
        # ? ATTENTION-WEIGHTED SUM — learned focus
        scores = self.attention_weights(sequence)
        weights = functional.softmax(scores, dim=1)
        attn_pooled = (weights * sequence).sum(dim=1)

        # ? MEAN POOLING — guaranteed full-window coverage
        mean_pooled = sequence.mean(dim=1)

        # ? LAST TIMESTEP — preserves immediate state
        last_pooled = sequence[:, -1, :]

        fused = torch.cat([attn_pooled, mean_pooled, last_pooled], dim=-1)
        return self.fuse(fused)


class MultiHeadTradingModel(nn.Module):
    """
    Causal Transformer with attention pooling for multi-task trading prediction.

    Outputs:
      - direction: logits for [LONG, NEUTRAL, SHORT] classification
      - sizing: sigmoid-bounded [qty_ratio, tp_pct_scaled, sl_pct_scaled]
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
        dropout = dropout if dropout is not None else config.model.DROPOUT

        # Input projection with normalization
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
            norm_first=True,  # Pre-norm: more stable training for deeper models
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._sequence_max_len = max_sequence_len

        # Attention pooling (replaces last-timestep pooling)
        self.attention_pool = AttentionPool(hidden_dim)

        # Shared representation with residual connection
        self.shared_linear = nn.Linear(hidden_dim, hidden_dim)
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.shared_activation = nn.GELU()
        self.shared_dropout = nn.Dropout(dropout)

        # HEAD 1: Direction prediction (LONG=0, NEUTRAL=1, SHORT=2)
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),  # Head-specific dropout
            nn.Linear(64, 3),
        )

        # HEAD 2: Position sizing — qty_ratio, TP%, SL% (all 0-1 via Sigmoid)
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),  # Head-specific dropout
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, window_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            window_features: [batch, seq_len, n_features]
        Returns:
            dict with 'direction' logits and 'sizing' outputs
        """
        if window_features.size(1) > self._sequence_max_len:
            raise ValueError(
                f"Sequence length {window_features.size(1)} exceeds "
                f"positional encoding max {self._sequence_max_len}"
            )

        # Input projection with norm and dropout
        projected = self.input_projection(window_features)
        projected = self.input_norm(projected)
        projected = self.input_dropout(projected)
        encoded_positions = self.positional_encoding(projected)

        # Causal mask: timestep i cannot attend to j > i (no future leakage)
        sequence_length = encoded_positions.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sequence_length, device=encoded_positions.device
        )
        encoded_sequence = self.encoder(encoded_positions, mask=causal_mask)

        # Attention-weighted pooling over all timesteps (replaces last-step only)
        pooled = self.attention_pool(encoded_sequence)

        # Shared layer with RESIDUAL connection
        shared_out = self.shared_linear(pooled)
        shared_out = self.shared_norm(shared_out)
        shared_out = self.shared_activation(shared_out)
        shared_out = self.shared_dropout(shared_out)
        shared = pooled + shared_out  # Residual: skip connection preserves gradient flow

        return {
            "direction": self.signal_head(shared),
            "sizing": self.sizing_head(shared),  # [qty_ratio, tp_scaled, sl_scaled]
        }


def direction_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Apply inference softmax temperature (config.strategy.INFERENCE_DIRECTION_TEMPERATURE).

    Raw 3-class logits are often too flat; max softmax stays below confidence gates (e.g. 0.60)
    on almost every row. Dividing logits by T<1 before softmax sharpens the distribution so
    gates match real trading-filter semantics. Training loss still uses raw logits.
    """
    temperature = float(config.strategy.INFERENCE_DIRECTION_TEMPERATURE)
    if temperature > 1e-6 and abs(temperature - 1.0) > 1e-6:
        logits = logits / temperature
    return functional.softmax(logits, dim=1)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — down-weights easy examples to focus on hard ones.

    For a 3-class problem with imbalance (Buy underrepresented), CrossEntropy gives
    equal weight to easy Neutral predictions and hard Buy predictions. Focal Loss
    multiplies each sample's loss by (1 - p_correct)^gamma, so already-confident
    predictions contribute less and misclassified examples dominate learning.

    Args:
        gamma: Focus parameter. Higher = more focus on hard examples. Default 2.0.
        alpha: Per-class weight tensor. If None, all classes weighted equally.
        reduction: 'mean' or 'sum' or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes] raw logits (not softmaxed)
            targets: [batch] integer class labels
        Returns:
            Focal loss scalar
        """
        probabilities = functional.softmax(logits, dim=1)
        # Gather the probability of the correct class for each sample
        target_probs = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Standard cross-entropy per sample
        ce_loss = functional.cross_entropy(logits, targets, reduction="none")

        # Focal modulation: (1 - p_correct)^gamma
        focal_weight = (1.0 - target_probs) ** self.gamma

        # ? DIRECTIONAL PENALTY (HARDENED)
        # ? IF TARGET IS BUY(0) AND WE PREDICT SELL(2), OR VICE VERSA, APPLY A 4.0X PENALTY.
        # ? PREVIOUS 2.5X WAS NOT ENOUGH — CONFUSION MATRIX SHOWED 41% FLIP RATE.
        # ? WE ALSO ADD A SOFT-PROBABILITY OPPOSITE PENALTY SO THE MODEL IS PUNISHED
        # ? PROPORTIONALLY TO HOW CONFIDENT THE WRONG DIRECTIONAL PROB IS.
        pred_labels = torch.argmax(logits, dim=1)
        is_opposite = ((targets == 0) & (pred_labels == 2)) | ((targets == 2) & (pred_labels == 0))
        directional_multiplier = torch.ones_like(focal_weight)
        directional_multiplier[is_opposite] = 4.0

        # ? SOFT OPPOSITE-PROBABILITY PENALTY — DIFFERENTIABLE COMPONENT
        # ? FOR LONG TARGETS, PENALIZE PROB OF SELL; FOR SHORT TARGETS, PENALIZE PROB OF LONG.
        opposite_prob = torch.zeros_like(target_probs)
        long_mask = (targets == 0)
        short_mask = (targets == 2)
        if long_mask.any():
            opposite_prob[long_mask] = probabilities[long_mask, 2]
        if short_mask.any():
            opposite_prob[short_mask] = probabilities[short_mask, 0]
        # ? ADDED AS A DIFFERENTIABLE FACTOR (1 + 2 * P_OPPOSITE) — SCALES WITH WRONG CONFIDENCE
        directional_multiplier = directional_multiplier * (1.0 + 2.0 * opposite_prob)

        # ? TRUE LABEL IS NEUTRAL BUT MODEL OUTPUTS LONG OR SHORT — MAIN DRIVER OF LOW BUY/SELL PRECISION
        false_trade_signal_on_neutral = (targets == 1) & (pred_labels != 1)
        neutral_violation_scale = float(config.training.FOCAL_NEUTRAL_VIOLATION_SCALE)
        if neutral_violation_scale > 1.0:
            directional_multiplier = torch.where(
                false_trade_signal_on_neutral,
                directional_multiplier * neutral_violation_scale,
                directional_multiplier,
            )

        # Per-class alpha weighting (optional)
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_weight

        focal_loss = focal_weight * ce_loss * directional_multiplier

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TradingLoss(nn.Module):
    """
    3-component loss for the 2-phase training pipeline:

    Phase 1 (gamma=0):
      L = alpha * FocalLoss(direction) + beta * MSE(sizing)

    Phase 2 (gamma=0.05):
      L += gamma * PnL_Effect
      where PnL_Effect penalizes high-confidence wrong trades
      and rewards high-confidence winning trades.

    CHANGES vs previous version:
      1. FocalLoss replaces CrossEntropyLoss — focuses on hard/misclassified examples
      2. No label smoothing — Focal Loss and label smoothing don't combine well
      3. PnL effect clamped to [-1, 1] before mean
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.3, gamma: float = 0.05,
                 class_weights: torch.Tensor | None = None,
                 focal_gamma: float = 2.0,
                 use_focal: bool = True,
                 consistency_weight: float = 0.10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # ? CONSISTENCY WEIGHT — STRENGTH OF THE SIGNAL/SIZING SYNC TERM (NEW)
        self.consistency_weight = consistency_weight

        if use_focal:
            self.direction_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        else:
            self.direction_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: model output dict with 'direction' and 'sizing'
            targets: label dict with 'direction', 'qty_ratio', 'take_profit_pct',
                     'stop_loss_pct', 'actual_pnl_pct'
        Returns:
            dict with 'total' loss and individual component losses
        """
        signal_logits = predictions["direction"]
        sizing_out = predictions["sizing"]
        true_signal = targets["direction"].long()

        strategy = config.strategy
        max_tp = strategy.LABEL_TP_PCT_MAX
        max_sl = strategy.LABEL_SL_PCT_MAX

        # Scale TP/SL to [0, 1] range to match Sigmoid output
        true_qty = targets["qty_ratio"]
        true_tp = targets["take_profit_pct"] / max_tp
        true_sl = targets["stop_loss_pct"] / max_sl
        true_sizing = torch.stack([true_qty, true_tp, true_sl], dim=1)

        # --- Loss 1: Direction correctness (Focal Loss or CrossEntropy) ---
        signal_loss = self.direction_loss(signal_logits, true_signal)

        # --- Loss 2: Sizing accuracy ---
        sizing_loss = self.mse(sizing_out, true_sizing)

        # --- Loss 3: PnL-based consequence learning (Phase 2 only when gamma > 0) ---
        actual_pnl = targets["actual_pnl_pct"]
        pred_qty = sizing_out[:, 0]

        is_loss = (actual_pnl < 0).float()
        is_profit = (actual_pnl > 0).float()

        per_sample_pnl_effect = (
            is_loss * actual_pnl.abs() * pred_qty          # PENALTY: confident + wrong = bad
            - is_profit * actual_pnl.abs() * (1.0 - pred_qty)  # REWARD: winning but shy = also penalized
        )
        # Clamp per-sample before mean to prevent any single bad sample from exploding loss
        pnl_effect = per_sample_pnl_effect.clamp(-1.0, 1.0).mean()

        # ? CONSISTENCY LOSS — IF SIGNAL HEAD IS UNCERTAIN (HIGH NEUTRAL PROB),
        # ? SIZING HEAD SHOULD PROPOSE SMALL POSITIONS, AND VICE VERSA.
        # ? FORMULATION: ENCOURAGE PRED_QTY ≈ DIRECTIONAL_EDGE.
        # ? DIRECTIONAL_EDGE = MAX(P_LONG, P_SHORT) - P_NEUTRAL ∈ [-1, 1].
        with_softmax = functional.softmax(signal_logits, dim=1)
        directional_edge = torch.maximum(with_softmax[:, 0], with_softmax[:, 2]) - with_softmax[:, 1]
        # ? CLAMP TO [0, 1] AND USE AS THE SIZING TARGET (DIFFERENTIABLE)
        target_qty_from_signal = directional_edge.clamp(0.0, 1.0)
        consistency_loss = self.mse(sizing_out[:, 0], target_qty_from_signal.detach())

        total = (
            self.alpha * signal_loss
            + self.beta * sizing_loss
            + self.gamma * pnl_effect
            + self.consistency_weight * consistency_loss
        )

        return {
            "total": total,
            "signal_loss": signal_loss,
            "sizing_loss": sizing_loss,
            "pnl_effect": pnl_effect,
            "consistency_loss": consistency_loss,
        }
