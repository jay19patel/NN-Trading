# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Self-attention mechanism to learn temporal patterns"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.shape[0]
        qkv = self.qkv(x).reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        return self.fc_out(out)

class TradingTransformerBlock(nn.Module):
    """A single Transformer block"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        forwarded = self.ffn(x)
        x = self.norm2(x + forwarded)
        return x

class MultiHeadTradingModel(nn.Module):
    """Multi-task transformer model for trading"""
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.transformer_blocks = nn.ModuleList([
            TradingTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.upside_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.downside_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1) 
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.squeeze(1)
        shared = self.shared_layer(x)
        return {
            'confidence': self.confidence_head(shared),
            'upside': self.upside_head(shared),
            'downside': self.downside_head(shared),
            'risk': self.risk_head(shared)
        }

class RiskAwareLoss(nn.Module):
    """Custom loss for risk-aware training"""
    def __init__(self, confidence_weight=1.0, upside_weight=2.0, downside_weight=2.0, risk_weight=1.0):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
        self.risk_weight = risk_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets):
        upside_loss = self.mse_loss(predictions['upside'].squeeze(), targets['upside'])
        confidence_weight = 1 + predictions['confidence'].squeeze() * 2
        upside_loss = (upside_loss * confidence_weight).mean()

        downside_loss = self.mse_loss(predictions['downside'].squeeze(), targets['downside'])
        downside_loss = (downside_loss * confidence_weight).mean()

        pred_is_long = predictions['upside'].squeeze() > torch.abs(predictions['downside'].squeeze())
        act_is_long = targets['upside'] > torch.abs(targets['downside'])
        sig_move = torch.max(targets['upside'], torch.abs(targets['downside'])) > 0.5
        confidence_target = ((pred_is_long == act_is_long) & sig_move).float()
        confidence_loss = self.mse_loss(predictions['confidence'].squeeze(), confidence_target).mean()

        risk_target = torch.clamp(torch.abs(targets['future_drawdown']) / 20.0, 0, 1)
        risk_loss = self.mse_loss(predictions['risk'].squeeze(), risk_target).mean()

        total_loss = (
            self.confidence_weight * confidence_loss +
            self.upside_weight * upside_loss +
            self.downside_weight * downside_loss +
            self.risk_weight * risk_loss
        )
        return {'total': total_loss, 'confidence': confidence_loss, 'upside': upside_loss, 'downside': downside_loss, 'risk': risk_loss}
