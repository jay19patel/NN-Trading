# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEncoding(nn.Module):
    """
    Fixed: Bug #8 - Added positional info so Transformer knows the order of bars.
    """
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """
    Self-attention mechanism to learn temporal patterns.
    Fixed: Bug #2 - Added causal mask to prevent future data leakage.
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # [Batch, Heads, Seq, Seq]
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 🛡️ FIXED: BUG #2 - CAUSAL MASK (Prevent looking ahead)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        energy = energy.masked_fill(mask, float('-inf'))
        
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
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
    """
    Multi-task transformer model for trading.
    FIXED: BUG #1 - Now handles 3D sequences [Batch, Seq, Features].
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        # FIXED: Bug #1 - Architecture properly projects sequence steps
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
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
        
        # Reduced complexity for stability
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.upside_head = nn.Linear(hidden_dim, 1)
        self.downside_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 3) 

    def forward(self, x):
        # x: [Batch, Seq, Features]
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
            
        # 🎯 FIXED: BUG #1 - Take only the LAST bar for prediction
        x = x[:, -1, :]
        
        shared = self.shared_layer(x)
        return {
            'confidence': torch.sigmoid(self.confidence_head(shared)),
            'upside': self.upside_head(shared),
            'downside': self.downside_head(shared),
            'risk': torch.sigmoid(self.risk_head(shared)),
            'direction': self.direction_head(shared)
        }

class RiskAwareLoss(nn.Module):
    """
    Custom loss for risk-aware training with label balancing.
    FIXED: BUG #5 & BUG #9 - Balanced weights and grounded confidence targets.
    """
    def __init__(self, confidence_weight=0.5, upside_weight=1.5, downside_weight=1.5, risk_weight=0.5, dir_weight=1.0):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
        self.risk_weight = risk_weight
        self.dir_weight = dir_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # 📈 Regression Losses
        upside_loss = self.mse_loss(predictions['upside'].squeeze(), targets['upside'])
        confidence_weight = 1 + predictions['confidence'].squeeze()
        upside_loss = (upside_loss * confidence_weight).mean()

        downside_loss = self.mse_loss(predictions['downside'].squeeze(), targets['downside'])
        downside_loss = (downside_loss * confidence_weight).mean()

        # 🚦 Classification Loss
        direction_loss = self.ce_loss(predictions['direction'], targets['direction'].long())

        # 🧠 FIXED: BUG #5 - CONFIDENCE TARGET (Grounded in Ground Truth, not predictions)
        # Confidence target: True move > 1.2% profit target
        sig_move = torch.max(targets['upside'], torch.abs(targets['downside'])) > 1.0
        strong_bull = targets['upside'] > torch.abs(targets['downside']) * 1.5
        strong_bear = torch.abs(targets['downside']) > targets['upside'] * 1.5
        confidence_target = (sig_move & (strong_bull | strong_bear)).float()
        
        confidence_loss = self.mse_loss(predictions['confidence'].squeeze(), confidence_target).mean()

        risk_target = torch.clamp(torch.abs(targets['future_drawdown']) / 20.0, 0, 1)
        risk_loss = self.mse_loss(predictions['risk'].squeeze(), risk_target).mean()

        total_loss = (
            self.confidence_weight * confidence_loss +
            self.upside_weight * upside_loss +
            self.downside_weight * downside_loss +
            self.risk_weight * risk_loss +
            self.dir_weight * direction_loss
        )
        return {
            'total': total_loss, 
            'confidence': confidence_loss, 
            'upside': upside_loss, 
            'downside': downside_loss, 
            'risk': risk_loss,
            'direction': direction_loss
        }
