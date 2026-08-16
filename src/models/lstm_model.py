"""
LSTM Forecaster — built from scratch in PyTorch.

Architecture:
    Input (lookback, features)
        → LSTM encoder (multi-layer, optional bidirectional)
        → LayerNorm
        → FC decoder (hidden → hidden/2 → horizon)
        → Output (horizon,)

This is intentionally built without any forecasting library so you can
explain every architectural decision in an interview:
  - Why LayerNorm? Stabilizes hidden state magnitudes across sequences.
  - Why gradient clipping? LSTMs suffer from exploding gradients on long sequences.
  - Why Huber loss option? PM2.5 has extreme outliers (Diwali spikes);
    Huber is less sensitive than MSE to these.
  - Why dropout between LSTM layers but not in the last layer?
    Regularization during encoding, clean signal for decoding.

Usage:
    from src.models.lstm_model import LSTMForecaster
    model = LSTMForecaster(input_size=80, hidden_size=128, horizon=24)
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM for multi-step time-series forecasting.
    
    Args:
        input_size: number of input features per timestep
        hidden_size: LSTM hidden state dimension
        num_layers: number of stacked LSTM layers
        dropout: dropout rate between LSTM layers and in FC decoder
        horizon: number of future timesteps to predict
        bidirectional: use bidirectional LSTM (doubles hidden size)
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, horizon=24, bidirectional=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ── LSTM Encoder ──
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Normalization 
        lstm_output_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # FC Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for FC layers, orthogonal for LSTM."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with long-term memory)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, lookback, input_size)
        
        Returns:
            predictions: (batch_size, horizon)
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, lookback, hidden * directions)
        
        # Take the last timestep's output
        last_output = lstm_out[:, -1, :]  # (batch, hidden * directions)
        
        # Normalize
        last_output = self.layer_norm(last_output)
        
        # Decode to forecast
        predictions = self.decoder(last_output)  # (batch, horizon)
        
        return predictions
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMForecasterWithAttention(nn.Module):
    """
    LSTM with temporal attention — attends to all lookback timesteps
    instead of just the last one.
    
    This is a step toward the Temporal Fusion Transformer and shows
    interviewers you understand attention mechanisms.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, horizon=24):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, lookback, features)
        Returns:
            predictions: (batch, horizon)
        """
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden)
        
        # Attention weights
        attn_scores = self.attention(lstm_out)  # (batch, lookback, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, lookback, 1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden)
        
        # Store attention weights for visualization
        self.last_attn_weights = attn_weights.detach()
        
        context = self.layer_norm(context)
        predictions = self.decoder(context)
        
        return predictions
    
    def get_attention_weights(self):
        """Get the last computed attention weights for visualization."""
        return self.last_attn_weights.squeeze(-1).cpu().numpy()
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
