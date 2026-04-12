import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    """
    Pillar 1: Long-Term Dynamics (Longitudinal)
    Bi-directional LSTM with self-attention for tracking stress accumulation,
    recovery, and circadian shifts over 24-hour sequences.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Self-Attention logic - hidden_dim * 2 because of Bi-LSTM
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        # Initial states: num_layers * 2 for bi-directional
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Calculate attention weights over the sequence length
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted context vector from bi-directional hidden states
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc(context_vector)
        return out, attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# Circadian extension — 24-hour window support
# ──────────────────────────────────────────────────────────────────────────────

class CircadianPositionalEncoding(nn.Module):
    """
    Learnable positional encoding that injects time-of-day information
    into a sequence.  Unlike fixed sinusoidal PE, this adapts to the
    specific circadian patterns present in the training data.
    """
    def __init__(self, d_model: int, max_len: int = 90_000):
        super().__init__()
        # Learnable embedding lookup: position -> d_model-dim vector
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.pe(positions)


class CircadianAttentionLSTM(nn.Module):
    """
    Pillar 1 (Extended): 24-Hour Longitudinal Stress Tracker

    Extends AttentionLSTM for full day-length sequences (~86 400 steps at 1 Hz).
    Key additions:
      - Input projection layer: maps raw features to a wider embedding space
        before the LSTM, giving the model more representational capacity for
        long contexts.
      - Learnable positional encoding: injects circadian phase information
        so the attention head can distinguish 'morning baseline' from 'afternoon
        peak' without relying solely on recurrent state.
      - Imbalance head: a dedicated output that estimates cumulative
        stress-recovery imbalance (a monotonically increasing scalar per day).

    Outputs:
        stress_index : (batch, 1)          -- instantaneous Stress Index [0, 1]
        imbalance    : (batch, 1)          -- cumulative imbalance score  [0, inf)
        attn_weights : (batch, seq_len, 1) -- for Pillar 4 clustering
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        max_seq_len: int = 90_000,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project raw features to embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # Circadian positional encoding
        self.pos_enc = CircadianPositionalEncoding(embed_dim, max_len=max_seq_len)

        # Bi-directional LSTM over embedded + positionally-encoded sequence
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2

        # Self-attention scoring
        self.attention = nn.Linear(lstm_out_dim, 1)

        # Instantaneous stress index prediction
        self.stress_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, output_dim),
            nn.Sigmoid(),  # constrain output to [0, 1]
        )

        # Cumulative stress-recovery imbalance (always >= 0)
        self.imbalance_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 1),
            nn.Softplus(),  # smooth approximation of ReLU, output in [0, inf)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            stress_index : (batch, output_dim)
            imbalance    : (batch, 1)
            attn_weights : (batch, seq_len, 1)
        """
        # 1. Project to embedding space + add circadian positional encoding
        x_emb = self.input_proj(x)       # (B, T, embed_dim)
        x_emb = self.pos_enc(x_emb)      # (B, T, embed_dim)

        # 2. BiLSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device)
        lstm_out, _ = self.lstm(x_emb, (h0, c0))   # (B, T, hidden_dim*2)

        # 3. Compute attention weights (soft-max over time axis)
        attn_scores  = self.attention(lstm_out)         # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)    # (B, T, 1)

        # 4. Attention-weighted context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden_dim*2)

        # 5. Dual prediction heads
        stress_index = self.stress_head(context)   # (B, output_dim)
        imbalance    = self.imbalance_head(context) # (B, 1)

        return stress_index, imbalance, attn_weights
