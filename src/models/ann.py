"""
ann.py — Pillar (Baseline): Standalone MLP for Physiological Stress Prediction

StressMLP is a simple fully-connected feed-forward network that maps a flattened
physiological feature window to a stress scalar.  It acts as the benchmarking
baseline against which LSTM and TCN performance is compared.

Architecturally it is a dense ANN — this also satisfies the stated requirement
for an "ANN" model component alongside TCN, LSTM, and Autoencoders.
"""

import torch
import torch.nn as nn
from typing import List


class StressMLP(nn.Module):
    """
    Standalone Artificial Neural Network (MLP) baseline for short-term
    physiological stress prediction.

    Input  : (batch_size, input_dim)  — expects a *flattened* feature window
             OR  (batch_size, seq_len, n_features) which is pooled globally.
    Output : (batch_size, 1)          — continuous stress index ∈ [0, 1].

    Args:
        input_dim   : number of input features (or seq_len * n_features if flat).
        hidden_dims : list of hidden-layer widths (default=[128, 64, 32]).
        dropout     : dropout probability applied after each hidden layer.
        pool_input  : if True, the network accepts (B, T, F) tensors and
                      applies global-average pooling over the time dimension
                      before the MLP.  Set False when the caller already flattens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        pool_input: bool = True,
    ):
        super(StressMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.pool_input = pool_input

        layers: List[nn.Module] = []
        current_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h

        # Final head: stress scalar in [0, 1]
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, F) if pool_input=True, else (B, input_dim).
        Returns:
            stress_index : (B, 1), values in [0, 1].
        """
        if self.pool_input and x.dim() == 3:
            # Global average pooling over the time dimension: (B, T, F) → (B, F)
            x = x.mean(dim=1)
        return self.network(x)
