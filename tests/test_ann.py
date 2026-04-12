"""
test_ann.py — Tests for StressMLP (standalone ANN baseline)

Run with:
    pytest tests/test_ann.py -v
"""

import torch
import pytest

from src.models.ann import StressMLP
from src.data.integrated_loader import FEATURE_COLS

N_FEATURES = len(FEATURE_COLS)   # 7
BATCH_SIZE = 16
SEQ_LEN    = 60


class TestStressMLPShapes:

    def test_output_shape_flat_input(self):
        """Model accepts (B, F) flat tensors when pool_input=False."""
        model = StressMLP(input_dim=N_FEATURES, pool_input=False)
        model.eval()
        x = torch.randn(BATCH_SIZE, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"

    def test_output_shape_sequence_input(self):
        """Model accepts (B, T, F) tensors when pool_input=True (default)."""
        model = StressMLP(input_dim=N_FEATURES, pool_input=True)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"

    def test_output_in_zero_one(self):
        """StressMLP uses Sigmoid, so outputs must be in [0, 1]."""
        model = StressMLP(input_dim=N_FEATURES)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES) * 10  # large inputs
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= 0.0, "Output below 0"
        assert out.max().item() <= 1.0, "Output above 1"

    def test_output_dtype(self):
        model = StressMLP(input_dim=N_FEATURES)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_custom_hidden_dims(self):
        """Non-default hidden_dims should still produce correct output shapes."""
        model = StressMLP(input_dim=N_FEATURES, hidden_dims=[256, 128, 64, 32])
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 1)

    def test_no_nan_output(self):
        """No NaN in output even for adversarial random inputs."""
        model = StressMLP(input_dim=N_FEATURES)
        model.eval()
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
            with torch.no_grad():
                out = model(x)
            assert not torch.isnan(out).any(), f"NaN in output (seed={seed})"

    def test_backward_pass(self):
        """Gradients should flow through the entire MLP without error."""
        model = StressMLP(input_dim=N_FEATURES)
        x     = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        out   = model(x)
        loss  = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestStressMLPPooling:

    def test_pool_input_false_rejects_3d(self):
        """With pool_input=False, passing a 3D tensor should raise a shape error."""
        model = StressMLP(input_dim=N_FEATURES * SEQ_LEN, pool_input=False)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        with pytest.raises(Exception):
            model(x)   # linear layer dimension mismatch

    def test_global_average_pooling_reduces_time_dim(self):
        """Pool over T: output should be independent of seq_len."""
        model = StressMLP(input_dim=N_FEATURES, pool_input=True)
        model.eval()
        for t in (30, 60, 120):
            x = torch.randn(4, t, N_FEATURES)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (4, 1), f"Failed for T={t}: {out.shape}"
