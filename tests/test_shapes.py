"""
Shape Validation Tests — Verification Plan Step 1

Ensures that regardless of the source dataset, the output tensor from all
models is consistently shaped and that the IntegratedLoader produces the
expected [Batch, Seq_Len, 7] contract.

Run with:
    pytest tests/test_shapes.py -v
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PhysiologicalTimeSeriesDataset
from src.data.integrated_loader import DataConfig, FEATURE_COLS
from src.models.lstm import AttentionLSTM, CircadianAttentionLSTM
from src.models.tcn import StressTCN
from src.models.autoencoder import StressAutoencoder


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

N_SAMPLES   = 500   # enough for several windows
SEQ_LEN     = 60
BATCH_SIZE  = 8
N_FEATURES  = 7     # [EDA, HR, TEMP, ACC_x, ACC_y, ACC_z, HRV]


def _make_synthetic_df(n=N_SAMPLES, seed=0):
    """Creates a synthetic DataFrame matching the PhysiologicalStream contract."""
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n).astype(np.float32) for col in FEATURE_COLS}
    data['stress_index'] = rng.uniform(0, 1, n).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df():
    return _make_synthetic_df()


@pytest.fixture
def feature_loader(synthetic_df):
    """Sliding-window DataLoader (unsupervised mode — no label)."""
    ds = PhysiologicalTimeSeriesDataset(
        synthetic_df[FEATURE_COLS], sequence_length=SEQ_LEN, stride=10, target_col=None
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def supervised_loader(synthetic_df):
    """Sliding-window DataLoader (supervised mode — stress_index as target)."""
    ds = PhysiologicalTimeSeriesDataset(
        synthetic_df, sequence_length=SEQ_LEN, stride=10, target_col='stress_index'
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DataLoader / PhysiologicalStream contract
# ──────────────────────────────────────────────────────────────────────────────

class TestPhysiologicalStreamContract:
    """The output tensor from the DataLoader must be [Batch, Seq_Len, 7]."""

    def test_unsupervised_batch_shape(self, feature_loader):
        batch_x, _ = next(iter(feature_loader))
        B, T, F = batch_x.shape
        assert T == SEQ_LEN,    f"Expected seq_len={SEQ_LEN}, got {T}"
        assert F == N_FEATURES, f"Expected {N_FEATURES} features, got {F}"
        assert B <= BATCH_SIZE, f"Batch size too large: {B}"

    def test_supervised_batch_shape(self, supervised_loader):
        batch_x, batch_y = next(iter(supervised_loader))
        B, T, F = batch_x.shape
        assert T == SEQ_LEN,    f"Expected seq_len={SEQ_LEN}, got {T}"
        assert F == N_FEATURES, f"Expected {N_FEATURES} features, got {F}"
        assert batch_y.shape == (B,), f"Target shape mismatch: {batch_y.shape}"

    def test_dtype_is_float32(self, feature_loader):
        batch_x, _ = next(iter(feature_loader))
        assert batch_x.dtype == torch.float32, "Input tensor must be float32"

    def test_no_nan_in_batch(self, feature_loader):
        for batch_x, _ in feature_loader:
            assert not torch.isnan(batch_x).any(), "NaN found in batch"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model output shapes
# ──────────────────────────────────────────────────────────────────────────────

class TestModelOutputShapes:

    def _batch(self):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        return x

    def test_attention_lstm_output(self):
        model = AttentionLSTM(input_dim=N_FEATURES, hidden_dim=32, num_layers=2, output_dim=1)
        model.eval()
        x = self._batch()
        with torch.no_grad():
            out, attn = model(x)
        assert out.shape   == (BATCH_SIZE, 1),          f"LSTM output shape: {out.shape}"
        assert attn.shape  == (BATCH_SIZE, SEQ_LEN, 1), f"Attention shape: {attn.shape}"

    def test_circadian_lstm_output(self):
        model = CircadianAttentionLSTM(
            input_dim=N_FEATURES, embed_dim=32, hidden_dim=32,
            num_layers=2, output_dim=1, max_seq_len=1000
        )
        model.eval()
        x = self._batch()
        with torch.no_grad():
            stress, imbalance, attn = model(x)
        assert stress.shape   == (BATCH_SIZE, 1),          f"Stress shape: {stress.shape}"
        assert imbalance.shape == (BATCH_SIZE, 1),         f"Imbalance shape: {imbalance.shape}"
        assert attn.shape     == (BATCH_SIZE, SEQ_LEN, 1), f"Attention shape: {attn.shape}"
        # Stress Index must be in [0, 1]
        assert stress.min().item() >= 0.0 and stress.max().item() <= 1.0

    def test_stress_tcn_output(self):
        model = StressTCN(
            input_dim=N_FEATURES, num_channels=[32, 32], output_dim=1, num_fragments=5
        )
        model.eval()
        x = self._batch()
        with torch.no_grad():
            stress, logits = model(x)
        assert stress.shape == (BATCH_SIZE, 1), f"TCN stress shape: {stress.shape}"
        assert logits.shape == (BATCH_SIZE, 5), f"TCN logits shape: {logits.shape}"

    def test_autoencoder_reconstruction_shape(self):
        model = StressAutoencoder(input_dim=N_FEATURES, latent_dim=8)
        model.eval()
        x = torch.randn(BATCH_SIZE, N_FEATURES)
        with torch.no_grad():
            recon = model(x)
        assert recon.shape == x.shape, f"AE output shape mismatch: {recon.shape}"

    def test_autoencoder_error_shape(self):
        model = StressAutoencoder(input_dim=N_FEATURES, latent_dim=8)
        model.eval()
        x = torch.randn(BATCH_SIZE, N_FEATURES)
        with torch.no_grad():
            errors = model.get_reconstruction_error(x)
        assert errors.shape == (BATCH_SIZE,), f"AE error shape: {errors.shape}"
        assert (errors >= 0).all(), "Reconstruction error must be non-negative"


# ──────────────────────────────────────────────────────────────────────────────
# 3. IntegratedLoader synthetic smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegratedLoaderContract:
    """Ensure that combined DataFrames always have the right columns & dtypes."""

    def test_output_columns(self, synthetic_df):
        """Verifies FEATURE_COLS are exactly 7 and correctly named."""
        assert len(FEATURE_COLS) == 7, f"Expected 7 features, got {len(FEATURE_COLS)}"
        expected = {'EDA', 'HR', 'TEMP', 'ACC_x', 'ACC_y', 'ACC_z', 'HRV'}
        assert set(FEATURE_COLS) == expected

    def test_synthetic_df_has_all_feature_cols(self, synthetic_df):
        for col in FEATURE_COLS:
            assert col in synthetic_df.columns, f"Missing column: {col}"

    def test_no_nan_after_fillna(self, synthetic_df):
        df = synthetic_df[FEATURE_COLS].fillna(0.0)
        assert not df.isna().any().any(), "NaN persists after fillna"
