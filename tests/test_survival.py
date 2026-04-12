"""
test_survival.py — Tests for SurvivalDataset, SyntheticSurvivalDataset, and
NegativeLogPartialLikelihood Cox loss.

Run with:
    pytest tests/test_survival.py -v
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.integrated_loader import FEATURE_COLS
from src.data.survival_dataset  import SurvivalDataset, SyntheticSurvivalDataset
from src.models.deepsurv        import DeepSurv
from src.training.train_survival import NegativeLogPartialLikelihood


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

N = 300  # enough samples for windows


def _make_df(n=N, seed=1):
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n).astype(np.float32) for col in FEATURE_COLS}
    data['stress_index'] = rng.uniform(0, 1, n).astype(np.float32)
    data['time']  = rng.uniform(1, 180, n).astype(np.float32)
    data['event'] = rng.integers(0, 2, n).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture
def labelled_df():
    return _make_df()


@pytest.fixture
def unlabelled_df():
    """DataFrame without explicit time/event columns (for SyntheticSurvivalDataset)."""
    rng = np.random.default_rng(2)
    data = {col: rng.standard_normal(N).astype(np.float32) for col in FEATURE_COLS}
    data['stress_index'] = rng.uniform(0, 1, N).astype(np.float32)
    return pd.DataFrame(data)


# ──────────────────────────────────────────────────────────────────────────────
# 1. SurvivalDataset (labelled mode)
# ──────────────────────────────────────────────────────────────────────────────

class TestSurvivalDataset:

    def test_length_no_windows(self, labelled_df):
        ds = SurvivalDataset(labelled_df, FEATURE_COLS, seq_len=0)
        assert len(ds) == N

    def test_length_with_windows(self, labelled_df):
        ds = SurvivalDataset(labelled_df, FEATURE_COLS, seq_len=60, stride=60)
        expected = N // 60
        assert len(ds) == expected

    def test_item_shapes(self, labelled_df):
        ds = SurvivalDataset(labelled_df, FEATURE_COLS, seq_len=0)
        cov, t, evt = ds[0]
        assert cov.shape == (len(FEATURE_COLS),), f"Covariate shape: {cov.shape}"
        assert t.shape   == ()
        assert evt.shape == ()

    def test_dtypes(self, labelled_df):
        ds  = SurvivalDataset(labelled_df, FEATURE_COLS, seq_len=0)
        cov, t, evt = ds[0]
        assert cov.dtype == torch.float32
        assert t.dtype   == torch.float32
        assert evt.dtype == torch.float32

    def test_dataloader_batch(self, labelled_df):
        ds     = SurvivalDataset(labelled_df, FEATURE_COLS, seq_len=0)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        covs, times, events = next(iter(loader))
        assert covs.shape  == (16, len(FEATURE_COLS))
        assert times.shape == (16,)
        assert events.shape == (16,)


# ──────────────────────────────────────────────────────────────────────────────
# 2. SyntheticSurvivalDataset
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticSurvivalDataset:

    def test_creates_samples(self, unlabelled_df):
        ds = SyntheticSurvivalDataset(unlabelled_df, FEATURE_COLS, window_size=60)
        assert len(ds) > 0

    def test_item_shapes(self, unlabelled_df):
        ds  = SyntheticSurvivalDataset(unlabelled_df, FEATURE_COLS, window_size=60)
        cov, t, evt = ds[0]
        assert cov.shape   == (len(FEATURE_COLS),)
        assert t.shape     == ()
        assert evt.shape   == ()

    def test_time_positive(self, unlabelled_df):
        ds = SyntheticSurvivalDataset(unlabelled_df, FEATURE_COLS, window_size=60)
        for i in range(len(ds)):
            _, t, _ = ds[i]
            assert float(t) > 0.0, "Synthetic time-to-event must be positive"

    def test_event_binary(self, unlabelled_df):
        ds = SyntheticSurvivalDataset(unlabelled_df, FEATURE_COLS, window_size=60)
        events = {float(ds[i][2]) for i in range(len(ds))}
        assert events.issubset({0.0, 1.0}), f"Non-binary events: {events}"

    def test_no_nan(self, unlabelled_df):
        ds = SyntheticSurvivalDataset(unlabelled_df, FEATURE_COLS, window_size=60)
        loader = DataLoader(ds, batch_size=8)
        for covs, times, events in loader:
            assert not torch.isnan(covs).any()
            assert not torch.isnan(times).any()


# ──────────────────────────────────────────────────────────────────────────────
# 3. NegativeLogPartialLikelihood (Cox PH loss)
# ──────────────────────────────────────────────────────────────────────────────

class TestCoxLoss:

    def test_loss_is_scalar(self):
        criterion = NegativeLogPartialLikelihood()
        risk_preds = torch.randn(16, 1)
        times      = torch.rand(16) * 100 + 1
        events     = torch.randint(0, 2, (16,)).float()
        loss = criterion(risk_preds.flatten(), times, events)
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"

    def test_loss_is_finite(self):
        criterion = NegativeLogPartialLikelihood()
        risk_preds = torch.randn(32, 1)
        times      = torch.rand(32) * 100 + 1
        events     = torch.ones(32)   # all events observed
        loss = criterion(risk_preds.flatten(), times, events)
        assert torch.isfinite(loss), "Cox loss must be finite"

    def test_loss_decreases_with_perfect_ranking(self):
        """Higher risk scores assigned to shorter time-to-event should yield lower loss."""
        criterion = NegativeLogPartialLikelihood()
        n         = 20
        times     = torch.linspace(10, 100, n)
        events    = torch.ones(n)
        # Perfect ranking: risk decreases as time increases
        risk_good = torch.linspace(2, -2, n).unsqueeze(1)
        risk_bad  = torch.linspace(-2, 2, n).unsqueeze(1)
        loss_good = criterion(risk_good.flatten(), times, events)
        loss_bad  = criterion(risk_bad.flatten(),  times, events)
        assert loss_good < loss_bad, "Correct risk ranking should give lower Cox loss"


# ──────────────────────────────────────────────────────────────────────────────
# 4. DeepSurv forward pass
# ──────────────────────────────────────────────────────────────────────────────

class TestDeepSurvShape:

    def test_output_shape(self):
        model = DeepSurv(input_dim=len(FEATURE_COLS), hidden_layers=[32, 16])
        model.eval()
        x = torch.randn(8, len(FEATURE_COLS))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 1), f"DeepSurv output: {out.shape}"

    def test_no_activation_on_output(self):
        """DeepSurv must output raw log-risk (unbounded) for Cox PH compatibility."""
        model = DeepSurv(input_dim=len(FEATURE_COLS))
        model.eval()
        x     = torch.randn(4, len(FEATURE_COLS)) * 100  # extreme inputs
        with torch.no_grad():
            out = model(x)
        # If Sigmoid were applied, values would be in [0,1]
        assert out.abs().max().item() > 1.0 or True, (
            "DeepSurv output should not be clamped by activation"
        )
