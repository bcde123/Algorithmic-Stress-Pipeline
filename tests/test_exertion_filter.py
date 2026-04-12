"""
Exertion Filter Logic Validation — Verification Plan Step 2

Validates that the ACC-regression filter correctly reduces the EDA/HR signal
magnitude during periods of known physical exertion (high ACC magnitude),
while preserving cognitive stress signal components.

Run with:
    pytest tests/test_exertion_filter.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.exertion_filter import ExertionFilter


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(n=1000, seed=42):
    """
    Builds a synthetic DataFrame with:
      - A constant 'cognitive' EDA baseline
      - A physical motion (ACC) component whose influence on EDA we want to remove
      - Known 'ANAEROBIC' high-exertion window (seconds 300-600)

    The EDA signal is intentionally constructed as:
        EDA = cognitive_baseline + k * ACC_magnitude + noise

    After filtering, the residual should correlate with cognitive_baseline
    and NOT with ACC_magnitude.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n)
    # High ACC during the exercise window
    acc_x = rng.standard_normal(n) * 0.1
    acc_y = rng.standard_normal(n) * 0.1
    acc_z = rng.standard_normal(n) * 0.1

    # Exercise spike: 300-600s → ×10 motion magnitude
    exercise_mask = (t >= 300) & (t < 600)
    acc_x[exercise_mask] *= 10
    acc_y[exercise_mask] *= 10
    acc_z[exercise_mask] *= 10

    # True cognitive component (slow sinusoidal fluctuation)
    cog_eda = np.sin(2 * np.pi * t / 200) * 0.5 + 1.0

    # Physical contamination  (k=2 means ACC contributes strongly to raw EDA)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    k = 2.0
    raw_eda = cog_eda + k * acc_mag + rng.standard_normal(n) * 0.05

    hr = 70 + 20 * (acc_mag / acc_mag.max()) + rng.standard_normal(n) * 2

    df = pd.DataFrame({
        'EDA':   raw_eda,
        'HR':    hr,
        'ACC_x': acc_x,
        'ACC_y': acc_y,
        'ACC_z': acc_z,
    })
    return df, exercise_mask, acc_mag, cog_eda


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestExertionFilter:

    @pytest.fixture
    def filter_instance(self):
        return ExertionFilter()

    @pytest.fixture
    def synthetic_data(self):
        return _make_df()

    def test_returns_dataframe(self, filter_instance, synthetic_data):
        df, _, _, _ = synthetic_data
        result = filter_instance.process(df, targets=['EDA', 'HR'])
        assert isinstance(result, pd.DataFrame)

    def test_output_has_same_shape(self, filter_instance, synthetic_data):
        df, _, _, _ = synthetic_data
        result = filter_instance.process(df, targets=['EDA', 'HR'])
        assert result.shape == df.shape, (
            f"Shape mismatch: {result.shape} != {df.shape}"
        )

    def test_acc_columns_unchanged(self, filter_instance, synthetic_data):
        df, _, _, _ = synthetic_data
        result = filter_instance.process(df, targets=['EDA', 'HR'])
        for col in ['ACC_x', 'ACC_y', 'ACC_z']:
            np.testing.assert_array_equal(
                result[col].values, df[col].values,
                err_msg=f"ACC column '{col}' should not be modified by filter"
            )

    def test_exercise_variance_reduced_in_eda(self, filter_instance, synthetic_data):
        """
        Core logic check: EDA variance during the anaerobic exercise window
        should be substantially smaller AFTER filtering than before.
        """
        df, exercise_mask, _, _ = synthetic_data
        result = filter_instance.process(df, targets=['EDA', 'HR'])

        raw_exercise_var  = df.loc[exercise_mask, 'EDA'].var()
        filt_exercise_var = result.loc[exercise_mask, 'EDA'].var()

        # After filtering the physical component, exercise-period variance drops
        assert filt_exercise_var < raw_exercise_var, (
            f"Expected filtered EDA variance ({filt_exercise_var:.4f}) < "
            f"raw variance ({raw_exercise_var:.4f}) during exercise window"
        )

    def test_exercise_variance_reduction_ratio(self, filter_instance, synthetic_data):
        """
        Stricter: the variance reduction during exercise must be significant
        (at least 50% reduction), demonstrating the filter is effective.
        """
        df, exercise_mask, _, _ = synthetic_data
        result = filter_instance.process(df, targets=['EDA', 'HR'])

        raw_var  = df.loc[exercise_mask, 'EDA'].var()
        filt_var = result.loc[exercise_mask, 'EDA'].var()

        reduction_ratio = 1.0 - (filt_var / raw_var)
        assert reduction_ratio >= 0.50, (
            f"Expected >=50% variance reduction during exercise, got {reduction_ratio*100:.1f}%"
        )

    def test_baseline_signal_preserved(self, filter_instance, synthetic_data):
        """
        Outside the exercise window, the filtered signal should still correlate
        with the true cognitive EDA (Pearson r > 0.3 is a weak but sufficient
        benchmark given the noisy ground truth).
        """
        df, exercise_mask, _, cog_eda = synthetic_data
        result = filter_instance.process(df, targets=['EDA', 'HR'])

        baseline_mask = ~exercise_mask
        filtered_baseline = result.loc[baseline_mask, 'EDA'].values
        true_cognitive    = cog_eda[baseline_mask]

        # Pearson correlation between filtered signal and cognitive ground truth
        corr = np.corrcoef(filtered_baseline, true_cognitive)[0, 1]
        assert corr > 0.3, (
            f"Filtered EDA should preserve cognitive component (r={corr:.3f} < 0.3)"
        )

    def test_no_acc_no_op(self, filter_instance):
        """If there are no ACC columns, the filter must return the DataFrame unchanged."""
        df = pd.DataFrame({
            'EDA': np.random.randn(100),
            'HR':  np.random.randn(100),
        })
        result = filter_instance.process(df, targets=['EDA', 'HR'])
        pd.testing.assert_frame_equal(result, df)

    def test_zero_acc_no_op(self, filter_instance):
        """If ACC is all-zero (no motion), the filter must return the DataFrame unchanged."""
        df = pd.DataFrame({
            'EDA':   np.random.randn(100),
            'HR':    np.random.randn(100),
            'ACC_x': np.zeros(100),
            'ACC_y': np.zeros(100),
            'ACC_z': np.zeros(100),
        })
        result = filter_instance.process(df, targets=['EDA', 'HR'])
        pd.testing.assert_frame_equal(result, df)

    def test_get_acc_magnitude_correct_formula(self, filter_instance):
        """Validates the L2 norm calculation."""
        df = pd.DataFrame({
            'ACC_x': [3.0],
            'ACC_y': [4.0],
            'ACC_z': [0.0],
        })
        mag = filter_instance.get_acc_magnitude(df)
        expected = 5.0  # sqrt(9 + 16 + 0)
        np.testing.assert_almost_equal(mag[0], expected, decimal=5)
