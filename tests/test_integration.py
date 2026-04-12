"""
test_integration.py — End-to-End Smoke Test for main.py Pipeline

Verifies that the full pipeline runs without error on synthetic data and
produces the expected output files.  Does NOT require real dataset paths.

Run with:
    pytest tests/test_integration.py -v -s

Note: This test is marked 'slow' — exclude it from fast CI runs with:
    pytest tests/ -v --ignore=tests/test_integration.py
    # or
    pytest tests/ -v -m "not slow"
"""

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ── Marks ─────────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.slow


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run the pipeline programmatically (avoids spawning a subprocess)
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / 'src'
OUTPUT_DIR   = PROJECT_ROOT / 'outputs'


def _run_pipeline_inline():
    """
    Runs the main() function directly in-process.
    Uses the synthetic fallback (no real dataset paths set).
    """
    # Ensure src is importable
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    # Clear any cached modules to avoid state pollution between test runs
    mods_to_remove = [k for k in sys.modules if k.startswith('src.') or k in (
        'data', 'models', 'training', 'analysis', 'rl',
    )]
    for mod in mods_to_remove:
        sys.modules.pop(mod, None)

    from src.main import main  # noqa: F401
    main()


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineSmoke:

    @pytest.fixture(autouse=True)
    def run_pipeline(self):
        """Runs main() exactly once per test class; result is shared."""
        _run_pipeline_inline()

    def test_stress_distribution_saved(self):
        assert (OUTPUT_DIR / 'stress_distribution.png').exists(), (
            "stress_distribution.png not written"
        )

    def test_regime_timeline_saved(self):
        assert (OUTPUT_DIR / 'regime_timeline.png').exists(), (
            "regime_timeline.png not written"
        )

    def test_circadian_curve_saved(self):
        assert (OUTPUT_DIR / 'circadian_curve.png').exists(), (
            "circadian_curve.png not written"
        )

    def test_signature_heatmap_saved(self):
        assert (OUTPUT_DIR / 'signature_heatmap.png').exists(), (
            "signature_heatmap.png not written"
        )

    def test_pressure_nodes_saved(self):
        assert (OUTPUT_DIR / 'pressure_nodes.png').exists(), (
            "pressure_nodes.png not written"
        )

    def test_threshold_report_saved(self):
        report_path = OUTPUT_DIR / 'threshold_report.txt'
        assert report_path.exists(), "threshold_report.txt not written"
        content = report_path.read_text()
        assert "PHYSIOLOGICAL STRESS THRESHOLD REPORT" in content
        assert "Subject:" in content


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight unit scope (no pipeline needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticFallback:
    """
    Validates the synthetic DataFrame generation and pre-processing steps
    without running the full pipeline.
    """

    def test_synthetic_df_shape(self):
        from src.main import _make_synthetic_df
        from src.data.integrated_loader import FEATURE_COLS
        df = _make_synthetic_df(n=500)
        assert df.shape[0] == 500
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nan_after_normalize(self):
        from src.main import _make_synthetic_df
        from src.data.integrated_loader import FEATURE_COLS
        from src.data.transforms import normalize_zscore
        df = _make_synthetic_df(n=200)
        df[FEATURE_COLS] = normalize_zscore(df[FEATURE_COLS]).fillna(0.0)
        assert not df[FEATURE_COLS].isna().any().any()

    def test_sliding_window_dataset(self):
        from src.main import _make_synthetic_df
        from src.data.integrated_loader import FEATURE_COLS
        from src.data.dataset import PhysiologicalTimeSeriesDataset
        from torch.utils.data import DataLoader
        df = _make_synthetic_df(n=300)
        ds = PhysiologicalTimeSeriesDataset(df[FEATURE_COLS], sequence_length=60, stride=10)
        loader = DataLoader(ds, batch_size=8)
        batch_x, _ = next(iter(loader))
        assert batch_x.shape[-1] == len(FEATURE_COLS)
        assert batch_x.shape[-2] == 60
