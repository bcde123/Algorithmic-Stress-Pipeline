"""
test_research_design.py — Tests for theory, causal, intervention, robustness,
and external-validation helpers.

Run with:
    pytest tests/test_research_design.py -v
"""

import numpy as np
import pandas as pd
import torch.nn as nn

from src.analysis.research_design import (
    build_research_design_summary,
    build_theoretical_framework,
    estimate_high_stress_effect,
    format_research_design_summary,
    leave_one_dataset_out_validation,
    summarize_model_complexity,
    validate_intervention_contrast,
)
from src.data.integrated_loader import FEATURE_COLS


def _make_df(n=240, seed=11):
    rng = np.random.default_rng(seed)
    stress = np.r_[np.repeat(0.2, n // 2), np.repeat(0.85, n - n // 2)]
    rng.shuffle(stress)
    data = {col: rng.normal(0, 1, n).astype(np.float32) for col in FEATURE_COLS}
    data['HR'] = (60 + 20 * stress + rng.normal(0, 1, n)).astype(np.float32)
    data['HRV'] = (0.08 - 0.03 * stress + rng.normal(0, 0.002, n)).astype(np.float32)
    data['stress_index'] = stress.astype(np.float32)
    data['dataset'] = np.where(np.arange(n) < n // 2, 'A', 'B')
    data['subject_id'] = np.where(np.arange(n) < n // 2, 's1', 's2')
    data['intervention'] = np.where(stress >= 0.6, 'stress', 'recovery')
    return pd.DataFrame(data)


class TestResearchDesignFramework:

    def test_theory_constructs_present(self):
        constructs = build_theoretical_framework()
        assert len(constructs) >= 4
        names = {item.name for item in constructs}
        assert 'Algorithmic demand' in names
        assert 'Exit and productivity risk' in names

    def test_complexity_summary_counts_parameters(self):
        model = nn.Linear(7, 1)
        result = summarize_model_complexity({'linear': model}, n_samples=100)[0]
        assert result.model_name == 'linear'
        assert result.trainable_parameters == 8
        assert result.sample_to_parameter_ratio > 0


class TestCausalAndInterventionValidation:

    def test_estimate_high_stress_effect_detects_hrv_drop(self):
        df = _make_df()
        estimate = estimate_high_stress_effect(df, outcome_col='HRV', n_bootstrap=20)
        assert estimate.n_treated > 0
        assert estimate.n_control > 0
        assert estimate.effect < 0
        assert estimate.outcome == 'HRV'

    def test_intervention_contrast_detects_stress_increase(self):
        df = _make_df()
        result = validate_intervention_contrast(df, outcome_col='stress_index', n_bootstrap=20)
        assert result.n_treatment > 0
        assert result.n_control > 0
        assert result.effect > 0

    def test_external_validation_runs_leave_one_dataset_out(self):
        df = _make_df()
        results = leave_one_dataset_out_validation(df, FEATURE_COLS, min_train=20, min_holdout=20)
        assert len(results) == 2
        assert {item.holdout_dataset for item in results} == {'A', 'B'}

    def test_full_summary_formats(self):
        df = _make_df()
        summary = build_research_design_summary(
            df=df,
            feature_cols=FEATURE_COLS,
            models={'linear': nn.Linear(len(FEATURE_COLS), 1)},
            n_model_samples=100,
        )
        text = format_research_design_summary(summary)
        assert 'RESEARCH DESIGN SUMMARY' in text
        assert 'Theoretical framework' in text
        assert 'Policy and managerial implications' in text
