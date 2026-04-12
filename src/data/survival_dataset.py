"""
survival_dataset.py — SurvivalDataset for DeepSurv Training

Converts a physiological stream DataFrame into (covariates, time_to_event, event)
triplets required by the Cox PH loss (NegativeLogPartialLikelihood).

Session-level aggregation strategy
────────────────────────────────────
Real longitudinal survival data requires time-to-burnout labels, which are
unavailable in the raw physiological datasets.  This module provides two modes:

  1. ``SurvivalDataset``    — requires explicit ``time`` and ``event`` columns
                              (for use with labelled survival datasets).

  2. ``SyntheticSurvivalDataset`` — generates plausible synthetic survival
                              targets from the stress_index column:
                              • time_to_event  ∝ 1 / mean_stress (higher stress → shorter time)
                              • event indicator = 1 if mean_stress > threshold (0.6)
                              Useful for pipeline smoke-testing when real survival 
                              labels are absent.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# SurvivalDataset  (labelled mode)
# ─────────────────────────────────────────────────────────────────────────────

class SurvivalDataset(Dataset):
    """
    PyTorch Dataset for labelled survival data.

    Each sample is a session (sliding window or explicit row) described by:
        covariates   — physiological features (float32 tensor)
        time         — observed duration until event or censoring (float32 scalar)
        event        — 1 = event occurred (burnout/attrition), 0 = censored

    Args:
        df            : DataFrame with feature columns + ``time`` + ``event``.
        feature_cols  : list of feature column names (covariates).
        time_col      : name of the time-to-event column.
        event_col     : name of the event-indicator column.
        seq_len       : if > 0, use sliding windows and aggregate features via
                        mean-pooling over the window; otherwise treat each row
                        as one sample.
        stride        : stride for sliding windows (only used when seq_len > 0).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        time_col: str = 'time',
        event_col: str = 'event',
        seq_len: int = 0,
        stride: int = 1,
    ):
        self.feature_cols = feature_cols
        self.seq_len      = seq_len
        self.stride       = stride

        features = df[feature_cols].values.astype(np.float32)
        times    = df[time_col].values.astype(np.float32)
        events   = df[event_col].values.astype(np.float32)

        if seq_len > 0:
            self._samples = self._build_windows(features, times, events)
        else:
            self._samples = [
                (features[i], times[i], events[i])
                for i in range(len(features))
            ]

    def _build_windows(self, features, times, events):
        samples = []
        n = len(features)
        for start in range(0, n - self.seq_len + 1, self.stride):
            end   = start + self.seq_len
            cov   = features[start:end].mean(axis=0)   # mean-pool covariate
            t     = times[end - 1]
            evt   = events[end - 1]
            samples.append((cov, t, evt))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        cov, t, evt = self._samples[idx]
        return (
            torch.tensor(cov, dtype=torch.float32),
            torch.tensor(t,   dtype=torch.float32),
            torch.tensor(evt, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SyntheticSurvivalDataset  (no survival labels required)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticSurvivalDataset(Dataset):
    """
    Generates plausible survival targets from a physiological DataFrame
    when true time-to-burnout labels are unavailable.

    Survival target heuristic
    ──────────────────────────
    • Group rows into non-overlapping windows of ``window_size`` seconds.
    • For each window compute ``mean_stress = mean(stress_index)``.
    • ``time_to_event  = max_time * (1 - mean_stress) + noise``
      where max_time defaults to 180 days (in abstract units).
    • ``event = 1 if mean_stress > burnout_threshold else 0``

    This produces a Cox-compatible dataset that exercises the survival loss
    without requiring external survival annotations.

    Args:
        df                : DataFrame with FEATURE_COLS + ``stress_index``.
        feature_cols      : list of feature column names.
        window_size       : rows to aggregate per survival sample.
        burnout_threshold : stress_index above which event=1.
        max_time          : upper bound for synthetic time-to-event.
        seed              : random seed for noise.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int = 60,
        burnout_threshold: float = 0.6,
        max_time: float = 180.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        features = df[feature_cols].values.astype(np.float32)
        stress   = df['stress_index'].values.astype(np.float32)

        samples = []
        n = len(features)
        for start in range(0, n - window_size + 1, window_size):
            end         = start + window_size
            cov         = features[start:end].mean(axis=0)
            mean_stress = float(stress[start:end].mean())

            # Synthetic time-to-event: inverse stress with small noise
            t   = max_time * (1.0 - mean_stress) + rng.normal(0, 5)
            t   = max(1.0, t)  # time must be positive

            evt = 1.0 if mean_stress > burnout_threshold else 0.0
            samples.append((cov, np.float32(t), np.float32(evt)))

        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        cov, t, evt = self._samples[idx]
        return (
            torch.tensor(cov, dtype=torch.float32),
            torch.tensor(t,   dtype=torch.float32),
            torch.tensor(evt, dtype=torch.float32),
        )
