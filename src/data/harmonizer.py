import pandas as pd
import numpy as np
from scipy import signal

class SignalHarmonizer:
    """
    Standardizes all physiological signals to a common sampling rate and 
    maps disparate dataset labels to a unified [0, 1] Stress Index.
    """
    def __init__(self, target_fs=1.0):
        self.target_fs = target_fs

    def resample_series(self, series, original_fs):
        """Standardizes a series to the target frequency."""
        if original_fs <= 0: return series
        if original_fs == self.target_fs: return series
            
        duration = len(series) / original_fs
        num_samples = int(duration * self.target_fs)
        if num_samples <= 0: return np.array([])
        
        resampled = signal.resample(series, num_samples)
        return resampled

    def map_labels(self, labels, dataset_type):
        """
        Maps source labels to a Unified Stress Index [0, 1].
        0.0: Baseline/Recovery, 1.0: Peak Stress.
        """
        unified = np.zeros_like(labels, dtype=float)
        
        if dataset_type == 'WESAD':
            # 1: Baseline(0), 2: Stress(1), 3: Amusement(0.2), 4: Meditation(0)
            mapping = {1: 0.0, 2: 1.0, 3: 0.2, 4: 0.0}
            unified = np.array([mapping.get(int(l), 0) for l in labels])
            
        elif dataset_type == 'SWELL':
            mapping = {'no stress': 0.0, 'interruption': 0.8, 'time pressure': 1.0}
            unified = np.array([mapping.get(str(l).lower(), 0) for l in labels])
            
        elif dataset_type == 'MMASH':
            # 1: Sleep(0), 4: Work(0.5), 6: Heavy(0.8)
            mapping = {1: 0.0, 4: 0.5, 6: 0.8}
            unified = np.array([mapping.get(int(l), 0) for l in labels])

        elif dataset_type == 'InducedStress':
            # Labels are already continuous floats from InducedStressLoader
            # STRESS=1.0, AEROBIC=0.3, ANAEROBIC=0.5
            unified = np.array([float(l) for l in labels])

        return unified

    def normalize_subject(self, df):
        """Subject-specific Z-score normalization."""
        return (df - df.mean()) / (df.std() + 1e-6)
