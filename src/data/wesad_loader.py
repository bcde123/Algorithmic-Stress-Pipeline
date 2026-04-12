import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from .transforms import apply_butter_bandpass

class WESADLoader:
    """
    Ingests WESAD subject pickles containing synchronized chest (RespiBAN) 
    and wrist (Empatica E4) signals.
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_subject(self, subject_id):
        """
        Loads the .pkl file for a given subject (e.g., 'S2').
        Returns a dictionary of dataframes for each sensor.
        """
        subject_file = self.data_dir / subject_id / f"{subject_id}.pkl"
        if not subject_file.exists():
            raise FileNotFoundError(f"Pickle not found for {subject_id}")

        with open(subject_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Wrist data (E4)
        wrist_data = data['signal']['wrist']
        labels = data['label']

        # Aligns labels to the wrist signal.
        # WESAD E4 sampling rates: ACC(32Hz), BVP(64Hz), EDA(4Hz), TEMP(4Hz)
        # ── Pre-resample bandpass filter ──────────────────────────────────────
        # Apply at native fs BEFORE downsampling to 1 Hz so artifacts don't
        # alias into the resampled signal (requirement from review doc §3 / §10).
        raw_eda = wrist_data['EDA'].flatten().astype(float)
        eda_filtered = apply_butter_bandpass(raw_eda, lowcut=0.05, highcut=1.5, fs=4, order=4)

        raw_bvp = wrist_data['BVP'].flatten().astype(float)
        bvp_filtered = apply_butter_bandpass(raw_bvp, lowcut=0.5, highcut=8.0, fs=64, order=4)

        df_eda  = pd.DataFrame(eda_filtered,  columns=['EDA'])
        df_bvp  = pd.DataFrame(bvp_filtered,  columns=['BVP'])
        df_temp = pd.DataFrame(wrist_data['TEMP'], columns=['TEMP'])
        df_acc  = pd.DataFrame(wrist_data['ACC'],  columns=['ACC_x', 'ACC_y', 'ACC_z'])
        
        return {
            'eda': df_eda,
            'bvp': df_bvp,
            'temp': df_temp,
            'acc': df_acc,
            'labels': pd.Series(labels, name='label'),
            'fs': {'eda': 4, 'bvp': 64, 'acc': 32, 'temp': 4, 'label': 700}
        }
