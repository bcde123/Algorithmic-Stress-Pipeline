import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def apply_butter_bandpass(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter. 
    As specified in Methodology Step 1, this removes movement artifacts from BVP and EDA signals.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def extract_rmssd(bvp_signal, sampling_rate=64):
    """
    Extracts time-domain features (like RMSSD for HRV) from the BVP signal.
    Methodology Step 2.
    """
    # In a full production pipeline, we run a peak detection alg here (e.g. neurokit2/scipy)
    # to find the inter-beat intervals (RR intervals).
    # For now, this is a proxy metric approximating standard deviation of successive intervals.
    return np.std(np.diff(bvp_signal))

def normalize_zscore(df_patient_session):
    """
    Subject-specific Z-score normalization (Methodology Step 2).
    Addresses baseline interpersonal physiological differences so models learn systemic stress changes.
    """
    normalized_df = df_patient_session.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_mean = normalized_df[col].mean()
        col_std = normalized_df[col].std()
        if col_std > 0:
            normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        else:
            normalized_df[col] = 0.0
            
    return normalized_df
