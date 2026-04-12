import os
from pathlib import Path
import pandas as pd
import numpy as np
from .transforms import extract_rmssd

class EmpaticaDataLoader:
    """
    Handles reading and preprocessing raw Empatica E4 output files.
    Usually provided in separate CSVs for each sensor (EDA.csv, HR.csv, TEMP.csv, ACC.csv)
    with the first row containing the start timestamp and the second the sampling rate.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def _read_e4_csv(self, filename):
        """Standard method to read an E4 csv file and convert it into a time-indexed series."""
        filepath = (Path(self.data_dir) / filename).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"{filename} not found in {self.data_dir}")
            
        # First row is initial timestamp (or date string), second row is frequency Hz
        raw_df = pd.read_csv(filepath, header=None)
        
        # Robust parsing of the start time (handles both Unix floats and string datetimes)
        try:
            start_val = raw_df.iloc[0, 0]
            # If it's a string that doesn't look like a float, parse it as a date
            if isinstance(start_val, str) and not str(start_val).replace('.','',1).isdigit():
                start_dt = pd.to_datetime(start_val)
            else:
                start_dt = pd.to_datetime(float(start_val), unit='s')
        except Exception:
            start_dt = pd.to_datetime(raw_df.iloc[0, 0])

        sample_rate = float(raw_df.iloc[1, 0])
        
        # Data starts from row 2. Cast to float for mathematical operations.
        data = raw_df.iloc[2:].values.astype(float)
        
        # Create datetime index
        duration = len(data) / sample_rate
        time_index = pd.date_range(
            start=start_dt, 
            periods=len(data), 
            freq=f'{1000/sample_rate}ms'
        )
        
        return pd.DataFrame(data, index=time_index, columns=[f"ch_{i}" for i in range(data.shape[1])])

    def load_user_session(self, user_id=None, session_id=None, session_dir=None):
        """
        Loads and aligns the data for a specific user and session.
        If session_dir is provided, it uses that directly.
        """
        if session_dir is None:
            session_dir = os.path.join(self.data_dir, f"user_{user_id}", f"session_{session_id}")
        
        # Temporarily update current data_dir to point to the specific session
        original_dir = self.data_dir
        self.data_dir = session_dir
        
        try:
            # Load streams
            eda = self._read_e4_csv("EDA.csv")
            hr = self._read_e4_csv("HR.csv")
            temp = self._read_e4_csv("TEMP.csv")
            acc = self._read_e4_csv("ACC.csv")
            bvp = self._read_e4_csv("BVP.csv")
        finally:
            self.data_dir = original_dir
        
        # Resample all to 1 Hz so they can be merged evenly
        eda_resampled = eda.resample('1s').mean()
        hr_resampled = hr.resample('1s').mean()
        temp_resampled = temp.resample('1s').mean()
        # For ACC, we might want to take the magnitude first, but for now just average
        acc_resampled = acc.resample('1s').mean()
        
        # Calculate HRV (RMSSD) from BVP for each 1-second window
        hrv_resampled = bvp.resample('1s').apply(lambda x: extract_rmssd(x.values.flatten() if isinstance(x, pd.DataFrame) else x.values))
        hrv_resampled.columns = ['HRV']
        
        # Merge
        merged = pd.concat([eda_resampled, hr_resampled, temp_resampled, acc_resampled, hrv_resampled], axis=1)
        merged.columns = ['EDA', 'HR', 'TEMP', 'ACC_x', 'ACC_y', 'ACC_z', 'HRV']
        
        # Interpolate missing values due to resampling offsets
        merged = merged.interpolate(method='time').dropna()
        
        return merged
