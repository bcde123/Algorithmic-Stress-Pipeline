import pandas as pd
import numpy as np
from pathlib import Path

from .transforms import apply_butter_bandpass


class InducedStressLoader:
    """
    Ingests the PhysioNet 'Wearable Device Dataset from Induced Stress and
    Structured Exercise Sessions' (36 subjects, CSV-based Empatica E4 output).

    Expected directory structure:
        data_dir/
            Wearable_Dataset/
                STRESS/
                    f01/ EDA.csv  HR.csv  TEMP.csv  ACC.csv  BVP.csv
                    f02/ ...
                AEROBIC/
                    ...
                ANAEROBIC/
                    ...

    Each CSV follows the standard E4 format:
        Row 0 — Unix start timestamp (or datetime string)
        Row 1 — Sampling rate (Hz)
        Rows 2+ — Signal values
    """

    SESSION_LABELS = {
        'STRESS':    1.0,
        'AEROBIC':   0.3,   # Physical exertion, not cognitive stress
        'ANAEROBIC': 0.5,   # High physical, some cognitive load
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_e4_csv(self, filepath: Path, col_names=None):
        """Reads a standard Empatica E4 CSV and returns a time-indexed DataFrame."""
        raw = pd.read_csv(filepath, header=None)
        # Parse start time (handles both Unix float and datetime string)
        try:
            start_val = raw.iloc[0, 0]
            if isinstance(start_val, str) and not str(start_val).replace('.', '', 1).isdigit():
                start_dt = pd.to_datetime(start_val)
            else:
                start_dt = pd.to_datetime(float(start_val), unit='s')
        except Exception:
            start_dt = pd.to_datetime(raw.iloc[0, 0])

        sample_rate = float(raw.iloc[1, 0])
        data = raw.iloc[2:].values.astype(float)

        time_index = pd.date_range(
            start=start_dt,
            periods=len(data),
            freq=f'{1000 / sample_rate:.6f}ms'
        )
        n_cols = data.shape[1] if data.ndim > 1 else 1
        if col_names is None:
            col_names = [f'ch_{i}' for i in range(n_cols)]
        return pd.DataFrame(data, index=time_index, columns=col_names)

    def _load_session_dir(self, session_dir: Path, stress_label: float):
        """
        Loads all E4 streams from a single session folder, resamples to 1 Hz,
        computes RMSSD-based HRV, and attaches the unified stress label.
        Returns a DataFrame with columns [EDA, HR, TEMP, ACC_x, ACC_y, ACC_z, HRV, label].
        """
        eda  = self._read_e4_csv(session_dir / 'EDA.csv',  ['EDA'])
        hr   = self._read_e4_csv(session_dir / 'HR.csv',   ['HR'])
        temp = self._read_e4_csv(session_dir / 'TEMP.csv', ['TEMP'])
        acc  = self._read_e4_csv(session_dir / 'ACC.csv',  ['ACC_x', 'ACC_y', 'ACC_z'])
        bvp  = self._read_e4_csv(session_dir / 'BVP.csv',  ['BVP'])

        # ── Pre-resample bandpass filter (at native E4 rates) ──────────────────
        # EDA at 4 Hz: bandpass 0.05–1.5 Hz removes DC drift & high-freq noise.
        # BVP at 64 Hz: bandpass 0.5–8.0 Hz isolates the pulse wave.
        try:
            eda_rate = float(pd.read_csv(session_dir / 'EDA.csv', header=None).iloc[1, 0])
            raw_eda = eda['EDA'].values.astype(float)
            if eda_rate > 2.0:  # Only filter if Nyquist > highcut
                eda['EDA'] = apply_butter_bandpass(raw_eda, lowcut=0.05, highcut=1.5,
                                                    fs=eda_rate, order=4)
        except Exception:
            pass  # Non-fatal: proceed with raw EDA if filter fails

        try:
            bvp_rate = float(pd.read_csv(session_dir / 'BVP.csv', header=None).iloc[1, 0])
            raw_bvp = bvp['BVP'].values.astype(float)
            if bvp_rate > 16.0:
                bvp['BVP'] = apply_butter_bandpass(raw_bvp, lowcut=0.5, highcut=8.0,
                                                    fs=bvp_rate, order=4)
        except Exception:
            pass  # Non-fatal

        # Resample to 1 Hz
        eda_r  = eda.resample('1s').mean()
        hr_r   = hr.resample('1s').mean()
        temp_r = temp.resample('1s').mean()
        acc_r  = acc.resample('1s').mean()

        # HRV: RMSSD from BVP (successive difference of BVP peaks approximation)
        def _rmssd(x):
            diffs = np.diff(x)
            return np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else np.nan

        hrv_r = bvp['BVP'].resample('1s').apply(_rmssd).rename('HRV').to_frame()

        merged = pd.concat([eda_r, hr_r, temp_r, acc_r, hrv_r], axis=1)
        merged.columns = ['EDA', 'HR', 'TEMP', 'ACC_x', 'ACC_y', 'ACC_z', 'HRV']
        merged = merged.interpolate(method='time').dropna()
        merged['label'] = stress_label
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_subject(self, subject_id: str, sessions=('STRESS', 'AEROBIC', 'ANAEROBIC')):
        """
        Loads all requested sessions for a subject and returns a concatenated DataFrame.

        Args:
            subject_id: e.g. 'f01'
            sessions:   iterable of session names to include

        Returns:
            pd.DataFrame with columns [EDA, HR, TEMP, ACC_x, ACC_y, ACC_z, HRV, label]
        """
        frames = []
        wearable_root = self.data_dir / 'Wearable_Dataset'
        for session in sessions:
            session_dir = wearable_root / session / subject_id
            if not session_dir.exists():
                continue
            label = self.SESSION_LABELS.get(session.upper(), 0.5)
            try:
                df = self._load_session_dir(session_dir, label)
                df['session'] = session
                frames.append(df)
            except FileNotFoundError:
                pass  # Missing files for this session → skip silently

        if not frames:
            raise FileNotFoundError(
                f"No valid session data found for subject '{subject_id}' "
                f"in {wearable_root}"
            )
        return pd.concat(frames, ignore_index=True)

    def list_subjects(self, session='STRESS'):
        """Returns a sorted list of subjects available for the given session."""
        session_root = self.data_dir / 'Wearable_Dataset' / session
        if not session_root.exists():
            return []
        return sorted([p.name for p in session_root.iterdir() if p.is_dir()])
