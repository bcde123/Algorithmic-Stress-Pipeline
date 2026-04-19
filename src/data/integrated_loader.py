"""
IntegratedLoader — Pillar 0: Data Ingestion & Harmonization

Unifies all four physiological datasets (WESAD, InducedStress, MMASH, SWELL)
into a standardised ``PhysiologicalStream`` output with a fixed 7-feature tensor:

    [EDA, HR, TEMP, ACC_x, ACC_y, ACC_z, HRV]   →   (float32, per-second)

Every returned DataFrame also carries a ``stress_index`` column (Unified Stress
Index ∈ [0, 1]) and a ``dataset`` tag for provenance tracking.

Usage
-----
>>> loader = IntegratedLoader(config)
>>> streams = loader.load_all()          # list[pd.DataFrame]
>>> df      = loader.combine()           # single concatenated DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .wesad_loader          import WESADLoader
from .induced_stress_loader import InducedStressLoader
from .mmash_loader          import MMASHLoader
from .swell_loader          import SWELLLoader
from .harmonizer            import SignalHarmonizer
from .exertion_filter       import ExertionFilter

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """
    Paths and options for each dataset.
    Pass ``None`` (default) for any dataset you do not have locally.
    """
    wesad_dir:          Optional[str] = None   # dir containing S2/, S3/, ...
    induced_stress_dir: Optional[str] = None   # dir containing Wearable_Dataset/
    mmash_dir:          Optional[str] = None   # dir containing DataPaper/
    swell_dir:          Optional[str] = None   # dir containing 'hrv dataset'/

    # Which subjects / users to load (None = all available)
    wesad_subjects:          Optional[List[str]] = None   # e.g. ['S2', 'S3']
    induced_stress_subjects: Optional[List[str]] = None   # e.g. ['f01', 'f02']
    mmash_users:             Optional[List[str]] = None   # e.g. ['user_1']
    swell_subjects:          Optional[List[str]] = None   # e.g. ['p1', 'p2']

    apply_exertion_filter: bool = True
    target_fs:             float = 1.0          # Hz — common resampling rate


# ──────────────────────────────────────────────────────────────────────────────
# Canonical feature columns (output contract)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ['EDA', 'HR', 'TEMP', 'ACC_x', 'ACC_y', 'ACC_z', 'HRV']
OUTPUT_COLS  = FEATURE_COLS + ['stress_index', 'dataset', 'subject_id']


# ──────────────────────────────────────────────────────────────────────────────
# IntegratedLoader
# ──────────────────────────────────────────────────────────────────────────────

class IntegratedLoader:
    """
    Unified ingestor for the four-dataset physiological stress corpus.

    The output of every ``_load_*`` method conforms to the PhysiologicalStream
    contract:
        - Columns : OUTPUT_COLS
        - dtype   : float32 (feature columns) + str (dataset / subject_id)
        - Sampling: ``config.target_fs`` Hz (default 1 Hz)
        - Stress  : ``stress_index`` ∈ [0, 1] (Unified Stress Index)
    """

    def __init__(self, config: DataConfig):
        self.config     = config
        self.harmonizer = SignalHarmonizer(target_fs=config.target_fs)
        self.exertion_f = ExertionFilter()

    # ── public API ──────────────────────────────────────────────────────────

    def load_all(self) -> List[pd.DataFrame]:
        """Load all configured datasets; returns a list of stream DataFrames."""
        streams: List[pd.DataFrame] = []
        loaders = [
            (self.config.wesad_dir,          self._load_wesad,          "WESAD"),
            (self.config.induced_stress_dir, self._load_induced_stress, "InducedStress"),
            (self.config.mmash_dir,          self._load_mmash,          "MMASH"),
            (self.config.swell_dir,          self._load_swell,          "SWELL"),
        ]
        for data_dir, loader_fn, name in loaders:
            if data_dir is None:
                logger.info("Skipping %s — no path configured.", name)
                continue
            try:
                dfs = loader_fn(data_dir)
                streams.extend(dfs)
                logger.info("Loaded %d stream(s) from %s.", len(dfs), name)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", name, exc)
        return streams

    def combine(self) -> pd.DataFrame:
        """
        Load all datasets and concatenate into a single DataFrame.
        Ensures correct dtypes and resets the integer index.
        """
        streams = self.load_all()
        if not streams:
            raise RuntimeError("No datasets were loaded. Check your DataConfig paths.")

        combined = pd.concat(streams, ignore_index=True)
        for col in FEATURE_COLS:
            if col in combined.columns:
                combined[col] = combined[col].astype(np.float32)
        combined['stress_index'] = combined['stress_index'].astype(np.float32)
        return combined

    # ── private loaders — one per dataset ───────────────────────────────────

    def _load_wesad(self, data_dir: str) -> List[pd.DataFrame]:
        subjects   = self.config.wesad_subjects or self._wesad_available_subjects(data_dir)
        streams    = []

        for sid in subjects:
            # _wesad_available_subjects may return full absolute paths when
            # it auto-detects a nested WESAD/ subdirectory.
            sid_path = Path(sid)
            if sid_path.is_absolute():
                effective_dir = str(sid_path.parent)
                effective_sid = sid_path.name
            else:
                effective_dir = data_dir
                effective_sid = sid

            loader = WESADLoader(effective_dir)
            try:
                raw = loader.load_subject(effective_sid)
            except FileNotFoundError:
                logger.warning("WESAD subject %s not found — skipping.", effective_sid)
                continue

            # Resample EDA & BVP signals to target_fs; labels to target_fs
            eda   = self.harmonizer.resample_series(raw['eda']['EDA'].values, original_fs=raw['fs']['eda'])
            bvp   = self.harmonizer.resample_series(raw['bvp']['BVP'].values, original_fs=raw['fs']['bvp'])
            acc_x = self.harmonizer.resample_series(raw['acc']['ACC_x'].values, original_fs=raw['fs']['acc'])
            acc_y = self.harmonizer.resample_series(raw['acc']['ACC_y'].values, original_fs=raw['fs']['acc'])
            acc_z = self.harmonizer.resample_series(raw['acc']['ACC_z'].values, original_fs=raw['fs']['acc'])
            temp  = self.harmonizer.resample_series(raw['temp']['TEMP'].values, original_fs=raw['fs']['temp'])
            lbl   = self.harmonizer.resample_series(raw['labels'].values.astype(float), original_fs=raw['fs']['label'])

            # Derive HR from BVP (very rough peak-to-peak → bpm approx)
            hr = np.full_like(eda, 70.0)  # fallback; wrist HR not in WESAD E4

            # Derive HRV (RMSSD of resampled BVP)
            hrv = np.array([
                np.sqrt(np.mean(np.diff(bvp[max(0, i-4):i+5]) ** 2))
                if i > 0 else np.nan
                for i in range(len(bvp))
            ])

            min_len = min(len(eda), len(hr), len(temp), len(acc_x), len(hrv), len(lbl))
            df = pd.DataFrame({
                'EDA':   eda[:min_len],
                'HR':    hr[:min_len],
                'TEMP':  temp[:min_len],
                'ACC_x': acc_x[:min_len],
                'ACC_y': acc_y[:min_len],
                'ACC_z': acc_z[:min_len],
                'HRV':   hrv[:min_len],
                'stress_index': self.harmonizer.map_labels(lbl[:min_len], 'WESAD'),
                'dataset':    'WESAD',
                'subject_id': effective_sid,
            })
            df = self._post_process(df)
            streams.append(df)

        return streams

    def _load_induced_stress(self, data_dir: str) -> List[pd.DataFrame]:
        loader   = InducedStressLoader(data_dir)
        subjects = self.config.induced_stress_subjects or loader.list_subjects()
        streams  = []

        for sid in subjects:
            try:
                raw = loader.load_subject(sid)   # already at 1 Hz, columns=FEATURE_COLS+label
            except FileNotFoundError:
                logger.warning("InducedStress subject %s not found — skipping.", sid)
                continue

            raw['stress_index'] = raw['label'].astype(np.float32)
            raw['dataset']      = 'InducedStress'
            raw['subject_id']   = sid
            df = raw[OUTPUT_COLS].copy()
            df = self._post_process(df)
            streams.append(df)

        return streams

    def _load_mmash(self, data_dir: str) -> List[pd.DataFrame]:
        loader = MMASHLoader(data_dir)
        users  = self.config.mmash_users or self._mmash_available_users(data_dir)
        streams = []

        for uid in users:
            try:
                raw = loader.load_user(uid)
            except Exception:
                logger.warning("MMASH user %s not found — skipping.", uid)
                continue

            acti     = raw['actigraph']
            activity = raw['activity']

            # Actigraph CSV assumed to have [HR, Axis1, Axis2, Axis3, ...] columns
            # Column mapping varies; attempt best-effort rename
            acti_cols = list(acti.columns)
            col_map: Dict[str, str] = {}
            for c in acti_cols:
                cl = c.lower()
                if 'hr' in cl and 'HR' not in col_map.values():
                    col_map[c] = 'HR'
                elif 'axis1' in cl or ('x' in cl and 'ACC_x' not in col_map.values()):
                    col_map[c] = 'ACC_x'
                elif 'axis2' in cl or ('y' in cl and 'ACC_y' not in col_map.values()):
                    col_map[c] = 'ACC_y'
                elif 'axis3' in cl or ('z' in cl and 'ACC_z' not in col_map.values()):
                    col_map[c] = 'ACC_z'
            acti = acti.rename(columns=col_map)

            # Build activity labels aligned to the actigraph length
            n = len(acti)
            if 'activity_type' in activity.columns:
                lbl_raw = np.full(n, 0.0)
                stress_idx = self.harmonizer.map_labels(activity['activity_type'].values, 'MMASH')
                # Simple broadcast: distribute activity labels evenly
                chunk = max(1, n // max(1, len(stress_idx)))
                for i, s in enumerate(stress_idx):
                    lbl_raw[i * chunk: (i + 1) * chunk] = s
            else:
                lbl_raw = np.zeros(n)

            df = pd.DataFrame({
                'EDA':   np.zeros(n, dtype=np.float32),   # MMASH has no EDA
                'HR':    acti.get('HR', pd.Series(np.zeros(n))).values[:n],
                'TEMP':  np.zeros(n, dtype=np.float32),   # MMASH has no TEMP
                'ACC_x': acti.get('ACC_x', pd.Series(np.zeros(n))).values[:n],
                'ACC_y': acti.get('ACC_y', pd.Series(np.zeros(n))).values[:n],
                'ACC_z': acti.get('ACC_z', pd.Series(np.zeros(n))).values[:n],
                'HRV':   np.zeros(n, dtype=np.float32),   # computed from RR separately
                'stress_index': lbl_raw[:n],
                'dataset':    'MMASH',
                'subject_id': uid,
            })
            df = self._post_process(df)
            streams.append(df)

        return streams

    def _load_swell(self, data_dir: str) -> List[pd.DataFrame]:
        loader  = SWELLLoader(data_dir)
        streams = []

        try:
            train_df, test_df = loader.load_hrv_features()
            hrv_df = pd.concat([train_df, test_df], ignore_index=True)
        except FileNotFoundError:
            logger.warning("SWELL HRV feature files not found in %s — skipping.", data_dir)
            return streams

        # SWELL HRV feature CSV — columns vary by version; attempt best-effort mapping
        condition_col = next((c for c in hrv_df.columns if 'condition' in c.lower()), None)

        stress_index = self.harmonizer.map_labels(
            hrv_df[condition_col].values if condition_col else np.zeros(len(hrv_df)),
            'SWELL'
        )

        n = len(hrv_df)
        df = pd.DataFrame({
            'EDA':   np.zeros(n, dtype=np.float32),
            'HR':    np.zeros(n, dtype=np.float32),
            'TEMP':  np.zeros(n, dtype=np.float32),
            'ACC_x': np.zeros(n, dtype=np.float32),
            'ACC_y': np.zeros(n, dtype=np.float32),
            'ACC_z': np.zeros(n, dtype=np.float32),
            'HRV':   hrv_df.get('RMSSD', pd.Series(np.zeros(n))).values,
            'stress_index': stress_index,
            'dataset':    'SWELL',
            'subject_id': 'pooled',
        })
        df = self._post_process(df)
        streams.append(df)
        return streams

    # ── helper utilities ────────────────────────────────────────────────────

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply exertion filtering, subject-level Z-score normalization,
        and NaN fill on feature columns.
        """
        if self.config.apply_exertion_filter:
            df = self.exertion_f.process(df, targets=['EDA', 'HR'])

        # Normalize feature columns (subject-specific)
        df[FEATURE_COLS] = self.harmonizer.normalize_subject(df[FEATURE_COLS])
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
        return df

    @staticmethod
    def _wesad_available_subjects(data_dir: str) -> List[str]:
        """
        Returns the list of WESAD subject folders (e.g. ['S2', 'S3', ...]).

        Auto-detects the common one-level nesting that occurs when the WESAD
        archive is unzipped: data/raw/wesad/WESAD/S2/... The loader transparently
        descends into the nested folder if no subject folders are found at the
        top level.
        """
        root = Path(data_dir)
        subjects = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('S')])
        if not subjects:
            # Check for a nested WESAD/ subdirectory (common extraction layout)
            nested = root / 'WESAD'
            if nested.exists():
                sub_names = sorted([p.name for p in nested.iterdir() if p.is_dir() and p.name.startswith('S')])
                if sub_names:
                    logger.info("Auto-detected nested WESAD layout — using %s", nested)
                    # Return full absolute paths so _load_wesad can split dir / name correctly
                    return [str(nested / s) for s in sub_names]
        return subjects

    @staticmethod
    def _mmash_available_users(data_dir: str) -> List[str]:
        """Returns sorted list of MMASH user folders."""
        paper_root = Path(data_dir) / 'DataPaper'
        if not paper_root.exists():
            return []
        return sorted([p.name for p in paper_root.iterdir() if p.is_dir()])
