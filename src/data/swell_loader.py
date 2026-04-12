import pandas as pd
import numpy as np
from pathlib import Path

class SWELLLoader:
    """
    Ingests the SWELL Knowledge Work dataset.
    Supports both precomputed HRV features and raw RRI processing.
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_hrv_features(self):
        """Loads the pre-calculated HRV features from the 'final' folder."""
        train_path = self.data_dir / "hrv dataset" / "data" / "final" / "train.csv"
        test_path = self.data_dir / "hrv dataset" / "data" / "final" / "test.csv"
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        return df_train, df_test

    def load_raw_rri(self, subject_id):
        """Loads raw inter-beat interval streams for a specific participant (e.g. 'p1')."""
        rri_file = self.data_dir / "hrv dataset" / "data" / "raw" / "rri" / f"{subject_id}.txt"
        label_file = self.data_dir / "hrv dataset" / "data" / "raw" / "labels" / f"{subject_id}.txt"
        
        if not rri_file.exists():
            raise FileNotFoundError(f"RRI file not found for {subject_id}")
            
        rri = pd.read_csv(rri_file, header=None, names=['ibi_ms'])
        labels = pd.read_csv(label_file, header=None, names=['condition'])
        
        return rri, labels
