import pandas as pd
import numpy as np
from pathlib import Path

class MMASHLoader:
    """
    Handles MMASH 24-hour recorded data.
    Captures multi-day rollover and integrates HR and Actigraphy.
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_user(self, user_id):
        """
        Loads data for a specific user (e.g., 'user_1').
        """
        user_path = self.data_dir / "DataPaper" / user_id
        
        # Load RR intervals (beat-to-beat)
        rr_df = pd.read_csv(user_path / "RR.csv")
        # Load Actigraphy (1Hz heart rate and 3-axis movement)
        acti_df = pd.read_csv(user_path / "Actigraph.csv")
        # Load Activity labels (contextual)
        activity_df = pd.read_csv(user_path / "Activity.csv")
        
        return {
            'rr': rr_df,
            'actigraph': acti_df,
            'activity': activity_df,
            'fs': {'actigraph': 1, 'rr': 'event-based', 'activity': 'event-based'}
        }
