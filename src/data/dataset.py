import torch
from torch.utils.data import Dataset
import numpy as np

class PhysiologicalTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for extracting sliding-windows across physiological streams.
    Used for Autoencoder (unsupervised reconstruction) and LSTM (supervised classification).
    """
    def __init__(self, dataframe, sequence_length=60, stride=1, target_col=None):
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_col = target_col
        
        # Pull feature matrix (exclude target column, time, and potential labels if not the target)
        exclude = [target_col, 'time', 'label'] if target_col != 'label' else [target_col, 'time']
        feature_cols = [c for c in dataframe.columns if c not in exclude]
        self.X = dataframe[feature_cols].values.astype(np.float32)
        
        if target_col and target_col in dataframe.columns:
            self.y = dataframe[target_col].values.astype(np.float32)
        else:
            self.y = None
            
        # Generates starting indices for building rolling windows
        self.window_starts = list(range(0, len(self.X) - sequence_length + 1, stride))
        
    def __len__(self):
        return len(self.window_starts)
        
    def __getitem__(self, idx):
        start_idx = self.window_starts[idx]
        end_idx = start_idx + self.sequence_length
        
        x_window = self.X[start_idx:end_idx]
        inputs = torch.tensor(x_window)
        
        if self.y is not None:
            # Supervised setup (LSTM predicting the state ending the window)
            y_window = self.y[end_idx - 1]
            targets = torch.tensor(y_window)
            return inputs, targets
        else:
            # Unsupervised setup (Autoencoder returning inputs as targets)
            return inputs, inputs

class DeepSurvDataset(Dataset):
    """
    PyTorch Dataset specially constructed to feed into the NegativeLogPartialLikelihood loss.
    Yields (covariates, time_to_event, event).
    """
    def __init__(self, covariates_df, times, events):
        self.covariates = covariates_df.values.astype(np.float32)
        self.times = np.array(times).astype(np.float32)
        self.events = np.array(events).astype(np.float32)
        
    def __len__(self):
        return len(self.covariates)
        
    def __getitem__(self, idx):
        covariates = torch.tensor(self.covariates[idx])
        time = torch.tensor(self.times[idx])
        event = torch.tensor(self.events[idx])
        
        return covariates, time, event
