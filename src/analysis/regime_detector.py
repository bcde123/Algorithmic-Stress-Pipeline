import torch
import numpy as np
from src.models.autoencoder import StressAutoencoder

class RegimeDetector:
    """
    Pillar 3: Latent State Detection (Autoencoder-based)
    Identifies "invisible" stress regimes and algorithmic pressures
    by calculating deviations from a subject's recovery baseline.
    """
    def __init__(self, model_path, input_dim, latent_dim, device='cpu'):
        self.device = device
        self.model = StressAutoencoder(input_dim, latent_dim).to(device)
        # Load pre-trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.baseline_threshold = 0.0

    def calibrate(self, baseline_data):
        """Sets the reconstruction threshold using a confirmed baseline/recovery period."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(baseline_data).to(self.device)
            # MSE between input and reconstruction
            errors = self.model.get_reconstruction_error(x).cpu().numpy()
            # Threshold set at 95th percentile of normal recovery fluctuation
            self.baseline_threshold = np.percentile(errors, 95)
        return self.baseline_threshold

    def get_stress_flags(self, session_data):
        """Flag regions exceeding the baseline threshold as 'Latent Stress Regimes'."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(session_data).to(self.device)
            errors = self.model.get_reconstruction_error(x).cpu().numpy()
            
            # Flags: 0=Baseline/Regulated, 1=Latent Pressure
            flags = (errors > self.baseline_threshold).astype(int)
        return flags, errors
