import torch
import torch.nn as nn

class StressAutoencoder(nn.Module):
    """
    Deep Autoencoder for unsupervised regime detection.
    Learns to reconstruct the physiological baseline (e.g., during recovery).
    High reconstruction error indicates a deviation to a latent stress regime 
    (invisible algorithmic pressure).
    """
    def __init__(self, input_dim, latent_dim):
        super(StressAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Identity()  # Z-score inputs range beyond [0,1]; Identity + MSE loss is correct
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_reconstruction_error(self, x):
        """Calculates Mean Squared Error between input and reconstruction."""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse
