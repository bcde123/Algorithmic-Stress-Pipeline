import torch
from torch.utils.data import DataLoader
from src.main import _make_synthetic_df, FEATURE_COLS
from src.data.dataset import PhysiologicalTimeSeriesDataset
from src.models.autoencoder import StressAutoencoder
from src.training.train_autoencoder import train_autoencoder

def main():
    print("=" * 60)
    print("  Isolated Training: StressAutoencoder (Anomaly/Regime)")
    print("=" * 60)

    # 1. Generate synthetic data
    print("Generating synthetic stress dataset (Unsupervised)...")
    df = _make_synthetic_df(n=2000)
    # The Autoencoder only scales features, zero targets needed for unsupervised reconstruction
    subset_df = df[FEATURE_COLS].copy()

    # 2. Build datasets
    dataset = PhysiologicalTimeSeriesDataset(
        subset_df, sequence_length=60, stride=10, target_col=None
    )
    
    train_size = int(0.8 * len(dataset))
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=16, shuffle=False)

    # 3. Initialize Model
    print("Initializing StressAutoencoder...")
    autoencoder = StressAutoencoder(
        input_dim=len(FEATURE_COLS), 
        latent_dim=16
    )

    # 4. Trigger training loop
    print("Starting Training Loop...")
    config = {'epochs': 3, 'learning_rate': 1e-3, 'patience': 2}
    trained_ae, history = train_autoencoder(
        autoencoder, train_loader, val_loader, config
    )

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
