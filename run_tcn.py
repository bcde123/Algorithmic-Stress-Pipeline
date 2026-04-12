import torch
from torch.utils.data import DataLoader
from src.main import _make_synthetic_df, FEATURE_COLS
from src.data.dataset import PhysiologicalTimeSeriesDataset
from src.models.tcn import StressTCN
from src.training.train_tcn import train_tcn

def main():
    print("=" * 60)
    print("  Isolated Training: StressTCN (Short-Term Reactions)")
    print("=" * 60)

    # 1. Generate some synthetic data to play with
    print("Generating synthetic stress dataset...")
    df = _make_synthetic_df(n=2000)
    subset_df = df[FEATURE_COLS + ['stress_index']].copy()

    # 2. Build datasets
    dataset = PhysiologicalTimeSeriesDataset(
        subset_df, sequence_length=60, stride=10, target_col='stress_index'
    )
    
    train_size = int(0.8 * len(dataset))
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=16, shuffle=False)

    # 3. Initialize the TCN Architecture
    print("Initializing StressTCN...")
    tcn_model = StressTCN(
        input_dim=len(FEATURE_COLS), 
        num_channels=[32, 32], 
        output_dim=1, 
        num_fragments=5
    )

    # 4. Trigger the isolated training loop
    print("Starting Training Loop...")
    config = {
        'epochs': 3, 
        'learning_rate': 1e-3, 
        'patience': 2,
        'alpha': 0.7, 
        'beta': 0.3
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_tcn, history = train_tcn(
        tcn_model, train_loader, val_loader, config, device=device
    )

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
