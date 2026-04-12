import torch
from torch.utils.data import DataLoader
from src.main import _make_synthetic_df, FEATURE_COLS
from src.data.dataset import PhysiologicalTimeSeriesDataset
from src.models.lstm import AttentionLSTM
from src.training.train_lstm import train_attention_lstm

def main():
    print("=" * 60)
    print("  Isolated Training: AttentionLSTM (Long-Term Dynamics)")
    print("=" * 60)

    # 1. Generate synthetic data
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

    # 3. Initialize Model
    print("Initializing AttentionLSTM...")
    lstm_model = AttentionLSTM(
        input_dim=len(FEATURE_COLS), 
        hidden_dim=32, 
        num_layers=2, 
        output_dim=1
    )

    # 4. Trigger training loop
    print("Starting Training Loop...")
    config = {'epochs': 3, 'learning_rate': 1e-3, 'patience': 2}
    trained_lstm, history = train_attention_lstm(
        lstm_model, train_loader, val_loader, config
    )

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
