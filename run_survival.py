import torch
from torch.utils.data import DataLoader
from src.main import _make_synthetic_df, FEATURE_COLS
from src.data.survival_dataset import SyntheticSurvivalDataset
from src.models.deepsurv import DeepSurv
from src.training.train_survival import train_survival_model

def main():
    print("=" * 60)
    print("  Isolated Training: DeepSurv (Attrition / Burnout Risk)")
    print("=" * 60)

    # 1. Generate synthetic data
    print("Generating synthetic survival dataset...")
    df = _make_synthetic_df(n=2000)

    # 2. Build hazard datasets (uses its own dataset struct for durations and censor checks)
    dataset = SyntheticSurvivalDataset(
        df, 
        feature_cols=FEATURE_COLS, 
        window_size=60,
        burnout_threshold=0.6, 
        max_time=180.0
    )
    
    train_size = int(0.8 * len(dataset))
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=16, shuffle=False)

    # 3. Initialize Model
    print("Initializing DeepSurv Hazard Engine...")
    deepsurv_model = DeepSurv(
        input_dim=len(FEATURE_COLS), 
        hidden_layers=[64, 32]
    )

    # 4. Trigger training loop
    print("Starting Training Loop...")
    config = {'epochs': 3, 'learning_rate': 1e-4}
    trained_surv, history = train_survival_model(
        deepsurv_model, train_loader, val_loader, config
    )

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
