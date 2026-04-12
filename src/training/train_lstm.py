import torch
import torch.nn as nn
import torch.optim as optim
from .trainer import BaseTrainer

def train_attention_lstm(model, train_loader, val_loader, config):
    """
    Training loop for Attention-based LSTM.
    Aligns with Step 4 of the Methodology Plan.
    Focuses identifying which features (EDA spikes vs HRV drops) contribute heavily to the transition into an imbalanced continuous state.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Binary Cross Entropy for classifying balanced vs imbalanced (fatigued) states.
    # Assumes tasks are labeled intensive vs resting based on dataset annotations.
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    trainer = BaseTrainer(model, device)
    
    print("Starting Attention-LSTM Training for State Sequence Modeling...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.get('epochs', 50),
        patience=config.get('patience', 10)
    )
    return model, history
