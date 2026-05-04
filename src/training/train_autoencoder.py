import torch
import torch.nn as nn
import torch.optim as optim
from .trainer import BaseTrainer

def train_autoencoder(model, train_loader, val_loader, config, device=None):
    """
    Training loop specifically for the Deep Autoencoder to establish a physiological baseline.
    The goal is to minimize reconstruction error on 'recovery/rest' segments (Step 3).
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # We use MSELoss to measure reconstruction accuracy
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    trainer = BaseTrainer(model, device)
    
    print("Starting Autoencoder Training on Rest/Recovery Baseline...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.get('epochs', 50),
        patience=config.get('patience', 10)
    )
    return model, history

def calculate_reconstruction_error(model, dataloader, device=None):
    """
    Post-training: Measure reconstruction errors on continuous streams.
    High error periods are clustered as 'stress regimes' (Step 3 of Methodology).
    """
    device = torch.device(device or next(model.parameters()).device)
    model.to(device)
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction='none') 
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Calculate error per sample over the sequence/feature dimensions
            batch_errors = criterion(outputs, inputs).mean(dim=[1, 2])
            errors.extend(batch_errors.cpu().tolist())
            
    return errors
