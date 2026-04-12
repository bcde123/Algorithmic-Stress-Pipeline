import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any, Callable

class BaseTrainer:
    """
    Base trainer class for standard PyTorch training loops.
    Supports early stopping, model checkpointing, and validation.
    """
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')

    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: Callable) -> float:
        self.model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle models that return multiple values (e.g. Attention weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Ensure target shape matches output shape (e.g. BCEWithLogitsLoss or MSELoss)
            if outputs.shape != targets.shape:
                if outputs.dim() == targets.dim() + 1 and outputs.size(-1) == 1:
                    targets = targets.unsqueeze(-1)
                elif outputs.shape == inputs.shape: # Autoencoder reconstruction mode
                    targets = inputs
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader: DataLoader, criterion: Callable) -> float:
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.shape != targets.shape:
                    if outputs.dim() == targets.dim() + 1 and outputs.size(-1) == 1:
                        targets = targets.unsqueeze(-1)
                    elif outputs.shape == inputs.shape:
                        targets = inputs
                
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: Callable, num_epochs: int, patience: int = 5) -> Dict[str, Any]:
        """
        Full training orchestration with early stopping.
        """
        history = {'train_loss': [], 'val_loss': []}
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(self.best_model_wts)
        return history
