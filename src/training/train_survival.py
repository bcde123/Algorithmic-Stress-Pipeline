import torch
import torch.nn as nn
import torch.optim as optim

class NegativeLogPartialLikelihood(nn.Module):
    """
    Cox Proportional Hazards Loss for DeepSurv.
    This loss function calculates the negative log partial likelihood of the proportional hazards model.
    """
    def __init__(self):
        super(NegativeLogPartialLikelihood, self).__init__()

    def forward(self, risk_preds, times, events):
        # Sort by right-censored time (descending)
        sorted_indices = torch.argsort(times, descending=True)
        events = events[sorted_indices]
        risk_preds = risk_preds[sorted_indices]

        hazard_ratio = torch.exp(risk_preds)
        
        # Cumulative sum of hazard ratios
        # Helps calculate the denominator of the likelihood equation (sum over individuals still at risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_preds - log_risk
        
        # Only sum the log likelihood for events that actually occurred (uncensored = 1)
        # Add epsilon to prevent div by zero
        loss = -torch.sum(uncensored_likelihood * events) / (torch.sum(events) + 1e-7)
        return loss

def train_survival_model(model, train_loader, val_loader, config, device=None):
    """
    Training loop for Deep Survival Model (DeepSurv).
    Aligns with Step 5: Treat sustained imbalance periods as covariates and predict the hazard ratios (attrition/burnout risk).
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    criterion = NegativeLogPartialLikelihood()
    # Survival models usually benefit from L2 Regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4), weight_decay=1e-4)
    
    model = model.to(device)

    print("Starting DeepSurv Model Training (Attrition Modeling)...")
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config.get('epochs', 50)):
        model.train()
        train_loss = 0.0
        
        # Assuming the dataloader yields continuous physiological covariates, time-to-event, and event indicators
        for covariates, times, events in train_loader:
            covariates = covariates.to(device)
            times = times.to(device)
            events = events.to(device)

            optimizer.zero_grad()
            risk_preds = model(covariates)
            
            loss = criterion(risk_preds.flatten(), times, events)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for covariates, times, events in val_loader:
                covariates = covariates.to(device)
                times = times.to(device)
                events = events.to(device)
                risk_preds = model(covariates)
                val_loss += criterion(risk_preds.flatten(), times, events).item()
        val_loss /= max(len(val_loader), 1)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.get('epochs', 50)} - DeepSurv Risk Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model, history
