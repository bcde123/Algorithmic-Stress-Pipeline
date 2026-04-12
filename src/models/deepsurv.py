import torch
import torch.nn as nn

class DeepSurv(nn.Module):
    """
    Deep Survival Model mapping duration of a worker operating under stress imbalance
    to their probability of exiting the workforce (attrition risk / hazard ratio).
    Methodology Step 5.
    
    Inputs are physiological and embedded covariates.
    Outputs a single scalar for each sample representing the proportional log-risk.
    """
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout=0.2):
        super(DeepSurv, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Construct dense layers mapping covariates to hazard
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            # ReLU activation is extremely standard in modern deep survival setups
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
            
        # Final output layer evaluates the prognostic index (risk score).
        # We do not use an activation function here, because the Loss function 
        # (NegativeLogPartialLikelihood) takes the raw risk score.
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Output shape is strictly (batch_size, 1)
        risk_score = self.network(x)
        return risk_score
