"""
Training loop for the multi-head StressTCN (Pillar 2).

Loss:
    total_loss = alpha * regression_loss(MSE) + beta * classification_loss(CE)
    Default: alpha=0.7, beta=0.3 (regression-weighted, since stress index is
    the primary target and classification head is auxiliary).
"""

from __future__ import annotations

import copy
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_tcn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train the multi-head StressTCN.

    Expected batch format from DataLoader:
        (x, stress_target, fragment_target)
        where:
            x               — (B, seq_len, input_dim) float32
            stress_target   — (B,) or (B, 1) float32  [Stress Index 0-1]
            fragment_target — (B,) int64               [Fragment class label]

    If the loader yields only (x, target) tuples (e.g. when stress index is
    the only label), the classification loss is disabled automatically.

    Config keys:
        epochs        — number of training epochs       (default 10)
        learning_rate — Adam lr                          (default 1e-3)
        patience      — early stopping patience          (default 5)
        alpha         — regression loss weight           (default 0.7)
        beta          — classification loss weight        (default 0.3)

    Returns:
        (best_model, history)
        history keys: train_loss, val_loss, train_reg_loss, train_cls_loss
    """
    model = model.to(device)
    best_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    epochs  = config.get('epochs', 10)
    lr      = config.get('learning_rate', 1e-3)
    patience= config.get('patience', 5)
    alpha   = config.get('alpha', 0.7)
    beta    = config.get('beta', 0.3)

    optimizer     = optim.Adam(model.parameters(), lr=lr)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    history: Dict[str, list] = {
        'train_loss': [], 'val_loss': [],
        'train_reg_loss': [], 'train_cls_loss': [],
    }

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        epoch_total = epoch_reg = epoch_cls = 0.0
        n_batches = 0

        for batch in train_loader:
            has_cls = len(batch) == 3
            if has_cls:
                x, stress_tgt, frag_tgt = batch
                frag_tgt = frag_tgt.long().to(device)
            else:
                x, stress_tgt = batch[:2]

            x          = x.to(device)
            stress_tgt = stress_tgt.float().to(device)
            if stress_tgt.dim() == 1:
                stress_tgt = stress_tgt.unsqueeze(-1)

            optimizer.zero_grad()
            stress_pred, frag_logits = model(x)

            reg_loss = reg_criterion(stress_pred, stress_tgt)

            if has_cls:
                cls_loss  = cls_criterion(frag_logits, frag_tgt)
                total_loss = alpha * reg_loss + beta * cls_loss
            else:
                cls_loss  = torch.tensor(0.0)
                total_loss = reg_loss

            total_loss.backward()
            # Gradient clipping to stabilise TCN training on long sequences
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_reg   += reg_loss.item()
            epoch_cls   += cls_loss.item()
            n_batches   += 1

        avg_train_loss = epoch_total / max(n_batches, 1)
        history['train_loss'].append(avg_train_loss)
        history['train_reg_loss'].append(epoch_reg / max(n_batches, 1))
        history['train_cls_loss'].append(epoch_cls / max(n_batches, 1))

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                has_cls = len(batch) == 3
                if has_cls:
                    x, stress_tgt, frag_tgt = batch
                    frag_tgt = frag_tgt.long().to(device)
                else:
                    x, stress_tgt = batch[:2]

                x          = x.to(device)
                stress_tgt = stress_tgt.float().to(device)
                if stress_tgt.dim() == 1:
                    stress_tgt = stress_tgt.unsqueeze(-1)

                stress_pred, frag_logits = model(x)
                reg_loss = reg_criterion(stress_pred, stress_tgt)

                if has_cls:
                    cls_loss   = cls_criterion(frag_logits, frag_tgt)
                    batch_loss = alpha * reg_loss + beta * cls_loss
                else:
                    batch_loss = reg_loss

                val_total += batch_loss.item()
                n_val     += 1

        avg_val_loss = val_total / max(n_val, 1)
        history['val_loss'].append(avg_val_loss)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train: {avg_train_loss:.4f} (reg={history['train_reg_loss'][-1]:.4f}, "
            f"cls={history['train_cls_loss'][-1]:.4f}) | "
            f"Val: {avg_val_loss:.4f}"
        )

        # ── Early stopping ───────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_weights)
    return model, history


def evaluate_tcn(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Evaluate a trained StressTCN; returns stress predictions and regression metrics.

    Returns:
        dict with keys: stress_preds, fragment_preds, mse, mae
    """
    model.eval()
    model.to(device)

    stress_preds  = []
    fragment_preds = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            stress_pred, frag_logits = model(x)
            stress_preds.append(stress_pred.cpu())
            fragment_preds.append(frag_logits.argmax(dim=-1).cpu())

    stress_preds   = torch.cat(stress_preds, dim=0).numpy()
    fragment_preds = torch.cat(fragment_preds, dim=0).numpy()

    return {
        'stress_preds':   stress_preds,
        'fragment_preds': fragment_preds,
    }
