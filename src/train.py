"""
SynechismCore v21.0 — Training Utilities
=========================================
v21.0 changes vs v20.0:
    - DataParallel-aware: get_base_model(), is_ode_model(), compute_loss()
      all correctly unwrap nn.DataParallel before attribute checks.
    - KL beta annealing: beta ramps from 0→target over first 30 epochs.
      Prevents posterior collapse where the encoder learns nothing useful
      in early epochs (the main cause of low latent utilization at rho=40+).
    - train_model() passes current epoch to compute_loss() for annealing.
    - Gradient clipping (max_norm=1.0) unchanged — already in v20.

Loss function:
    Total = MSE(pred, target)
          + beta_eff * KL(mu, logvar)    [for ODE variants, annealed]
          + gamma * agent_regularization [for hybrid variant]

Author: Paul E. Harris IV — SynechismCore v21.0
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# DATAPARALLEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_base_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel to access the underlying model's attributes."""
    return model.module if isinstance(model, nn.DataParallel) else model


def is_ode_model(model: nn.Module) -> bool:
    """Check if model is an ODE variant (returns pred, mu, logvar tuple)."""
    return hasattr(get_base_model(model), 'ode_func')


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(model, xb, yb, beta=0.0005, epoch=0, warmup_epochs=30):
    """
    Unified loss for all model types.

    KL annealing: beta_eff = beta * min(1, epoch / warmup_epochs)
    This prevents posterior collapse in early training where the encoder
    has not yet learned meaningful representations but the KL penalty
    pushes mu→0, logvar→0 (posterior = prior, information destroyed).

    Without annealing: encoder collapses at high-chaos rho values.
    With annealing: encoder has 30 epochs to build useful representations
    before the full KL penalty kicks in.
    """
    base = get_base_model(model)

    # KL annealing: ramp from 0 → beta over warmup_epochs
    beta_eff = beta * min(1.0, epoch / max(1, warmup_epochs))

    if is_ode_model(model):
        pred, mu, logvar = model(xb)
        min_steps = min(pred.shape[1], yb.shape[1])
        recon = nn.functional.mse_loss(pred[:, :min_steps], yb[:, :min_steps])
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss  = recon + beta_eff * kl

        # HyperAgent regularization (uses base model to avoid DataParallel wrapper)
        if hasattr(base, 'use_agent') and base.use_agent:
            loss = loss + base.agent_regularization_loss(xb)
    else:
        pred = model(xb)
        min_steps = min(pred.shape[1], yb.shape[1])
        loss = nn.functional.mse_loss(pred[:, :min_steps], yb[:, :min_steps])

    return loss


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model:        nn.Module,
    X_train:      torch.Tensor,
    Y_train:      torch.Tensor,
    lr:           float,
    epochs:       int,
    batch_size:   int   = 64,
    name:         str   = 'model',
    verbose:      bool  = True,
    device:       torch.device = None,
    beta:         float = 0.0005,
    patience:     int   = 30,
    warmup_epochs: int  = 30,
) -> float:
    """
    Train model with cosine LR schedule, gradient clipping, and KL annealing.

    DataParallel note: pass the DataParallel-wrapped model in; this function
    handles it correctly via get_base_model() in compute_loss().

    Returns best training loss achieved.
    """
    if device is None:
        device = DEVICE

    # Move model to device (DataParallel models are already on device,
    # but this is safe to call again)
    if not isinstance(model, nn.DataParallel):
        model = model.to(device)

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    dataset   = TensorDataset(X_train, Y_train)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           drop_last=True, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss  = float('inf')
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = compute_loss(model, xb, yb, beta=beta,
                                epoch=epoch, warmup_epochs=warmup_epochs)
            loss.backward()
            # Gradient clipping — prevents LSTM/ODE explosion on chaotic attractors
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        scheduler.step()

        if avg < best_loss:
            best_loss  = avg
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    [{name:>18}] early stop @ epoch {epoch+1} | loss={best_loss:.6f}")
            break

        if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"    [{name:>18}] epoch {epoch+1:>3}/{epochs} | loss={avg:.6f}")

    return best_loss


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model:      nn.Module,
    X_test:     torch.Tensor,
    Y_test:     torch.Tensor,
    device:     torch.device = None,
    batch_size: int = 128,
) -> tuple:
    """
    Evaluate model on test set. Returns (predictions, true_values) as numpy arrays.
    Shape: (N, T, D). Pass to stats.compute_full_stats() for p-values.

    DataParallel-safe: eval mode propagates to all replicas automatically.
    """
    if device is None:
        device = DEVICE

    if not isinstance(model, nn.DataParallel):
        model = model.to(device)
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            yb = Y_test[i:i+batch_size]

            if is_ode_model(model):
                pred, _, _ = model(xb)
            else:
                pred = model(xb)

            pred = pred.cpu()
            min_steps = min(pred.shape[1], yb.shape[1])
            all_preds.append(pred[:, :min_steps].numpy())
            all_true.append(yb[:, :min_steps].numpy())

    return np.concatenate(all_preds), np.concatenate(all_true)
