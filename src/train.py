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
# PERIODIC-CORRECTION TRAINING (for CorrectionGate variants only)
# ══════════════════════════════════════════════════════════════════════════════

def train_model_with_periodic_correction(
    model:           nn.Module,
    traj_norm:       torch.Tensor,
    seq_len:         int   = 50,
    pred_steps:      int   = 20,
    lookahead_steps: int   = 20,
    lr:              float = 1e-3,
    epochs:          int   = 100,
    batch_size:      int   = 64,
    name:            str   = 'model',
    verbose:         bool  = True,
    device:          torch.device = None,
    beta:            float = 0.0005,
    patience:        int   = 30,
    warmup_epochs:   int   = 30,
) -> float:
    """
    Trains a CorrectionGate variant (e.g. make_v23('v23_hybrid_gated', ...))
    by exposing it to a simulated real-data correction event during a
    short rollout, instead of only the single-window supervised fit that
    train_model() does. Only variants with use_correction_gate=True need
    this - every other variant keeps using the standard train_model().

    Design: each training example simulates exactly one correction event,
    at the boundary of the first predicted window (i.e. from a single
    training example's perspective, correction happens every pred_steps).
    The actual test-time correction cadence (e.g. every ~40 steps, less
    frequent than every window) is a coherence-rollout scheduling detail
    handled separately in run_coherence_test_with_correction() - this
    training procedure only needs to teach the gate "a correction just
    happened here, here's the consequence one window later," which is the
    same lesson regardless of how often that happens at test time. That's
    why correction_interval isn't a parameter here.

    BPTT is truncated to exactly two windows (the correction window, plus
    one lookahead window lookahead_steps long) per training example, in
    the spirit of the "pushforward trick" (Brandstetter et al., 2022) for
    stabilizing autoregressive rollout training without unbounded
    backprop-through-time - this is not an implementation of that exact
    algorithm, just the same underlying idea: train on the model's own
    generated (corrected) rollout state without paying for long BPTT.

    Regularization (KL annealing, HyperAgent sparsity/magnitude penalty)
    mirrors compute_loss() exactly, applied to both windows, so the only
    experimental variable this introduces relative to train_model() is
    the correction-gate exposure - not a different training recipe that
    could confound comparison with the fixed-schedule/zero-correction
    reference variants.

    NOTE on DataParallel: forward_with_states() is called directly on the
    unwrapped base module (get_base_model(model)), not through `model`,
    because nn.DataParallel only dispatches to .forward() and has no
    mechanism for routing other method calls across replicas. This means
    the rollout computation in this function runs on a single device even
    if N_GPU > 1 - a deliberate simplification for this new, specialized
    training path, not an oversight.
    """
    if device is None:
        device = DEVICE

    base_model = get_base_model(model)
    if not (hasattr(base_model, 'use_correction_gate') and base_model.use_correction_gate):
        raise ValueError(
            "train_model_with_periodic_correction() requires a model built "
            "with use_correction_gate=True (e.g. make_v23('v23_hybrid_gated', ...))."
        )

    if not isinstance(model, nn.DataParallel):
        model = model.to(device)
    base_model = get_base_model(model)

    traj_norm   = traj_norm.to(device)
    T           = traj_norm.shape[0]
    window_span = seq_len + pred_steps + lookahead_steps
    max_start   = T - window_span
    if max_start <= 0:
        raise ValueError(
            f"traj_norm too short ({T} steps) for seq_len={seq_len} + "
            f"pred_steps={pred_steps} + lookahead_steps={lookahead_steps}."
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss       = float('inf')
    no_improve      = 0
    steps_per_epoch = max(1, max_start // batch_size)

    # model.eval() first, then correction_gate.train() specifically:
    # everything else in this model (the frozen, already-trained base
    # weights) must not drift at all, including buffers that update on
    # forward passes independent of requires_grad - e.g. spectral_norm's
    # power-iteration vectors inside ElasticAttractorODE.nonlinear, which
    # only refresh while their owning module is in train mode. Calling
    # plain model.train() here would silently let those buffers keep
    # moving even though their weights are frozen, undermining the
    # "identical underlying weights otherwise" isolation this training
    # path exists to guarantee. correction_gate itself has no
    # train/eval-sensitive layers (no BatchNorm/Dropout/spectral_norm),
    # so putting it in train mode has no side effect beyond signaling
    # intent - it's set explicitly only for clarity and convention.
    #
    # This is called exactly ONCE, here, before the epoch loop - not
    # inside it. If a future edit adds a per-epoch or per-batch
    # model.train() call anywhere below, this eval()/train() sequence
    # must be reapplied at that same point, or the fix is silently
    # undone partway through training.
    model.eval()
    base_model.correction_gate.train()

    for epoch in range(epochs):
        beta_eff = beta * min(1.0, epoch / max(1, warmup_epochs))
        total = 0.0

        for _ in range(steps_per_epoch):
            starts = torch.randint(0, max_start, (batch_size,)).tolist()
            optimizer.zero_grad()

            ctx     = torch.stack([traj_norm[s:s + seq_len] for s in starts])
            gt1     = torch.stack([traj_norm[s + seq_len:s + seq_len + pred_steps] for s in starts])
            gt2_off = seq_len + pred_steps
            gt2     = torch.stack([traj_norm[s + gt2_off:s + gt2_off + lookahead_steps] for s in starts])

            # Window 1: raw prediction, then correction at its last step
            pred1, mu1, logvar1, h_traj1 = base_model.forward_with_states(ctx, pred_steps)
            loss1 = nn.functional.mse_loss(pred1, gt1)
            kl1   = -0.5 * torch.mean(1 + logvar1 - mu1.pow(2) - logvar1.exp())

            last_pred   = pred1[:, -1, :]
            last_real   = gt1[:, -1, :]
            discrepancy = last_real - last_pred
            h_last      = h_traj1[-1]
            gate        = base_model.correction_gate(h_last, discrepancy)
            corrected_last = last_pred + gate * discrepancy

            corrected_window1 = torch.cat(
                [pred1[:, :-1, :], corrected_last.unsqueeze(1)], dim=1)

            # Window 2 (lookahead): built from the corrected rollout -
            # gradient stays connected back through corrected_last into
            # the gate. This is the only place the gate gets a training
            # signal about the consequence of its own decision.
            new_ctx = torch.cat([ctx[:, pred_steps:, :], corrected_window1], dim=1)
            pred2, mu2, logvar2, _ = base_model.forward_with_states(new_ctx, lookahead_steps)
            loss2 = nn.functional.mse_loss(pred2, gt2)
            kl2   = -0.5 * torch.mean(1 + logvar2 - mu2.pow(2) - logvar2.exp())

            loss = loss1 + loss2 + beta_eff * (kl1 + kl2)
            if base_model.use_agent:
                loss = loss + base_model.agent_regularization_loss(ctx)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()

        avg = total / steps_per_epoch
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

    NaN/Inf protection: detects and replaces invalid values in predictions.
    Warning printed if any found (must not silently hide the problem).
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

    preds = np.concatenate(all_preds)
    true_vals = np.concatenate(all_true)

    # NaN/Inf protection: count invalid values
    nan_count = np.isnan(preds).sum()
    inf_count = np.isinf(preds).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  WARNING: {nan_count} NaN + {inf_count} Inf values in predictions, replacing with nan_to_num")
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)

    return preds, true_vals
