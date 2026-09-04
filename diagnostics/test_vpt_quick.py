#!/usr/bin/env python3
"""
Quick test to verify VPT fixes work correctly.
Runs a single experiment (Lorenz) with minimal training.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

os.environ['WANDB_DISABLED'] = 'true'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import make_lorenz_dataset, make_ks_dataset, make_weather_dataset, make_finance_dataset, make_robotics_dataset
from models import FairTransformer
from train import train_model, evaluate_model
from chaotic_metrics import compute_chaotic_metrics, LYAPUNOV_EXPONENTS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_vpt_all_experiments():
    """Test VPT computation for all experiments"""
    print("="*70)
    print("VPT Fix Verification Test")
    print("="*70)

    experiments = [
        ('lorenz', make_lorenz_dataset([18, 20, 22], n_traj=5, seq_len=50, pred_steps=20), 'lorenz63_rho28', 0.02),
        ('ks_pde', make_ks_dataset(nu=1.0, n_traj=2, seq_len=64, pred_steps=16), 'ks_pde', 0.25),
        ('weather', make_weather_dataset([3, 4], n_traj=2, seq_len=50, pred_steps=10), 'lorenz96', 0.05),
        ('finance', make_finance_dataset(regime='calm', n_steps=2000), None, None),  # Finance skip VPT
        ('robotics', make_robotics_dataset(gamma=0.5, n_traj=2, seq_len=50, pred_steps=20), 'lorenz96', 0.02),
    ]

    for exp_name, (_, X_tr, Y_tr), sys_key, dt in experiments:
        print(f"\n{'-'*70}")
        print(f"Experiment: {exp_name.upper()}")
        print(f"{'-'*70}")

        # Create a simple model
        in_dim = X_tr.shape[-1]
        model = FairTransformer(in_dim, in_dim, 64, Y_tr.shape[1])
        model = model.to(DEVICE)

        # Quick training
        print(f"Training on {len(X_tr)} samples for 50 epochs...")
        train_model(model, X_tr, Y_tr, lr=1e-3, epochs=50, batch_size=16,
                   verbose=False, device=DEVICE)

        # Evaluate
        print(f"Evaluating...")
        preds, trues = evaluate_model(model, X_tr[:20], Y_tr[:20], device=DEVICE)

        print(f"Predictions shape: {preds.shape}, True shape: {trues.shape}")

        # Compute metrics
        if exp_name == 'finance':
            # Finance: skip VPT, compute only MAE/nRMSE
            mae = float(np.abs(preds[0] - trues[0]).mean())
            nrmse_1 = float(np.sqrt(((preds[0][:1] - trues[0][:1]) ** 2).mean()))
            print(f"  MAE: {mae:.6f}")
            print(f"  nRMSE@1: {nrmse_1:.6f}")
            print(f"  VPT: N/A (stochastic regime-switching, not chaotic)")
        else:
            # Chaotic systems: compute VPT
            cm = compute_chaotic_metrics(preds[0], trues[0], system=sys_key, dt=dt)
            print(f"  MAE: {cm['mae']:.6f}")
            print(f"  VPT: {cm['vpt_lyap']:.4f} Lyapunov times")
            print(f"  Lambda: {cm['lambda']:.4f}")
            print(f"  nRMSE@1: {cm['nrmse_1']:.6f}")
            print(f"  nRMSE@20: {cm['nrmse_20']:.6f}")

            # Check VPT is not stuck at 0
            if cm['vpt_lyap'] < 0.001:
                print(f"  WARNING: VPT is very small ({cm['vpt_lyap']:.6f})")
            else:
                print(f"  OK: VPT is non-zero (good!)")

if __name__ == '__main__':
    test_vpt_all_experiments()
    print("\n" + "="*70)
    print("VPT test complete!")
    print("="*70)
