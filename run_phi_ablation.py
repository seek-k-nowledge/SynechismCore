#!/usr/bin/env python3
"""
SynechismCore v23.0.1 — φ vs √2 vs e Ablation (Claim 4)
=========================================================
Tests whether φ specifically beats other irrationals, or whether any
irrational base provides equivalent benefit over uniform sampling.

Theoretical prediction: φ wins (Hurwitz's theorem — φ is the hardest
real number to approximate rationally, giving lowest star discrepancy).
Empirical confirmation: this script.

If φ does NOT specifically win, the claim is revised to:
'irrational sampling beats uniform sampling' — still publishable and
still confirmed by v17.2 (p=0.0000).

Usage:
    python run_phi_ablation.py --system lorenz --seeds 42 0 1 7 100
    python run_phi_ablation.py --quick
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data import make_lorenz_dataset, make_ks_dataset
from train import train_model, evaluate_model
from v23_components import SynechismV23, compare_shutters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./results/fresh_run', exist_ok=True)

PHI   = (1.0 + 5.0 ** 0.5) / 2.0
BASES = {
    'phi':     PHI,
    'sqrt2':   np.sqrt(2),
    'e':       np.e,
    'uniform': 1.0,
}


def run_phi_ablation(system='lorenz', seeds=None, epochs=100, quick=False):
    if quick:
        seeds, epochs = [42], 30
        print("Quick mode")
    seeds = seeds or [42, 0, 1, 7, 100]

    print(f"\n{'='*60}")
    print(f"  φ ABLATION  system={system}  seeds={seeds}  epochs={epochs}")
    print(f"{'='*60}")

    # Theoretical comparison first
    print("\nTheoretical: Star Discrepancy D* (lower = more uniform)")
    for name, d in sorted(compare_shutters(50).items(), key=lambda x: x[1]):
        bar = '█' * int(d * 200)
        print(f"  {name:<25} D*={d:.4f}  {bar}")

    if system == 'lorenz':
        _, X_tr, Y_tr = make_lorenz_dataset(
            [18, 20, 22, 24, 26, 28], n_traj=80, seq_len=50, pred_steps=20)
        _, X_te, Y_te = make_lorenz_dataset(
            [35], n_traj=30, seq_len=50, pred_steps=20, seed=999)
        in_dim = 3
    else:
        _, X_tr, Y_tr = make_ks_dataset(nu=1.0)
        _, X_te, Y_te = make_ks_dataset(nu=0.5)
        in_dim = 64

    all_maes = {name: [] for name in BASES}

    for seed in seeds:
        print(f"\n  ── seed {seed} ──")
        np.random.seed(seed); torch.manual_seed(seed)

        for base_name, base_val in BASES.items():
            model = SynechismV23(
                in_dim=in_dim, out_dim=in_dim, hidden=128,
                pred_steps=20, system=system,
                use_elastic_manifold=False,
                use_irrational_shutter=True,
                use_laminar_bypass=False,
                phi_base=base_val,
            ).to(DEVICE)

            t0 = time.time()
            train_model(model, X_tr, Y_tr, lr=1e-3, epochs=epochs,
                        batch_size=64, verbose=False, device=DEVICE)
            preds, trues = evaluate_model(model, X_te, Y_te, device=DEVICE)
            mae = float(np.abs(preds - trues).mean())
            all_maes[base_name].append(mae)
            print(f"    {base_name:<12} MAE={mae:.4f}  ({time.time()-t0:.0f}s)")

    # Results
    print(f"\n  {'Base':<12}  {'Mean MAE':>10}  {'Std':>8}  {'vs uniform':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*10}")
    uniform_mean = np.array(all_maes['uniform']).mean()
    final = {}
    for name, maes in all_maes.items():
        arr  = np.array(maes)
        mean = arr.mean()
        std  = arr.std()
        ratio = uniform_mean / (mean + 1e-8)
        final[name] = {'mean': mean, 'std': std, 'ratio': ratio}
        print(f"  {name:<12}  {mean:>10.4f}  {std:>8.4f}  {ratio:>9.2f}×")

    # Statistical test
    if len(seeds) > 1:
        from scipy.stats import mannwhitneyu
        phi_arr = np.array(all_maes['phi'])
        unif_arr = np.array(all_maes['uniform'])
        _, p = mannwhitneyu(phi_arr, unif_arr, alternative='less')
        final['p_phi_vs_uniform'] = p
        sig = '✅ significant' if p < 0.05 else '⚠️  not significant'
        print(f"\n  φ vs uniform: p={p:.4e}  {sig}")

        # Also test phi vs sqrt2 and phi vs e
        for other in ['sqrt2', 'e']:
            _, p2 = mannwhitneyu(phi_arr, np.array(all_maes[other]),
                                 alternative='less')
            final[f'p_phi_vs_{other}'] = p2
            sig2 = '✅' if p2 < 0.05 else '⚠️'
            print(f"  φ vs {other:<6}: p={p2:.4e}  {sig2}")

    path = f'./results/fresh_run/phi_ablation_{system}.json'
    with open(path, 'w') as f:
        json.dump({'system': system, 'seeds': seeds, 'epochs': epochs,
                   'results': final, 'raw_maes': all_maes}, f, indent=2)
    print(f"\n  Saved: {path}")
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', default='lorenz',
                        choices=['lorenz', 'ks_pde'])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 0, 1, 7, 100])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_phi_ablation(system=args.system, seeds=args.seeds,
                     epochs=args.epochs, quick=args.quick)
