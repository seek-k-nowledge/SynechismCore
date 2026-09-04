#!/usr/bin/env python3
"""
SynechismCore v23.0.1 — v23 Benchmark + Coherence Test
========================================================
Runs head-to-head: v22 baseline vs v23 variants vs baselines.
Also runs the 25,000-step coherence rollout test.

PATCH P6 applied: .detach() in coherence context window update.
  Without .detach(), tensor references accumulate across 25,000 iterations.
  Even inside torch.no_grad(), some PyTorch versions accumulate references.
  .detach() guarantees no OOM at step ~20,000.

Usage:
    python run_v23_benchmark.py --quick
    python run_v23_benchmark.py --experiment lorenz robotics --seeds 0 1 2 3 4
    python run_v23_benchmark.py --coherence --max-steps 25000 --seeds 42
    python run_v23_benchmark.py --experiment ks_pde --seeds 0 1 2 3 4 5 6 7 8 9
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault('WANDB_DISABLED', 'true')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import (make_lorenz_dataset, make_ks_dataset,
                  make_finance_dataset, make_weather_dataset,
                  make_robotics_dataset)
from models import FairTransformer, FairLSTM, FairMamba, make_synechism
from train import train_model, evaluate_model, get_base_model
from stats import compute_full_stats
from v23_components import make_v23
from chaotic_metrics import compute_chaotic_metrics, print_sota_comparison

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU  = torch.cuda.device_count() if torch.cuda.is_available() else 0

os.makedirs('./results/fresh_run', exist_ok=True)
os.makedirs('./results/v23', exist_ok=True)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap(model):
    model = model.to(DEVICE)
    if N_GPU > 1:
        model = nn.DataParallel(model)
    return model


def get_experiment(name, seed=42):
    if name == 'lorenz':
        rho_train = [18, 20, 22, 24, 26, 28]
        rho_test  = [35, 40, 45, 50]
        _, X_tr, Y_tr = make_lorenz_dataset(
            rho_train, n_traj=100, seq_len=50, pred_steps=20, seed=seed)
        _, X_te, Y_te = make_lorenz_dataset(
            rho_test, n_traj=30, seq_len=50, pred_steps=20, seed=seed + 1000)
        return X_tr, Y_tr, X_te, Y_te, 3, 'Lorenz-63 (ρ: 18-28 → 35-50)'
    elif name == 'ks_pde':
        _, X_tr, Y_tr = make_ks_dataset(nu=1.0)
        _, X_te, Y_te = make_ks_dataset(nu=0.5)
        return X_tr, Y_tr, X_te, Y_te, 64, 'KS-PDE (ν: 1.0→0.5)'
    elif name == 'finance':
        _, X_tr, Y_tr = make_finance_dataset(regime='calm')
        _, X_te, Y_te = make_finance_dataset(regime='crisis')
        return X_tr, Y_tr, X_te, Y_te, X_tr.shape[-1], 'Finance (calm→crisis)'
    elif name == 'weather':
        _, X_tr, Y_tr = make_weather_dataset(F_values=[6, 7, 8, 9, 10])
        _, X_te, Y_te = make_weather_dataset(F_values=[12, 16])
        return X_tr, Y_tr, X_te, Y_te, 40, 'Weather L96 (F: 6-10→12-16)'
    elif name == 'robotics':
        _, X_tr, Y_tr = make_robotics_dataset(gamma=0.5)
        _, X_te, Y_te = make_robotics_dataset(gamma=0.05)
        return X_tr, Y_tr, X_te, Y_te, X_tr.shape[-1], 'Robotics (γ: 0.5→0.05)'
    else:
        raise ValueError(f"Unknown experiment: {name}")


V23_VARIANTS = ['v22_baseline', 'shutter_only', 'elastic_only',
                'bypass_only', 'v23_full', 'v23_hybrid']

LRS = {
    'v22_baseline': 1e-3, 'shutter_only': 1e-3, 'elastic_only': 1e-3,
    'bypass_only':  1e-3, 'v23_full': 1e-3,      'v23_hybrid': 8e-4,
    'transformer':  6e-4, 'lstm': 1e-3,           'mamba': 5e-4,
}


def build_models(variants, in_dim, pred_steps=20, system='default', hidden=128):
    models = {}
    for v in variants:
        if v in V23_VARIANTS:
            models[v] = make_v23(v, in_dim, in_dim, hidden=hidden,
                                 pred_steps=pred_steps, system=system)
        elif v == 'transformer':
            models[v] = FairTransformer(in_dim, in_dim, hidden, pred_steps)
        elif v == 'lstm':
            models[v] = FairLSTM(in_dim, in_dim, hidden, pred_steps)
        elif v == 'mamba':
            models[v] = FairMamba(in_dim, in_dim, hidden, pred_steps)
    return models


def run_experiment(exp_name, variants, seeds, epochs, batch_size=512):
    print(f"\n{'='*62}")
    print(f"  {exp_name.upper()} | variants={variants} | seeds={seeds}")
    print(f"{'='*62}")

    seed_maes  = {v: [] for v in variants}
    timing     = {v: [] for v in variants}
    chaos_data = {v: [] for v in variants}

    for seed in seeds:
        print(f"\n  ── seed {seed} ──")
        set_seed(seed)
        X_tr, Y_tr, X_te, Y_te, in_dim, label = get_experiment(exp_name, seed)
        system = exp_name if exp_name in ('ks_pde', 'weather') else 'lorenz63_rho28'
        models = build_models(variants, in_dim, system=system)

        for v, model in models.items():
            t0 = time.time()
            m  = wrap(model)
            train_model(m, X_tr, Y_tr, lr=LRS.get(v, 1e-3), epochs=epochs,
                        batch_size=batch_size, verbose=False, device=DEVICE)
            preds, trues = evaluate_model(m, X_te, Y_te, device=DEVICE)
            mae     = float(np.abs(preds - trues).mean())
            elapsed = time.time() - t0

            # Chaotic metrics for SOTA comparison
            sys_key = {'lorenz': 'lorenz63_rho28', 'ks_pde': 'ks_pde',
                       'weather': 'lorenz96', 'finance': 'ks_pde',
                       'robotics': 'lorenz96'}.get(exp_name, 'lorenz63_rho28')
            dt_map = {'lorenz': 0.02, 'ks_pde': 0.25, 'finance': 0.01,
                      'weather': 0.05, 'robotics': 0.02}
            dt = dt_map.get(exp_name, 0.02)
            cm = compute_chaotic_metrics(preds[0], trues[0],
                                         system=sys_key, dt=dt)

            seed_maes[v].append(mae)
            timing[v].append(elapsed)
            chaos_data[v].append(cm)
            print(f"    {v:<22}  MAE={mae:.4f}  "
                  f"VPT={cm['vpt_lyap']:.2f}TL  "
                  f"nRMSE1={cm['nrmse_1']:.4f}  ({elapsed:.0f}s)")

    # Summary
    print(f"\n  {'Variant':<22}  {'Mean MAE':>10}  {'Std':>8}  "
          f"{'vs TF':>8}  {'Mean VPT':>10}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")

    summary = {}
    tf_maes  = np.array(seed_maes.get('transformer', [1.0]))
    tf_mean  = tf_maes.mean()

    for v in variants:
        arr  = np.array(seed_maes[v])
        mean, std = arr.mean(), arr.std()
        vs_tf = tf_mean / (mean + 1e-8)
        mean_vpt = np.mean([c['vpt_lyap'] for c in chaos_data[v]])
        summary[v] = {'mean': mean, 'std': std, 'vs_tf': vs_tf,
                      'mean_vpt': mean_vpt, 'maes': seed_maes[v],
                      'chaos': chaos_data[v]}
        print(f"  {v:<22}  {mean:>10.4f}  {std:>8.4f}  "
              f"{vs_tf:>7.2f}×  {mean_vpt:>10.2f}")

    # Significance: v23_full vs transformer
    if 'v23_full' in seed_maes and 'transformer' in seed_maes and len(seeds) > 1:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(np.array(seed_maes['v23_full']),
                            np.array(seed_maes['transformer']),
                            alternative='less')
        summary['p_v23_vs_tf'] = p
        sig = '✅ significant' if p < 0.05 else '⚠️  not significant'
        print(f"\n  v23_full vs transformer: p={p:.4e}  {sig}")

    # SOTA comparison printout
    if exp_name in ('lorenz', 'ks_pde') and 'v23_full' in summary:
        best = {'mae': summary['v23_full']['mean'],
                'vpt_lyap': summary['v23_full']['mean_vpt'],
                'nrmse_1': np.mean([c['nrmse_1']
                                    for c in chaos_data['v23_full']]),
                'nrmse_20': np.mean([c['nrmse_20']
                                     for c in chaos_data['v23_full']]),
                'smape_10': np.mean([c['smape_10']
                                     for c in chaos_data['v23_full']])}
        sota_sys = 'lorenz63' if exp_name == 'lorenz' else 'ks_pde'
        print_sota_comparison(best, sota_sys)

    # Save
    path = f'./results/fresh_run/{exp_name}.json'
    with open(path, 'w') as f:
        json.dump({
            'experiment': exp_name, 'label': label,
            'seeds': seeds, 'epochs': epochs,
            'summary': {k: {kk: vv for kk, vv in v.items()
                            if kk not in ('maes', 'chaos')}
                        for k, v in summary.items() if isinstance(v, dict)},
            'raw_maes': seed_maes,
        }, f, indent=2)
    print(f"\n  Saved: {path}")
    return summary


def run_coherence_test(variants, max_steps=25000, seed=42):
    """
    PATCH P6: .detach() added to context window update.
    Prevents OOM accumulation across 25,000 iterations.
    """
    from data import generate_lorenz63

    print(f"\n{'='*62}")
    print(f"  COHERENCE TEST — {max_steps:,} steps  seed={seed}")
    print(f"  (P6 patch: .detach() in context window update)")
    print(f"{'='*62}")

    set_seed(seed)
    rho = 28.0
    dt  = 0.02
    traj = generate_lorenz63(rho, max_steps + 300, dt=dt,
                              warmup=2000, seed=seed)
    traj_t = torch.FloatTensor(traj[300:])
    mu  = traj_t[:500].mean(0)
    std = traj_t[:500].std(0).clamp(min=1e-6)
    traj_norm = (traj_t - mu) / std

    seq_len, pred_steps = 50, 20
    X_list, Y_list = [], []
    for s in range(0, min(5000, len(traj_norm)) - seq_len - pred_steps, 3):
        X_list.append(traj_norm[s:s + seq_len])
        Y_list.append(traj_norm[s + seq_len:s + seq_len + pred_steps])
    X_tr = torch.stack(X_list)
    Y_tr = torch.stack(Y_list)

    trained = {}
    for v in variants:
        print(f"\n  Training {v}...")
        if v in V23_VARIANTS:
            model = make_v23(v, 3, 3, hidden=128, pred_steps=pred_steps)
        elif v == 'transformer':
            model = FairTransformer(3, 3, 128, pred_steps)
        elif v == 'lstm':
            model = FairLSTM(3, 3, 128, pred_steps)
        elif v == 'mamba':
            model = FairMamba(3, 3, 128, pred_steps)
        else:
            model = make_synechism(v, 3, 3, hidden=128, pred_steps=pred_steps)
        m = wrap(model)
        train_model(m, X_tr, Y_tr, lr=1e-3, epochs=100,
                    batch_size=64, verbose=False, device=DEVICE)
        trained[v] = m

    results = {}
    for v, model in trained.items():
        model.eval()
        context = traj_norm[:seq_len].unsqueeze(0).to(DEVICE)
        coherent_steps = 0
        bad_windows    = 0
        step = 0

        with torch.no_grad():
            while step < max_steps - pred_steps:
                base = get_base_model(model)
                if hasattr(base, 'ode_func') or hasattr(base, 'encoder'):
                    pred, _, _ = model(context)
                else:
                    pred = model(context)
                pred = pred[0]  # (pred_steps, 3)

                gt_start = seq_len + step
                gt_end   = gt_start + pred_steps
                if gt_end >= len(traj_norm):
                    break
                gt  = traj_norm[gt_start:gt_end].to(DEVICE)
                mae = (pred - gt).abs().mean().item()

                if mae > 1.5:
                    bad_windows += 1
                    if bad_windows >= 5:
                        break
                else:
                    bad_windows = 0

                coherent_steps += pred_steps
                step           += pred_steps

                # PATCH P6: .detach() prevents OOM at step ~20,000
                context = torch.cat([
                    context[:, pred_steps:, :],
                    pred.unsqueeze(0).clamp(-10, 10).detach()  # PATCH P6
                ], dim=1)

        results[v] = coherent_steps
        mark = "🏆" if coherent_steps >= 19940 else "✅" if coherent_steps > 5000 else "⚠️"
        print(f"  {mark} {v:<22}  {coherent_steps:>8,} steps")

    print(f"\n  {'Variant':<22}  {'Steps':>8}  {'vs LSTM':>8}")
    lstm_steps = results.get('lstm', max(results.values(), default=1))
    for v, s in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {v:<22}  {s:>8,}  {s / max(lstm_steps, 1):>7.1f}×")

    os.makedirs('./results/fresh_run', exist_ok=True)
    with open('./results/fresh_run/coherence_test.json', 'w') as f:
        json.dump({'rho': rho, 'max_steps': max_steps,
                   'seed': seed, 'results': results}, f, indent=2)
    print("\n  Saved: ./results/fresh_run/coherence_test.json")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', nargs='+',
                        default=['lorenz', 'ks_pde', 'finance',
                                 'weather', 'robotics'])
    parser.add_argument('--variants', nargs='+',
                        default=['v22_baseline', 'v23_full', 'v23_hybrid',
                                 'transformer', 'lstm', 'mamba'])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 0, 1, 7, 100])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--quick', action='store_true',
                        help='1 seed, 30 epochs')
    parser.add_argument('--coherence', action='store_true',
                        help='Run coherence rollout test (25k steps)')
    parser.add_argument('--max-steps', type=int, default=25000)
    args = parser.parse_args()

    if args.quick:
        args.seeds  = [42]
        args.epochs = 30
        print("Quick mode: 1 seed, 30 epochs")

    print(f"Device: {DEVICE}  GPUs: {N_GPU}")

    if args.coherence:
        run_coherence_test(args.variants, args.max_steps, seed=args.seeds[0])
    else:
        for exp in args.experiment:
            run_experiment(exp, args.variants, args.seeds, args.epochs)
