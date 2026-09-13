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

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys, json, time, gc, argparse
from collections import deque
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

        for v in list(models.keys()):
            model = models[v]
            t0 = time.time()
            m  = wrap(model)
            train_model(m, X_tr, Y_tr, lr=LRS.get(v, 1e-3), epochs=epochs,
                        batch_size=batch_size, verbose=False, device=DEVICE)
            preds, trues = evaluate_model(m, X_te, Y_te, device=DEVICE)
            mae     = float(np.abs(preds - trues).mean())
            elapsed = time.time() - t0

            # Chaotic metrics for SOTA comparison
            # (Finance is stochastic regime-switching, not deterministic chaos — skip VPT)
            if exp_name == 'finance':
                cm = {
                    'mae': float(np.abs(preds[0] - trues[0]).mean()),
                    'vpt_lyap': float('nan'),
                    'nrmse_1': float(np.sqrt(((preds[0][:1] - trues[0][:1]) ** 2).mean())),
                    'nrmse_20': float(np.sqrt(((preds[0][min(19, len(preds[0])-1):min(20, len(preds[0]))] -
                                               trues[0][min(19, len(trues[0])-1):min(20, len(trues[0]))]) ** 2).mean())),
                    'smape_10': float(200.0 * np.mean(np.abs(preds[0][:min(10, len(preds[0]))] -
                                                              trues[0][:min(10, len(trues[0]))]) /
                                                      (np.abs(preds[0][:min(10, len(preds[0]))]) +
                                                       np.abs(trues[0][:min(10, len(trues[0]))]) + 1e-8))),
                    'system': 'finance_regime_switching',
                    'lambda': float('nan'),
                }
            else:
                sys_key = {'lorenz': 'lorenz63_rho28', 'ks_pde': 'ks_pde',
                           'weather': 'lorenz96', 'robotics': 'lorenz96'}.get(exp_name, 'lorenz63_rho28')
                dt_map = {'lorenz': 0.02, 'ks_pde': 0.25, 'weather': 0.05, 'robotics': 0.02}
                dt = dt_map.get(exp_name, 0.02)
                cm = compute_chaotic_metrics(preds[0], trues[0],
                                             system=sys_key, dt=dt)

            seed_maes[v].append(mae)
            timing[v].append(elapsed)
            chaos_data[v].append(cm)

            # GPU memory cleanup: release this variant's model before the next one starts
            del models[v]
            del m
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vpt_str = f"{cm['vpt_lyap']:.2f}TL" if not np.isnan(cm['vpt_lyap']) else "N/A"
            print(f"    {v:<22}  MAE={mae:.4f}  "
                  f"VPT={vpt_str:<7}  "
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
        mean_vpt = np.nanmean([c['vpt_lyap'] for c in chaos_data[v]])
        summary[v] = {'mean': mean, 'std': std, 'vs_tf': vs_tf,
                      'mean_vpt': mean_vpt, 'maes': seed_maes[v],
                      'chaos': chaos_data[v]}
        vpt_print = f"{mean_vpt:.2f}" if not np.isnan(mean_vpt) else "N/A"
        print(f"  {v:<22}  {mean:>10.4f}  {std:>8.4f}  "
              f"{vs_tf:>7.2f}×  {vpt_print:>10}")

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


def run_coherence_test(variants, max_steps=1_000_000, seeds=None):
    """
    PATCH P6: .detach() added to context window update.
    Prevents OOM accumulation across long iterations.

    Safety ceiling is max_steps. If any (variant, seed) reaches it without
    diverging, that result is flagged for manual inspection (possible metric
    blind spot or model collapse).

    Divergence criterion (replaces old flat "mae > 1.5 for 5 consecutive
    windows" rule, which two problems confirmed was unreliable):
      - Per-window error is normalized by that window's ground-truth std,
        reusing the exact normalization compute_vpt() uses in
        chaotic_metrics.py (RMSE / truth_std, threshold=0.4) instead of a
        flat absolute MAE cutoff that isn't calibrated to the system's
        natural scale.
      - "Diverged" now requires the FRACTION of bad windows within a
        rolling window of the last DIVERGENCE_WINDOW_SIZE windows to exceed
        DIVERGENCE_FRACTION_THRESHOLD, instead of a lucky streak of 5
        consecutive bad windows in a row (measured to occur by chance
        alone, independent of real divergence).
    """
    if seeds is None:
        seeds = [42]

    # Divergence criterion tuning (see docstring above).
    # VPT_NORM_THRESHOLD reuses compute_vpt()'s default `threshold=0.4` in
    # chaotic_metrics.py, applied to the same RMSE/truth_std normalization,
    # rather than inventing a new cutoff for a differently-scaled quantity.
    VPT_NORM_THRESHOLD = 0.4
    # Rolling window size (within the 15-25 range) and bad-fraction cutoff.
    # Measured chance-level single-window "bad" rate was ~25%; 50% is double
    # that and, for a window of 20 independent draws at p=0.25, sits ~2.6
    # standard deviations above the chance-level mean (binomial: mean=5,
    # std=1.94 bad windows out of 20) - comfortably out of chance-oscillation
    # range while still reachable by a genuinely diverged model.
    DIVERGENCE_WINDOW_SIZE = 20
    DIVERGENCE_FRACTION_THRESHOLD = 0.5

    # Force deterministic CUDA kernels so "same seed" actually reproduces the
    # same trained weights and the same autoregressive rollout. Without this,
    # cuDNN algorithm selection and non-deterministic reductions introduce
    # tiny floating-point differences that get exponentially amplified by
    # the chaotic dynamics in this test. Scoped here (not in set_seed()) so
    # it only affects the coherence test, not the regular benchmark runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from data import generate_lorenz63

    print(f"\n{'='*62}")
    print(f"  COHERENCE TEST — {max_steps:,} steps  seeds={seeds}")
    print(f"  (P6 patch: .detach() in context window update)")
    print(f"{'='*62}")

    rho = 28.0
    dt  = 0.02
    seq_len, pred_steps = 50, 20

    # Store results as list of dicts: (variant, seed, coherent_steps, stopped_reason, needs_manual_review)
    all_results = []
    per_variant_steps = {v: [] for v in variants}

    for seed in seeds:
        print(f"\n  ── seed {seed} ──")
        set_seed(seed)

        # Generate trajectory for this seed
        traj = generate_lorenz63(rho, max_steps + 300, dt=dt,
                                  warmup=2000, seed=seed)
        traj_t = torch.FloatTensor(traj[300:])
        mu  = traj_t[:500].mean(0)
        std = traj_t[:500].std(0).clamp(min=1e-6)
        traj_norm = (traj_t - mu) / std

        # Build training data
        X_list, Y_list = [], []
        for s in range(0, min(5000, len(traj_norm)) - seq_len - pred_steps, 3):
            X_list.append(traj_norm[s:s + seq_len])
            Y_list.append(traj_norm[s + seq_len:s + seq_len + pred_steps])
        X_tr = torch.stack(X_list)
        Y_tr = torch.stack(Y_list)

        # Train all variants on this seed's data
        trained = {}
        for v in variants:
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

        # Run coherence test for each variant
        # NOTE: iterate over a snapshot of keys (not .items()) so each
        # variant's model can be deleted from `trained` immediately after
        # its rollout finishes, rather than staying resident on the GPU
        # for the rest of the loop.
        for v in list(trained.keys()):
            model = trained[v]
            model.eval()
            context = traj_norm[:seq_len].unsqueeze(0).to(DEVICE)
            coherent_steps = 0
            recent_windows = deque(maxlen=DIVERGENCE_WINDOW_SIZE)
            step = 0
            stopped_reason = None
            step_start_time = time.time()

            # Collect trajectories for inspection (seed 0 only, select variants)
            save_trajectory = (seed == 0 and v in ['v22_baseline', 'v23_full', 'v23_hybrid', 'lstm'])
            pred_trajectory = []
            gt_trajectory = []

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

                    # Normalized error: same approach as compute_vpt() in
                    # chaotic_metrics.py - RMSE divided by this window's
                    # ground-truth std, instead of a flat absolute MAE
                    # threshold that isn't calibrated to the system's scale.
                    window_std = gt.std().item()
                    if window_std < 1e-8:
                        window_std = 1.0
                    normalized_error = torch.sqrt(((pred - gt) ** 2).mean()).item() / window_std

                    # Collect trajectory data for inspection
                    if save_trajectory:
                        pred_trajectory.append(pred.detach().cpu().numpy())
                        gt_trajectory.append(gt.detach().cpu().numpy())

                    recent_windows.append(normalized_error > VPT_NORM_THRESHOLD)
                    if (len(recent_windows) == DIVERGENCE_WINDOW_SIZE and
                            sum(recent_windows) / DIVERGENCE_WINDOW_SIZE > DIVERGENCE_FRACTION_THRESHOLD):
                        stopped_reason = "diverged"
                        break

                    coherent_steps += pred_steps
                    step           += pred_steps

                    # PATCH P6: .detach() prevents OOM at step ~20,000
                    context = torch.cat([
                        context[:, pred_steps:, :],
                        pred.unsqueeze(0).clamp(-10, 10).detach()  # PATCH P6
                    ], dim=1)

                    # Progress report every 1,000 steps
                    if step % 1000 == 0:
                        elapsed = time.time() - step_start_time
                        rate = step / max(elapsed, 0.1)
                        print(f"      step {step:>7,}/{max_steps:,} — {elapsed:>6.0f}s elapsed — {rate:>6.0f} steps/sec")

                # If loop exited normally without diverging, we hit max_steps ceiling
                if stopped_reason is None:
                    stopped_reason = "max_steps_reached"

            # Save trajectories for manual inspection (seed 0 only, select variants)
            if save_trajectory and pred_trajectory:
                os.makedirs('./results/coherence', exist_ok=True)
                pred_array = np.concatenate(pred_trajectory, axis=0)
                gt_array = np.concatenate(gt_trajectory, axis=0)
                save_path = f'./results/coherence/trajectory_debug_{v}_seed0.npz'
                np.savez(save_path, predicted=pred_array, ground_truth=gt_array,
                         coherent_steps=coherent_steps, stopped_reason=stopped_reason)

            needs_review = (stopped_reason == "max_steps_reached")
            mark = "🏆" if coherent_steps >= 19940 else "✅" if coherent_steps > 5000 else "⚠️"
            if needs_review:
                mark = "⚠️🔍"
            print(f"    {mark} {v:<20}  {coherent_steps:>8,} steps  ({stopped_reason})")

            all_results.append({
                'variant': v,
                'seed': seed,
                'coherent_steps': coherent_steps,
                'stopped_reason': stopped_reason,
                'needs_manual_review': needs_review
            })
            per_variant_steps[v].append((coherent_steps, needs_review))

            # GPU memory cleanup: release this variant's model before the next one starts.
            # Without this, every variant's model for the current seed stays resident on
            # the GPU for the entire coherence loop (up to max_steps each), so memory
            # builds up variant-over-variant and seed-over-seed with no error until the
            # process is killed (often silently, with no Python-level traceback).
            del trained[v]
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary table: per-variant stats across all seeds
    # Separate normal results from those needing manual review
    normal_steps = {v: [s for s, review in per_variant_steps[v] if not review]
                    for v in variants}
    review_steps = {v: [(seeds[i], s) for i, (s, review) in enumerate(per_variant_steps[v]) if review]
                    for v in variants}

    print(f"\n  {'Variant':<22}  {'Mean Steps':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

    summary = {}
    for v in variants:
        if normal_steps[v]:
            arr = np.array(normal_steps[v])
            mean, std = arr.mean(), arr.std()
            min_val, max_val = arr.min(), arr.max()
            summary[v] = {'mean': float(mean), 'std': float(std),
                         'min': int(min_val), 'max': int(max_val)}
            print(f"  {v:<22}  {mean:>12.0f}  {std:>10.1f}  {min_val:>10}  {max_val:>10}")

    # Separately list any results that hit the ceiling (need manual review)
    has_review_cases = any(review_steps.values())
    if has_review_cases:
        print(f"\n  ⚠️  Results flagged for manual review (hit --max-steps ceiling):")
        print(f"  {'Variant':<22}  {'Seed':>6}  {'Steps':>10}")
        print(f"  {'─'*22}  {'─'*6}  {'─'*10}")
        for v in variants:
            for seed_val, s in review_steps[v]:
                print(f"  {v:<22}  {seed_val:>6}  {s:>10,}  ← verify: real coherence or metric blind spot?")

    # Save results: both individual (variant, seed) records and summary
    os.makedirs('./results/coherence', exist_ok=True)
    with open('./results/coherence/coherence_results.json', 'w') as f:
        json.dump({
            'rho': rho,
            'max_steps': max_steps,
            'seeds': seeds,
            'individual_results': all_results,
            'summary': summary,
            'note': 'Results with needs_manual_review=true hit the --max-steps ceiling; '
                    'verify they are genuinely coherent, not metric blind spots.'
        }, f, indent=2)
    print(f"\n  Saved: ./results/coherence/coherence_results.json")
    return all_results, summary


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
    parser.add_argument('--max-steps', type=int, default=1_000_000)
    args = parser.parse_args()

    if args.quick:
        args.seeds  = [42]
        args.epochs = 30
        print("Quick mode: 1 seed, 30 epochs")

    print(f"Device: {DEVICE}  GPUs: {N_GPU}")

    if args.coherence:
        run_coherence_test(args.variants, args.max_steps, seeds=args.seeds)
    else:
        for exp in args.experiment:
            run_experiment(exp, args.variants, args.seeds, args.epochs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
