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
from train import (train_model, evaluate_model, get_base_model,
                   train_model_with_periodic_correction)
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


def run_coherence_test_with_correction(max_steps=1_000_000, seeds=None,
                                        correction_interval=40, fixed_c=0.8,
                                        lookahead_steps=20, gate_train_epochs=100):
    """
    NEW, ADDITIONAL coherence test mode: periodic real-data correction.
    Does NOT modify run_coherence_test() above in any way - this is a
    separate function, duplicating its boilerplate deliberately (per
    review decision) rather than factoring out shared helpers, so there
    is zero risk of this new experiment disturbing the already-verified
    zero-correction test.

    Compares four conditions, matching Fan et al. (2020)'s reservoir-
    computing periodic-correction cadence (~40 steps) against a new
    learned-gate alternative:
      - v23_hybrid_zero    — no correction ever (existing behavior,
                              re-run here on the SAME trained weights used
                              for v23_hybrid_fixed, so the only difference
                              between those two conditions is the
                              correction strategy, not a different
                              training run)
      - v23_hybrid_fixed   — fixed-schedule correction, c=fixed_c
                              (Fan et al. baseline replication)
      - v23_hybrid_gated   — learned CorrectionGate, trained via
                              train_model_with_periodic_correction()
                              (the new contribution)
      - v22_baseline_fixed — outside reference: fixed-schedule correction
                              on the plain v22 architecture, to gauge how
                              much periodic correction helps independent
                              of the v23 components

    Divergence is measured against each window's RAW, pre-correction
    prediction (confirmed design decision) - so periodic correction
    can't mask genuine model failure in the reported "coherent steps"
    metric. The correction only affects what gets fed forward as context
    for the next window, never what the divergence check compares.

    Correction only touches the LAST timestep of a window when that
    window's end lands on a correction_interval boundary (e.g. every
    2nd window at pred_steps=20, correction_interval=40) - the other
    pred_steps-1 positions in that window are always the model's raw,
    uncorrected prediction.
    """
    if seeds is None:
        seeds = [42]

    VPT_NORM_THRESHOLD = 0.4
    DIVERGENCE_WINDOW_SIZE = 20
    DIVERGENCE_FRACTION_THRESHOLD = 0.5

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from data import generate_lorenz63

    print(f"\n{'='*62}")
    print(f"  COHERENCE TEST WITH PERIODIC CORRECTION — {max_steps:,} steps")
    print(f"  seeds={seeds}  correction_interval={correction_interval}  fixed_c={fixed_c}")
    print(f"{'='*62}")

    rho = 28.0
    dt  = 0.02
    seq_len, pred_steps = 50, 20

    CONDITIONS = [
        {'name': 'v23_hybrid_zero',    'base_variant': 'v23_hybrid',       'correction_mode': 'zero'},
        {'name': 'v23_hybrid_fixed',   'base_variant': 'v23_hybrid',       'correction_mode': 'fixed'},
        {'name': 'v23_hybrid_gated',   'base_variant': 'v23_hybrid_gated', 'correction_mode': 'gated'},
        {'name': 'v22_baseline_fixed', 'base_variant': 'v22_baseline',     'correction_mode': 'fixed'},
    ]
    base_variants = sorted(set(c['base_variant'] for c in CONDITIONS))
    uses_remaining = {bv: sum(1 for c in CONDITIONS if c['base_variant'] == bv)
                      for bv in base_variants}

    all_results = []
    per_condition_steps = {c['name']: [] for c in CONDITIONS}

    for seed in seeds:
        print(f"\n  ── seed {seed} ──")
        set_seed(seed)

        traj = generate_lorenz63(rho, max_steps + 300, dt=dt,
                                  warmup=2000, seed=seed)
        traj_t = torch.FloatTensor(traj[300:])
        mu  = traj_t[:500].mean(0)
        std = traj_t[:500].std(0).clamp(min=1e-6)
        traj_norm = (traj_t - mu) / std

        X_list, Y_list = [], []
        for s in range(0, min(5000, len(traj_norm)) - seq_len - pred_steps, 3):
            X_list.append(traj_norm[s:s + seq_len])
            Y_list.append(traj_norm[s + seq_len:s + seq_len + pred_steps])
        X_tr = torch.stack(X_list)
        Y_tr = torch.stack(Y_list)

        remaining = dict(uses_remaining)
        trained_bases = {}

        # Process 'v23_hybrid' before 'v23_hybrid_gated' explicitly (not
        # relying on incidental alphabetical ordering) - v23_hybrid_gated
        # needs v23_hybrid's already-trained weights below.
        ordered_variants = [bv for bv in base_variants if bv != 'v23_hybrid_gated']
        if 'v23_hybrid_gated' in base_variants:
            ordered_variants.append('v23_hybrid_gated')

        for bv in ordered_variants:
            model = make_v23(bv, 3, 3, hidden=128, pred_steps=pred_steps)

            if bv == 'v23_hybrid_gated':
                # Start from v23_hybrid's exact trained weights (same seed),
                # not from scratch - isolates the correction strategy as the
                # ONLY difference between v23_hybrid_fixed/zero and
                # v23_hybrid_gated. strict=False because v23_hybrid_gated
                # has one extra module (correction_gate) that v23_hybrid's
                # state dict has no entry for.
                base_hybrid = get_base_model(trained_bases['v23_hybrid'])
                model.load_state_dict(base_hybrid.state_dict(), strict=False)

                # Freeze everything except correction_gate - only the gate
                # itself should learn from train_model_with_periodic_correction().
                # This is what actually isolates the comparison: the base
                # model's dynamics stay bit-for-bit the ones v23_hybrid
                # already trained, so any difference vs v23_hybrid_fixed is
                # attributable to the correction strategy alone.
                for param_name, param in model.named_parameters():
                    param.requires_grad = param_name.startswith('correction_gate.')

            m = wrap(model)
            if bv == 'v23_hybrid_gated':
                train_model_with_periodic_correction(
                    m, traj_norm[:5000], seq_len=seq_len, pred_steps=pred_steps,
                    lookahead_steps=lookahead_steps, epochs=gate_train_epochs,
                    batch_size=64, verbose=False, device=DEVICE, name=bv,
                )
            else:
                train_model(m, X_tr, Y_tr, lr=1e-3, epochs=100,
                            batch_size=64, verbose=False, device=DEVICE)
            trained_bases[bv] = m

        for cond in CONDITIONS:
            name  = cond['name']
            mode  = cond['correction_mode']
            model = trained_bases[cond['base_variant']]
            base  = get_base_model(model)
            model.eval()

            context = traj_norm[:seq_len].unsqueeze(0).to(DEVICE)
            coherent_steps = 0
            recent_windows = deque(maxlen=DIVERGENCE_WINDOW_SIZE)
            step = 0
            stopped_reason = None
            step_start_time = time.time()

            save_trajectory = (seed == 0)
            pred_trajectory, gt_trajectory = [], []
            correction_flags, gate_values = [], []

            with torch.no_grad():
                while step < max_steps - pred_steps:
                    # forward_with_states() is called on the unwrapped base
                    # module directly (see train_model_with_periodic_correction's
                    # docstring for why - DataParallel only dispatches .forward()).
                    pred, _, _, h_traj = base.forward_with_states(context)
                    pred   = pred[0]        # (pred_steps, 3)
                    h_traj = h_traj[:, 0, :]  # (pred_steps, hidden)

                    gt_start = seq_len + step
                    gt_end   = gt_start + pred_steps
                    if gt_end >= len(traj_norm):
                        break
                    gt = traj_norm[gt_start:gt_end].to(DEVICE)

                    # Divergence check on the RAW, pre-correction prediction -
                    # confirmed design decision, so correction can't mask
                    # genuine model failure in this metric.
                    window_std = gt.std().item()
                    if window_std < 1e-8:
                        window_std = 1.0
                    normalized_error = torch.sqrt(((pred - gt) ** 2).mean()).item() / window_std

                    recent_windows.append(normalized_error > VPT_NORM_THRESHOLD)
                    if (len(recent_windows) == DIVERGENCE_WINDOW_SIZE and
                            sum(recent_windows) / DIVERGENCE_WINDOW_SIZE > DIVERGENCE_FRACTION_THRESHOLD):
                        stopped_reason = "diverged"
                        break

                    coherent_steps += pred_steps
                    step           += pred_steps

                    # Correction only touches the LAST timestep of this
                    # window, only when its end lands on a correction
                    # boundary, and only for non-zero correction modes.
                    next_window = pred.clone()
                    is_correction_step = (mode != 'zero' and step % correction_interval == 0)
                    gate_val = float('nan')
                    if is_correction_step:
                        last_pred   = pred[-1]
                        last_real   = gt[-1]
                        discrepancy = last_real - last_pred
                        if mode == 'fixed':
                            gate_val = fixed_c
                        elif mode == 'gated':
                            h_last = h_traj[-1].unsqueeze(0)
                            disc_b = discrepancy.unsqueeze(0)
                            gate_val = base.correction_gate(h_last, disc_b).item()
                        next_window[-1] = last_pred + gate_val * discrepancy

                    if save_trajectory:
                        pred_trajectory.append(pred.detach().cpu().numpy())
                        gt_trajectory.append(gt.detach().cpu().numpy())
                        correction_flags.append(bool(is_correction_step))
                        gate_values.append(gate_val)

                    context = torch.cat([
                        context[:, pred_steps:, :],
                        next_window.unsqueeze(0).clamp(-10, 10).detach()
                    ], dim=1)

                    if step % 1000 == 0:
                        elapsed = time.time() - step_start_time
                        rate = step / max(elapsed, 0.1)
                        print(f"      [{name}] step {step:>7,}/{max_steps:,} — "
                              f"{elapsed:>6.0f}s elapsed — {rate:>6.0f} steps/sec")

                if stopped_reason is None:
                    stopped_reason = "max_steps_reached"

            if save_trajectory and pred_trajectory:
                os.makedirs('./results/coherence', exist_ok=True)
                np.savez(
                    f'./results/coherence/trajectory_debug_correction_{name}_seed0.npz',
                    predicted=np.concatenate(pred_trajectory, axis=0),
                    ground_truth=np.concatenate(gt_trajectory, axis=0),
                    correction_flags=np.array(correction_flags),
                    gate_values=np.array(gate_values),
                    coherent_steps=coherent_steps, stopped_reason=stopped_reason,
                )

            needs_review = (stopped_reason == "max_steps_reached")
            mark = "🏆" if coherent_steps >= 19940 else "✅" if coherent_steps > 5000 else "⚠️"
            if needs_review:
                mark = "⚠️🔍"
            print(f"    {mark} {name:<20}  {coherent_steps:>8,} steps  ({stopped_reason})")

            all_results.append({
                'condition': name,
                'base_variant': cond['base_variant'],
                'correction_mode': mode,
                'seed': seed,
                'coherent_steps': coherent_steps,
                'stopped_reason': stopped_reason,
                'needs_manual_review': needs_review,
            })
            per_condition_steps[name].append((coherent_steps, needs_review))

            # GPU cleanup only once every condition sharing this base
            # variant (v23_hybrid is reused by both the zero and fixed
            # conditions) has finished its rollout.
            bv = cond['base_variant']
            remaining[bv] -= 1
            if remaining[bv] == 0:
                del trained_bases[bv]
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    normal_steps = {n: [s for s, review in per_condition_steps[n] if not review]
                    for n in per_condition_steps}
    review_steps = {n: [(seeds[i], s) for i, (s, review) in enumerate(per_condition_steps[n]) if review]
                    for n in per_condition_steps}

    print(f"\n  {'Condition':<22}  {'Mean Steps':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

    summary = {}
    for name in per_condition_steps:
        if normal_steps[name]:
            arr = np.array(normal_steps[name])
            mean, std = arr.mean(), arr.std()
            min_val, max_val = arr.min(), arr.max()
            summary[name] = {'mean': float(mean), 'std': float(std),
                             'min': int(min_val), 'max': int(max_val)}
            print(f"  {name:<22}  {mean:>12.0f}  {std:>10.1f}  {min_val:>10}  {max_val:>10}")

    has_review_cases = any(review_steps.values())
    if has_review_cases:
        print(f"\n  ⚠️  Results flagged for manual review (hit --max-steps ceiling):")
        print(f"  {'Condition':<22}  {'Seed':>6}  {'Steps':>10}")
        print(f"  {'─'*22}  {'─'*6}  {'─'*10}")
        for name in per_condition_steps:
            for seed_val, s in review_steps[name]:
                print(f"  {name:<22}  {seed_val:>6}  {s:>10,}  ← verify: real coherence or metric blind spot?")

    os.makedirs('./results/coherence', exist_ok=True)
    with open('./results/coherence/coherence_correction_results.json', 'w') as f:
        json.dump({
            'rho': rho,
            'max_steps': max_steps,
            'seeds': seeds,
            'correction_interval': correction_interval,
            'fixed_c': fixed_c,
            'lookahead_steps': lookahead_steps,
            'individual_results': all_results,
            'summary': summary,
            'note': 'Results with needs_manual_review=true hit the --max-steps ceiling; '
                    'verify they are genuinely coherent, not metric blind spots.'
        }, f, indent=2)
    print(f"\n  Saved: ./results/coherence/coherence_correction_results.json")
    return all_results, summary


# Standard Lorenz-63 largest Lyapunov exponent, per user-specified value
# (the codebase's own chaotic_metrics.py stores 0.9056 for 'lorenz63_rho28',
# effectively the same value to 3 significant figures).
LORENZ63_LYAPUNOV_EXPONENT = 0.906


def steps_to_lyapunov_times(steps, dt, lyapunov_exponent=LORENZ63_LYAPUNOV_EXPONENT):
    """1 Lyapunov time = 1 / (lyapunov_exponent * dt) steps."""
    return steps * lyapunov_exponent * dt


# ── First-crossing VPT helpers (used only by run_coherence_test_head_to_head
#    and run_coherence_test_gate_retrain) ────────────────────────────────────
VPT_FIRST_CROSSING_THRESHOLDS = (0.3, 0.4)


def _std_ddof1(values):
    """Sample standard deviation (ddof=1); None when fewer than 2 values."""
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=1)) if arr.size >= 2 else None


def _fmt_opt(x, spec):
    return format(x, spec) if x is not None else format('n/a', '>' + spec.split('.')[0].lstrip('>'))


class _FirstCrossingVPT:
    """
    Per-step first-crossing valid prediction time (VPT) tracker.

    Definition (exact):
      sigma_d  = per-dimension standard deviation (ddof=1) of traj_norm[:5000]
                 (the normalized trajectory's first 5000 steps; fewer if the
                 trajectory is shorter).
      e(t)     = sqrt( mean over dimensions d of ((pred_t[d] - truth_t[d]) / sigma_d[d])^2 )
                 for every individual rollout step t = 0, 1, 2, ... (t = 0 is
                 the first predicted step, truth index seq_len + t). pred_t is
                 the model's emitted prediction for step t, BEFORE any
                 correction is blended into the next context window. A
                 non-finite e(t) (NaN/inf prediction) is treated as +inf,
                 i.e. as an exceedance.
      VPT(th)  = the first step t at which e(t) > th, i.e. the number of
                 steps predicted before the error first exceeds th.
      Censoring: if e(t) never exceeds th within the evaluated horizon,
                 VPT(th) = horizon (number of steps evaluated) and
                 censored = True. Summaries count censored runs at their
                 horizon as lower bounds (see _summarize_vpt).
    """

    def __init__(self, traj_norm, device, thresholds=VPT_FIRST_CROSSING_THRESHOLDS):
        self.sigma_d = traj_norm[:5000].std(0, unbiased=True).clamp(min=1e-8).to(device)
        self.thresholds = tuple(thresholds)
        self.first_cross = {th: None for th in self.thresholds}
        self.chunks = []
        self.n_steps = 0

    def update(self, pred, gt):
        e = torch.sqrt((((pred - gt) / self.sigma_d) ** 2).mean(dim=-1))
        e = e.detach().cpu().numpy().astype(np.float64)
        e = np.where(np.isfinite(e), e, np.inf)
        for th in self.thresholds:
            if self.first_cross[th] is None:
                idx = np.nonzero(e > th)[0]
                if idx.size:
                    self.first_cross[th] = self.n_steps + int(idx[0])
        self.chunks.append(e.astype(np.float32))
        self.n_steps += len(e)

    def all_crossed(self):
        return all(v is not None for v in self.first_cross.values())

    def series(self):
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)

    def results(self, dt):
        out = {'vpt_horizon_steps': self.n_steps}
        for th in self.thresholds:
            censored = self.first_cross[th] is None
            steps = self.n_steps if censored else self.first_cross[th]
            out[f'vpt_{th}_steps'] = int(steps)
            out[f'vpt_{th}_lyapunov_times'] = float(steps_to_lyapunov_times(steps, dt))
            out[f'vpt_{th}_censored'] = bool(censored)
        return out

    def save_series(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.series())
        return path


def _summarize_vpt(results, key_field, dt, thresholds=VPT_FIRST_CROSSING_THRESHOLDS):
    """Per-condition VPT summary. Censored runs are never silently dropped.

    For each condition and threshold:
      n                       all runs
      n_uncensored/n_censored runs that did / did not cross the threshold
      median_steps            median over ALL runs, censored runs counted at
                              their horizon (valid as long as fewer than half
                              are censored; otherwise it is itself a lower bound)
      median_is_lower_bound   True when half or more of the runs are censored
      lower_bound_mean_steps  mean over ALL runs with censored runs counted at
                              their horizon. The true mean is >= this value
                              whenever any run is censored.
      mean_steps / std_steps  mean and std (ddof=1) over UNCENSORED runs only
                              (kept for reference; biased low if the best runs
                              are the censored ones)
    """
    summary = {}
    names = sorted({r[key_field] for r in results}, key=lambda n: [r[key_field] for r in results].index(n))
    for name in names:
        rows = [r for r in results if r[key_field] == name]
        summary[name] = {}
        for th in thresholds:
            all_steps = np.asarray([r[f'vpt_{th}_steps'] for r in rows], dtype=float)
            unc = [r[f'vpt_{th}_steps'] for r in rows if not r[f'vpt_{th}_censored']]
            cen = [(r['seed'], r[f'vpt_{th}_steps']) for r in rows if r[f'vpt_{th}_censored']]
            s = {'n': len(rows), 'n_uncensored': len(unc), 'n_censored': len(cen),
                 'censored_runs_seed_horizon': cen}
            if len(all_steps):
                med = float(np.median(all_steps))
                lb_mean = float(all_steps.mean())
                s.update({
                    'median_steps': med,
                    'median_lyapunov_times': float(steps_to_lyapunov_times(med, dt)),
                    'median_is_lower_bound': bool(len(cen) * 2 >= len(rows)),
                    'lower_bound_mean_steps': lb_mean,
                    'lower_bound_mean_lyapunov_times': float(steps_to_lyapunov_times(lb_mean, dt)),
                    'lower_bound_mean_is_exact': len(cen) == 0,
                })
            if unc:
                arr = np.asarray(unc, dtype=float)
                std = _std_ddof1(arr)
                s.update({
                    'mean_steps': float(arr.mean()), 'std_steps': std,
                    'min_steps': int(arr.min()), 'max_steps': int(arr.max()),
                    'mean_lyapunov_times': float(steps_to_lyapunov_times(arr.mean(), dt)),
                    'std_lyapunov_times': (float(steps_to_lyapunov_times(std, dt))
                                           if std is not None else None),
                })
            summary[name][str(th)] = s
    return summary


def _print_vpt_summary(vpt_summary, label_width, thresholds=VPT_FIRST_CROSSING_THRESHOLDS):
    for th in thresholds:
        print(f"\n  First-crossing VPT({th}) — all runs; censored runs counted at their horizon")
        print(f"  {'Condition':<{label_width}}  {'n':>3}  {'cens':>4}  {'Median':>9}  "
              f"{'LB mean':>9}  {'Med TL':>7}  | {'Unc mean':>9}  {'Unc std':>8}")
        for name, per_th in vpt_summary.items():
            s = per_th[str(th)]
            med = f"{s['median_steps']:>9.1f}" + ("+" if s['median_is_lower_bound'] else " ")
            lb = f"{s['lower_bound_mean_steps']:>9.1f}" + (" " if s['lower_bound_mean_is_exact'] else "+")
            if s['n_uncensored']:
                unc = f"{s['mean_steps']:>9.1f}  {_fmt_opt(s['std_steps'], '>8.1f')}"
            else:
                unc = f"{'(none)':>9}  {'':>8}"
            print(f"  {name:<{label_width}}  {s['n']:>3}  {s['n_censored']:>4}  {med[:-1]}{med[-1]} "
                  f"{lb[:-1]}{lb[-1]} {s['median_lyapunov_times']:>6.2f}  | {unc}")
            for seed_val, horizon in s['censored_runs_seed_horizon']:
                print(f"      censored: seed={seed_val} never crossed {th} within {horizon:,} steps")
        print("  '+' = lower bound (censored runs present).  'Unc' = uncensored runs only, for reference.")


def run_coherence_test_head_to_head(max_steps=1_000_000, seeds=None,
                                     correction_interval=40, fixed_c=0.8):
    """
    Experiment 3 (NEW, ADDITIVE): head-to-head comparison of five
    architectures under IDENTICAL fixed-schedule correction. Does NOT
    modify run_coherence_test(), run_coherence_test_with_correction(), or
    the corrected divergence metric in any way - separate function,
    duplicated boilerplate, same convention as those two.

    Architectures: v23_hybrid, v22_baseline (both via make_v23()), lstm,
    transformer, mamba (src/models.py). Each is trained ONCE per seed via
    the standard train_model(), then evaluated under BOTH zero-correction
    and fixed-correction rollouts from the SAME trained weights - isolating
    the correction strategy as the only variable per architecture (extra
    evaluation, not extra training).

    Correction mechanism is identical to last night's v22_baseline_fixed
    condition in run_coherence_test_with_correction(): corrected_value =
    prediction + fixed_c * (real_value - prediction), applied only to the
    last timestep of a window when that window's end lands on a
    correction_interval boundary. Divergence criterion (normalized
    RMSE/window_std vs 0.4, rolling window >50% of last 20) is the exact
    same corrected metric.

    TWO METRICS are reported per run:

    1. rolling_window_steps (unchanged; also stored as coherent_steps for
       backward compatibility): per 20-step window, normalized error =
       RMSE(pred, truth) / std(truth window); the run "diverges" once the
       last 20 windows are full and >50% of them exceed 0.4.
       coherent_steps grows by 20 only after a window passes that check,
       so the smallest reportable value is 19 x 20 = 380 steps (a floor).

    2. First-crossing VPT (floor-free), computed per time step:
         sigma_d = per-dimension standard deviation (ddof=1) of
                   traj_norm[:5000].
         e(t)    = sqrt(mean_d(((pred_t[d] - truth_t[d]) / sigma_d[d])^2))
                   for every rollout step t = 0, 1, 2, ... (t = 0 is the
                   first predicted step); pred_t is the emitted prediction
                   before any correction is blended into the next context.
                   Non-finite e(t) counts as +inf (an exceedance).
         VPT(th) = the first step t where e(t) > th, for th = 0.3 and 0.4
                   (= number of steps predicted before the first
                   exceedance).
         If e(t) never exceeds th within the evaluated horizon, VPT(th) =
         horizon (steps evaluated) and censored = True; censored runs are
         counted at their horizon in the median and lower-bound mean (labeled as lower bounds), and listed separately.
       The rollout keeps running after the rolling-window metric stops
       (rolling_window_steps is frozen at its stop value) until VPT(0.4)
       has crossed or the horizon ends, so VPT is not cut short by the
       rolling metric. The full e(t) series of every run is saved to
       results/coherence/error_series/head_to_head/<arch>_<mode>_seed<seed>.npy.
       VPT is reported in steps and in Lyapunov times via
       steps_to_lyapunov_times() (~55 steps per Lyapunov time at dt=0.02).
       All summary standard deviations use ddof=1 (None when n < 2).
    """
    if seeds is None:
        seeds = [42]
    from v23_components import SynechismV23

    VPT_NORM_THRESHOLD = 0.4
    DIVERGENCE_WINDOW_SIZE = 20
    DIVERGENCE_FRACTION_THRESHOLD = 0.5

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from data import generate_lorenz63

    print(f"\n{'='*62}")
    print(f"  EXPERIMENT 3: HEAD-TO-HEAD UNDER IDENTICAL CORRECTION — {max_steps:,} steps")
    print(f"  seeds={seeds}  correction_interval={correction_interval}  fixed_c={fixed_c}")
    print(f"{'='*62}")

    rho = 28.0
    dt  = 0.02
    seq_len, pred_steps = 50, 20

    ARCHITECTURES = ['v23_hybrid', 'v22_baseline', 'lstm', 'transformer', 'mamba']
    CORRECTION_MODES = ['zero', 'fixed']

    all_results = []
    per_condition_steps = {f"{a}_{m}": [] for a in ARCHITECTURES for m in CORRECTION_MODES}

    for seed in seeds:
        print(f"\n  ── seed {seed} ──")
        set_seed(seed)

        traj = generate_lorenz63(rho, max_steps + 300, dt=dt,
                                  warmup=2000, seed=seed)
        traj_t = torch.FloatTensor(traj[300:])
        mu  = traj_t[:500].mean(0)
        std = traj_t[:500].std(0).clamp(min=1e-6)
        traj_norm = (traj_t - mu) / std

        X_list, Y_list = [], []
        for s in range(0, min(5000, len(traj_norm)) - seq_len - pred_steps, 3):
            X_list.append(traj_norm[s:s + seq_len])
            Y_list.append(traj_norm[s + seq_len:s + seq_len + pred_steps])
        X_tr = torch.stack(X_list)
        Y_tr = torch.stack(Y_list)

        trained = {}
        for arch in ARCHITECTURES:
            if arch in ('v23_hybrid', 'v22_baseline'):
                model = make_v23(arch, 3, 3, hidden=128, pred_steps=pred_steps)
            elif arch == 'lstm':
                model = FairLSTM(3, 3, 128, pred_steps)
            elif arch == 'transformer':
                model = FairTransformer(3, 3, 128, pred_steps)
            elif arch == 'mamba':
                model = FairMamba(3, 3, 128, pred_steps)
            m = wrap(model)
            train_model(m, X_tr, Y_tr, lr=1e-3, epochs=100,
                        batch_size=64, verbose=False, device=DEVICE)
            trained[arch] = m

        for arch in list(trained.keys()):
            model = trained[arch]
            base  = get_base_model(model)
            model.eval()
            # SynechismV23 models (v23_hybrid, v22_baseline) return a 3-tuple;
            # the plain baselines (lstm, transformer, mamba) return a bare
            # tensor. isinstance instead of attribute duck-typing: an
            # `encoder` attribute is not a safe marker (FairTransformer's
            # encoder stack is named `transformer`, not `encoder`, today,
            # but nothing guarantees that).
            is_synechism = isinstance(base, SynechismV23)

            for mode in CORRECTION_MODES:
                name = f"{arch}_{mode}"
                context = traj_norm[:seq_len].unsqueeze(0).to(DEVICE)
                coherent_steps = 0
                recent_windows = deque(maxlen=DIVERGENCE_WINDOW_SIZE)
                step = 0
                stopped_reason = None
                rolling_stopped = False
                vpt = _FirstCrossingVPT(traj_norm, DEVICE)
                step_start_time = time.time()

                with torch.no_grad():
                    while step < max_steps - pred_steps:
                        if is_synechism:
                            pred, _, _ = model(context)
                        else:
                            pred = model(context)
                        pred = pred[0]

                        gt_start = seq_len + step
                        gt_end   = gt_start + pred_steps
                        if gt_end >= len(traj_norm):
                            break
                        gt = traj_norm[gt_start:gt_end].to(DEVICE)

                        vpt.update(pred, gt)

                        if not rolling_stopped:
                            window_std = gt.std().item()
                            if window_std < 1e-8:
                                window_std = 1.0
                            normalized_error = torch.sqrt(((pred - gt) ** 2).mean()).item() / window_std

                            recent_windows.append(normalized_error > VPT_NORM_THRESHOLD)
                            if (len(recent_windows) == DIVERGENCE_WINDOW_SIZE and
                                    sum(recent_windows) / DIVERGENCE_WINDOW_SIZE > DIVERGENCE_FRACTION_THRESHOLD):
                                stopped_reason = "diverged"
                                rolling_stopped = True

                        # Stop once the rolling metric has stopped AND every
                        # VPT threshold has crossed; otherwise keep rolling
                        # out (rolling_window_steps stays frozen).
                        if rolling_stopped and vpt.all_crossed():
                            break

                        if not rolling_stopped:
                            coherent_steps += pred_steps
                        step           += pred_steps

                        next_window = pred.clone()
                        is_correction_step = (mode == 'fixed' and step % correction_interval == 0)
                        if is_correction_step:
                            last_pred   = pred[-1]
                            last_real   = gt[-1]
                            discrepancy = last_real - last_pred
                            next_window[-1] = last_pred + fixed_c * discrepancy

                        context = torch.cat([
                            context[:, pred_steps:, :],
                            next_window.unsqueeze(0).clamp(-10, 10).detach()
                        ], dim=1)

                        if step % 1000 == 0:
                            elapsed = time.time() - step_start_time
                            rate = step / max(elapsed, 0.1)
                            print(f"      [{name}] step {step:>7,}/{max_steps:,} — "
                                  f"{elapsed:>6.0f}s elapsed — {rate:>6.0f} steps/sec")

                    if stopped_reason is None:
                        stopped_reason = "max_steps_reached"

                needs_review = (stopped_reason == "max_steps_reached")
                lyap_times = steps_to_lyapunov_times(coherent_steps, dt)
                mark = "🏆" if coherent_steps >= 19940 else "✅" if coherent_steps > 5000 else "⚠️"
                if needs_review:
                    mark = "⚠️🔍"
                print(f"    {mark} {name:<20}  {coherent_steps:>8,} steps  "
                      f"({lyap_times:.2f} TL)  ({stopped_reason})")

                vpt_res = vpt.results(dt)
                vpt_status = ("vpt_censored" if any(vpt_res[f'vpt_{th}_censored']
                                                    for th in VPT_FIRST_CROSSING_THRESHOLDS)
                              else "vpt_crossed")
                series_path = vpt.save_series(
                    f'./results/coherence/error_series/head_to_head/{name}_seed{seed}.npy')
                vpt_str = "  ".join(
                    f"VPT({th})={vpt_res[f'vpt_{th}_steps']:,} steps "
                    f"({vpt_res[f'vpt_{th}_lyapunov_times']:.2f} TL)"
                    + (" [CENSORED]" if vpt_res[f'vpt_{th}_censored'] else "")
                    for th in VPT_FIRST_CROSSING_THRESHOLDS)
                print(f"        {vpt_str}  ({vpt_status})")

                all_results.append({
                    'architecture': arch,
                    'correction_mode': mode,
                    'condition': name,
                    'seed': seed,
                    'rolling_window_steps': coherent_steps,
                    'coherent_steps': coherent_steps,
                    'coherent_lyapunov_times': lyap_times,
                    'stopped_reason': stopped_reason,
                    'vpt_status': vpt_status,
                    'needs_manual_review': needs_review,
                    **vpt_res,
                    'error_series_path': series_path,
                })
                per_condition_steps[name].append((coherent_steps, needs_review))

            del trained[arch]
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    normal_steps = {n: [s for s, review in per_condition_steps[n] if not review]
                    for n in per_condition_steps}
    review_steps = {n: [(seeds[i], s) for i, (s, review) in enumerate(per_condition_steps[n]) if review]
                    for n in per_condition_steps}

    print(f"\n  {'Architecture x Mode':<24}  {'Mean Steps':>12}  {'Std':>10}  "
          f"{'Mean TL':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*24}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    summary = {}
    for name in per_condition_steps:
        if normal_steps[name]:
            arr = np.array(normal_steps[name])
            mean, std = arr.mean(), _std_ddof1(arr)
            min_val, max_val = arr.min(), arr.max()
            mean_lt = steps_to_lyapunov_times(mean, dt)
            summary[name] = {'mean': float(mean), 'std': std,
                             'min': int(min_val), 'max': int(max_val),
                             'mean_lyapunov_times': float(mean_lt)}
            print(f"  {name:<24}  {mean:>12.1f}  {_fmt_opt(std, '>10.1f')}  {mean_lt:>10.2f}  {min_val:>8}  {max_val:>8}")

    has_review_cases = any(review_steps.values())
    if has_review_cases:
        print(f"\n  ⚠️  Results flagged for manual review (hit --max-steps ceiling):")
        for name in per_condition_steps:
            for seed_val, s in review_steps[name]:
                print(f"  {name:<24}  seed={seed_val:>3}  {s:>10,} steps  ← verify")

    vpt_summary = _summarize_vpt(all_results, 'condition', dt)
    _print_vpt_summary(vpt_summary, 24)

    os.makedirs('./results/coherence', exist_ok=True)
    with open('./results/coherence/coherence_head_to_head_results.json', 'w') as f:
        json.dump({
            'rho': rho, 'dt': dt,
            'lyapunov_exponent': LORENZ63_LYAPUNOV_EXPONENT,
            'max_steps': max_steps,
            'seeds': seeds,
            'correction_interval': correction_interval,
            'fixed_c': fixed_c,
            'individual_results': all_results,
            'summary': summary,
            'summary_rolling_window_steps': summary,
            'summary_first_crossing_vpt': vpt_summary,
            'vpt_thresholds': list(VPT_FIRST_CROSSING_THRESHOLDS),
            'note': 'Results with needs_manual_review=true hit the --max-steps ceiling; '
                    'verify they are genuinely coherent, not metric blind spots. '
                    'rolling_window_steps has a 380-step reporting floor; '
                    'first-crossing VPT is floor-free (see function docstring). '
                    'Censored VPT runs are counted at their horizon in median_steps and '
                    'lower_bound_mean_steps (lower bounds); mean_steps/std_steps are '
                    'uncensored-only, for reference. Std uses ddof=1.'
        }, f, indent=2)
    print(f"\n  Saved: ./results/coherence/coherence_head_to_head_results.json")
    return all_results, summary


def run_coherence_test_gate_retrain(max_steps=1_000_000, seeds=None,
                                     correction_interval=40, fixed_c=0.8,
                                     lookahead_steps=20, gate_train_epochs=100,
                                     new_n_cycles=5):
    """
    Experiment 4 (NEW, ADDITIVE): gate retrain. Does NOT modify
    run_coherence_test(), run_coherence_test_with_correction(),
    run_coherence_test_head_to_head(), or the corrected divergence metric
    in any way - separate function, duplicated boilerplate, same
    convention as those.

    Tests whether last night's learned CorrectionGate was genuinely
    undertrained/stuck (bias init -3.0 -> sigmoid ~0.05, one correction
    event + one lookahead window per training example) rather than a
    real negative result, by comparing two conditions per seed:

      - v23_hybrid_gated_old_relogged: EXACT same training settings as
        last night's v23_hybrid_gated (init_bias=-3.0 default, n_cycles=1
        default -> train_model_with_periodic_correction()'s unchanged
        original code path). Only difference from last night: gate-value
        logging is now on during the coherence rollout, so we can see
        directly whether the gate was actually stuck near 0.05.
      - v23_hybrid_gated_v2: init_bias=logit(fixed_c) (~=+1.386 for
        fixed_c=0.8, starting near the Fan et al. fixed constant instead
        of near 0), trained with n_cycles=new_n_cycles multi-cycle
        rollout training (Bocquet et al., NPG 2026-inspired - see
        train_model_with_periodic_correction()'s docstring in train.py).
        new_n_cycles defaults to 5; this count wasn't specified in the
        brief, so it's exposed as a parameter rather than silently fixed.

    Both conditions keep last night's isolation exactly: the gated model
    starts from that seed's trained v23_hybrid weights (load_state_dict,
    strict=False), every parameter except correction_gate.* is frozen,
    and train_model_with_periodic_correction() internally does
    model.eval() + correction_gate.train() (not a blanket model.train())
    so frozen spectral-norm buffers can't drift during gate training.

    Gate-value logging (NEW for this experiment): the gate's actual
    output (sigmoid, in [0,1]) is recorded at every correction step, for
    every seed (not just seed 0), stored per-step in individual_results
    and summarized as mean/min/max per (condition, seed). Gate values are
    logged only while the rolling-window metric is still running, so the
    logged gate statistics cover the same span as before.

    TWO METRICS are reported per run:

    1. rolling_window_steps (unchanged; also stored as coherent_steps for
       backward compatibility): per 20-step window, normalized error =
       RMSE(pred, truth) / std(truth window); the run "diverges" once the
       last 20 windows are full and >50% of them exceed 0.4.
       coherent_steps grows by 20 only after a window passes that check,
       so the smallest reportable value is 19 x 20 = 380 steps (a floor).

    2. First-crossing VPT (floor-free), computed per time step:
         sigma_d = per-dimension standard deviation (ddof=1) of
                   traj_norm[:5000].
         e(t)    = sqrt(mean_d(((pred_t[d] - truth_t[d]) / sigma_d[d])^2))
                   for every rollout step t = 0, 1, 2, ... (t = 0 is the
                   first predicted step); pred_t is the emitted prediction
                   before any correction is blended into the next context.
                   Non-finite e(t) counts as +inf (an exceedance).
         VPT(th) = the first step t where e(t) > th, for th = 0.3 and 0.4
                   (= number of steps predicted before the first
                   exceedance).
         If e(t) never exceeds th within the evaluated horizon, VPT(th) =
         horizon (steps evaluated) and censored = True; censored runs are
         counted at their horizon in the median and lower-bound mean (labeled as lower bounds), and listed separately.
       The rollout keeps running after the rolling-window metric stops
       (rolling_window_steps is frozen at its stop value) until VPT(0.4)
       has crossed or the horizon ends. The full e(t) series of every run
       is saved to
       results/coherence/error_series/gate_retrain/<condition>_seed<seed>.npy.
       VPT is reported in steps and in Lyapunov times via
       steps_to_lyapunov_times() (~55 steps per Lyapunov time at dt=0.02).
       All summary standard deviations use ddof=1 (None when n < 2).
    """
    if seeds is None:
        seeds = [42]

    VPT_NORM_THRESHOLD = 0.4
    DIVERGENCE_WINDOW_SIZE = 20
    DIVERGENCE_FRACTION_THRESHOLD = 0.5

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from data import generate_lorenz63
    import math
    logit_fixed_c = math.log(fixed_c / (1 - fixed_c))

    print(f"\n{'='*62}")
    print(f"  EXPERIMENT 4: GATE RETRAIN — {max_steps:,} steps")
    print(f"  seeds={seeds}  correction_interval={correction_interval}  new_n_cycles={new_n_cycles}")
    print(f"{'='*62}")

    rho = 28.0
    dt  = 0.02
    seq_len, pred_steps = 50, 20

    CONDITIONS = [
        {'name': 'v23_hybrid_gated_old_relogged', 'init_bias': -3.0,        'n_cycles': 1},
        {'name': 'v23_hybrid_gated_v2',           'init_bias': logit_fixed_c, 'n_cycles': new_n_cycles},
    ]

    all_results = []
    per_condition_steps = {c['name']: [] for c in CONDITIONS}

    for seed in seeds:
        print(f"\n  ── seed {seed} ──")
        set_seed(seed)

        traj = generate_lorenz63(rho, max_steps + 300, dt=dt,
                                  warmup=2000, seed=seed)
        traj_t = torch.FloatTensor(traj[300:])
        mu  = traj_t[:500].mean(0)
        std = traj_t[:500].std(0).clamp(min=1e-6)
        traj_norm = (traj_t - mu) / std

        X_list, Y_list = [], []
        for s in range(0, min(5000, len(traj_norm)) - seq_len - pred_steps, 3):
            X_list.append(traj_norm[s:s + seq_len])
            Y_list.append(traj_norm[s + seq_len:s + seq_len + pred_steps])
        X_tr = torch.stack(X_list)
        Y_tr = torch.stack(Y_list)

        # Train v23_hybrid once per seed - both gated conditions start from
        # its exact trained weights (same isolation as
        # run_coherence_test_with_correction()).
        hybrid_model = make_v23('v23_hybrid', 3, 3, hidden=128, pred_steps=pred_steps)
        hybrid_m = wrap(hybrid_model)
        train_model(hybrid_m, X_tr, Y_tr, lr=1e-3, epochs=100,
                    batch_size=64, verbose=False, device=DEVICE)
        base_hybrid = get_base_model(hybrid_m)

        for cond in CONDITIONS:
            cname = cond['name']
            gated_model = make_v23('v23_hybrid_gated', 3, 3, hidden=128, pred_steps=pred_steps,
                                    correction_gate_init_bias=cond['init_bias'])
            gated_model.load_state_dict(base_hybrid.state_dict(), strict=False)
            for pname, p in gated_model.named_parameters():
                p.requires_grad = pname.startswith('correction_gate.')

            m = wrap(gated_model)
            train_model_with_periodic_correction(
                m, traj_norm[:5000], seq_len=seq_len, pred_steps=pred_steps,
                lookahead_steps=lookahead_steps, epochs=gate_train_epochs,
                batch_size=64, verbose=False, device=DEVICE, name=cname,
                n_cycles=cond['n_cycles'], correction_interval=correction_interval,
                detach_between_cycles=True,
            )

            base = get_base_model(m)
            m.eval()

            context = traj_norm[:seq_len].unsqueeze(0).to(DEVICE)
            coherent_steps = 0
            recent_windows = deque(maxlen=DIVERGENCE_WINDOW_SIZE)
            step = 0
            stopped_reason = None
            rolling_stopped = False
            vpt = _FirstCrossingVPT(traj_norm, DEVICE)
            step_start_time = time.time()
            gate_log = []  # every gate value at every correction step, this seed

            with torch.no_grad():
                while step < max_steps - pred_steps:
                    pred, _, _, h_traj = base.forward_with_states(context)
                    pred   = pred[0]
                    h_traj = h_traj[:, 0, :]

                    gt_start = seq_len + step
                    gt_end   = gt_start + pred_steps
                    if gt_end >= len(traj_norm):
                        break
                    gt = traj_norm[gt_start:gt_end].to(DEVICE)

                    vpt.update(pred, gt)

                    if not rolling_stopped:
                        window_std = gt.std().item()
                        if window_std < 1e-8:
                            window_std = 1.0
                        normalized_error = torch.sqrt(((pred - gt) ** 2).mean()).item() / window_std

                        recent_windows.append(normalized_error > VPT_NORM_THRESHOLD)
                        if (len(recent_windows) == DIVERGENCE_WINDOW_SIZE and
                                sum(recent_windows) / DIVERGENCE_WINDOW_SIZE > DIVERGENCE_FRACTION_THRESHOLD):
                            stopped_reason = "diverged"
                            rolling_stopped = True

                    # Stop once the rolling metric has stopped AND every VPT
                    # threshold has crossed; otherwise keep rolling out
                    # (rolling_window_steps stays frozen).
                    if rolling_stopped and vpt.all_crossed():
                        break

                    if not rolling_stopped:
                        coherent_steps += pred_steps
                    step           += pred_steps

                    next_window = pred.clone()
                    is_correction_step = (step % correction_interval == 0)
                    if is_correction_step:
                        last_pred   = pred[-1]
                        last_real   = gt[-1]
                        discrepancy = last_real - last_pred
                        h_last = h_traj[-1].unsqueeze(0)
                        disc_b = discrepancy.unsqueeze(0)
                        gate_val = base.correction_gate(h_last, disc_b).item()
                        if not rolling_stopped:
                            gate_log.append(gate_val)
                        next_window[-1] = last_pred + gate_val * discrepancy

                    context = torch.cat([
                        context[:, pred_steps:, :],
                        next_window.unsqueeze(0).clamp(-10, 10).detach()
                    ], dim=1)

                    if step % 1000 == 0:
                        elapsed = time.time() - step_start_time
                        rate = step / max(elapsed, 0.1)
                        print(f"      [{cname}] step {step:>7,}/{max_steps:,} — "
                              f"{elapsed:>6.0f}s elapsed — {rate:>6.0f} steps/sec")

                if stopped_reason is None:
                    stopped_reason = "max_steps_reached"

            needs_review = (stopped_reason == "max_steps_reached")
            lyap_times = steps_to_lyapunov_times(coherent_steps, dt)
            gate_arr = np.array(gate_log) if gate_log else np.array([float('nan')])
            gate_mean = float(np.nanmean(gate_arr))
            gate_min  = float(np.nanmin(gate_arr))
            gate_max  = float(np.nanmax(gate_arr))

            mark = "🏆" if coherent_steps >= 19940 else "✅" if coherent_steps > 5000 else "⚠️"
            if needs_review:
                mark = "⚠️🔍"
            print(f"    {mark} {cname:<30}  {coherent_steps:>8,} steps  ({lyap_times:.2f} TL)  "
                  f"gate[mean/min/max]={gate_mean:.3f}/{gate_min:.3f}/{gate_max:.3f}  ({stopped_reason})")

            vpt_res = vpt.results(dt)
            vpt_status = ("vpt_censored" if any(vpt_res[f'vpt_{th}_censored']
                                                for th in VPT_FIRST_CROSSING_THRESHOLDS)
                          else "vpt_crossed")
            series_path = vpt.save_series(
                f'./results/coherence/error_series/gate_retrain/{cname}_seed{seed}.npy')
            vpt_str = "  ".join(
                f"VPT({th})={vpt_res[f'vpt_{th}_steps']:,} steps "
                f"({vpt_res[f'vpt_{th}_lyapunov_times']:.2f} TL)"
                + (" [CENSORED]" if vpt_res[f'vpt_{th}_censored'] else "")
                for th in VPT_FIRST_CROSSING_THRESHOLDS)
            print(f"        {vpt_str}  ({vpt_status})")

            all_results.append({
                'condition': cname,
                'seed': seed,
                'rolling_window_steps': coherent_steps,
                **vpt_res,
                'error_series_path': series_path,
                'coherent_steps': coherent_steps,
                'coherent_lyapunov_times': lyap_times,
                'stopped_reason': stopped_reason,
                'vpt_status': vpt_status,
                'needs_manual_review': needs_review,
                'n_correction_steps': len(gate_log),
                'gate_values': gate_log,
                'gate_value_mean': gate_mean,
                'gate_value_min': gate_min,
                'gate_value_max': gate_max,
            })
            per_condition_steps[cname].append((coherent_steps, needs_review))

            del m, base, gated_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del hybrid_m, base_hybrid, hybrid_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    normal_steps = {n: [s for s, review in per_condition_steps[n] if not review]
                    for n in per_condition_steps}

    print(f"\n  {'Condition':<32}  {'Mean Steps':>12}  {'Std':>10}  {'Mean TL':>10}")
    print(f"  {'─'*32}  {'─'*12}  {'─'*10}  {'─'*10}")
    summary = {}
    for name in per_condition_steps:
        if normal_steps[name]:
            arr = np.array(normal_steps[name])
            mean, std = arr.mean(), _std_ddof1(arr)
            mean_lt = steps_to_lyapunov_times(mean, dt)
            gate_means = [r['gate_value_mean'] for r in all_results if r['condition'] == name]
            summary[name] = {
                'mean_steps': float(mean), 'std_steps': std,
                'mean_lyapunov_times': float(mean_lt),
                'gate_value_mean_across_seeds': float(np.nanmean(gate_means)) if gate_means else None,
            }
            print(f"  {name:<32}  {mean:>12.1f}  {_fmt_opt(std, '>10.1f')}  {mean_lt:>10.2f}")

    vpt_summary = _summarize_vpt(all_results, 'condition', dt)
    _print_vpt_summary(vpt_summary, 32)

    os.makedirs('./results/coherence', exist_ok=True)
    with open('./results/coherence/coherence_gate_retrain_results.json', 'w') as f:
        json.dump({
            'rho': rho, 'dt': dt,
            'lyapunov_exponent': LORENZ63_LYAPUNOV_EXPONENT,
            'max_steps': max_steps,
            'seeds': seeds,
            'correction_interval': correction_interval,
            'fixed_c_reference': fixed_c,
            'new_n_cycles': new_n_cycles,
            'individual_results': all_results,
            'summary': summary,
            'summary_rolling_window_steps': summary,
            'summary_first_crossing_vpt': vpt_summary,
            'vpt_thresholds': list(VPT_FIRST_CROSSING_THRESHOLDS),
            'note': 'rolling_window_steps has a 380-step reporting floor; '
                    'first-crossing VPT is floor-free (see function docstring). '
                    'Censored VPT runs are counted at their horizon in median_steps and '
                    'lower_bound_mean_steps (lower bounds); mean_steps/std_steps are '
                    'uncensored-only, for reference. Std uses ddof=1.',
        }, f, indent=2)
    print(f"\n  Saved: ./results/coherence/coherence_gate_retrain_results.json")
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
    parser.add_argument('--coherence-correction', action='store_true',
                        help='Run periodic real-data correction coherence test '
                             '(fixed-schedule vs learned-gate vs zero-correction)')
    parser.add_argument('--coherence-head-to-head', action='store_true',
                        help='Experiment 3: five architectures under identical '
                             'fixed-schedule correction (zero vs fixed, same trained weights)')
    parser.add_argument('--coherence-gate-retrain', action='store_true',
                        help='Experiment 4: retrain CorrectionGate with logit(fixed_c) init '
                             'and multi-cycle training, vs last-night settings relogged')
    parser.add_argument('--max-steps', type=int, default=1_000_000)
    parser.add_argument('--correction-interval', type=int, default=40,
                        help='Steps between real-data correction events (Fan et al. 2020)')
    parser.add_argument('--fixed-c', type=float, default=0.8,
                        help='Fixed correction blend constant (Fan et al. 2020 Lorenz setup)')
    parser.add_argument('--lookahead-steps', type=int, default=20,
                        help='BPTT lookahead window for training the learned CorrectionGate')
    parser.add_argument('--gate-n-cycles', type=int, default=5,
                        help='Experiment 4: number of full correction cycles to train '
                             'the new-settings gate across')
    args = parser.parse_args()

    if args.quick:
        args.seeds  = [42]
        args.epochs = 30
        print("Quick mode: 1 seed, 30 epochs")

    print(f"Device: {DEVICE}  GPUs: {N_GPU}")

    if args.coherence:
        run_coherence_test(args.variants, args.max_steps, seeds=args.seeds)
    elif args.coherence_correction:
        run_coherence_test_with_correction(
            args.max_steps, seeds=args.seeds,
            correction_interval=args.correction_interval,
            fixed_c=args.fixed_c, lookahead_steps=args.lookahead_steps,
        )
    elif args.coherence_head_to_head:
        run_coherence_test_head_to_head(
            args.max_steps, seeds=args.seeds,
            correction_interval=args.correction_interval, fixed_c=args.fixed_c,
        )
    elif args.coherence_gate_retrain:
        run_coherence_test_gate_retrain(
            args.max_steps, seeds=args.seeds,
            correction_interval=args.correction_interval, fixed_c=args.fixed_c,
            lookahead_steps=args.lookahead_steps, new_n_cycles=args.gate_n_cycles,
        )
    else:
        for exp in args.experiment:
            run_experiment(exp, args.variants, args.seeds, args.epochs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
