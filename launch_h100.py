#!/usr/bin/env python3
"""
SynechismCore v23.0.1 — GPU Launch Script
==========================================
PATCH P5 applied: torch.set_float32_matmul_precision('high')
  Enables TF32 on H100/A100 Tensor Cores.
  2-3x free speedup on Hopper architecture.
  Zero measurable accuracy impact for ODE training.
  Must be set before any torch operations.

Usage:
    python launch_h100.py                         # all phases, 5 seeds
    python launch_h100.py --seeds 0 1 2 3 4 5 6 7 8 9   # 10 seeds
    python launch_h100.py --quick                 # 1 seed, 30 epochs
    python launch_h100.py --phase phi             # phi ablation only
    python launch_h100.py --phase v23             # v23 components only
    python launch_h100.py --phase coherence       # 25k step rollout only
    python launch_h100.py --phase ks_pde          # KS-PDE only
"""

import torch
# PATCH P5: Enable TF32 — must be first torch operation after import
# H100/A100 Tensor Cores use TF32 (10-bit mantissa vs 23-bit FP32) for matmuls.
# For ODE training this has zero measurable impact on accuracy and gives
# 2-3x free speedup on Hopper architecture.
torch.set_float32_matmul_precision('high')

import os, sys, subprocess, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def detect_gpu():
    if not torch.cuda.is_available():
        print("No GPU detected — running on CPU")
        return False
    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*56}")
    print(f"  GPU:      {name}")
    print(f"  Compute:  {cap[0]}.{cap[1]}")
    print(f"  Memory:   {mem:.1f} GB")
    print(f"  TF32:     enabled (P5 patch)")
    print(f"  Batch:    {recommended_batch_size()} (auto-scaled to memory)")
    print(f"{'='*56}\n")
    return True


def recommended_batch_size():
    if not torch.cuda.is_available():
        return 64
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem >= 80: return 512
    if mem >= 40: return 256
    if mem >= 16: return 128
    return 64


def run_phase(cmd, name):
    print(f"\n{'─'*56}")
    print(f"  Starting: {name}")
    print(f"{'─'*56}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "✅ DONE" if result.returncode == 0 else "❌ FAILED"
    print(f"\n  {status}: {name} ({elapsed/60:.1f} min)")
    return result.returncode == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',
                        choices=['phi', 'v23', 'coherence', 'ks_pde', 'all'],
                        default='all')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 0, 1, 7, 100])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--quick', action='store_true',
                        help='1 seed, 30 epochs (sanity check)')
    args = parser.parse_args()

    detect_gpu()
    os.makedirs('./results/fresh_run', exist_ok=True)

    if args.quick:
        args.seeds  = [42]
        args.epochs = 30
        print("Quick mode: 1 seed, 30 epochs (~5 min on H100)")

    seeds_str = [str(s) for s in args.seeds]
    t_start   = time.time()
    ok        = []

    phases = [args.phase] if args.phase != 'all' \
             else ['phi', 'v23', 'coherence', 'ks_pde']

    for phase in phases:
        if phase == 'phi':
            ok.append(run_phase(
                [sys.executable, 'run_phi_ablation.py',
                 '--system', 'lorenz', '--seeds'] + seeds_str +
                ['--epochs', str(args.epochs)],
                'φ vs √2 vs e ablation (Claim 4)'
            ))

        elif phase == 'v23':
            ok.append(run_phase(
                [sys.executable, 'run_v23_benchmark.py',
                 '--experiment', 'lorenz', 'robotics',
                 '--variants', 'v22_baseline', 'elastic_only', 'shutter_only',
                               'v23_full', 'v23_hybrid',
                               'transformer', 'lstm', 'mamba',
                 '--seeds'] + seeds_str + ['--epochs', str(args.epochs)],
                'v23 components: Lorenz + Robotics'
            ))

        elif phase == 'coherence':
            ok.append(run_phase(
                [sys.executable, 'run_v23_benchmark.py',
                 '--coherence', '--max-steps', '25000',
                 '--variants', 'v23_full', 'v22_baseline', 'lstm', 'transformer',
                 '--seeds', str(args.seeds[0])],
                '25,000-step coherence rollout'
            ))

        elif phase == 'ks_pde':
            ok.append(run_phase(
                [sys.executable, 'run_v23_benchmark.py',
                 '--experiment', 'ks_pde',
                 '--variants', 'v22_baseline', 'v23_full',
                               'transformer', 'lstm', 'mamba',
                 '--seeds'] + seeds_str + ['--epochs', str(args.epochs)],
                'KS-PDE headline confirmation'
            ))

    total = (time.time() - t_start) / 60
    passed = sum(ok)
    print(f"\n{'='*56}")
    print(f"  ALL PHASES DONE: {passed}/{len(ok)} passed  ({total:.1f} min)")
    print(f"  Results in: ./results/fresh_run/")
    print(f"{'='*56}")
