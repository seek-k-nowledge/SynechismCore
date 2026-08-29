"""
SynechismCore v23.0.1 — Chaotic Systems Metrics
=================================================
Metrics used by 2025-2026 SOTA papers so results are directly comparable.

VPT    — Valid Prediction Time in Lyapunov times (PhyxMamba standard)
nRMSE  — Normalized RMSE at 1-step and 20-step (PDE-Transformer standard)
sMAPE  — Symmetric MAPE at horizon 10 (PhyxMamba standard)
D_frac — Attractor fractal dimension proxy error
D_stsp — Attractor state-space volume error

SOTA reference values (March 2026):
    KS-PDE:    PDE-Transformer-L nRMSE-1=0.0111, nRMSE-20=0.7357
    Lorenz-63: PhyxMamba VPT=5.06 TL, sMAPE@10=67.29
    Lorenz-96: PhyxMamba VPT=1.66 TL
"""

import numpy as np
from typing import Optional

LYAPUNOV_EXPONENTS = {
    'lorenz63_rho28': 0.9056,
    'lorenz63_rho35': 0.9800,
    'lorenz63_rho40': 1.050,
    'lorenz96':       1.670,
    'ks_pde':         0.080,
}

SOTA_REFERENCE = {
    'ks_pde': {
        'PDE-Transformer-L (Holzschuh, ICML 2025)': {
            'nrmse_1': 0.0111, 'nrmse_20': 0.7357, 'hardware': 'A100'},
        'MNO (Cheng, JCP 2026)': {
            'rmse_reduction': '40-89% vs TF', 'hardware': 'A100/V100'},
    },
    'lorenz63': {
        'PhyxMamba (Liu, arXiv 2025)': {
            'vpt_lyap': 5.06, 'smape_10': 67.29,
            'd_frac': 0.060, 'd_stsp': 1.133},
    },
    'lorenz96': {
        'PhyxMamba (Liu, arXiv 2025)': {'vpt_lyap': 1.66},
    },
}


def compute_vpt(predictions, ground_truth, dt, lyapunov_exponent,
                threshold=0.4):
    T = min(len(predictions), len(ground_truth))
    truth_std = ground_truth[:T].std(axis=0).mean()
    if truth_std < 1e-8:
        truth_std = 1.0
    errors = np.sqrt(((predictions[:T] - ground_truth[:T]) ** 2
                      ).mean(axis=-1)) / truth_std
    exceeded = np.where(errors > threshold)[0]
    t_star = exceeded[0] * dt if len(exceeded) > 0 else T * dt
    return float(t_star * lyapunov_exponent)


def compute_nrmse(predictions, ground_truth, step=None):
    if step is not None:
        preds = predictions[..., step, :]
        truth = ground_truth[..., step, :]
    else:
        preds, truth = predictions, ground_truth
    rmse = np.sqrt(((preds - truth) ** 2).mean())
    norm = truth.std()
    return float(rmse / max(norm, 1e-8))


def compute_smape(predictions, ground_truth, horizon=10):
    T = min(horizon, len(predictions), len(ground_truth))
    denom = np.abs(predictions[:T]) + np.abs(ground_truth[:T]) + 1e-8
    return float(200.0 * np.mean(np.abs(predictions[:T] - ground_truth[:T]) / denom))


def compute_chaotic_metrics(predictions, ground_truth,
                            system='lorenz63_rho28', dt=0.02):
    lam = LYAPUNOV_EXPONENTS.get(system, 0.9056)
    return {
        'mae':       float(np.abs(predictions - ground_truth).mean()),
        'vpt_lyap':  compute_vpt(predictions, ground_truth, dt, lam),
        'nrmse_1':   compute_nrmse(predictions, ground_truth, step=0),
        'nrmse_20':  compute_nrmse(predictions, ground_truth,
                                   step=min(19, len(predictions) - 1)),
        'smape_10':  compute_smape(predictions, ground_truth, horizon=10),
        'system':    system,
        'lambda':    lam,
    }


def print_sota_comparison(your_results: dict, system: str):
    print(f"\n  {'='*58}")
    print(f"  SOTA COMPARISON — {system.upper()}")
    print(f"  {'='*58}")
    print(f"  SynechismCore v23 (this run):")
    print(f"    VPT:       {your_results.get('vpt_lyap', 0):.2f} Lyapunov times")
    print(f"    nRMSE-1:   {your_results.get('nrmse_1', 0):.4f}")
    print(f"    nRMSE-20:  {your_results.get('nrmse_20', 0):.4f}")
    print(f"    sMAPE@10:  {your_results.get('smape_10', 0):.2f}%")
    print(f"    MAE:       {your_results.get('mae', 0):.4f}")
    sota = SOTA_REFERENCE.get(system, {})
    if sota:
        print(f"\n  Literature (March 2026):")
        for paper, m in sota.items():
            vals = {k: v for k, v in m.items() if k != 'hardware'}
            print(f"    {paper}: {vals}")


def compare_shutters(n_steps: int = 50) -> dict:
    """
    Compare star discrepancy D* for phi, sqrt2, e, uniform.
    Lower D* = more uniform = less resonance with attractor frequencies.
    Theoretical prediction (Hurwitz): phi should have lowest D*.
    """
    from v23_components import compare_shutters as _cs
    return _cs(n_steps)
