"""
SynechismCore v19.0 — Statistics Module
========================================
THE BUG THAT BROKE v18.5:
  The previous version computed p-values by comparing two SCALAR values
  (mean MAE of model A vs mean MAE of model B), which always gives
  p=0.50 or p=1.00 — statistically meaningless.

THE FIX:
  Compute p-values over DISTRIBUTIONS of per-sample errors.
  Mann-Whitney U test over N error samples (non-parametric, robust).
  This is the correct approach and what the v17.2 paper used.

  v17.2 result: p < 1e-43 because we had thousands of error samples.
  v18.5 result: p = 0.50/1.00 because we compared two numbers.

Author: Paul E. Harris IV — SynechismCore v19.0
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List


def compute_significance(
    ode_errors: np.ndarray,
    baseline_errors: np.ndarray,
    alternative: str = 'less'
) -> Tuple[float, float]:
    """
    Compute Mann-Whitney U test between two error distributions.

    Args:
        ode_errors: per-sample absolute errors for SynechismODE (shape: N,)
        baseline_errors: per-sample absolute errors for baseline (shape: N,)
        alternative: 'less' = test if ODE errors are LOWER (ODE wins)

    Returns:
        (statistic, p_value)

    Note: Mann-Whitney U is non-parametric — no normality assumption.
    This is more robust than t-test for MAE distributions which are skewed.
    """
    assert len(ode_errors) > 1, "Need >1 sample for significance test"
    assert len(baseline_errors) > 1, "Need >1 sample for significance test"

    stat, p = stats.mannwhitneyu(
        ode_errors.flatten(),
        baseline_errors.flatten(),
        alternative=alternative
    )
    return float(stat), float(p)


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p < 1e-100:
        return "p<1e-100"
    elif p < 1e-10:
        exp = int(np.floor(np.log10(p)))
        return f"p<1e{exp}"
    elif p < 0.001:
        return f"p={p:.2e}"
    elif p < 0.05:
        return f"p={p:.4f}*"
    else:
        return f"p={p:.4f} ns"


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def compute_full_stats(
    ode_preds: np.ndarray,
    baseline_preds: np.ndarray,
    true_values: np.ndarray,
    model_name: str = "ODE",
    baseline_name: str = "Transformer"
) -> Dict:
    """
    Compute full statistical comparison between ODE and a baseline.

    Args:
        ode_preds: (N, T, D) ODE predictions
        baseline_preds: (N, T, D) baseline predictions
        true_values: (N, T, D) ground truth

    Returns:
        dict with MAE, ratio, p-value, and significance
    """
    # Per-sample MAE
    ode_errors = np.abs(ode_preds - true_values).mean(axis=(1, 2))        # (N,)
    baseline_errors = np.abs(baseline_preds - true_values).mean(axis=(1, 2))  # (N,)

    ode_mae = ode_errors.mean()
    baseline_mae = baseline_errors.mean()
    ratio = baseline_mae / ode_mae  # >1 means ODE wins

    _, p_val = compute_significance(ode_errors, baseline_errors, alternative='less')

    wins = ode_mae < baseline_mae
    significant = p_val < 0.05

    return {
        "model": model_name,
        "baseline": baseline_name,
        f"{model_name.lower()}_mae": float(ode_mae),
        f"{baseline_name.lower()}_mae": float(baseline_mae),
        "ratio": float(ratio),
        "p_value": float(p_val),
        "p_formatted": format_p_value(p_val),
        "significant": significant,
        "ode_wins": wins,
        "status": "✅ WINS" if (wins and significant) else ("⚠️ MARGINAL" if wins else "❌ LOSES"),
        "n_samples": len(ode_errors),
    }


def aggregate_multi_seed(seed_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds.

    Args:
        seed_results: list of result dicts, one per seed

    Returns:
        dict with mean, std, and consistency across seeds
    """
    ratios = [r["ratio"] for r in seed_results]
    p_vals = [r["p_value"] for r in seed_results]
    wins   = [r["ode_wins"] for r in seed_results]

    return {
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std":  float(np.std(ratios)),
        "ratio_min":  float(np.min(ratios)),
        "ratio_max":  float(np.max(ratios)),
        "p_value_min": float(np.min(p_vals)),
        "p_value_max": float(np.max(p_vals)),
        "wins_all_seeds": all(wins),
        "wins_count": sum(wins),
        "n_seeds": len(seed_results),
        "consistent": all(wins) or not any(wins),  # False if mixed wins/losses
    }


def print_results_table(results: Dict, experiment_name: str):
    """Print a formatted results table."""
    print(f"\n{'='*60}")
    print(f"  {experiment_name}")
    print(f"{'='*60}")
    print(f"  ODE MAE:      {results.get('ode_mae', results.get('synechismode_mae', 0)):.4f}")
    print(f"  Baseline MAE: {results.get('transformer_mae', 0):.4f}")
    print(f"  Ratio:        {results['ratio']:.3f}×")
    print(f"  p-value:      {results['p_formatted']}")
    print(f"  Status:       {results['status']}")
    print(f"  N samples:    {results['n_samples']:,}")
    print(f"{'='*60}")
