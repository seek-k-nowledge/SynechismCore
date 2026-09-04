#!/usr/bin/env python3
"""
Diagnostic script to test VPT computation with various scenarios.
This doesn't require GPU or full training.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from chaotic_metrics import compute_vpt, LYAPUNOV_EXPONENTS

def test_vpt_scenario(name, predictions, ground_truth, dt, lyapunov_exp, threshold=0.4):
    """Test a VPT scenario and print diagnostics"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Shapes: pred={predictions.shape}, truth={ground_truth.shape}")
    print(f"dt={dt}, lambda={lyapunov_exp}, threshold={threshold}")

    # Manually trace through compute_vpt logic
    T = min(len(predictions), len(ground_truth))
    print(f"T (min length) = {T}")

    truth_std = ground_truth[:T].std(axis=0).mean()
    print(f"truth_std (spatial std averaged over time): {truth_std:.6f}")

    if truth_std < 1e-8:
        print(f"  -> truth_std too small, using 1.0")
        truth_std = 1.0

    # Compute per-timestep RMSE
    errors_raw = np.sqrt(((predictions[:T] - ground_truth[:T]) ** 2).mean(axis=-1))
    print(f"Raw per-timestep RMSE: min={errors_raw.min():.6f}, max={errors_raw.max():.6f}, mean={errors_raw.mean():.6f}")

    errors = errors_raw / truth_std
    print(f"Normalized errors (RMSE/truth_std): min={errors.min():.6f}, max={errors.max():.6f}, mean={errors.mean():.6f}")

    exceeded = np.where(errors > threshold)[0]
    print(f"Timesteps exceeding threshold {threshold}: {exceeded} (count={len(exceeded)})")

    t_star = exceeded[0] * dt if len(exceeded) > 0 else T * dt
    print(f"t_star = {t_star:.6f} (first exceed or max time)")

    vpt = float(t_star * lyapunov_exp)
    print(f"VPT = t_star * lambda = {t_star:.6f} * {lyapunov_exp} = {vpt:.4f} Lyapunov times")

    # Compare with actual function
    vpt_actual = compute_vpt(predictions, ground_truth, dt, lyapunov_exp, threshold)
    print(f"Actual compute_vpt result: {vpt_actual:.4f}")

    if abs(vpt - vpt_actual) > 1e-5:
        print(f"  ⚠️  MISMATCH between manual and function!")

    return vpt_actual

# Test 1: Lorenz-like data with good predictions
print("\nTest 1: Lorenz with good predictions (small error)")
np.random.seed(42)
pred_steps, dim = 20, 3
ground_truth = np.random.randn(pred_steps, dim) * 10  # Lorenz scale
predictions = ground_truth + np.random.randn(pred_steps, dim) * 0.1  # Small noise
vpt1 = test_vpt_scenario(
    "Lorenz - small error",
    predictions, ground_truth,
    dt=0.02,
    lyapunov_exp=LYAPUNOV_EXPONENTS['lorenz63_rho28'],
    threshold=0.4
)

# Test 2: Lorenz with large error (exceeds threshold)
print("\n\nTest 2: Lorenz with large error (should exceed threshold)")
predictions2 = ground_truth + np.random.randn(pred_steps, dim) * 5.0  # Large noise
vpt2 = test_vpt_scenario(
    "Lorenz - large error",
    predictions2, ground_truth,
    dt=0.02,
    lyapunov_exp=LYAPUNOV_EXPONENTS['lorenz63_rho28'],
    threshold=0.4
)

# Test 3: KS-PDE with wrong dt (0.25 vs correct 0.25, but spatial std issue)
print("\n\nTest 3: KS-PDE with spatial variability")
pred_steps, dim = 16, 64
ground_truth3 = np.random.randn(pred_steps, dim) * 3  # KS-PDE spatial scale
predictions3 = ground_truth3 + np.random.randn(pred_steps, dim) * 0.1
vpt3 = test_vpt_scenario(
    "KS-PDE - small error",
    predictions3, ground_truth3,
    dt=0.25,
    lyapunov_exp=LYAPUNOV_EXPONENTS['ks_pde'],
    threshold=0.4
)

# Test 4: Weather (Lorenz96) - using WRONG dt from run_v23_benchmark
print("\n\nTest 4: Weather/Lorenz96 with WRONG dt (0.25 instead of 0.05)")
pred_steps, dim = 10, 40
ground_truth4 = np.random.randn(pred_steps, dim) * 5  # L96 scale
predictions4 = ground_truth4 + np.random.randn(pred_steps, dim) * 0.1
vpt4_wrong = test_vpt_scenario(
    "Weather - WRONG dt=0.25",
    predictions4, ground_truth4,
    dt=0.25,  # WRONG
    lyapunov_exp=LYAPUNOV_EXPONENTS['lorenz96'],
    threshold=0.4
)

print("\n\nTest 5: Weather/Lorenz96 with CORRECT dt (0.05)")
vpt4_correct = test_vpt_scenario(
    "Weather - CORRECT dt=0.05",
    predictions4, ground_truth4,
    dt=0.05,  # CORRECT
    lyapunov_exp=LYAPUNOV_EXPONENTS['lorenz96'],
    threshold=0.4
)

# Test 6: Edge case - zero ground truth (would cause std=0)
print("\n\nTest 6: Edge case - constant ground truth")
predictions5 = np.ones((10, 3)) * 1.0
ground_truth5 = np.ones((10, 3)) * 1.0  # Constant
vpt5 = test_vpt_scenario(
    "Constant truth (std=0)",
    predictions5, ground_truth5,
    dt=0.02,
    lyapunov_exp=LYAPUNOV_EXPONENTS['lorenz63_rho28'],
    threshold=0.4
)

# Test 7: NaN check - what if predictions contain NaN?
print("\n\nTest 7: Predictions with NaN values")
predictions6 = ground_truth.copy()
predictions6[5, 1] = np.nan
try:
    vpt6 = test_vpt_scenario(
        "Predictions with NaN",
        predictions6, ground_truth,
        dt=0.02,
        lyapunov_exp=LYAPUNOV_EXPONENTS['lorenz63_rho28'],
        threshold=0.4
    )
except Exception as e:
    print(f"Error: {e}")
    print(f"This might be the issue if NaN/Inf are in predictions!")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Lorenz small error VPT: {vpt1:.4f} Lyapunov times")
print(f"Lorenz large error VPT: {vpt2:.4f} Lyapunov times")
print(f"KS-PDE small error VPT: {vpt3:.4f} Lyapunov times")
print(f"Weather WRONG dt VPT: {vpt4_wrong:.4f} Lyapunov times")
print(f"Weather CORRECT dt VPT: {vpt4_correct:.4f} Lyapunov times")
print(f"Constant truth VPT: {vpt5:.4f} Lyapunov times")
print("\nIf any VPT is 0.00, that indicates a bug in compute_vpt or how it's called.")
