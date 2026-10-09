#!/usr/bin/env python3
"""
Test VPT computation logic directly without training.
Verifies that the VPT metric is working correctly with synthetic data.
"""

import sys
import numpy as np

sys.path.insert(0, 'src')
from chaotic_metrics import compute_vpt

print("="*70)
print("VPT Computation Logic Test")
print("="*70)

# Test 1: Perfect predictions (should give max VPT)
print("\nTest 1: Perfect predictions (error = 0 always)")
pred1 = np.random.randn(20, 3) * 10
truth1 = pred1.copy()
vpt1 = compute_vpt(pred1, truth1, dt=0.02, lyapunov_exponent=0.9056, threshold=0.4)
print(f"  Predictions: shape={pred1.shape}, range=[{pred1.min():.2f}, {pred1.max():.2f}]")
print(f"  Ground truth: shape={truth1.shape}, std={truth1.std():.4f}")
print(f"  Error threshold: 0.4")
print(f"  VPT result: {vpt1:.4f} Lyapunov times")
print(f"  Expected: 20 * 0.02 * 0.9056 = {20*0.02*0.9056:.4f} (max time)")

# Test 2: Small errors (should exceed after several steps)
print("\nTest 2: Small prediction errors")
np.random.seed(42)
truth2 = np.random.randn(20, 3) * 3
pred2 = truth2 + np.random.randn(20, 3) * 0.5  # Small noise
truth_std2 = truth2.std()
errors2 = np.sqrt(((pred2 - truth2) ** 2).mean(axis=-1)) / truth_std2
vpt2 = compute_vpt(pred2, truth2, dt=0.02, lyapunov_exponent=0.9056, threshold=0.4)
print(f"  Ground truth std: {truth_std2:.4f}")
print(f"  Error range: [{errors2.min():.4f}, {errors2.max():.4f}]")
print(f"  Timestep exceeding 0.4 threshold: {np.where(errors2 > 0.4)[0]}")
print(f"  VPT result: {vpt2:.4f} Lyapunov times")

# Test 3: Large immediate error (should give VPT ~0)
print("\nTest 3: Large prediction error (immediate threshold exceed)")
truth3 = np.random.randn(20, 3) * 3
pred3 = truth3 + np.random.randn(20, 3) * 10  # Large noise
truth_std3 = truth3.std()
errors3 = np.sqrt(((pred3 - truth3) ** 2).mean(axis=-1)) / truth_std3
vpt3 = compute_vpt(pred3, truth3, dt=0.02, lyapunov_exponent=0.9056, threshold=0.4)
print(f"  Ground truth std: {truth_std3:.4f}")
print(f"  Error range: [{errors3.min():.4f}, {errors3.max():.4f}]")
print(f"  Timestep exceeding 0.4 threshold: {np.where(errors3 > 0.4)[0]}")
print(f"  VPT result: {vpt3:.4f} Lyapunov times")
if vpt3 < 0.001:
    print(f"  -> Error exceeded immediately, VPT ~0 (expected for bad predictions)")

# Test 4: KS-PDE style (64 spatial dimensions)
print("\nTest 4: KS-PDE (64 spatial dimensions)")
np.random.seed(43)
truth4 = np.random.randn(16, 64) * 2
pred4 = truth4 + np.random.randn(16, 64) * 0.1  # Small error
truth_std4 = truth4.std()
errors4 = np.sqrt(((pred4 - truth4) ** 2).mean(axis=-1)) / truth_std4
vpt4 = compute_vpt(pred4, truth4, dt=0.25, lyapunov_exponent=0.08, threshold=0.4)
print(f"  Ground truth std (global): {truth_std4:.4f}")
print(f"  Error range: [{errors4.min():.4f}, {errors4.max():.4f}]")
print(f"  Timestep exceeding 0.4 threshold: {np.where(errors4 > 0.4)[0]}")
print(f"  VPT result: {vpt4:.4f} Lyapunov times")
print(f"  Expected: ~{16*0.25*0.08:.4f} if no error exceeds threshold (max time)")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"Test 1 (perfect): {vpt1:.4f} TL (should be ~0.3622)")
print(f"Test 2 (small error): {vpt2:.4f} TL (should be > 0)")
print(f"Test 3 (large error): {vpt3:.4f} TL (should be ~0 if immediate exceed)")
print(f"Test 4 (KS-PDE): {vpt4:.4f} TL (should be > 0 with small error)")
print("\nIf all tests show non-zero VPT (except immediate failure), the fix works!")
print("="*70)
