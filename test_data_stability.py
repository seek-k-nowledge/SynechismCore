#!/usr/bin/env python3
"""
Test whether Weather (Lorenz96) and Robotics (damped oscillator) generators
can produce NaN/Inf values at extreme test-regime parameters.

This is a CPU-only test to check stability before relying on these for GPU runs.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from data import generate_lorenz96, generate_oscillator

def test_generator(name, generator_func, test_params, n_trials=10):
    """Test a generator multiple times for NaN/Inf production"""
    print(f"\n{'='*70}")
    print(f"Testing {name}")
    print(f"{'='*70}")
    print(f"Parameters: {test_params}")
    print(f"Number of trials: {n_trials}")

    nan_trajectories = 0
    inf_trajectories = 0
    successful_trajectories = 0

    for trial in range(n_trials):
        try:
            traj = generator_func(**test_params, seed=42 + trial)

            has_nan = np.isnan(traj).any()
            has_inf = np.isinf(traj).any()

            if has_nan:
                nan_count = np.isnan(traj).sum()
                print(f"  Trial {trial}: ⚠️  {nan_count} NaN values")
                nan_trajectories += 1
            elif has_inf:
                inf_count = np.isinf(traj).sum()
                print(f"  Trial {trial}: ⚠️  {inf_count} Inf values")
                inf_trajectories += 1
            else:
                print(f"  Trial {trial}: ✓ OK (shape={traj.shape}, "
                      f"range=[{traj.min():.2f}, {traj.max():.2f}])")
                successful_trajectories += 1

        except Exception as e:
            print(f"  Trial {trial}: ❌ ERROR: {e}")

    print(f"\nResults for {name}:")
    print(f"  ✓ Successful:  {successful_trajectories}/{n_trials}")
    print(f"  ⚠️  NaN:        {nan_trajectories}/{n_trials}")
    print(f"  ⚠️  Inf:        {inf_trajectories}/{n_trials}")

    is_stable = (nan_trajectories == 0 and inf_trajectories == 0)
    if is_stable:
        print(f"  Status: ✅ STABLE")
    else:
        print(f"  Status: ❌ UNSTABLE - needs P8-style retry logic")

    return is_stable


# Test 1: Lorenz96 (Weather) at extreme forcing (F=22, which is test regime)
print("TEST 1: WEATHER (Lorenz96) STABILITY")
print("="*70)
print("Context: In the paper, test uses F=22 (high chaos)")
print("In data.py, make_weather_dataset has generate_lorenz96(..., n_steps=3000)")

weather_stable = test_generator(
    "generate_lorenz96(F=22) - HIGH CHAOS REGIME",
    generate_lorenz96,
    {"F": 22, "N": 40, "n_steps": 3000, "seed": 42},
    n_trials=10
)

# Also test lower F values for comparison
print("\n--- For comparison: lower forcing values ---")
weather_stable_low = test_generator(
    "generate_lorenz96(F=6) - training regime",
    generate_lorenz96,
    {"F": 6, "N": 40, "n_steps": 3000, "seed": 42},
    n_trials=3
)

# Test 2: Damped Oscillator (Robotics) at extreme damping (gamma=0.05, test regime)
print("\n\nTEST 2: ROBOTICS (Damped Oscillator) STABILITY")
print("="*70)
print("Context: In the paper, test uses gamma=0.05 (near-failure)")
print("In data.py, make_robotics_dataset has generate_oscillator(..., n_steps=6000)")

robotics_stable = test_generator(
    "generate_oscillator(gamma=0.05) - NEAR-FAILURE REGIME",
    generate_oscillator,
    {"gamma": 0.05, "n_steps": 6000, "seed": 42},
    n_trials=10
)

# Also test normal operation for comparison
print("\n--- For comparison: normal damping ---")
robotics_stable_normal = test_generator(
    "generate_oscillator(gamma=0.5) - normal operation",
    generate_oscillator,
    {"gamma": 0.5, "n_steps": 6000, "seed": 42},
    n_trials=3
)

# Test 3: Edge case - even more extreme
print("\n\nTEST 3: EXTREME EDGE CASES")
print("="*70)

print("\nWeather with F=30 (extremely high):")
test_generator(
    "generate_lorenz96(F=30) - EXTREME",
    generate_lorenz96,
    {"F": 30, "N": 40, "n_steps": 3000, "seed": 42},
    n_trials=3
)

print("\nRobotics with gamma=0.01 (even more extreme):")
test_generator(
    "generate_oscillator(gamma=0.01) - EXTREME",
    generate_oscillator,
    {"gamma": 0.01, "n_steps": 6000, "seed": 42},
    n_trials=3
)

# Summary
print("\n\n" + "="*70)
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)

if weather_stable:
    print("✓ Weather (F=22) is STABLE")
else:
    print("❌ Weather (F=22) is UNSTABLE - recommend applying P8-style fix")
    print("   (Retry on NaN/Inf, similar to KS-PDE patch P8)")

if robotics_stable:
    print("✓ Robotics (gamma=0.05) is STABLE")
else:
    print("❌ Robotics (gamma=0.05) is UNSTABLE - recommend applying P8-style fix")
    print("   (Retry on NaN/Inf, similar to KS-PDE patch P8)")

if weather_stable and robotics_stable:
    print("\n✅ Both generators are stable at extreme test regimes")
    print("   No additional patches needed")
else:
    print("\n⚠️  At least one generator is unstable")
    print("   Should apply retry-on-failure logic before H100 run")
