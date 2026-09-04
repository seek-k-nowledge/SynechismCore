# VPT (Valid Prediction Time) Diagnostic Report

## Issue Summary
VPT metric is returning 0.00 Lyapunov times across all model variants and all experiments. This indicates a critical bug in how VPT is computed, not a model training issue.

## Root Cause Analysis

### Bug #1: Incorrect Normalization for Spatial Systems (CRITICAL)

**Location:** `src/chaotic_metrics.py`, line 50

```python
truth_std = ground_truth[:T].std(axis=0).mean()
errors = np.sqrt(((predictions[:T] - ground_truth[:T]) ** 2).mean(axis=-1)) / truth_std
```

**Problem:**
- `.std(axis=0)` computes standard deviation ALONG the time axis (axis 0), giving per-location temporal variability
- For spatial PDEs like KS-PDE, this gives very small values (~0.1-0.5), not the overall signal scale
- Normalizing errors by this tiny value makes even small prediction errors > threshold (0.4)
- Results in `exceeded` array catching timestep 0, making `t_star = 0 * dt = 0`

**Example for KS-PDE:**
```
Ground truth shape: (16 timesteps, 64 spatial points)
ground_truth[:T].std(axis=0) = [0.15, 0.12, ..., 0.18]  (temporal std at each location)
truth_std = mean of above ≈ 0.15  (very small!)

Prediction error: ~0.10 (reasonable for first step)
Normalized error: 0.10 / 0.15 = 0.667
0.667 > 0.4 threshold → exceeds at timestep 0 → t_star = 0 → VPT = 0
```

**For Lorenz (works by accident):**
```
Ground truth shape: (20 timesteps, 3 dimensions)  
ground_truth[:T].std(axis=0) = [~2.0, ~2.1, ~1.9]  (temporal range of each dimension)
truth_std = mean ≈ 2.0  (matches typical Lorenz scale)

Prediction error: ~0.05-0.10
Normalized error: 0.05 / 2.0 = 0.025
0.025 < 0.4 threshold → no exceed until much later → VPT > 0
```

**Why this happened:** The metric was calibrated on temporal systems (Lorenz, Finance) where std(axis=0) captures scale. Breaks on spatial systems.

---

### Bug #2: Wrong dt Values for Weather and Robotics

**Location:** `run_v23_benchmark.py`, line 140

```python
dt = 0.02 if exp_name == 'lorenz' else 0.25
```

**Problem:** dt is hardcoded to 0.25 for non-Lorenz experiments, but actual generators use:
- Lorenz: dt=0.02 ✓
- KS-PDE: dt=0.25 ✓
- **Weather (Lorenz96): dt=0.05 ✗ (using 0.25 is 5× wrong)**
- **Robotics: dt=0.02 ✗ (using 0.25 is 12.5× wrong)**
- Finance: dt=N/A (time-stepped returns)

**Impact:**
Even if VPT calculated correctly, weather and robotics would report:
- VPT_reported = VPT_actual * (dt_wrong / dt_correct)
- Weather: off by factor of 5×
- Robotics: off by factor of 12.5×

---

### Bug #3: Incorrect System Key Mapping

**Location:** `run_v23_benchmark.py`, line 138-139

```python
sys_key = {'lorenz': 'lorenz63_rho28', 'ks_pde': 'ks_pde',
           'weather': 'lorenz96'}.get(exp_name, 'lorenz63_rho28')
```

**Problem:** Finance and Robotics default to `'lorenz63_rho28'` with λ=0.9056, which is inappropriate:
- **Finance:** Uses regime-switching, not a single chaotic system. Using Lorenz λ is nonsensical.
- **Robotics (damped oscillator):** Has different dynamics than Lorenz. Should have distinct λ or be computed differently.

**Impact:** Wrong Lyapunov exponents being applied to non-matching systems.

---

## Summary of Bugs

| Bug | Severity | Location | Impact |
|-----|----------|----------|--------|
| Normalization method (spatial std vs global std) | **CRITICAL** | chaotic_metrics.py:50 | VPT→0 for all spatial PDEs and multi-dimensional systems |
| Wrong dt for Weather | High | run_v23_benchmark.py:140 | Weather VPT off by 5× |
| Wrong dt for Robotics | High | run_v23_benchmark.py:140 | Robotics VPT off by 12.5× |
| Wrong sys_key for Finance/Robotics | Medium | run_v23_benchmark.py:138 | Wrong Lyapunov exponents used |

---

## Why VPT Returns 0.00 (Not Some Small Value)

The formatting in line 148 uses `.2f` (2 decimal places), so 0.00 could actually be values < 0.005. But the real issue is:

1. Error normalization by spatial std makes all errors large
2. First timestep's error exceeds threshold (0.4)
3. `exceeded[0] = 0` (first index is 0)
4. `t_star = 0 * dt = 0`
5. `VPT = 0 * λ = 0` or rounds to 0.00

---

## Recommended Fixes

### Fix #1 (Critical): Use Global Normalization
```python
def compute_vpt(predictions, ground_truth, dt, lyapunov_exponent, threshold=0.4):
    T = min(len(predictions), len(ground_truth))
    # Use global std, not per-location temporal std
    truth_std = ground_truth[:T].std()  # RMS of entire signal
    if truth_std < 1e-8:
        truth_std = 1.0
    errors = np.sqrt(((predictions[:T] - ground_truth[:T]) ** 2).mean(axis=-1)) / truth_std
    exceeded = np.where(errors > threshold)[0]
    t_star = exceeded[0] * dt if len(exceeded) > 0 else T * dt
    return float(t_star * lyapunov_exponent)
```

### Fix #2: Correct dt Values
```python
dt_map = {
    'lorenz': 0.02,
    'ks_pde': 0.25,
    'finance': 0.01,  # synthetic returns, default interval
    'weather': 0.05,  # generate_lorenz96 default
    'robotics': 0.02,  # generate_oscillator default
}
dt = dt_map.get(exp_name, 0.02)
```

### Fix #3: System Key Mapping
```python
sys_key_map = {
    'lorenz': 'lorenz63_rho28',
    'ks_pde': 'ks_pde',
    'weather': 'lorenz96',
    'finance': 'finance',  # Add to LYAPUNOV_EXPONENTS or skip VPT
    'robotics': 'robotics',  # Add to LYAPUNOV_EXPONENTS or use L96
}
```

---

## Paper Impact

This is a **CRITICAL issue for the paper:**

1. **VPT values are all wrong** — The claimed comparison to PhyxMamba SOTA (VPT=5.06 for Lorenz) is comparing broken metric to literature SOTA
2. **Not just a scaling factor** — The root cause (spatial vs global std) means the metric isn't measuring what we think
3. **Affects all experiments** — Even the KS-PDE results (which we thought were good) are compromised
4. **Must fix before H100 run** — Otherwise all new results will be corrupted with the same bug

---

## Data Generator Stability (Second Investigation)

Code review of generate_lorenz96 and generate_oscillator shows they should be numerically stable at extreme regimes (F=22, gamma=0.05) based on:
- Both use scipy.integrate.solve_ivp with conservative tolerances
- Neither have divisions by parameters or exponential blowup
- However, should still add P8-style retry logic as precaution

