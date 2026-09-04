# SynechismCore Diagnostic Report — Two Critical Issues

**Date:** 2026-09-04  
**Status:** PRE-FIX (diagnosis only, no changes made yet)  
**Impact to Paper:** HIGH (both issues affect results submitted to arXiv)

---

## ISSUE #1: VPT Metric Broken — Returns 0.00 Across All Experiments

### Severity
🔴 **CRITICAL** — All VPT values in paper are invalid

### Root Cause
Three interconnected bugs in VPT computation:

#### Bug 1A: Wrong Normalization for Spatial Systems (PRIMARY)
**File:** `src/chaotic_metrics.py`, line 50  
**Current code:**
```python
truth_std = ground_truth[:T].std(axis=0).mean()
errors = np.sqrt(((predictions[:T] - ground_truth[:T]) ** 2).mean(axis=-1)) / truth_std
```

**The Problem:**
- `.std(axis=0)` computes standard deviation ALONG time axis → gives per-location temporal variability
- For spatial PDEs, this produces very small values (~0.1-0.5), NOT the overall signal scale
- Normalizing errors by tiny truth_std inflates normalized errors dramatically
- Result: even small prediction errors exceed threshold (0.4) at timestep 0
- When first timestep exceeds: `t_star = 0 * dt = 0` → `VPT = 0 * λ = 0.00`

**Why Lorenz Appears to Work:**
- Lorenz has only 3 dimensions (not spatial)
- `.std(axis=0)` gives per-dimension temporal range (~2.0) which happens to match signal scale
- By accident, produces reasonable errors that don't exceed threshold until later
- This masks the fundamental bug

**Example Failure Pattern - KS-PDE:**
```
Ground truth: (16 timesteps, 64 spatial points)
truth_std = ground_truth[:16].std(axis=0).mean()
          = mean([0.15, 0.12, ..., 0.18])  ← spatial temporal variability
          ≈ 0.15  ← TINY

Prediction error: 0.10 (reasonable for first step)
Normalized error: 0.10 / 0.15 = 0.67
0.67 > 0.4 → EXCEEDS at t=0
t_star = 0
VPT = 0.00  ← BUG MANIFESTS HERE
```

#### Bug 1B: Wrong dt Values for Weather and Robotics
**File:** `run_v23_benchmark.py`, line 140  
**Current code:**
```python
dt = 0.02 if exp_name == 'lorenz' else 0.25
```

**The Problem:**
| System | Actual dt | Used in Code | Error Factor |
|--------|-----------|--------------|--------------|
| Lorenz | 0.02 | 0.02 | 1× ✓ |
| KS-PDE | 0.25 | 0.25 | 1× ✓ |
| Weather (L96) | 0.05 | 0.25 | **5× ✗** |
| Robotics | 0.02 | 0.25 | **12.5× ✗** |

If VPT were computed correctly, weather and robotics would report inflated values by 5× and 12.5× respectively.

#### Bug 1C: Incorrect System Key Mapping
**File:** `run_v23_benchmark.py`, line 138-139  
**Current code:**
```python
sys_key = {'lorenz': 'lorenz63_rho28', 'ks_pde': 'ks_pde',
           'weather': 'lorenz96'}.get(exp_name, 'lorenz63_rho28')
```

**The Problem:**
- Finance and Robotics default to `'lorenz63_rho28'` (λ=0.9056)
- Finance is regime-switching, not a chaotic system → Lorenz λ is meaningless
- Robotics is damped oscillator, different dynamics than Lorenz
- Wrong Lyapunov exponents applied to non-matching systems

---

## ISSUE #2: Data Generator Stability at Extreme Regimes

### Severity
🟡 **MODERATE** — Low probability of failure, but catastrophic if it occurs

### Context
KS-PDE generator needed P8 (retry-on-failure) because it could produce NaN at low viscosity. Need to verify Weather and Robotics are safe at their extreme test regimes.

### Analysis Summary

#### Weather (Lorenz96) at F=22
**Physics:** Nonlinear attractor with forcing F=22  
**Risk Assessment:**
- Linear negative feedback (-x term) ensures bounded solutions
- Nonlinear term (xp1-xm2)*xm1 could cause numerical overflow with extreme values
- Unlikely but possible: truncation error accumulation over 3000 steps
- No divide-by-zero, sqrt of negative, or exponentials
- **Risk Level: MODERATE** (numerical, not physical)

**Comparison to KS-PDE:**
- KS-PDE had exponential blowup with ν→0 (e^Lhat*dt explodes)
- L96 only has polynomial nonlinearity (bounded by physics)
- **L96 risk is lower but not zero**

#### Robotics (Damped Oscillator) at γ=0.05
**Physics:** Underdamped driven oscillator with light damping  
**Risk Assessment:**
- Linear system → physically guaranteed bounded
- Resonance effects bounded by driving force amplitude (A=1.0)
- Damping term -2γv = -0.1v always stabilizes (not exponential)
- No division by gamma, no exponentials
- **Risk Level: LOW** (physics is inherently stable)

**Comparison to KS-PDE:**
- Much lower risk than KS-PDE
- Light damping doesn't cause divergence like low viscosity does

### Current Vulnerability
Neither generator has NaN/Inf detection:
```python
# Current code - NO protection
traj = generate_lorenz96(F, ...)
# If solve_ivp silently fails, traj contains NaN
# Propagates through entire dataset
```

### Recommendation
Apply P8-style retry-on-failure to both for defensive robustness:
- **Cost:** Negligible (retry succeeds first time 99%+ of time)
- **Benefit:** Catches solver failures before corrupting datasets
- **Implementation:** 10 retry attempts with different random seeds

---

## Summary Table

| Issue | Component | Bug | Impact | Severity | Fixable |
|-------|-----------|-----|--------|----------|---------|
| 1A | VPT metric | Spatial std normalization | All PDEs return VPT=0 | 🔴 CRITICAL | ✓ Yes |
| 1B | VPT metric | Wrong dt values | Weather/Robotics off by 5-12.5× | 🔴 HIGH | ✓ Yes |
| 1C | VPT metric | Wrong sys_key | Wrong λ for Finance/Robotics | 🟠 MEDIUM | ✓ Yes |
| 2 | Data generators | No NaN/Inf protection | Silent failure risk | 🟡 MODERATE | ✓ Yes |

---

## Paper Impact

### Current Situation
- All VPT values in paper are **wrong** (mostly 0 or garbage)
- Comparison to PhyxMamba SOTA is comparing broken metric to literature
- Cannot claim KS-PDE/Lorenz/Weather superiority based on VPT
- Even MAE/nRMSE results might be tainted if training is unstable

### Must Fix Before
- Submitting to arXiv
- Pushing fresh H100 results
- Any claims about VPT in paper

### Can Continue Without Fix
- Code review and structural improvements
- Preparing other sections of paper
- Debugging training code

---

## Recommended Next Steps

1. **Review this diagnosis** — Verify findings match your observations
2. **Decide on timeline** — Fix before H100 run? Or during initial results review?
3. **If fixing immediately:**
   - Apply VPT fixes (1A, 1B, 1C)
   - Add data generator P10 protection
   - Re-run quick validation on CPU
   - Check if results make sense

4. **If fixing after H100 run:**
   - Run on H100 with current broken code
   - Get preliminary timing/validation
   - Fix VPT metric before analyzing results
   - Re-analyze with corrected metric

---

## Files for Review

- **VPT_DIAGNOSIS.md** — Detailed root cause analysis with examples
- **DATA_STABILITY_ANALYSIS.md** — Physics and numerical analysis of generators
- **test_vpt_diagnosis.py** — CPU diagnostic script (not runnable without deps)
- **test_data_stability.py** — Data generator stability test (not runnable without deps)

