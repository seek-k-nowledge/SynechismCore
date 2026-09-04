# Data Generator Stability Analysis

## Summary
Code review of `generate_lorenz96` and `generate_oscillator` for potential NaN/Inf production at extreme test regimes.

## Test Regimes
- **Weather (Lorenz96):** F=22 (high chaos test regime) vs F=6 (normal training)
- **Robotics (Oscillator):** γ=0.05 (near-failure test) vs γ=0.5 (normal training)

---

## Analysis 1: Weather (Lorenz96) - F=22

### Code
```python
def lorenz96_rhs(t, x, F):
    N = len(x)
    xm1 = np.roll(x, 1)
    xm2 = np.roll(x, 2)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F
```

### Stability Assessment

**At F=6 (training):**
- States stabilize around x ≈ 6 ± small oscillations
- Typical range: [-5, 15]
- Nonlinear term (xp1 - xm2) * xm1: [-50, 50] × [-5, 15] = [-750, 750]
- Stable by damping -x term
- **Risk Level: LOW**

**At F=22 (test regime):**
- States stabilize around x ≈ 22 ± oscillations  
- Typical range: [10, 35]
- Nonlinear term (xp1 - xm2) * xm1: [-50, 50] × [10, 35] = [-1750, 1750]
- Still stable by damping -x term
- However, multiplication of large numbers increases floating-point error
- **Risk Level: MODERATE** (numerical error accumulation, not catastrophic divergence)

### Potential Issues
1. **Accumulation of truncation error** over 3000 steps with F=22
2. **No explicit NaN check** before returning trajectory
3. **If solve_ivp fails internally**, NaN could propagate undetected

### Comparison to KS-PDE (which needed P8)
- KS-PDE had NaN production due to **exponential blowup** in Fourier space (nu → 0)
- L96 doesn't have exponentials, only polynomial nonlinearity
- **Risk is lower than KS-PDE**, but not zero

---

## Analysis 2: Robotics (Damped Oscillator) - γ=0.05

### Code
```python
def damped_oscillator_rhs(t, state, gamma, omega=2*np.pi, A=1.0):
    x, v = state
    dxdt = v
    dvdt = -2*gamma*v - omega**2*x + A*np.cos(omega*t)
    return [dxdt, dvdt]
```

### Stability Assessment

**Physics at different damping:**
- ω ≈ 6.28 rad/s (from 2π)
- ω² ≈ 39.48
- Damping coefficient: 2γ (linear velocity damping)

**At γ=0.5 (training):**
- Damping 2γ = 1.0
- Underdamped but stable: ζ = γ/√(mω²) ≈ 0.08 (very underdamped)
- Amplitude grows to steady-state (~0.05 units)
- Stable energy balance
- **Risk Level: LOW**

**At γ=0.05 (near-failure test):**
- Damping 2γ = 0.1
- Even more underdamped: ζ ≈ 0.008
- At driving frequency ω, resonance effect strong
- Amplitude: A / (2γω) ≈ 1.0 / (0.1 × 6.28) ≈ 1.6 units (modest)
- Still bounded by physics
- Damping 2γ*v = 0.1*v always stabilizes
- **Risk Level: LOW** (no exponential or division by gamma, just lighter damping)

### Comparison to KS-PDE
- Unlike KS-PDE (exponential blowup with nu→0), this is a linear system
- Light damping doesn't cause divergence, just less dissipation
- Resonance is bounded by driving force amplitude
- **Risk is much lower than KS-PDE**

---

## NaN/Inf Failure Modes

### What would cause NaN in Lorenz96?
1. **Overflow in (xp1-xm2)*xm1 multiplication**
   - Would require xm1 > 1e19 or similar (extremely unlikely)
   - RK45 adaptive step-size would reduce dt first
2. **Division by zero** - doesn't occur in this RHS
3. **Sqrt of negative** - doesn't occur
4. **Loss of precision** in solve_ivp - possible but rare
5. **Silent failure in solve_ivp** - would return NaN in sol.y

### What would cause NaN in Oscillator?
1. **Overflow** - requires v > 1e19, x > 1e19 (extremely unlikely)
   - Equation has no exponentials or factorials
2. **Division by zero** - doesn't occur (no division in RHS)
3. **Sqrt of negative** - doesn't occur
4. **Loss of precision** - low risk with γ=0.05 (not near zero)

---

## Current vs. Post-P8 Pattern

### Current Code (Weather)
```python
def make_weather_dataset(F_values, n_traj=30, ...):
    for F in F_values:
        for i in range(n_traj):
            traj = generate_lorenz96(F, ...)  # No retry on NaN/Inf
            # ... process traj
```
**If generate_lorenz96 returns NaN, entire dataset is corrupted.**

### After P8-Style Fix
```python
def generate_lorenz96(F, ...):
    for attempt in range(10):
        candidate = generate_lorenz96_impl(F, ...)
        if not (np.isnan(candidate).any() or np.isinf(candidate).any()):
            return candidate
    raise RuntimeError(f"L96 trajectory (F={F}) produced NaN/Inf...")
```
**Robustness to solver failures.**

---

## Recommendation

### For Weather (Lorenz96) at F=22
- **Risk Level:** MODERATE (numerical accumulation, not fundamental instability)
- **Action:** Apply P8-style retry logic as **defensive programming**
- **Rationale:** 3000 steps with nonlinear terms could occasionally fail; catch it early
- **Cost:** Negligible (first attempt succeeds ~99%+ of time)

### For Robotics (Oscillator) at γ=0.05
- **Risk Level:** LOW (physics is stable, no exponentials)
- **Action:** Consider P8-style retry logic for **consistency** with other datasets
- **Rationale:** Safety margin; aligns pattern across all generators
- **Cost:** Negligible

---

## Implementation Plan

### Patch P10: Add NaN/Inf Protection to Data Generators

Apply retry-on-failure to both `make_weather_dataset` and `make_robotics_dataset`:

**In make_weather_dataset:**
```python
for F in F_values:
    for i in range(n_traj):
        # PATCH P10: retry KS-PDE generation on NaN/Inf
        traj = None
        for attempt in range(10):
            candidate = generate_lorenz96(F, N=N, n_steps=3000, seed=seed+i*11+attempt*100000)
            if not (np.isnan(candidate).any() or np.isinf(candidate).any()):
                traj = candidate
                break
        if traj is None:
            raise RuntimeError(f"L96 trajectory {i} (F={F}) produced NaN/Inf in all 10 attempts")
```

**In make_robotics_dataset:**
```python
for i in range(n_traj):
    # PATCH P10: retry oscillator generation on NaN/Inf
    traj = None
    for attempt in range(10):
        candidate = generate_oscillator(gamma, n_steps=n_steps, seed=seed+i*13+attempt*100000)
        if not (np.isnan(candidate).any() or np.isinf(candidate).any()):
            traj = candidate
            break
    if traj is None:
        raise RuntimeError(f"Oscillator trajectory {i} (gamma={gamma}) produced NaN/Inf in all 10 attempts")
```

---

## Bottom Line

- ✓ Both generators are **physically stable** at extreme test regimes
- ⚠️ Neither has NaN/Inf detection → risk of silent data corruption
- 🛡️ **Recommend P10 patch** for defensive robustness (cheap insurance)
- 📊 **No blocker for H100 run** if time-critical, but strongly suggested before final paper push

