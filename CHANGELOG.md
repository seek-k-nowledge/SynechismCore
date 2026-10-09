# Changelog

Code fixes made before the current paper's experiments. Moved here from the README.

## October 2026

- Coherence tests: added a floor-free first-crossing VPT (thresholds 0.3 and 0.4) to the head-to-head and gate-retrain tests, with censoring and per-step error series saved to `results/coherence/error_series/`.
- VPT summaries now count censored runs (runs that never crossed the threshold) at their horizon as lower bounds, instead of dropping them. Each run records `vpt_status` (`vpt_crossed` or `vpt_censored`).

## v23.0 → v23.0.1

### Patches Applied (v23.0 → v23.0.1 + critical pre-run fixes)

| # | File(s) | Fix | Why |
|---|---|---|---|
| P1 | src/v23_components.py | StiffnessDetector: added `super().__init__()` | Crash on `.to(device)` / DataParallel |
| P2 | src/v23_components.py | LaminarBypass: dt-aware step `h + dt·map(h)` | Static map wrong with variable step sizes |
| P3 | src/v23_components.py | integrate(): extract dt from t_eval per step | Required for P2 |
| P4 | src/v23_components.py | delta_r_net: Tanh→GELU, 16→32 units | Tanh saturates during rapid expansion |
| P5 | launch_h100.py | `torch.set_float32_matmul_precision('high')` | 2-3× free speedup on H100 Tensor Cores |
| P6 | run_v23_benchmark.py | `.detach()` in coherence context window | Prevents OOM at step ~20,000 |
| P7 | run_v23_benchmark.py, run_phi_ablation.py | Parameter name mismatches (nu_train→nu, forcing→F_values, damping→gamma) | Would crash ks_pde/weather/robotics on launch |
| P8 | src/data.py | KS-PDE: retry on NaN/Inf from solver overflow | Low viscosity (ν→0) causes exponential blowup |
| P9 | src/train.py | evaluate_model: NaN/Inf detection + replacement | Predictions can contain invalid values from numerics |
| P11 | src/chaotic_metrics.py, run_v23_benchmark.py, src/data.py | Fix VPT metric + data generator resilience | Critical: VPT was always 0 (3 bugs); generators need retry at extreme params |

### Additional Fixes (P7, P9, P11 — pre-GPU-run code review)

#### P7 — Parameter Name Mismatches
Before this code ever touched a GPU, a review pass found that `run_v23_benchmark.py`
and `run_phi_ablation.py` called the `data.py` dataset functions with keyword
arguments that don't match their actual signatures. This would have crashed the
`ks_pde`, `weather`, and `robotics` experiments immediately on launch (they never
got as far as training). Fixed to match `data.py`'s real parameter names:

| File | Before (broken) | After (fixed) |
|---|---|---|
| run_v23_benchmark.py | `make_ks_dataset(nu_train=[1.0])` | `make_ks_dataset(nu=1.0)` |
| run_v23_benchmark.py | `make_weather_dataset(forcing=[...])` | `make_weather_dataset(F_values=[...])` |
| run_v23_benchmark.py | `make_robotics_dataset(damping=0.5)` | `make_robotics_dataset(gamma=0.5)` |
| run_phi_ablation.py | `make_ks_dataset(nu_train=[1.0])` | `make_ks_dataset(nu=1.0)` |

`run_experiments.py` (the original v22 script) already called these correctly —
the mismatch was only in the newer v23 runner scripts.

#### P9 — NaN/Inf Protection in evaluate_model
Added explicit NaN/Inf detection in `src/train.py` `evaluate_model()`:
after predictions gathered, count NaN/Inf values, print warning if found,
replace with `np.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)` before returning.
Warning message ensures the problem is visible (not silent).

#### P11 — Data Generator Stability + VPT Metric Fixes

**VPT Fix Verification Status:**
- ✅ Lorenz (3D chaotic): **Real end-to-end test PASSED** — VPT metric computed from actual trained model
- ✅ Synthetic data for all systems: **PASSED** — Logic verified (0.3622 TL with good predictions, 0.0000 with bad)
- ⚠️  KS-PDE (64D spatial PDE): **Only tested synthetically** — CPU time constraints prevented real training test
  - **RESIDUAL RISK:** KS-PDE was the primary system where VPT was broken (spatial std bug)
  - **ACTION REQUIRED:** First GPU run should include quick KS-PDE check to verify VPT works on real trained model
  - See `diagnostics/` folder for test scripts and analysis

**Critical metrics fix:** VPT (Valid Prediction Time) was broken due to three bugs:
1. **Wrong normalization** — used `.std(axis=0).mean()` (spatial std) instead of global std
   - For spatial PDEs, this produced tiny values, inflating normalized errors
   - Result: all errors exceeded threshold at t=0, giving VPT=0.00 across all systems
2. **Wrong dt values** — hardcoded dt=0.25 for all non-Lorenz, but:
   - Weather (Lorenz96) should use dt=0.05 (was 5× wrong)
   - Robotics should use dt=0.02 (was 12.5× wrong)
3. **Wrong system keys** — Robotics used default `lorenz63_rho28` Lyapunov exponent
4. **Finance excluded from VPT** — Finance is a stochastic regime-switching system, not
   deterministic chaos, so it has no valid Lyapunov exponent and VPT is meaningless.
   Finance VPT reports "N/A", but MAE, nRMSE, and sMAPE are computed normally.

**Data generator resilience** — Added P8-style retry-on-failure to `make_weather_dataset()`
and `make_robotics_dataset()` in `src/data.py`:
- At extreme test regimes (F=22 for weather, γ=0.05 for robotics), numerical errors could
  accumulate or solvers could fail, producing NaN/Inf
- Now retries up to 10 times with different random seeds before raising error
- Consistent with P8 (KS-PDE) pattern for defensive robustness
