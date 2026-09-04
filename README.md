# SynechismCore v23.0.1

**Latent Neural ODEs with Aperiodic φ-Scaling for Chaotic Dynamical Systems**
Paul E. Harris IV · Independent Researcher, Mashantucket Pequot Nation
github.com/seek-k-nowledge/SynechismCore · MIT License · Target: arXiv · ML4PS @ NeurIPS 2026

---

## Overview

SynechismCore explores whether encoding φ (the golden ratio) as an aperiodic scaling structure inside a Neural ODE improves its ability to model chaotic dynamical systems — systems like turbulent fluids, atmospheric weather, and robotic control, where small differences compound unpredictably over time. Standard sequence models (LSTMs, Transformers) tend to lose accuracy quickly on these systems as errors accumulate; this work tests whether φ-scaled dynamics hold up longer and more accurately across several real benchmark domains.

Results are mixed by design — the goal is honest benchmarking, not a clean win. SynechismCore currently outperforms baselines on turbulence (KS-PDE) and chaotic bifurcation (Lorenz) modeling, is competitive on finance and weather forecasting, and underperforms on robotics control. Fresh H100 benchmark runs (in progress) will confirm whether these results hold beyond the original Kaggle free-tier testing.

---

## What This Package Is

Complete benchmark suite for SynechismCore v23.0.1 — the patched version
documented in the whitepaper §1. All six patches are applied in the code.
Run this on a rented GPU to produce the results that fill the PENDING
cells in the paper's tables.

**Prior results (Kaggle):** saved in `results/kaggle_confirmed/`
**Fresh run results:** will save to `results/fresh_run/`

---

## Patches Applied (v23.0 → v23.0.1 + critical pre-run fixes)

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

## Additional Fixes (P7, P9, P11 — pre-GPU-run code review)

### P7 — Parameter Name Mismatches
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

### P9 — NaN/Inf Protection in evaluate_model
Added explicit NaN/Inf detection in `src/train.py` `evaluate_model()`:
after predictions gathered, count NaN/Inf values, print warning if found,
replace with `np.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)` before returning.
Warning message ensures the problem is visible (not silent).

### P11 — Data Generator Stability + VPT Metric Fixes

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

---

## Confirmed Results (Kaggle Free Tier — What This Run Must Match)

| Experiment | SynechismCore | Baseline | Ratio | Status |
|---|---|---|---|---|
| KS-PDE (ν: 1.0→0.5) | 0.2952±0.0021 | 0.4207 TF | **1.43×** | ✅ WIN |
| Lorenz bifurcation (v17.2 vs LSTM) | 0.1269–0.8037 | 0.1772–0.8658 | **1.08–1.40×** | ✅ WIN |
| Coherence | 19,940 steps | ~1,260 steps | **15.8×** | ✅ CONFIRMED |
| φ significance | p=0.0000 | uniform | — | ✅ CONFIRMED |
| Robotics | 12.8109 | 6.6470 TF | 0.52× | ❌ TF WINS |
| Finance | 5.6783 | 5.8804 TF | 1.04× | ⚠️ MARGINAL |
| Weather L96 | 1.0550 | 1.0508 TF | 1.00× | ⚠️ TIED |

---

## File Structure

```
synechism_v23_patched/
├── src/
│   ├── models.py            # v22: SynechismV20, FairTransformer, FairLSTM, FairMamba
│   ├── v23_components.py    # v23.0.1 (all 4 code patches applied)
│   ├── chaotic_metrics.py   # VPT, nRMSE, sMAPE for SOTA comparison
│   ├── data.py              # 5 experiment data generators
│   ├── train.py             # Training loop, evaluate_model
│   ├── stats.py             # Correct Mann-Whitney U significance
│   ├── quantum_lattice.py   # PhiLattice, FibonacciLattice, HaltonLattice
│   ├── hyperagent.py        # Event detector + jump correction
│   ├── hyevo.py             # Evolutionary hyperparameter search
│   └── __init__.py
├── run_experiments.py       # Full v22 5-experiment benchmark (unchanged)
├── run_phi_ablation.py      # φ vs √2 vs e ablation
├── run_v23_benchmark.py     # v23 vs v22 + coherence test (P6 applied)
├── launch_h100.py           # Master launcher (P5 applied)
├── requirements.txt
├── COMPETITORS.md           # 2026 SOTA reference values
├── SynechismCore_v23_whitepaper.docx
└── results/
    ├── kaggle_confirmed/    # Prior confirmed results (reference)
    └── fresh_run/           # New results go here
```

---

**Note:** GPU rental provider below (Lambda Labs) reflects the original setup path. Currently migrating to a debit-card-friendly provider (Paperspace) — this section will be updated once that's confirmed working.

## Part 1: Renting a GPU (First Time)

**Provider: Lambda Labs** — https://lambdalabs.com
Recommended for first-time renters: transparent pricing, simple setup,
PyTorch pre-installed.

**Cost:** ~$2.50–3.00/hr for H100 80GB. Full suite: under 3 hours. Total ~$8–10.

**BILLING STARTS WHEN YOU LAUNCH. Always TERMINATE (not Stop) when done.**

### Step 1 — Create account
Go to lambdalabs.com → Sign Up → add payment method → verify email.

### Step 2 — Generate SSH key on your laptop

Open Terminal on your laptop:
- **Mac:** Applications → Utilities → Terminal
- **Windows:** Search for PowerShell

```bash
# Check if you already have a key
ls ~/.ssh/id_ed25519.pub

# If file not found, generate one:
ssh-keygen -t ed25519 -C "your@email.com"
# Press Enter three times (accept all defaults)

# Print your public key — copy ALL of this output:
cat ~/.ssh/id_ed25519.pub
```

In Lambda Labs dashboard: SSH Keys → Add SSH Key → paste the output → Save.

### Step 3 — Launch instance
Dashboard → Instances → Launch Instance
- Select: **1× H100 SXM (80GB)**. If unavailable: 1× A100 (40GB) works.
- Image: **PyTorch 2.x**
- SSH key: select yours
- Click Launch. Wait 1–2 minutes for status to show "Running".
- Note the IP address (looks like: 192.222.51.47)

---

## Part 2: Connecting From Your Laptop

```bash
ssh ubuntu@YOUR_IP_ADDRESS
```

Example:
```bash
ssh ubuntu@192.222.51.47
```

If asked "Are you sure you want to continue connecting?" → type `yes`

Your prompt changes to `ubuntu@gpu-instance:~$` — you are now on the server.

### Verify GPU is working
```bash
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('Memory:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

Expected output:
```
GPU: NVIDIA H100 80GB HBM3
Memory: 79.9 GB
```

---

## Part 3: Upload Code and Install

### Install dependencies (on the server)
```bash
pip install torchdiffeq scipy matplotlib pandas -q
```

### Upload your zip (open a NEW terminal on your laptop)
```bash
scp ~/Downloads/SynechismCore_v23_FRESH_patched.zip ubuntu@YOUR_IP:/home/ubuntu/
```

Replace `~/Downloads/` with wherever you saved the file.

### Extract on the server
```bash
cd /home/ubuntu
unzip SynechismCore_v23_FRESH_patched.zip
cd synechism_v23_patched
ls
```

You should see: `launch_h100.py  run_experiments.py  run_phi_ablation.py
run_v23_benchmark.py  src/  results/`

---

## Part 4: Running Experiments

### Use screen BEFORE starting any long run

`screen` keeps your scripts running if your laptop sleeps or internet drops.

```bash
# Start a persistent session
screen -S synechism

# Your prompt changes slightly — you are now inside screen
# Start your run here
```

To **detach** (leave running, close laptop): press `Ctrl+A` then `D`
To **reattach** later: `ssh` back in, then `screen -r synechism`

### Quick sanity check first (~5 min)
```bash
python run_v23_benchmark.py --quick
```
Runs 1 seed, 30 epochs. If numbers appear in output — everything works.

### Full suite with 10 seeds (~2.5 hr on H100)
```bash
python launch_h100.py --seeds 0 1 2 3 4 5 6 7 8 9
```

### Individual phases
```bash
# Phase 1: φ ablation (~30 min)
python run_phi_ablation.py --system lorenz --seeds 0 1 2 3 4 5 6 7 8 9

# Phase 2: v23 on Lorenz + Robotics (~45 min)
python run_v23_benchmark.py \
    --experiment lorenz robotics \
    --variants v22_baseline elastic_only shutter_only v23_full v23_hybrid \
               transformer lstm mamba \
    --seeds 0 1 2 3 4 5 6 7 8 9

# Phase 3: 25k coherence test (~20 min)
python run_v23_benchmark.py \
    --coherence --max-steps 25000 \
    --variants v23_full v22_baseline lstm transformer \
    --seeds 42

# Phase 4: KS-PDE headline (~45 min)
python run_v23_benchmark.py \
    --experiment ks_pde \
    --variants v22_baseline v23_full transformer lstm mamba \
    --seeds 0 1 2 3 4 5 6 7 8 9
```

---

## Part 5: Download Results and Shut Down

### Download results to your laptop
Open a NEW terminal window on your laptop:
```bash
scp -r ubuntu@YOUR_IP:/home/ubuntu/synechism_v23_patched/results/ ~/Desktop/synechism_results/
```

Verify the JSON files are on your Desktop before the next step.

### TERMINATE the instance
1. Go to Lambda Labs dashboard in your browser
2. Instances → find your instance → click **Terminate**
3. Confirm termination
4. Instance disappears from dashboard — billing stops immediately

**Common mistake:** clicking Stop instead of Terminate.
Stopped instances may still bill. Always Terminate.

---

## Part 6: Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large | Add `--quick` or reduce batch in script |
| `ModuleNotFoundError: torchdiffeq` | Not installed | `pip install torchdiffeq -q` |
| `Connection refused` | Instance still booting | Wait 1–2 min, retry |
| `Permission denied (publickey)` | SSH key mismatch | Check public key in Lambda Labs dashboard |
| `NaN loss after epoch 1` | LR too high | Add `--lr 1e-4` |
| `results/fresh_run/ is empty` | Run crashed | Check: `cat nohup.out \| tail -50` |

---

## What Results Go in the Paper

After the run, `results/fresh_run/` contains JSON files.
These numbers fill the PENDING cells in Table 5.2 of the whitepaper.

Every result — win or loss — gets reported.
The paper already has confirmed negative results (Robotics 0.52×, Weather tied).
If v23 doesn't fix Robotics, that goes in the paper too.

---

*SynechismCore v23.0.1 · Paul E. Harris IV · Mashantucket Pequot Nation · April 2026*
*All prior confirmed results: Kaggle free-tier (P100, T4×2)*
*No H100 results exist yet — this run produces them*
*github.com/seek-k-nowledge/SynechismCore · MIT License*
