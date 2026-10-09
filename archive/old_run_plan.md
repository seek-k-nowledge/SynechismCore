# Earlier run plan (v23) — superseded

> This was the README's GPU guide before the paper was narrowed to one question. It used Lambda Labs and an H100 and ran a 10-seed full suite in four phases. It is kept for the record only. For the current runs, see the main [README](../README.md#running-the-experiments-paperspace-a100).

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

### Get the code (on the server)
```bash
cd /home/ubuntu
git clone https://github.com/seek-k-nowledge/SynechismCore.git
cd SynechismCore
ls
```

You should see: `launch_h100.py  run_experiments.py  run_phi_ablation.py
run_v23_benchmark.py  src/  results/  diagnostics/  paper/`

To update an existing copy later: `cd /home/ubuntu/SynechismCore && git pull`

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
scp -r ubuntu@YOUR_IP:/home/ubuntu/SynechismCore/results/ ~/Desktop/synechism_results/
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
