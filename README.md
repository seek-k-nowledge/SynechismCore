> **Status: work in progress.** The first full run (50 training epochs, 5 seeds) is done. A longer-training run (200 epochs for every model) is in progress and decides the main claim. Numbers below are preliminary.

# SynechismCore

**Can a continuous-time model keep chaotic forecasts accurate when data arrives at uneven times?**
Paul E. Harris IV · Independent Researcher, Mashantucket Pequot Nation
MIT License · Target: arXiv preprint, then a 2027 workshop

---

## Overview

Real-world data rarely arrives on a clock. Robot sensors drop readings, headset frame rates wobble, weather stations lose connection, and event logs come in bursts. A world model that forecasts from that data has to cope with uneven gaps.

Most sequence models step once per observation, as if readings were evenly spaced. The Transformer (the architecture behind ChatGPT), the LSTM and Mamba-style state-space models all work this way, even when they're given the gaps as extra input. **SynechismCore** is a latent neural ODE: it evolves a hidden state continuously in time, so it can be run forward over any gap, short or long.

This project compares SynechismV23 against Transformer, LSTM and Mamba forecasters on the chaotic Lorenz-63 system observed under four sampling regimes. All four regimes have the same average rate but increasingly uneven timing. Every model gets the same data and the same timing information, and we measure how long each model's free-running forecast stays accurate.

Full experiment description: [`docs/IRREGULAR_TIMING.md`](docs/IRREGULAR_TIMING.md). Scope: [`PAPER_SCOPE.md`](PAPER_SCOPE.md).

---

## Architecture and test

![SynechismV23 for irregular timing, and the test](docs/architecture_irregular.png)

**(a) SynechismV23 hybrid.**
- **Encoder:** turns the last 50 observations, and the gap before each one, into a starting hidden state.
- **Latent ODE:** evolves that state to each requested future time. It uses an elastic attractor term that keeps forecasts bounded, golden-ratio (φ) sub-steps inside each gap, and a cheap shortcut through smooth stretches.
- **HyperAgent:** corrects sudden jumps.
- **Decoder:** turns the hidden states into forecasts.

**(b) The test.** Every forecaster is told the past gaps and the future times it must predict. It runs forward on its own predictions, and accuracy is the valid prediction time (VPT) in Lyapunov times.

Component details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Results so far (preliminary: 50 training epochs, 5 seeds)

![Valid prediction time by sampling regime](docs/irregular_results_50ep.png)

The table shows mean valid prediction time in Lyapunov times. Each value is the mean of 5 seeds, and each seed's value is the median of 30 test forecasts. Higher is better.

| Model | Regular | Jitter | Half missing | Exponential |
|---|---|---|---|---|
| **SynechismV23** | **2.97** | **1.68** | **1.62** | **1.50** |
| LSTM | 2.59 | 1.28 | 1.32 | 1.26 |
| Transformer | 2.40 | 1.08 | 0.53 | 0.37 |
| Mamba | 1.69 | 0.96 | 0.47 | 0.32 |

**What this shows:** at an equal budget of 50 training epochs, SynechismV23 forecasts longest in every regime.
- **Transformer and Mamba** lose over 80% of their VPT going from regular to exponential sampling.
- **SynechismV23 and the LSTM** lose about 50%.

**What it does not show yet:**
- **No model had finished learning at 50 epochs.** Training loss was still falling 5–18% over the last 10 epochs.
- **The LSTM catches up with more training.** In a first seed trained for 200 epochs, it overtook the 50-epoch SynechismV23 under uneven sampling.
- **The main claim needs both trained to convergence.** It waits for SynechismV23 at 200 epochs, which is running now.

**Ablations at 50 epochs:**
- **Golden-ratio sub-steps:** no measurable effect on VPT.
- **Other V23 components:** beat the V22 configuration in 4 of 5 seeds in every regime.

Result files: [`results/irregular_50ep/`](results/irregular_50ep/). Each file holds the per-start VPTs and the per-epoch training loss for every model and seed.

---

## Earlier study (paused)

An earlier experiment asked whether a forecaster's architecture still matters once you keep correcting it with real data. Its results were floor-limited: the metric couldn't report anything below 380 steps. A correction test was also found to fail before its first correction. That study is paused; its summary is kept in [`results/periodic_correction_5seed_summary.md`](results/periodic_correction_5seed_summary.md), and its code is still in `run_v23_benchmark.py`. Earlier exploratory results (v17–v22, free Kaggle GPUs, an older flawed metric) are not part of this work.

---

## Repository

```
SynechismCore/
├── run_irregular.py             # Irregular-timing experiment: train, roll out, measure VPT
├── run_irregular_parallel.sh    # Full run on one GPU: one worker per sampling regime
├── run_v23_benchmark.py         # Earlier correction study (paused)
├── src/
│   ├── irregular_data.py        # Lorenz-63 observed under 4 sampling regimes
│   ├── irregular_models.py      # SynechismV23 for irregular timing + time-aware baselines
│   ├── v23_components.py        # SynechismV23 components (elastic ODE, bypass, shutter)
│   ├── models.py                # Encoder, Koopman lifting, baseline building blocks
│   ├── hyperagent.py            # Event detector + jump correction
│   └── ...                      # metrics, training and stats helpers from earlier work
├── docs/
│   ├── IRREGULAR_TIMING.md          # The experiment: data, models, metric, how to run
│   ├── ARCHITECTURE.md              # Model description
│   ├── architecture_irregular.png   # Figure 1 (make_architecture_irregular.py)
│   ├── irregular_results_50ep.png   # Results figure (make_results_figure.py)
│   └── architecture_v23.png         # Figure for the earlier correction study
├── results/
│   ├── irregular_50ep/          # 50-epoch run: results JSON + logs per regime
│   └── periodic_correction_5seed_summary.md
├── PAPER_SCOPE.md
└── requirements.txt
```

---

## Running the experiment (Paperspace A100)

The runs use a rented **Paperspace** machine with **1× A100**. It bills while it's running, so stop it when you're done. Stopping keeps the disk and your files.

```bash
cd ~/SynechismCore && git pull && pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     # should print True

# quick crash test (a few minutes)
python -u run_irregular.py --seeds 0 --epochs 1 --n-train-obs 600 --n-starts 2 --horizon-lt 1 --smoke

# full run: 4 workers side by side, one per regime; powers off when finished
bash run_irregular_parallel.sh --shutdown
```

- **How long:** about 7 hours at 50 epochs on an A100. The latent ODE dominates the time; the baselines take minutes.
- **Watch progress:** `tail results/irregular/log_*.txt`.
- **Output:** results go to `results/irregular/<regime>/`. The `curves/` folders hold the full error curve of every forecast; they are large and aren't kept in the repo.
- **Redraw the figures:**

```bash
python docs/make_results_figure.py results/irregular_50ep docs/irregular_results_50ep "50 epochs"
python docs/make_architecture_irregular.py
```

---

## Reporting rules

Every result gets reported, including:
- results that don't favour SynechismV23;
- censored runs (forecasts still accurate at the 10-Lyapunov-time horizon), reported as lower bounds;
- per-epoch losses, so readers can check every model finished learning.

---

*SynechismCore · Paul E. Harris IV · Independent Researcher · October 2026 · MIT License*
