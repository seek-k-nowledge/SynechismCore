# Irregular-timing experiment

**Question.** When observations arrive at uneven times, does a continuous-time
latent ODE (SynechismV23) keep chaotic forecasts accurate longer than
Transformer, LSTM and Mamba forecasters that are given the same timing
information?

## Data

Lorenz-63 (rho = 28) is integrated accurately once (DOP853, rtol 1e-10) and
observed at times drawn from four regimes. Every regime has the same mean
gap (0.02 time units), so they differ only in how uneven the timing is.

| Regime | Gap between observations | Unevenness (CV) |
|---|---|---|
| regular | always 0.02 | 0 (control) |
| jitter | 0.02 x U(0.5, 1.5) | ~0.29 |
| exponential | 0.02 x Exp(1), floored at 0.002 | ~1.0 |
| missing | 0.01 grid, each point kept with p = 0.5 | ~0.7 |

Code: `src/irregular_data.py`.

## Models (`src/irregular_models.py`)

Every model gets the last 50 observations, the gap before each one, and the
gaps to the 20 future times it must predict.

- **synechism**: SynechismV23 hybrid adapted for uneven timing. The encoder
  reads the gaps as an extra input channel. The latent ODE is integrated to
  the actual future times for each example, using RK4 sub-steps whose
  boundaries sit at golden-ratio (Weyl) points. The elastic attractor, the
  laminar bypass (decided per example), HyperAgent, SAGA, Koopman lifting and
  the decoder are unchanged. At evaluation the encoder mean is used, so
  forecasts are deterministic.
- **synechism_nophi**: ablation with equal sub-steps instead of golden-ratio sub-steps.
- **v22_baseline**: ablation with the elastic attractor, golden-ratio
  sub-steps, bypass and HyperAgent switched off.
- **transformer**: pre-norm, 8 heads, 4 layers. Positions are the actual
  observation times (sinusoidal encoding of continuous time).
- **lstm**: 2 layers. Reads (value, gap) pairs and decodes step by step,
  given each next gap.
- **mamba**: selective state-space block. Reads (value, gap) pairs, and its
  head gets the future gaps.

## Measurement (`run_irregular.py`)

From many start points on a separate test trajectory, each model runs forward
on its own predictions. The future observation times are known; the values
are not. **Valid prediction time (VPT)** is the elapsed physical time until the
normalized error first exceeds 0.4 (and 0.3), in Lyapunov times
(1 LT = 1/0.906 ≈ 1.10 time units). A run that never crosses the threshold
within the horizon is censored at the horizon and reported as a lower bound.

Training: Adam, cosine learning rate, batch size 64, gradient clipping 1.0,
50 epochs for every model. Every epoch's training loss is saved, so it can be
checked that each model had finished learning. The latent models add a KL term
(beta = 5e-4, annealed over 30 epochs) and the HyperAgent regularizer.

## Running

```bash
# quick crash test (minutes)
python -u run_irregular.py --seeds 0 --epochs 1 --n-train-obs 600 --n-starts 2 --horizon-lt 1 --smoke
# full run on one GPU: 4 workers side by side, one per timing regime
bash run_irregular_parallel.sh            # add --shutdown to power off when done
```

Outputs go to `results/irregular/<regime>/`: `main_results.json` with per-start
VPTs and per-epoch training loss, and `curves/` with every error curve.

## Status (29 Oct 2026)

**Main result: every model trained 200 epochs, 5 seeds** ([`results/irregular_200ep/`](../results/irregular_200ep/), figure [`irregular_results_200ep.png`](irregular_results_200ep.png)). Mean VPT(0.4) in Lyapunov times:

| Model | Regular | Jitter | Half missing | Exponential |
|---|---|---|---|---|
| SynechismV23 | 3.55 | 2.45 | 2.32 | 2.03 |
| LSTM | 2.08 | 2.24 | 2.10 | 1.78 |
| Transformer | 2.79 | 1.44 | 0.79 | 0.55 |
| Mamba | 2.77 | 1.20 | 0.69 | 0.50 |

SynechismV23 vs LSTM: 5/5 seeds (regular), 4/5 seeds in each uneven regime;
per-forecast wins 139, 88, 86, 86 of 150. Vs Transformer and Mamba: 5/5 seeds
everywhere. No run was censored. Training loss changed by at most 4.4%
(SynechismV23), 3.9% (LSTM), 6.3% (Mamba) and 11.2% (Transformer, regular)
over the last 20 epochs. The 200-epoch run was made with
`run_irregular.py --epochs 200`; baselines and SynechismV23 ran as separate
jobs with identical data and settings, and their result lists are merged per regime.

**Shorter budget: 50 epochs, 5 seeds** ([`results/irregular_50ep/`](../results/irregular_50ep/)), including ablations:

| Model | Regular | Jitter | Half missing | Exponential |
|---|---|---|---|---|
| SynechismV23 | 2.97 | 1.68 | 1.62 | 1.50 |
| synechism_nophi | 2.93 | 1.62 | 1.63 | 1.51 |
| v22_baseline | 2.68 | 1.67 | 1.50 | 1.41 |
| LSTM | 2.59 | 1.28 | 1.32 | 1.26 |
| Transformer | 2.40 | 1.08 | 0.53 | 0.37 |
| Mamba | 1.69 | 0.96 | 0.47 | 0.32 |

At 50 epochs training loss was still falling 5-18% over the last 10 epochs, so
these numbers are not converged. Ablations have not yet been rerun at 200 epochs.
