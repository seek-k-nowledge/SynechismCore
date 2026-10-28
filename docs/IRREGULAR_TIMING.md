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

## Status

Code complete and tested on CPU. The full 5-seed GPU run has not been done
yet; no results are claimed here.
