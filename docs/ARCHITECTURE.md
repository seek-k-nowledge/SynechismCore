# Model architecture

This page describes **SynechismV23** (the `v23_hybrid` variant) as used in the irregular-timing study and the October 2026 technical report.

## Overview

![SynechismV23 for irregular timing, and the test](architecture_irregular.png)

**Figure 1.** *(a)* SynechismV23 hybrid adapted for uneven sampling ([`src/irregular_models.py`](../src/irregular_models.py)). The U-Net encoder reads the last 50 observations plus the gap before each (4 channels); the posterior mean is used at test time. The latent ODE is integrated with RK4 from the last observation to each requested future time, with 4 sub-steps per gap whose boundaries sit at sorted golden-ratio Weyl points; the laminar bypass decides per example. HyperAgent and decoder are unchanged. *(b)* Every forecaster receives the past gaps and the future query times, runs forward on its own predictions, and is scored by valid prediction time in Lyapunov times. Full protocol: [`IRREGULAR_TIMING.md`](IRREGULAR_TIMING.md).

What changed from the regular-timing model: only the encoder input (an extra gap channel) and the integration (to the real future times instead of a fixed grid). Every other component below is the same.

---

## (a) SynechismV23 hybrid

| Stage | What it does | Code |
|---|---|---|
| Input | Last 50 observations of the Lorenz-63 state (*x*, *y*, *z*), normalized, plus the time gap before each one (4 channels) | `src/irregular_models.py` |
| Encoder | U-Net-style 1-D convolutional encoder (conv → pool → conv → pool → upsample with skip connections), variational output *z* with mean μ and log-variance | `UFOEncoder` |
| Initial state | *h*₀ = tanh(*W* [*z* ; *g*]), where *g* is a learned goal vector (SAGA) shared by all inputs; then a gated Koopman lifting (lift to ⌊φ·128⌋ dimensions and project back) | `SynechismV23.encode`, `KoopmanLifting` |
| Latent ODE | d*h*/d*t* = *L*(*h*) + *N*(*h*) + *F*(*h*) − α (‖*h*‖² − *R*<sub>eff</sub>(*h*)²) *h* | `ElasticAttractorODE` |
| | *L*: linear map. *N*: LayerNorm + spectral-normalized MLP. *F*: Fourier skip (learned-frequency sin/cos features). The last term pulls *h* toward a sphere of radius *R*<sub>eff</sub>, keeping forecasts bounded. | |
| Elastic radius | *R*<sub>eff</sub>(*h*) = *R*<sub>base</sub> + δ*R*(*h*) · *p*<sub>event</sub>(*h*): the sphere widens when an event detector fires | `ElasticAttractorODE.effective_radius` |
| Integration | **IrrationalShutter**: RK4 from the last observation to each requested future time, with 4 sub-steps per gap at sorted φ-Weyl points. **LaminarBypass**: in chunks where a curvature heuristic is low, a gated linear step *h* + Δ*t*·*M*(*h*) replaces RK4 | `IrrationalShutter`, `LaminarBypass` |
| HyperAgent | Event detector + learned jump correction added to each hidden state | `HyperAgent` |
| Decoder | Two-layer MLP from the 128-dimensional hidden state to the 3 outputs, at each requested future time | `SynechismV23.decoder` |

Hidden size is 128 throughout.

**Ablation variants** (`make_v23`):

| Variant | Elastic radius | Irrational shutter | Laminar bypass | HyperAgent |
|---|---|---|---|---|
| `v22_baseline` | off | off | off | off |
| `v23_full` | on | on | on | off |
| `v23_hybrid` (tested) | on | on | on | on |

With the three v23 components off, `v22_baseline` uses the earlier attractor ODE (fixed radius) integrated with an adaptive solver at φ-lattice evaluation points.

## (b) Baselines and test protocol

Every forecaster receives the same observations, the same past gaps and the same future query times:

- **Transformer** with continuous-time positional encodings.
- **LSTM** (2 layers) conditioned on each inter-observation gap.
- **Mamba-style** selective state-space model.

All models are trained to convergence (200 epochs) on Lorenz-63 observed under four sampling regimes with the same average rate (regular, jitter, half missing, exponential), over 5 seeds. Each runs forward on its own predictions and is scored by valid prediction time (VPT) in Lyapunov times. Full data, model and metric details: [`IRREGULAR_TIMING.md`](IRREGULAR_TIMING.md).

To regenerate Figure 1: `python docs/make_architecture_irregular.py`.
