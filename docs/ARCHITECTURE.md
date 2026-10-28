# Model architecture

This page describes **SynechismV23** (the `v23_hybrid` variant). The current paper uses it in the **irregular-timing study**; the original figure below comes from the earlier correction study.

## Irregular-timing version (current paper)

![SynechismV23 for irregular timing, and the test](architecture_irregular.png)

**Figure 1.** *(a)* SynechismV23 hybrid adapted for uneven sampling ([`src/irregular_models.py`](../src/irregular_models.py)). The U-Net encoder reads the last 50 observations plus the gap before each (4 channels); the posterior mean is used at test time. The latent ODE is integrated with RK4 from the last observation to each requested future time, with 4 sub-steps per gap whose boundaries sit at sorted golden-ratio Weyl points; the laminar bypass decides per example. HyperAgent and decoder are unchanged. *(b)* Every forecaster receives the past gaps and the future query times, runs forward on its own predictions, and is scored by valid prediction time in Lyapunov times. Full protocol: [`IRREGULAR_TIMING.md`](IRREGULAR_TIMING.md).

What changed from the regular-timing model: only the encoder input (an extra gap channel) and the integration (to the real future times instead of a fixed grid). Every other component below is the same.

## Original version (earlier correction study)

Taken from [`src/v23_components.py`](../src/v23_components.py), [`src/models.py`](../src/models.py) and [`run_v23_benchmark.py`](../run_v23_benchmark.py).

![SynechismV23 architecture and test protocol](architecture_v23.png)

**Earlier figure.** *(a)* SynechismV23 hybrid on regularly sampled windows, forecasting 20 steps on a fixed golden-ratio time grid. *(b)* The paused correction protocol: every 40 steps the latest predicted point was corrected toward the true value with strength *c* = 0, 0.8 or a learned gate.

---

## (a) SynechismV23 hybrid

| Stage | What it does | Code |
|---|---|---|
| Input | Last 50 observed steps of the Lorenz-63 state (*x*, *y*, *z*), normalized | `run_v23_benchmark.py` |
| Encoder | U-Net-style 1-D convolutional encoder (conv → pool → conv → pool → upsample with skip connections), variational output *z* with mean μ and log-variance | `UFOEncoder` |
| Initial state | *h*₀ = tanh(*W* [*z* ; *g*]), where *g* is a learned goal vector (SAGA) shared by all inputs; then a gated Koopman lifting (lift to ⌊φ·128⌋ dimensions and project back) | `SynechismV23.encode`, `KoopmanLifting` |
| Latent ODE | d*h*/d*t* = *L*(*h*) + *N*(*h*) + *F*(*h*) − α (‖*h*‖² − *R*<sub>eff</sub>(*h*)²) *h* | `ElasticAttractorODE` |
| | *L*: linear map. *N*: LayerNorm + spectral-normalized MLP. *F*: Fourier skip (learned-frequency sin/cos features). The last term pulls *h* toward a sphere of radius *R*<sub>eff</sub>, keeping forecasts bounded. | |
| Elastic radius | *R*<sub>eff</sub>(*h*) = *R*<sub>base</sub> + δ*R*(*h*) · *p*<sub>event</sub>(*h*): the sphere widens when an event detector fires | `ElasticAttractorODE.effective_radius` |
| Integration | **IrrationalShutter**: fixed-step RK4 at sorted φ-Weyl time points. **LaminarBypass**: in chunks where a curvature heuristic is low, a gated linear step *h* + Δ*t*·*M*(*h*) replaces RK4 | `IrrationalShutter`, `LaminarBypass` |
| HyperAgent | Event detector + learned jump correction added to each hidden state | `HyperAgent` |
| Decoder | Two-layer MLP from the 128-dimensional hidden state to the 3 outputs, for each of the 20 predicted steps | `SynechismV23.decoder` |

Hidden size is 128 throughout. The model returns the 20-step forecast together with the hidden-state trajectory, which the CorrectionGate uses.

**Ablation variants** (`make_v23`):

| Variant | Elastic radius | Irrational shutter | Laminar bypass | HyperAgent | CorrectionGate |
|---|---|---|---|---|---|
| `v22_baseline` | off | off | off | off | off |
| `v23_full` | on | on | on | off | off |
| `v23_hybrid` (tested) | on | on | on | on | off |
| `v23_hybrid_gated` | on | on | on | on | on |

With the three v23 components off, `v22_baseline` uses the earlier attractor ODE (fixed radius) integrated with an adaptive solver at φ-lattice evaluation points.

**CorrectionGate.** A small MLP with a sigmoid output, *G*(*h*, *x* − *x̂*, ‖*x* − *x̂*‖) ∈ [0, 1], taking the model's hidden state, the correction error and its size. It is trained with periodic correction events in the training rollout (`train_model_with_periodic_correction`). Because it needs the latent ODE's hidden state, the learned-correction condition applies to SynechismV23 only.

## (b) Baselines and test protocol

**Baselines** (`src/models.py`), all with hidden size 128, trained on the same data with the same procedure:

- **FairTransformer**: Transformer encoder, 8 heads, 4 layers, sinusoidal positional encoding, linear head to the 20-step forecast.
- **FairLSTM**: 2-layer LSTM, autoregressive output.
- **FairMamba**: selective state-space (Mamba-style) block with input-dependent step size and zero-order-hold discretization; linear head to the 20-step forecast.

**Protocol** (Experiments 3 and 4):

- System: Lorenz-63, ρ = 28, d*t* = 0.02. Training windows are taken from the first 5,000 steps of each trajectory; 5 seeds.
- Rollout: from a 50-step context, each forecaster predicts 20 steps, which are appended to the context; this repeats up to a 50,000-step cap.
- Correction: every 40 steps, the latest predicted point is replaced by *x̂* + *c* (*x* − *x̂*), with *c* = 0 (none), *c* = 0.8 (fixed, following Fan et al. 2020) or *c* = *G*(·) (learned).
- Measure: first-crossing valid prediction time (VPT), the first step at which the RMS error across dimensions, scaled by each dimension's standard deviation over the first 5,000 steps, exceeds 0.3 or 0.4. Runs that never cross within the cap are reported as censored lower bounds. The earlier rolling-window metric is reported alongside it.

To regenerate the figure: `python docs/make_architecture_figure.py`.
