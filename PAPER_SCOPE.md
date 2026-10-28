THESIS: When observations arrive at uneven times, a continuous-time latent ODE
(SynechismV23) keeps chaotic forecasts accurate longer than discrete-sequence
forecasters (Transformer, LSTM, Mamba) that are given the same timing
information. Tested on Lorenz-63 under four sampling regimes with the same mean
rate (regular, jitter, half missing, exponential), 5 seeds, valid prediction
time in Lyapunov times.

DONE (Oct 2026): (1) code for data, time-aware models and runner, tested on CPU
and A100; (2) full run at an equal 50-epoch budget: SynechismV23 longest in all
four regimes; Transformer and Mamba lose over 80% of VPT under exponential
sampling, SynechismV23 and LSTM about 50%; golden-ratio sub-steps show no VPT
effect; V23 parts beat V22 in 4/5 seeds; (3) found that no model had converged
at 50 epochs.

RUNNING: all models at 200 epochs (5 seeds). This decides the headline: in seed
0 the 200-epoch LSTM overtook the 50-epoch SynechismV23 under uneven sampling.

NEXT: step-by-step decoding versions of the Transformer and Mamba (one-shot vs
step-by-step decoding is a confound); ContiFormer as a continuous-time
Transformer baseline; read Krach and Teichmann (2024) and BITS (2026) in full;
a second system (Kuramoto-Sivashinsky), then robotics data.

PAUSED: the earlier periodic-correction study (results floor-limited; see
results/periodic_correction_5seed_summary.md).

OUT OF SCOPE (future work): weather (ERA5), dysts 135-system benchmark,
uncertainty calibration, Lyapunov-spectrum and invariant-measure checks.
