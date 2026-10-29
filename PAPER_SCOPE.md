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

DONE (29 Oct 2026): all models at 200 epochs, 5 seeds. SynechismV23 longest
VPT in all four regimes (exponential: 2.03 vs LSTM 1.78, Transformer 0.55,
Mamba 0.50); 5/5 seeds vs Transformer and Mamba everywhere; 4/5 seeds vs LSTM
in each uneven regime (modest edge); LSTM degrades least (-14% vs -43%).

NEXT: more seeds for the LSTM comparison; ablations at 200 epochs; step-by-step decoding versions of the Transformer and Mamba (one-shot vs
step-by-step decoding is a confound); ContiFormer as a continuous-time
Transformer baseline; read Krach and Teichmann (2024) and BITS (2026) in full;
a second system (Kuramoto-Sivashinsky), then robotics data.

PAUSED: the earlier periodic-correction study (results floor-limited; see
results/periodic_correction_5seed_summary.md).

OUT OF SCOPE (future work): weather (ERA5), dysts 135-system benchmark,
uncertainty calibration, Lyapunov-spectrum and invariant-measure checks.
