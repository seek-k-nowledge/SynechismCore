THESIS: Periodic real-data correction extends how long a chaotic forecast stays
accurate. This paper tests whether the forecaster's architecture (SynechismCore
latent ODE, v22, LSTM, Transformer, Mamba), and a learned correction gate, change
how much that correction helps.
DONE: (1) corrected coherence metric; (2) Exp 1 zero-correction runs, 6
architectures, all ~380-460 steps. 380 is that metric's reporting floor, so this
does NOT show a shared chaos limit; the earlier "no architecture escapes the
limit" reading is withdrawn pending floor-free VPT; (3) Exp 2 correction
strategies (results/periodic_correction_5seed_summary.md, also floor-limited);
(4) floor-free first-crossing VPT added to Exp 3 and Exp 4, with censored runs
reported as lower bounds.
TO RUN: Exp 3 head-to-head, all architectures under identical fixed 0.8
correction; Exp 4 gate retrain (log gate outputs, start at 0.8, train across
full correction cycles). KS, Weather and Robotics need NEW coherence-test code
(the coherence test currently exists only for Lorenz); whether they are in
scope is decided at the budget checkpoint after the first Lorenz session.
OUT OF SCOPE (future work): Finance; WP-NODE-style comparison (noise sweep, KL
divergence, Lorenz-96, ERA5); dysts 135-system benchmark; per-component
ablations; uncertainty calibration; Lyapunov-loss rebuild.
