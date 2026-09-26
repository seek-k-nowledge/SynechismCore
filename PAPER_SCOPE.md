THESIS: Using a corrected, luck-resistant coherence metric, no architecture
escapes the chaos (Lyapunov) limit without data. Periodic real-data correction
extends coherence. This paper tests whether SynechismCore's architecture, and a
learned correction gate, make correction work better than standard
architectures (v22, LSTM, Transformer, Mamba).
DONE: (1) corrected coherence metric; (2) Exp 1 zero-correction ceiling,
6 architectures, ~380-460 steps; (3) Exp 2 correction strategies (table above).
TO RUN: Exp 3 head-to-head, all architectures under identical fixed 0.8
correction; Exp 4 gate retrain (log gate outputs, start at 0.8, train across
full correction cycles); then KS, Weather, Robotics re-runs under the corrected
metric. Budget checkpoint after the first Lorenz session before the others.
OUT OF SCOPE (future work): Finance; WP-NODE-style comparison (noise sweep, KL
divergence, Lorenz-96, ERA5); dysts 135-system benchmark; per-component
ablations; uncertainty calibration; Lyapunov-loss rebuild.
