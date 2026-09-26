# Periodic Real-Data Correction — 5-Seed Coherence Test Summary

**Status: transcribed summary only.** The raw log (`run_log_correction_test2.txt`)
and results JSON from this run were produced on the Paperspace GPU machine and
have not yet been pulled onto this local machine — that machine is currently
stopped. This file exists so the numbers aren't lost in the meantime; it should
be replaced or supplemented with the actual raw log/JSON once retrieved.

## Setup

Lorenz-63 coherence test with periodic real-data correction, comparing four
conditions (see `run_coherence_test_with_correction()` in `run_v23_benchmark.py`):
fixed-schedule correction (Fan et al. 2020-style, c=0.8), a learned
`CorrectionGate`, and zero-correction, across 5 seeds (0–4).

## Results — steps until divergence

| Condition            | Mean ± Std   | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|-----------------------|--------------|--------|--------|--------|--------|--------|
| v23_hybrid_zero       | 384.0 ± 8.9  | 380    | 400    | 380    | 380    | 380    |
| v23_hybrid_fixed      | 464.0 ± 75.4 | 380    | 580    | 420    | 480    | 460    |
| v23_hybrid_gated      | 396.0 ± 26.1 | 380    | 400    | 440    | 380    | 380    |
| v22_baseline_fixed    | 448.0 ± 87.9 | 600    | 400    | 440    | 420    | 380    |

## Key finding

The fixed 0.8 blend won on average (`v23_hybrid_fixed`: 464.0 mean, vs. 384.0 for
zero-correction). The learned `CorrectionGate` (`v23_hybrid_gated`: 396.0 mean)
matched the zero-correction result in 4 of 5 seeds, suggesting the gate never
meaningfully opened during training — consistent with its bias being initialized
at -3.0 (sigmoid(-3.0) ≈ 0.05), which starts it very close to "always distrust the
correction." This motivates Experiment 4 (gate retrain): initializing the gate's
bias at `logit(0.8) ≈ +1.386` instead, and training across multiple full
correction cycles rather than a single correction window + one lookahead window.
