# SynechismCore v23.0.1 — Competitive Landscape (March 2026)

What your fresh run is competing against. All numbers from published papers.
Report your results honestly against these — wins and losses both.

---

## KS-PDE

### PDE-Transformer-L (Holzschuh et al., ICML 2025)
Current transformer SOTA on KS-PDE.
- nRMSE 1-step:  **0.0111**
- nRMSE 20-step: **0.7357**
- Beats DiT-S, UDiT-S, scOT-S, FactFormer by 44-80% relative error
- Hardware: A100

### MNO — Mamba Neural Operator (Cheng et al., JCP 2026)
- 40-89% RMSE reduction vs best Transformer on chaotic PDEs
- Long-horizon: relative L2 stays bounded (2× lower than TF)
- 10× FLOPs reduction, 8× faster inference
- Hardware: A100/V100

**Your confirmed Kaggle result:** 1.43× MAE over standard Transformer.
PDE-Transformer-L is a stronger baseline than a standard Transformer.
The fresh run will show whether the gap holds against it.

---

## Lorenz-63 Long-Horizon

### PhyxMamba (Liu et al., arXiv 2025)
Current SOTA for long-term chaotic prediction.
- VPT: **5.06 Lyapunov times** (vs 1.59 TL best baseline)
- sMAPE@10: **67.29** (vs 102.96 iTransformer)
- D_frac: 0.060, D_stsp: 1.133 (best attractor geometry)
- Uses: time-delay embedding + MMD regularization + student-forcing

**Your confirmed result (v17.2):**
- 19,940 steps at ρ=28, dt=0.02
- λ ≈ 0.9056 → 1 Lyapunov time = 1/0.9056 ≈ 1.10 time units
- 19,940 steps × 0.02 dt = 398.8 time units ÷ 1.10 = **~362 Lyapunov times**

This would vastly exceed PhyxMamba's 5.06 TL — but the metrics must
be computed identically to compare fairly. The fresh run uses
compute_vpt() with the standard 0.4 normalized-error threshold.

---

## Lorenz-96 (40D Weather Proxy)

### PhyxMamba: VPT = 1.66 TL on L96
Your confirmed result: 1.01-1.06× MAE at individual forcing levels.
Multi-seed aggregate: 1.00× tied.
L96 is your weakest experiment. Report honestly.

---

## Finance / Regime Shift

### FMamba (Wu et al. 2024): A100, AMZN+TSLA OHLCV
### CMDMamba (Qin et al. 2025): V100, 10.4% accuracy gain
### TSMamba (Ma et al. 2024): ETTm2, Weather, ILI forecasting

Your confirmed result: 1.04× marginal, not significant.
Finance is a negative result. The ElasticManifold may help, but
financial regime shifts are sharp discontinuities — the HyperAgent
(hybrid variant) is the better bet than elastic alone.

---

## Where SynechismCore Has A Real Shot

| Benchmark | Your edge | Honest risk |
|---|---|---|
| KS-PDE | φ-sampling + attractor stabilization | PDE-Transformer-L is strong |
| Lorenz-63 coherence | 19,940 steps confirmed | Need metric alignment for fair comparison |
| Lorenz bifurcation | p<1e-43 vs LSTM (v17.2) | v22 multi-seed lost at ρ=40 |
| Mamba collapse | Confirmed novel finding | Already in your results |
| Robotics | ElasticManifold (untested) | 0.52× loss was decisive |

---

## What Cannot Be Claimed

- SOTA on KS-PDE: Neural CDE (0.2884) already beats SynechismCore (0.2952)
- SOTA on Lorenz: Need metric-aligned VPT comparison vs PhyxMamba
- Universal superiority: Finance and Weather are honest negatives

The paper's contribution is the structural analysis — when continuous
latent dynamics help, and when they don't — with honest negative results.
That framing is stronger for ML4PS than performance claims would be.
