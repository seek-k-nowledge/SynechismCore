# SynechismCore — Live Results Log

Plain-English record of what's been run, what it means, and how it compares
to the originally confirmed results. Updated as each experiment finishes.

## How to read "good vs bad" in this log
- Lower MAE = more accurate (smaller mistakes)
- Higher VPT = stayed accurate longer before failing
- p-value close to 0 = a real, trustworthy difference
- p-value close to 1 = essentially no real difference (not significant)

---

## LORENZ — COMPLETE (10 seeds, real GPU run on A100, Sep 2026)

**What this experiment tests:** train on one range of chaos (rho 18-28), test on a harder, unseen range (rho 35-50) - checking if the model generalizes past a bifurcation, not just memorizes.

**Original confirmed result (Kaggle, v22, in the whitepaper):** ODE lost to Transformer, ratio 0.94x - an already-honest, already-published negative result.

**Tonight's real result (10 seeds, A100, all v23 variants added):**
- Transformer: MAE 6.39 (winner, very consistent, std only 0.12)
- v23_full: MAE 11.03 (best of our own variants)
- v23_hybrid: MAE 11.04
- v22_baseline: MAE 11.82
- p-value (v23_full vs transformer): 0.9999 - NOT significant, a clear, unambiguous loss, not a close call

**Verdict - GOOD, BAD, or SAME:**
- SAME as before: Transformer still wins on this test, matching the original confirmed v22 result's direction. Not new bad news.
- GOOD (new finding): v23_full and v23_hybrid both beat v22_baseline consistently (11.03-11.04 vs 11.82) - real, repeatable improvement from the new components, even though it's not enough to beat Transformer. This is new information the original paper didn't have.
- WATCH: shutter_only showed real numerical instability on several seeds (NaN warnings triggered, caught and clamped by the P10 fix) - a legitimate, citable finding about where this component is fragile.

**File:** results/fresh_run/lorenz.json

---

## KS-PDE — IN PROGRESS (switched to 5 seeds after 10-seed run proved too slow)

**What this experiment tests:** train on one viscosity (nu=1.0), test on lower viscosity (nu=0.5) - more turbulent, harder to predict.

**Original confirmed result:** SynechismCore WON - 1.43x better than Transformer (0.2952 vs 0.4207 MAE), 5 seeds, tight variance. This is the paper's headline positive result.

**Tonight's real timing data:** ~80 minutes for ONE full seed (all 8 variants) on the A100. 5 seeds estimated ~6.7 hours total.

**Status: not yet complete this session - relaunching at 5 seeds.** Will update this section once done. THIS IS THE MOST IMPORTANT REMAINING RESULT - it either reconfirms the paper's main win or doesn't.

---

## WEATHER, FINANCE, ROBOTICS — NOT YET RUN THIS SESSION

Original confirmed results for reference:
- Weather: tied, 1.00x (honest negative)
- Finance: marginal, 1.04x, not significant (honest negative)
- Robotics: LOST, 0.52x - this is the one ElasticManifold was specifically built to fix. Whether v23_hybrid beats v22_baseline here is the single most important open question in the whole project.

---

## COHERENCE TEST (19,940 steps) — NOT YET RE-RUN THIS SESSION

This is the paper's strongest, most attention-grabbing confirmed result (already validated on original Kaggle hardware, not touched or threatened by anything above). Plan: re-run with multiple seeds and possibly extend past 25,000 steps to strengthen the claim. Decided NOT to call this a "world record" - found a real paper (arXiv 2004.01258) claiming "practically infinite" horizon on the same Lorenz system, but using periodic real-data correction ("rare state updating" every ~40 steps) - a fundamentally easier setup than our pure, zero-correction autoregression. Correct framing: "exceeds prior benchmarks in the strict pure-autoregressive setting" - precise, defensible, cites the competing paper directly rather than ignoring it.
