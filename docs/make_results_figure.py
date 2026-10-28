"""Draws the irregular-timing results figure from saved result files.

    python docs/make_results_figure.py results/irregular_50ep docs/irregular_results_50ep "50 epochs"

Reads <results dir>/<regime>/main_results.json, takes the median VPT(0.4) over the
30 test starts for each seed, and plots the mean over seeds (dot) with the lowest
and highest seed (whisker), for each model across the four sampling regimes.
"""
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "results/irregular_50ep"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/irregular_results_50ep"
BUDGET = sys.argv[3] if len(sys.argv) > 3 else "50 epochs"

REGIMES = [("regular", "Regular", "0"), ("jitter", "Jitter", "0.29"),
           ("missing", "Half missing", "0.70"), ("exponential", "Exponential", "1.0")]
MODELS = [("synechism", "SynechismV23"), ("lstm", "LSTM"),
          ("transformer", "Transformer"), ("mamba", "Mamba")]
BLUE = "#2a78d6"; GREY = "#9b9a92"; INK = "#1b1b1a"; INK2 = "#5c5b55"; GRID = "#e6e5df"
STYLE = {"synechism": (BLUE, 1.0, 2.6, 7), "lstm": (GREY, 1.0, 1.6, 5.5),
         "transformer": (GREY, 0.7, 1.6, 5.5), "mamba": (GREY, 0.45, 1.6, 5.5)}

plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]

stats = {}
for key, _, _ in REGIMES:
    with open(os.path.join(RES, key, "main_results.json")) as f:
        rows = json.load(f)["results"]
    for m, _ in MODELS:
        per_seed = [st.median(r["vpt_0.4"]) for r in rows if r["model"] == m]
        if per_seed:
            stats[(m, key)] = (st.mean(per_seed), min(per_seed), max(per_seed), len(per_seed))

fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.subplots_adjust(left=0.08, right=0.74, top=0.80, bottom=0.14)
for mi, (m, name) in enumerate(MODELS):
    color, alpha, lw, ms = STYLE[m]
    xs, ys = [], []
    for ri, (key, _, _) in enumerate(REGIMES):
        if (m, key) not in stats:
            continue
        mean, lo, hi, _ = stats[(m, key)]
        x = ri + (mi - 1.5) * 0.05
        ax.plot([x, x], [lo, hi], color=color, alpha=alpha, lw=1.5, solid_capstyle="round")
        xs.append(x); ys.append(mean)
    ax.plot(xs, ys, color=color, alpha=alpha, lw=lw, marker="o", ms=ms, zorder=3)
    first, last = stats.get((m, "regular")), stats.get((m, "exponential"))
    if first and last:
        drop = round(100 * (1 - last[0] / first[0]))
        stats[(m, "label")] = (last[0], f"{name}  {last[0]:.2f}  (−{drop}%)")

# end labels, nudged apart
labels = sorted([(stats[(m, "label")], m) for m, _ in MODELS if (m, "label") in stats],
                key=lambda t: -t[0][0])
prev = None
for (yv, text), m in labels:
    y = yv if prev is None else min(yv, prev - 0.17)
    prev = y
    ax.annotate(text, xy=(3 + 0.1, yv), xytext=(3.25, y), textcoords="data", va="center",
                fontsize=10.5, color=INK if m == "synechism" else INK2,
                fontweight="bold" if m == "synechism" else "normal",
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1), annotation_clip=False)
ax.text(3.25, labels[0][0][0] + 0.25, "At exponential\n(change from regular)",
        fontsize=9.5, color=INK2, va="bottom")

ax.set_xticks(range(4))
ax.set_xticklabels([f"{lab}\nCV {cv}" for _, lab, cv in REGIMES], fontsize=10.5, color=INK)
ax.set_xlim(-0.3, 3.15); ax.set_ylim(0, 3.5)
ax.set_yticks([0, 1, 2, 3]); ax.tick_params(axis="y", labelsize=10, colors=INK2, length=0)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Valid prediction time (Lyapunov times)", fontsize=10.5, color=INK2)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)

n = max(v[3] for k, v in stats.items() if k[1] != "label")
fig.text(0.08, 0.95, f"SynechismV23 forecasts longest in every regime at {BUDGET}",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.08, 0.905, f"Lorenz-63 observed at uneven times. Dot = mean of {n} seeds (median of 30 test "
         "forecasts each); whisker = lowest to highest seed.", fontsize=10, color=INK2)
fig.text(0.08, 0.875, f"Every model trained for {BUDGET}. Regimes ordered by how uneven the gaps are (CV).",
         fontsize=10, color=INK2)
fig.text(0.08, 0.03, "SynechismCore · Paul E. Harris IV · github.com/seek-k-nowledge/SynechismCore",
         fontsize=8.5, color=GREY)
fig.savefig(OUT + ".png", facecolor="white")
print("saved", OUT + ".png")
