"""Draws docs/architecture_irregular.png and .svg (Figure 1, irregular-timing study).
Run from the repo root:
    python docs/make_architecture_irregular.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/architecture_irregular"
sys.path.insert(0, "src")
import numpy as np
from irregular_data import sample_gaps
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

INK = "#1b1b1a"; INK2 = "#45443f"; MUTED = "#7a7870"; LINE = "#9a988f"
C_IO = "#eef0f3"; C_ENC = "#e3eefb"; C_ODE = "#fdf0e3"; C_SUB = "#fff8ef"
C_POST = "#e6f4ee"; C_LOOP = "#f3eefb"; C_GATE = "#fbe9e9"
E_ENC = "#3987e5"; E_ODE = "#d97a26"; E_POST = "#199e70"; E_LOOP = "#7c5cc4"; E_GATE = "#c94848"; E_IO = "#8a8f99"

fig = plt.figure(figsize=(16, 10.6), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 106); ax.axis("off")


def box(x, y, w, h, title, body=None, fc=C_IO, ec=E_IO, ts=12.5, bs=10.5, lw=1.6, title_y=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - (title_y or 2.1), title, ha="center", va="top", fontsize=ts,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - (title_y or 2.1) - 3.0, body, ha="center", va="top", fontsize=bs,
                color=INK2, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", color=INK)


def arrow(x1, y1, x2, y2, label=None, color=LINE, lx=None, ly=None, ls="-", rad=0.0, ha="left"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=1.6, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2 + 1.2, ly if ly is not None else (y1 + y2) / 2,
                label, fontsize=9.5, color=MUTED, va="center", ha=ha, style="italic")


# ===================== (a) MODEL =====================
ax.text(3, 103, "(a)  SynechismV23 hybrid: the model being tested", fontsize=15, fontweight="bold",
        color=INK, va="top")
cx, W = 41, 64
X = cx - W / 2

box(X + 4, 91.5, W - 8, 5.5, "Input: last 50 observations (x, y, z) + the gap before each", fc=C_IO, ec=E_IO, ts=10.5)
arrow(cx, 91.5, cx, 88.6)
box(X, 79.5, W, 9, "UFO encoder (U-Net, 1-D convolutions)",
    "reads values and gaps (4 channels); conv → pool → upsample with skips;\nvariational output z (the mean μ is used at test time)",
    fc=C_ENC, ec=E_ENC)
arrow(cx, 79.5, cx, 76.6, "z", lx=cx + 1.2)
box(X, 67.5, W, 9, r"Initial state  $h_0 = \tanh(W\,[z;\,g])$,  then Koopman lifting",
    "g = SAGA, a learned goal vector shared by all inputs;\nKoopman: gated lift to φ×128 dims and back",
    fc=C_ENC, ec=E_ENC)
arrow(cx, 67.5, cx, 64.6, r"$h_0$  (128 dims)", lx=cx + 1.2)

# ODE block
oy, oh = 31.5, 33
ax.add_patch(FancyBboxPatch((X - 2, oy), W + 4, oh, boxstyle="round,pad=0,rounding_size=1.6",
                            fc=C_ODE, ec=E_ODE, lw=2.0))
ax.text(cx, oy + oh - 1.8, "Latent ODE: follows a learned rule for how h changes over time",
        ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
ax.text(cx, oy + oh - 6.2,
        r"$\frac{dh}{dt} = L(h) + N(h) + F(h) - \alpha\,(\|h\|^2 - R_{\mathrm{eff}}(h)^2)\,h$",
        ha="center", va="top", fontsize=15, color=INK)
ax.text(cx, oy + oh - 12.6,
        "L linear  ·  N nonlinear (spectral-norm MLP)  ·  F Fourier skip (sin/cos features)\n"
        "last term: pull toward a sphere of radius R_eff, so forecasts stay bounded",
        ha="center", va="top", fontsize=10, color=INK2, linespacing=1.35)
box(X + 1, oy + 1.5, W / 2 - 2, 12.2, "Elastic radius",
    "R_eff = R_base + δR(h) · p_event(h)\nsphere widens when an\nevent detector fires",
    fc=C_SUB, ec=E_ODE, ts=11, bs=9.6, lw=1.2)
box(cx + 1, oy + 1.5, W / 2 - 2, 12.2, "Integrated to the real times",
    "RK4 to each requested time, with\n4 golden-ratio (φ) sub-steps per gap;\nbypass: cheap step where smooth",
    fc=C_SUB, ec=E_ODE, ts=11, bs=9.6, lw=1.2)
arrow(cx, oy, cx, 28.4, "hidden states at the 20 requested future times", lx=cx + 1.2)

box(X, 19.5, W, 9, "HyperAgent: event detector + jump correction",
    "adds a learned correction to each hidden state when a sudden\nchange (discontinuity) is detected",
    fc=C_POST, ec=E_POST)
arrow(cx, 19.5, cx, 16.6)
box(X, 9.5, W, 7, r"Decoder (MLP)  →  forecast $\hat{y}$ at those 20 times × 3",
    fc=C_POST, ec=E_POST, ts=11.5)

ax.text(3, 5.2, "Ablations: synechism_nophi = equal sub-steps instead of golden-ratio ones.\n"
        "v22_baseline = elastic radius, φ sub-steps, bypass and HyperAgent switched off.",
        fontsize=10, color=MUTED, va="top", linespacing=1.4)

# divider
ax.plot([80.5, 80.5], [3, 103], color="#d9d7cf", lw=1.2)

# ===================== (b) PROTOCOL =====================
ax.text(85, 103, "(b)  Test: same data and same timing information for every model", fontsize=15,
        fontweight="bold", color=INK, va="top")
bx, bw = 87, 69
bcx = bx + bw / 2
# sampling regimes with real tick marks
ax.add_patch(FancyBboxPatch((bx, 70), bw, 27.5, boxstyle="round,pad=0,rounding_size=1.2",
                            fc=C_IO, ec=E_IO, lw=1.6))
ax.text(bcx, 96, "Lorenz-63 observed at uneven times (same average rate, 0.02)", ha="center",
        va="top", fontsize=12, fontweight="bold", color=INK)
rng = np.random.default_rng(3)
rows = [("regular", "Regular", "every 0.02"), ("jitter", "Jitter", "0.01–0.03"),
        ("missing", "Half missing", "half the readings lost"), ("exponential", "Exponential", "bursts and long gaps")]
span = 0.6
for k, (key, lab, note) in enumerate(rows):
    yy = 89.5 - k * 5.0
    ax.text(bx + 2, yy, lab, fontsize=10.5, color=INK, va="center", fontweight="bold")
    ax.text(bx + 2, yy - 2.0, note, fontsize=8.6, color=MUTED, va="center")
    t = np.cumsum(sample_gaps(200, key, 0.02, rng)); t = t[t < span]
    x0, x1 = bx + 21, bx + bw - 3
    ax.plot([x0, x1], [yy - 0.9, yy - 0.9], color="#c9c7bf", lw=1)
    for tt in t:
        xx = x0 + tt / span * (x1 - x0)
        ax.plot([xx, xx], [yy - 0.9, yy + 1.1], color=E_ENC, lw=1.3)
arrow(bcx, 70, bcx, 66.6)
box(bx, 47, bw, 19.5, "Forecasters, each told the gaps and the times to predict",
    "SynechismV23 (this work): latent ODE run over the real gaps\n"
    "Transformer (8 heads, 4 layers): continuous-time positional encoding\n"
    "LSTM: reads each gap; predicts one time at a time\n"
    "Mamba (state-space model): reads each gap\n"
    "plus two SynechismV23 ablations", fc=C_LOOP, ec=E_LOOP, bs=10.4)
arrow(bcx, 47, bcx, 43.6)
box(bx, 27, bw, 16.5, "Free-running forecast",
    "predict the values at the next 20 observation times\n(times known, values not), feed the predictions back\n"
    "as history, repeat out to 10 Lyapunov times;\n30 start points on held-out data, 5 seeds", fc=C_LOOP, ec=E_LOOP, bs=10.4)
arrow(bcx, 27, bcx, 23.6)
box(bx, 8.5, bw, 15, "What's measured",
    "valid prediction time (VPT): elapsed time until the error,\nscaled by the system's natural spread, first exceeds 0.4 (or 0.3),\n"
    "in Lyapunov times (1 LT ≈ 1.1 time units, the time for a\nsmall error to grow about 2.7×)", fc="#f6f6f3", ec=E_IO, bs=10.2)

ax.text(158, 1.5, "SynechismCore · Paul E. Harris IV", fontsize=9, color=MUTED, ha="right", va="bottom")
fig.savefig(OUT + ".png", dpi=200, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
print("ok")
