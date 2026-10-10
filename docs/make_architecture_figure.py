"""Draws docs/architecture_v23.png and .svg (Figure 1). Run from the repo root:
    python docs/make_architecture_figure.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/architecture_v23"
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

box(X + 12, 91.5, W - 24, 5.5, "Input window  x:  last 50 steps × 3 values (x, y, z)", fc=C_IO, ec=E_IO, ts=10.5)
arrow(cx, 91.5, cx, 88.6)
box(X, 79.5, W, 9, "UFO encoder (U-Net, 1-D convolutions)",
    "conv → pool → conv → pool → upsample with skip connections;\nvariational output: latent z with mean μ and log-variance",
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
box(cx + 1, oy + 1.5, W / 2 - 2, 12.2, "How it's integrated",
    "Irrational shutter: RK4 steps at\nφ (golden-ratio) Weyl time points\nLaminar bypass: cheap gated linear\nstep where curvature is low",
    fc=C_SUB, ec=E_ODE, ts=11, bs=9.6, lw=1.2)
arrow(cx, oy, cx, 28.4, "hidden states for the next 20 steps", lx=cx + 1.2)

box(X, 19.5, W, 9, "HyperAgent: event detector + jump correction",
    "adds a learned correction to each hidden state when a sudden\nchange (discontinuity) is detected",
    fc=C_POST, ec=E_POST)
arrow(cx, 19.5, cx, 16.6)
box(X, 9.5, W, 7, r"Decoder (MLP)  →  forecast $\hat{y}$:  next 20 steps × 3",
    fc=C_POST, ec=E_POST, ts=11.5)

ax.text(3, 5.2, "Ablations: v22_baseline = same model with the elastic radius, shutter and bypass switched off.\n"
        "v23_hybrid_gated = this model plus the CorrectionGate shown in (b).",
        fontsize=10, color=MUTED, va="top", linespacing=1.4)

# divider
ax.plot([80.5, 80.5], [3, 103], color="#d9d7cf", lw=1.2)

# ===================== (b) PROTOCOL =====================
ax.text(85, 103, "(b)  Test protocol: identical for every forecaster", fontsize=15, fontweight="bold",
        color=INK, va="top")
bx, bw = 91, 46
bcx = bx + bw / 2
box(bx, 85, bw, 9, "Context window", "the last 50 steps the forecaster sees", fc=C_IO, ec=E_IO)
arrow(bcx, 85, bcx, 81.6)
box(bx - 4, 62.5, bw + 8, 19, "Forecaster  (one of five, trained the same way)",
    "SynechismV23 hybrid  (this work)\n"
    "v22 baseline  (same code, new parts off)\n"
    "Transformer  (8 heads, 4 layers)\n"
    "LSTM  (2 layers)\n"
    "Mamba  (state-space model)",
    fc=C_LOOP, ec=E_LOOP, bs=10.6)
arrow(bcx, 62.5, bcx, 57.1, "predict", lx=bcx + 1.2)
box(bx, 48, bw, 9, "Next 20 predicted steps", "the forecast is rolled forward on its own output",
    fc=C_LOOP, ec=E_LOOP)
arrow(bcx, 48, bcx, 44.6)
box(bx - 4, 22.5, bw + 8, 22, "Every 40 steps: real-data correction",
    r"the latest predicted point $\hat{x}$ is nudged toward the" "\n" r"true value $x$:" "\n\n"
    r"$\hat{x} \leftarrow \hat{x} + c\,(x - \hat{x})$" "\n\n"
    "none:  c = 0       fixed:  c = 0.8\n"
    r"learned:  $c = G(h,\ x-\hat{x},\ |x-\hat{x}|)$",
    fc=C_LOOP, ec=E_LOOP, bs=10.6)
# loop-back arrow (drawn as a bracket on the left, inside panel b)
lx0 = 83.6
ax.plot([bx - 4, lx0, lx0], [33.5, 33.5, 89.5], color=E_LOOP, lw=1.8)
ax.add_patch(FancyArrowPatch((lx0, 89.5), (bx, 89.5), arrowstyle="-|>", mutation_scale=16,
                             color=E_LOOP, lw=1.8))
ax.text(lx0 - 1.3, 61, "append to context and repeat", fontsize=10, color=E_LOOP, ha="center",
        va="center", style="italic", rotation=90)

# gate box
box(bx + bw + 6.5, 26, 15.5, 15, "CorrectionGate G",
    "small MLP → [0, 1]\ninputs: hidden state h,\nerror, error size\n(learned variant only;\nneeds the ODE's h)",
    fc=C_GATE, ec=E_GATE, ts=10.5, bs=8.9)
arrow(bx + bw + 6.5, 33.5, bx + bw + 4.2, 33.5, color=E_GATE)

box(bx - 4, 7.5, bw + 8, 12.5, "What's measured",
    "valid prediction time (VPT): first step where the error,\nscaled by the system's natural spread, exceeds 0.3 or 0.4\n"
    "Lorenz-63 · ρ = 28 · dt = 0.02 · 5 seeds · 50,000-step cap",
    fc="#f6f6f3", ec=E_IO, bs=10.2)

ax.text(158, 1.5, "SynechismCore · Paul E. Harris IV", fontsize=9, color=MUTED, ha="right", va="bottom")
fig.savefig(OUT + ".png", dpi=200, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
print("ok")
