"""
Time-aware forecasters for the irregular-timing study.

Every model sees the same inputs:
    x_hist   (B, S, 3)  last S observed values
    dt_hist  (B, S, 1)  the gap before each of those observations
    dt_fut   (B, K)     the gaps between the K future times to predict
and returns predictions at those K future times, shape (B, K, 3).
Gaps are divided by MEAN_DT before entering any network, so they are ~1.

IrregularSynechism reuses SynechismV23's components (UFO encoder, SAGA,
Koopman lifting, elastic attractor ODE, laminar bypass, HyperAgent, decoder).
What changes for irregular data:
  * the encoder reads the gaps as a 4th input channel;
  * the latent ODE is integrated to the actual future times, per example,
    with RK4 sub-steps between consecutive targets;
  * the golden-ratio (phi) idea is kept for those sub-steps: sub-step
    boundaries sit at sorted Weyl points {k*phi mod 1}, so each gap is cut
    aperiodically (use_phi_substeps=False gives equal sub-steps);
  * the laminar bypass decides per example and per sub-step whether to take
    its gated linear step instead of RK4;
  * at evaluation (model.eval()) the encoder's mean is used, not a sample,
    so forecasts are deterministic like the baselines.

The baselines receive the gaps as an extra input feature and the future gaps
as an input to their output head (the LSTM gets each next gap as it steps), so
every model knows exactly when it is being asked to predict.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models import UFOEncoder, KoopmanLifting, AttractorODEFunc, _SSMBlock   # noqa: E402
from v23_components import ElasticAttractorODE, LaminarBypass, PHI           # noqa: E402
from hyperagent import HyperAgent, HyperAgentLoss                            # noqa: E402

MEAN_DT = 0.02


def phi_fractions(m, phi=PHI):
    """Sub-step boundaries in (0, 1]: sorted {k*phi mod 1}, k = 1..m-1, then 1."""
    pts = sorted(((k * phi) % 1.0) for k in range(1, m))
    return [p for p in pts] + [1.0]


# ══════════════════════════════════════════════════════════════════════════════
# SynechismV23 for irregular timing
# ══════════════════════════════════════════════════════════════════════════════

class IrregularSynechism(nn.Module):
    is_latent_ode = True

    def __init__(self, in_dim=3, out_dim=3, hidden=128, seq_len=50,
                 use_elastic=True, use_phi_substeps=True, use_bypass=True,
                 use_hyperagent=True, use_koopman=True, use_fourier_skip=True,
                 substeps=4, alpha=0.1, R_base=1.0, max_R_expansion=5.0,
                 laminar_threshold=0.05):
        super().__init__()
        self.use_bypass = use_bypass
        self.use_agent = use_hyperagent
        self.use_koopman = use_koopman

        self.encoder = UFOEncoder(in_dim + 1, hidden, seq_len=seq_len)
        self.saga = nn.Parameter(torch.randn(hidden) * 0.01)
        self.h0_proj = nn.Linear(hidden * 2, hidden)
        if use_koopman:
            self.koopman = KoopmanLifting(hidden)
        if use_elastic:
            self.ode_func = ElasticAttractorODE(hidden, alpha=alpha, R_base=R_base,
                                                max_expansion=max_R_expansion,
                                                use_fourier_skip=use_fourier_skip)
        else:
            self.ode_func = AttractorODEFunc(hidden, alpha=alpha, R=R_base,
                                             use_fourier_skip=use_fourier_skip)
        if use_bypass:
            self.bypass = LaminarBypass(hidden, threshold=laminar_threshold)
        if use_hyperagent:
            self.agent = HyperAgent(hidden)
            self.agent_loss_fn = HyperAgentLoss()
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, out_dim))
        fr = phi_fractions(substeps) if use_phi_substeps else \
            [(k + 1) / substeps for k in range(substeps)]
        widths = np.diff([0.0] + fr)
        self.register_buffer('sub_widths', torch.tensor(widths, dtype=torch.float32))

    def encode(self, x_hist, dt_hist):
        inp = torch.cat([x_hist, dt_hist / MEAN_DT], dim=-1)
        z, mu, logvar = self.encoder(inp)
        code = z if self.training else mu
        g = self.saga.unsqueeze(0).expand(code.shape[0], -1)
        h0 = torch.tanh(self.h0_proj(torch.cat([code, g], dim=-1)))
        if self.use_koopman:
            h0 = self.koopman(h0)
        return h0, mu, logvar

    def _f(self, h):
        return self.ode_func(None, h)

    def _rk4(self, h, dt):
        k1 = self._f(h)
        k2 = self._f(h + 0.5 * dt * k1)
        k3 = self._f(h + 0.5 * dt * k2)
        k4 = self._f(h + dt * k3)
        return h + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _substep(self, h, dt):
        if not self.use_bypass:
            return self._rk4(h, dt)
        with torch.no_grad():
            lam = self.bypass.detector.is_laminar(h, self._f(h)).unsqueeze(-1)
        h_rk4 = self._rk4(h, dt)
        h_lam = self.bypass._laminar_step(h, dt)
        return torch.where(lam, h_lam, h_rk4)

    def integrate(self, h0, dt_fut):
        """Follow the ODE to each future time; returns (K, B, hidden)."""
        h, out = h0, []
        for k in range(dt_fut.shape[1]):
            gap = dt_fut[:, k:k + 1]                       # (B, 1), physical time
            for w in self.sub_widths:
                h = self._substep(h, gap * w)
            out.append(h)
        return torch.stack(out, dim=0)

    def forward(self, x_hist, dt_hist, dt_fut):
        h0, mu, logvar = self.encode(x_hist, dt_hist)
        h_traj = self.integrate(h0, dt_fut)
        if self.use_agent:
            h_traj = h_traj + self.agent(h_traj)
        preds = self.decoder(h_traj).permute(1, 0, 2)
        return preds, mu, logvar, h0

    def extra_loss(self, h0):
        if not self.use_agent:
            return h0.new_zeros(())
        return self.agent_loss_fn(self.agent.event_prob(h0), self.agent(h0))


# ══════════════════════════════════════════════════════════════════════════════
# Baselines, given the same timing information
# ══════════════════════════════════════════════════════════════════════════════

def time_encoding(t, dim):
    """Sinusoidal encoding of continuous time t (B, S) -> (B, S, dim)."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
                      * (-math.log(1000.0) / half))
    ang = t.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TimeTransformer(nn.Module):
    """Pre-norm Transformer encoder (8 heads, 4 layers). Position = actual time."""
    is_latent_ode = False

    def __init__(self, in_dim=3, out_dim=3, hidden=128, pred_len=20,
                 nhead=8, nlayers=4):
        super().__init__()
        self.pred_len, self.out_dim = pred_len, out_dim
        self.proj = nn.Linear(in_dim + 1, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
                                           dim_feedforward=hidden * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers,
                                                 enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden + pred_len, hidden * 2), nn.GELU(),
                                  nn.Linear(hidden * 2, out_dim * pred_len))
        self.hidden = hidden

    def forward(self, x_hist, dt_hist, dt_fut):
        g = dt_hist / MEAN_DT                              # (B, S, 1)
        t_rel = torch.cumsum(g.squeeze(-1), dim=1)
        t_rel = t_rel - t_rel[:, -1:]                      # 0 at the last observation
        h = self.proj(torch.cat([x_hist, g], dim=-1)) + time_encoding(t_rel, self.hidden)
        h = self.transformer(h)[:, -1]
        out = self.head(torch.cat([h, dt_fut / MEAN_DT], dim=-1))
        return out.reshape(-1, self.pred_len, self.out_dim)


class TimeLSTM(nn.Module):
    """2-layer LSTM; reads (value, gap); decodes step by step given each next gap."""
    is_latent_ode = False

    def __init__(self, in_dim=3, out_dim=3, hidden=128, pred_len=20):
        super().__init__()
        self.lstm = nn.LSTM(in_dim + 1, hidden, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(hidden, out_dim)

    def forward(self, x_hist, dt_hist, dt_fut):
        _, (h, c) = self.lstm(torch.cat([x_hist, dt_hist / MEAN_DT], dim=-1))
        prev, preds = x_hist[:, -1:], []
        for k in range(dt_fut.shape[1]):
            inp = torch.cat([prev, dt_fut[:, k:k + 1, None] / MEAN_DT], dim=-1)
            out, (h, c) = self.lstm(inp, (h, c))
            prev = self.decoder(out)
            preds.append(prev)
        return torch.cat(preds, dim=1)


class TimeMamba(nn.Module):
    """Selective state-space block; reads (value, gap); head gets the future gaps."""
    is_latent_ode = False

    def __init__(self, in_dim=3, out_dim=3, hidden=128, pred_len=20,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.pred_len, self.out_dim = pred_len, out_dim
        self.proj_in = nn.Linear(in_dim + 1, hidden)
        self.ssm = _SSMBlock(hidden, d_state, d_conv, expand)
        self.head = nn.Sequential(nn.Linear(hidden + pred_len, hidden * 2), nn.GELU(),
                                  nn.Linear(hidden * 2, out_dim * pred_len))

    def forward(self, x_hist, dt_hist, dt_fut):
        h = self.ssm(self.proj_in(torch.cat([x_hist, dt_hist / MEAN_DT], dim=-1)))[:, -1]
        out = self.head(torch.cat([h, dt_fut / MEAN_DT], dim=-1))
        return out.reshape(-1, self.pred_len, self.out_dim)


MODEL_NAMES = ('synechism', 'synechism_nophi', 'v22_baseline',
               'transformer', 'lstm', 'mamba')


def make_model(name, seq_len=50, pred_len=20, hidden=128):
    if name == 'synechism':            # SynechismV23 hybrid, everything on
        return IrregularSynechism(hidden=hidden, seq_len=seq_len)
    if name == 'synechism_nophi':      # ablation: equal sub-steps instead of phi
        return IrregularSynechism(hidden=hidden, seq_len=seq_len, use_phi_substeps=False)
    if name == 'v22_baseline':         # ablation: v23 components off
        return IrregularSynechism(hidden=hidden, seq_len=seq_len, use_elastic=False,
                                  use_phi_substeps=False, use_bypass=False,
                                  use_hyperagent=False)
    if name == 'transformer':
        return TimeTransformer(hidden=hidden, pred_len=pred_len)
    if name == 'lstm':
        return TimeLSTM(hidden=hidden, pred_len=pred_len)
    if name == 'mamba':
        return TimeMamba(hidden=hidden, pred_len=pred_len)
    raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")
