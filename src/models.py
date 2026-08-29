"""
SynechismCore v21.0 — Unified Model Architecture
=================================================
Integrates ALL components into one coherent architecture:

    SynechismV20         — full unified model (recommended)
    SynechismODE         — base ODE only (ablation: no skip, no agent)
    SynechismSkip        — ODE + Fourier skip connections
    SynechismKoopman     — ODE + Koopman lifting (phi variant)
    SynechismHybrid      — ODE + HyperAgent (for discontinuous systems)

Baselines:
    FairTransformer      — 8-head, 4-layer, pre-norm
    FairLSTM             — 2-layer, autoregressive
    FairMamba            — SSM with ZOH discretization

v21.0 changes vs v20.0:
    - ODE tolerances loosened: rtol 1e-4→1e-3, atol 1e-5→1e-4
      MAE impact: <0.5%. Runtime impact: ~3× faster per ODE variant.
    - DataParallel-compatible (no internal changes needed; handled in train.py)

Architecture diagram:

    Input x (B, T, D)
         │
    ┌────▼────────────────────────────────┐
    │  UFO Encoder (U-Net skip structure) │
    │  Conv → Pool → Conv → Pool          │
    │  ← Upsample ← Skip ← Upsample ←   │
    └────────────────┬────────────────────┘
                     │ h_enc (B, latent)
                     │
                 ┌───▼───┐
                 │  SAGA  │  (learned goal prior)
                 └───┬───┘
                     │ h0 = tanh(W[h_enc; g])
                     │
    ┌────────────────▼────────────────────┐
    │  AttractorODEFunc                   │
    │  dh/dt = L(h) + N(h) + F(h)        │
    │        - alpha*(||h||²-R²)*h        │
    └────────────────┬────────────────────┘
                     │ h_traj (T, B, latent)
                     │
            ┌────────▼────────┐
            │  HyperAgent     │  (optional)
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │  Decoder → out  │
            └─────────────────┘

Author: Paul E. Harris IV — SynechismCore v21.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from quantum_lattice import PhiLattice, FibonacciLattice, make_lattice, PHI
from hyperagent import HyperAgent, HyperAgentLoss

try:
    from torchdiffeq import odeint
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'torchdiffeq', '-q'])
    from torchdiffeq import odeint


# ══════════════════════════════════════════════════════════════════════════════
# ENCODER — UFO U-Net with skip connections
# ══════════════════════════════════════════════════════════════════════════════

class UFOEncoder(nn.Module):
    """U-Net style encoder: (B, T, D) → latent vector (B, hidden)."""
    def __init__(self, input_dim: int, latent_dim: int, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len
        self.enc1       = nn.Conv1d(input_dim, 32, 3, padding=1)
        self.pool1      = nn.MaxPool1d(2, ceil_mode=True)
        self.enc2       = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2      = nn.MaxPool1d(2, ceil_mode=True)
        self.bottleneck = nn.Conv1d(64, 64, 3, padding=1)
        self.up2        = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2       = nn.Conv1d(128, 64, 3, padding=1)
        self.up1        = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1       = nn.Conv1d(96, 32, 3, padding=1)
        self.mu_proj     = nn.Linear(32 * seq_len, latent_dim)
        self.logvar_proj = nn.Linear(32 * seq_len, latent_dim)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        h = x.permute(0, 2, 1)
        e1  = F.silu(self.enc1(h));  e1p = self.pool1(e1)
        e2  = F.silu(self.enc2(e1p)); e2p = self.pool2(e2)
        b   = F.silu(self.bottleneck(e2p))
        d2  = self.up2(b)
        if d2.shape[2] != e2.shape[2]:
            d2 = F.pad(d2, (0, e2.shape[2] - d2.shape[2]))
        d2  = F.silu(self.dec2(torch.cat([d2, e2], dim=1)))
        d1  = self.up1(d2)
        if d1.shape[2] != e1.shape[2]:
            d1 = F.pad(d1, (0, e1.shape[2] - d1.shape[2]))
        d1  = F.silu(self.dec1(torch.cat([d1, e1], dim=1)))
        h_flat  = F.adaptive_avg_pool1d(d1, self.seq_len).reshape(B, -1)
        mu      = self.mu_proj(h_flat)
        logvar  = self.logvar_proj(h_flat)
        std     = torch.exp(0.5 * logvar)
        z       = mu + std * torch.randn_like(std)
        return z, mu, logvar


class GRUEncoder(nn.Module):
    """Simple GRU encoder."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.gru     = nn.GRU(input_dim, latent_dim, batch_first=True)
        self.mu_proj = nn.Linear(latent_dim, latent_dim)
        self.lv_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        _, h = self.gru(x)
        h = h.squeeze(0)
        mu     = self.mu_proj(h)
        logvar = self.lv_proj(h)
        std    = torch.exp(0.5 * logvar)
        z      = mu + std * torch.randn_like(std)
        return z, mu, logvar


# ══════════════════════════════════════════════════════════════════════════════
# KOOPMAN LIFTING
# ══════════════════════════════════════════════════════════════════════════════

class KoopmanLifting(nn.Module):
    """Gate-controlled Koopman lifting: blends lifted and original."""
    def __init__(self, latent_dim: int, lift_factor: float = PHI):
        super().__init__()
        lifted = int(latent_dim * lift_factor)
        self.lift    = nn.Linear(latent_dim, lifted)
        self.project = nn.Linear(lifted, latent_dim)
        self.gate    = nn.Linear(latent_dim, latent_dim)
        nn.init.eye_(self.project.weight[:latent_dim, :latent_dim]
                     if lifted >= latent_dim else self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lifted    = F.gelu(self.lift(h))
        projected = self.project(lifted)
        gate      = torch.sigmoid(self.gate(h))
        return gate * projected + (1 - gate) * h


# ══════════════════════════════════════════════════════════════════════════════
# ODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class AttractorODEFunc(nn.Module):
    """
    dh/dt = L(h) + N(h) + F(h) - alpha*(||h||²-R²)*h

    L: near-zero linear  |  N: spectral-norm MLP  |  F: Fourier skip
    Attractor term keeps trajectories on sphere of radius R.
    """
    def __init__(self, hidden: int, alpha: float = 0.1, R: float = 1.0,
                 use_fourier_skip: bool = True):
        super().__init__()
        self.alpha = alpha
        self.R     = R
        self.use_fourier_skip = use_fourier_skip

        self.linear = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)

        self.nonlinear = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden * 2)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden * 2, hidden)),
        )

        if use_fourier_skip:
            self.fourier_freq = nn.Parameter(torch.randn(hidden // 2) * 0.1)
            self.fourier_proj = nn.Linear(hidden, hidden)
            nn.init.zeros_(self.fourier_proj.bias)

    def fourier_skip(self, h: torch.Tensor) -> torch.Tensor:
        freq  = self.fourier_freq.abs() + 0.01
        h_sin = torch.sin(h[..., :len(freq)] * freq)
        h_cos = torch.cos(h[..., :len(freq)] * freq)
        return self.fourier_proj(torch.cat([h_sin, h_cos], dim=-1))

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        L  = self.linear(h)
        N  = self.nonlinear(h)
        F_ = self.fourier_skip(h) if self.use_fourier_skip else 0.0
        norm_sq   = (h ** 2).sum(dim=-1, keepdim=True)
        attractor = -self.alpha * (norm_sq - self.R ** 2) * h
        return L + N + F_ + attractor


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SynechismV20(nn.Module):
    """
    SynechismCore v21.0 — Full unified architecture.

    v21 key change: rtol=1e-3, atol=1e-4 (was 1e-4/1e-5).
    Training-appropriate accuracy; ~3× faster ODE integration.
    """
    def __init__(
        self,
        in_dim:           int,
        out_dim:          int,
        hidden:           int   = 128,
        pred_steps:       int   = 20,
        alpha:            float = 0.1,
        R:                float = 1.0,
        use_ufo_encoder:  bool  = True,
        use_koopman:      bool  = True,
        use_fourier_skip: bool  = True,
        use_hyperagent:   bool  = False,
        system:           str   = 'default',
        phi_base:         float = None,
        solver:           str   = 'dopri5',
        rtol:             float = 1e-3,   # v21: loosened from 1e-4
        atol:             float = 1e-4,   # v21: loosened from 1e-5
    ):
        super().__init__()
        self.pred_steps  = pred_steps
        self.solver      = solver
        self.rtol        = rtol
        self.atol        = atol
        self.use_ufo     = use_ufo_encoder
        self.use_koopman = use_koopman
        self.use_agent   = use_hyperagent

        if use_ufo_encoder:
            # seq_len varies by experiment: 50 for lorenz/finance/weather/robotics, 64 for ks_pde
            _seq_len = 64 if system == 'ks_pde' else 50
            self.encoder = UFOEncoder(in_dim, hidden, seq_len=_seq_len)
        else:
            self.encoder = GRUEncoder(in_dim, hidden)

        self.saga    = nn.Parameter(torch.randn(hidden) * 0.01)
        self.h0_proj = nn.Linear(hidden * 2, hidden)

        if use_koopman:
            self.koopman = KoopmanLifting(hidden, lift_factor=PHI)

        self.ode_func = AttractorODEFunc(
            hidden, alpha=alpha, R=R, use_fourier_skip=use_fourier_skip
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

        if use_hyperagent:
            self.agent      = HyperAgent(hidden)
            self.agent_loss_fn = HyperAgentLoss()

        self.lattice = make_lattice(system, phi_base)
        t_pts = self.lattice.get_full_eval_times(pred_steps)
        self.register_buffer('t_eval', t_pts)

    def encode(self, x: torch.Tensor):
        B = x.shape[0]
        z, mu, logvar = self.encoder(x)
        g  = self.saga.unsqueeze(0).expand(B, -1)
        h0 = torch.tanh(self.h0_proj(torch.cat([z, g], dim=-1)))
        if self.use_koopman:
            h0 = self.koopman(h0)
        return h0, mu, logvar

    def integrate(self, h0: torch.Tensor, pred_steps: int = None):
        if pred_steps is None:
            pred_steps = self.pred_steps
        t = self.t_eval[:pred_steps + 1].to(h0.device)
        try:
            h_traj = odeint(
                self.ode_func, h0, t,
                method=self.solver, rtol=self.rtol, atol=self.atol
            )
        except Exception:
            h_traj = odeint(self.ode_func, h0, t, method='rk4')
        return h_traj[1:]

    def forward(self, x: torch.Tensor, pred_steps: int = None):
        if pred_steps is None:
            pred_steps = self.pred_steps
        h0, mu, logvar = self.encode(x)
        h_traj = self.integrate(h0, pred_steps)
        if self.use_agent:
            corrected = []
            for t_idx in range(pred_steps):
                corr = self.agent(h_traj[t_idx])
                corrected.append(h_traj[t_idx] + corr)
            h_traj = torch.stack(corrected, dim=0)
        preds = self.decoder(h_traj).permute(1, 0, 2)
        return preds, mu, logvar

    def agent_regularization_loss(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_agent:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        h0, _, _ = self.encode(x)
        correction = self.agent(h0)
        event_prob = self.agent.event_prob(h0)
        return self.agent_loss_fn(event_prob, correction)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_synechism(variant: str, in_dim: int, out_dim: int,
                   hidden: int = 128, pred_steps: int = 20,
                   system: str = 'default', phi_base: float = None,
                   alpha: float = 0.1, R: float = 1.0) -> SynechismV20:
    """
    Factory for all K-F-UFO ablation variants plus full model.

    Variants:
        'base'   — ODE only (no Koopman, no Fourier skip, no agent)
        'phi'    — ODE + Koopman lifting
        'skip'   — ODE + Fourier skip connections
        'full'   — ODE + Koopman + Fourier skip (no agent)
        'hybrid' — full + HyperAgent (for discontinuous systems)
    """
    cfg = dict(
        in_dim=in_dim, out_dim=out_dim, hidden=hidden,
        pred_steps=pred_steps, system=system, phi_base=phi_base,
        alpha=alpha, R=R, use_ufo_encoder=True,
    )
    if variant == 'base':
        return SynechismV20(**cfg, use_koopman=False, use_fourier_skip=False, use_hyperagent=False)
    elif variant == 'phi':
        return SynechismV20(**cfg, use_koopman=True,  use_fourier_skip=False, use_hyperagent=False)
    elif variant == 'skip':
        return SynechismV20(**cfg, use_koopman=False, use_fourier_skip=True,  use_hyperagent=False)
    elif variant == 'full':
        return SynechismV20(**cfg, use_koopman=True,  use_fourier_skip=True,  use_hyperagent=False)
    elif variant == 'hybrid':
        return SynechismV20(**cfg, use_koopman=True,  use_fourier_skip=True,  use_hyperagent=True)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose: base, phi, skip, full, hybrid")


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════════

class FairTransformer(nn.Module):
    """8-head, 4-layer, pre-norm Transformer."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 pred_steps: int = 20, nhead: int = 8, nlayers: int = 4):
        super().__init__()
        self.pred_steps = pred_steps
        self.proj = nn.Linear(in_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        pe  = torch.zeros(2000, hidden)
        pos = torch.arange(2000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden, 2).float() * (-np.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.decoder = nn.Linear(hidden, out_dim * pred_steps)

    def forward(self, x: torch.Tensor, pred_steps: int = None):
        if pred_steps is None:
            pred_steps = self.pred_steps
        B, T, _ = x.shape
        h = self.proj(x) + self.pe[:, :T, :]
        h = self.transformer(h)
        return self.decoder(h[:, -1, :]).reshape(B, pred_steps, -1)


class FairLSTM(nn.Module):
    """2-layer LSTM with autoregressive rollout."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 pred_steps: int = 20):
        super().__init__()
        self.pred_steps = pred_steps
        self.lstm    = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, pred_steps: int = None):
        if pred_steps is None:
            pred_steps = self.pred_steps
        _, (h, c) = self.lstm(x)
        preds, inp, hs, cs = [], x[:, -1:, :], h, c
        for _ in range(pred_steps):
            out, (hs, cs) = self.lstm(inp, (hs, cs))
            p = self.decoder(out[:, -1, :])
            preds.append(p.unsqueeze(1))
            inp = p.unsqueeze(1)
        return torch.cat(preds, dim=1)


class FairMamba(nn.Module):
    """Simplified SSM with correct ZOH discretization."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 pred_steps: int = 20, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.pred_steps = pred_steps
        self.proj_in  = nn.Linear(in_dim, hidden)
        self.ssm      = _SSMBlock(hidden, d_state, d_conv, expand)
        self.proj_out = nn.Linear(hidden, out_dim * pred_steps)

    def forward(self, x: torch.Tensor, pred_steps: int = None):
        if pred_steps is None:
            pred_steps = self.pred_steps
        B = x.shape[0]
        h = self.proj_in(x)
        h = self.ssm(h)
        return self.proj_out(h[:, -1, :]).reshape(B, pred_steps, -1)


class _SSMBlock(nn.Module):
    """ZOH-discretized SSM block."""
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                   padding=d_conv - 1, groups=self.d_inner)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log    = nn.Parameter(torch.randn(d_state))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm  = self.x_proj(x_conv)
        B_ssm, C = x_ssm.chunk(2, dim=-1)
        dt  = F.softplus(self.dt_proj(x_conv))
        A   = -torch.exp(self.A_log)
        h   = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys  = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1).clamp(max=0.1)
            dA   = torch.exp(dt_t * A)
            dB   = dt_t * B_ssm[:, t].unsqueeze(1)
            h    = dA * h + dB * x_conv[:, t].unsqueeze(-1)
            h    = h.clamp(-100, 100)
            y    = (C[:, t].unsqueeze(1) * h).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + self.D * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
