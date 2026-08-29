"""
SynechismCore v23.0.1 — Three New Architectural Components (PATCHED)
=====================================================================
All six patches from the whitepaper §1 are applied here before any
benchmark run. Document each fix in the docstring so reviewers can
verify the code matches the paper's description.

PATCH RECORD (v23.0 → v23.0.1):
  P1  StiffnessDetector: added super().__init__() — prevents crash on
      .to(device) / DataParallel / .state_dict() calls.
  P2  LaminarBypass._laminar_step: dt-aware — h(t+dt) ≈ h(t) + dt·map(h)
      instead of static projection. Required because IrrationalShutter
      produces variable step sizes.
  P3  LaminarBypass.integrate: extracts dt from t_eval at each step and
      passes it to _laminar_step. Required for P2.
  P4  ElasticAttractorODE.delta_r_net: Tanh → GELU, 16 → 32 units.
      GELU has better gradient flow during rapid radius expansion vs
      Tanh saturation.
  P5  launch_h100.py: torch.set_float32_matmul_precision('high') —
      2-3× free speedup on H100/A100 Tensor Cores with no accuracy loss
      for ODE training.
  P6  run_v23_benchmark.py coherence rollout: .detach() added to context
      window update — prevents OOM accumulation at step ~20,000.

Author: Paul E. Harris IV — SynechismCore v23.0.1
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

try:
    from torchdiffeq import odeint
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchdiffeq', '-q'])
    from torchdiffeq import odeint

PHI = (1.0 + 5.0 ** 0.5) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IRRATIONAL SHUTTER
# ═══════════════════════════════════════════════════════════════════════════════

class IrrationalShutter:
    """
    Forces ODE integration to step at Weyl-sequence (φ-lattice) points.

    v22 problem: PhiLattice generated evaluation times for dopri5, but
    dopri5 uses these as OUTPUT points only. Its internal integration steps
    are chosen adaptively by error estimation, reintroducing effective uniform
    spacing and the resonance the φ-lattice was meant to prevent.

    Fix: switch from dopri5 to rk4 with φ-lattice points as the actual
    integration steps. With rk4, torchdiffeq evaluates the ODE function
    exactly at the specified t values — no internal subdivision.

    Trade-off: rk4 with variable step sizes accumulates more truncation error
    than dopri5 with adaptive steps. The bet is that avoiding resonance
    outweighs this at moderate chaos levels. The ablation confirms or refutes.

    Note: this class is stateless — it generates t_eval tensors.
    The caller must use method='rk4' (enforced in SynechismV23.integrate).
    """

    def __init__(self, phi_base: float = PHI, dt_scale: float = 0.02):
        self.phi_base = phi_base
        self.dt_scale = dt_scale

    def get_eval_times(self, n_steps: int,
                       device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns [0] + sorted Weyl sequence scaled to physical time.
        Shape: (n_steps + 1,). Starts at 0, ends at ~n_steps * dt_scale.
        """
        k = np.arange(1, n_steps + 1)
        weyl = (k * self.phi_base) % 1.0
        weyl_sorted = np.sort(weyl) * self.dt_scale * n_steps
        t = np.concatenate([[0.0], weyl_sorted])
        t_tensor = torch.FloatTensor(t)
        if device is not None:
            t_tensor = t_tensor.to(device)
        return t_tensor

    def discrepancy(self, n_steps: int = 50) -> float:
        """Star discrepancy D* — lower = more uniform = less resonance."""
        k = np.arange(1, n_steps + 1)
        pts = np.sort((k * self.phi_base) % 1.0)
        n = len(pts)
        upper = max((i + 1) / n - pts[i] for i in range(n))
        lower = max(pts[i] - i / n for i in range(n))
        return max(upper, lower)


def compare_shutters(n_steps: int = 50) -> dict:
    """
    Compare discrepancy: φ vs √2 vs e vs uniform.
    Lower D* = better aperiodic coverage = less resonance with attractor.
    Theoretical prediction (Hurwitz): φ should win.
    Empirical confirmation: run_phi_ablation.py.
    """
    bases = {
        'phi (golden ratio)': PHI,
        'sqrt2':              np.sqrt(2),
        'e (Euler)':          np.e,
        'uniform (baseline)': 1.0,
    }
    return {name: IrrationalShutter(base).discrepancy(n_steps)
            for name, base in bases.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LAMINAR BYPASS (PATCHED: P1, P2, P3)
# ═══════════════════════════════════════════════════════════════════════════════

class StiffnessDetector(nn.Module):
    """
    Estimates local curvature of the ODE trajectory in latent space.

    PATCH P1: super().__init__() added. Without it, any call to
    .to(device), .state_dict(), .parameters(), or DataParallel raises
    AttributeError. The bug was invisible during dev because this class
    has no trainable parameters and forward() only accesses self.threshold.

    Naming note (from whitepaper §3.3.3): 'StiffnessDetector' is a misnomer.
    True stiffness = ratio of largest to smallest Jacobian eigenvalue.
    This computes a gradient variance heuristic — closer to curvature.
    Consider renaming to CurvatureHeuristic before submission to avoid
    reviewer criticism.
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__()   # PATCH P1: was missing, caused crash on .to(device)
        self.threshold = threshold

    def is_laminar(self, h: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
        """
        Returns bool mask (B,) — True where trajectory curvature is low.
        Low curvature → laminar → safe to use linear bypass.
        """
        state_mag = h.norm(dim=-1).clamp(min=1e-6)
        grad_var  = torch.var(dh, dim=-1)
        curvature = grad_var / state_mag
        return curvature < self.threshold


class LaminarBypass(nn.Module):
    """
    Chunked integration with stiffness-aware solver switching.

    PATCH P2 + P3: dt-aware laminar step.

    Original: h_next = gate * bypass_map(h) + (1 - gate) * h
    Problem: static projection learns an average step size. IrrationalShutter
    produces variable step sizes (some 0.001, some 0.04). A static map
    introduces systematic error proportional to deviation from that average.

    Fixed: Euler approximation of a linear ODE:
        h(t+dt) ≈ h(t) + dt · bypass_map(h)
    giving: h_next = gate * (h + dt * bypass_map(h)) + (1 - gate) * h

    The integrate() loop now extracts dt from t_eval at each step and
    passes it to _laminar_step (P3).
    """

    def __init__(self, hidden: int, threshold: float = 0.05,
                 chunk_size: int = 10):
        super().__init__()
        self.detector   = StiffnessDetector(threshold)
        self.chunk_size = chunk_size

        # Near-identity initialization: stable bypass at training start
        self.bypass_map = nn.Linear(hidden, hidden, bias=True)
        nn.init.eye_(self.bypass_map.weight)
        nn.init.zeros_(self.bypass_map.bias)

        # Gate: starts at sigmoid(3) ≈ 0.95 (nearly full bypass in laminar)
        self.bypass_gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.bypass_gate.weight)
        nn.init.constant_(self.bypass_gate.bias, 3.0)

    def _laminar_step(self, h: torch.Tensor, dt: float) -> torch.Tensor:
        """
        PATCH P2: dt-aware linear step.
        h(t+dt) ≈ h(t) + dt · bypass_map(h), gated by bypass_gate.
        """
        gate = torch.sigmoid(self.bypass_gate(h))              # (B, 1)
        return gate * (h + dt * self.bypass_map(h)) + (1 - gate) * h

    def integrate(self, ode_func: nn.Module, h0: torch.Tensor,
                  t_eval: torch.Tensor, pred_steps: int) -> torch.Tensor:
        """
        Chunked integration: laminar phases use bypass, turbulent use RK4.
        PATCH P3: dt extracted from t_eval at each step, passed to _laminar_step.
        """
        results = []
        h_curr  = h0
        laminar_count = 0
        turbulent_count = 0
        step = 0

        while step < pred_steps:
            chunk_end = min(step + self.chunk_size, pred_steps)
            n = chunk_end - step

            with torch.no_grad():
                dh = ode_func(t_eval[step], h_curr)
                is_lam = self.detector.is_laminar(h_curr, dh)

            if is_lam.all():
                chunk = []
                h = h_curr
                for i in range(n):
                    # PATCH P3: extract dt from t_eval for this specific step
                    dt = (t_eval[step + i + 1] - t_eval[step + i]).item() \
                         if (step + i + 1) < len(t_eval) else 0.02
                    h = self._laminar_step(h, dt)   # PATCH P2: dt-aware
                    chunk.append(h)
                h_chunk = torch.stack(chunk, dim=0)
                laminar_count += n
            else:
                t_chunk = t_eval[step:step + n + 1]
                h_chunk = odeint(ode_func, h_curr, t_chunk, method='rk4')[1:]
                turbulent_count += n

            results.append(h_chunk)
            h_curr = h_chunk[-1]
            step   = chunk_end

        self._last_laminar_frac = laminar_count / max(pred_steps, 1)
        return torch.cat(results, dim=0)

    @property
    def laminar_fraction(self) -> float:
        return getattr(self, '_last_laminar_frac', 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ELASTIC MANIFOLD (PATCHED: P4)
# ═══════════════════════════════════════════════════════════════════════════════

class ElasticAttractorODE(nn.Module):
    """
    AttractorODEFunc with HyperAgent-coupled dynamic radius R.

    Problem: fixed R = 1.0 works for smooth systems but fights discontinuous
    ones. The attractor sphere constrains amplitudes that physically must grow
    in an underdamped robotics system.

    Solution: R_eff(h) = R_base + δR(h) · event_prob(h).
    During smooth dynamics: event_prob ≈ 0, R_eff ≈ R_base (normal).
    During discontinuity: event_prob ≈ 1, sphere expands by δR.
    After the jump: event_prob drops, manifold tightens.

    PATCH P4: delta_r_net activation Tanh → GELU, hidden 16 → 32.
    During a discontinuous regime shift — the exact scenario this targets —
    Tanh gradient saturates for large inputs, suppressing the expansion
    signal when it matters most. GELU does not saturate the same way.
    """

    def __init__(self, hidden: int, alpha: float = 0.1, R_base: float = 1.0,
                 max_expansion: float = 5.0, use_fourier_skip: bool = True):
        super().__init__()
        self.alpha         = alpha
        self.R_base        = R_base
        self.max_expansion = max_expansion
        self.use_fourier_skip = use_fourier_skip

        # Core ODE terms (same as v22 AttractorODEFunc)
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

        # Event detector: initialized to output ≈ 0.05 (sparse by default)
        self.event_detector = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.event_detector[-2].bias, -3.0)

        # PATCH P4: Tanh → GELU, 16 → 32 units for better gradient flow
        # during rapid manifold expansion in discontinuous regimes
        self.delta_r_net = nn.Sequential(
            nn.Linear(hidden, 32),   # was 16 — PATCH P4
            nn.GELU(),               # was Tanh — PATCH P4
            nn.Linear(32, 1),        # was 16 — PATCH P4
            nn.Softplus(),
        )
        nn.init.zeros_(self.delta_r_net[0].weight)
        nn.init.zeros_(self.delta_r_net[0].bias)

    def _fourier_skip(self, h: torch.Tensor) -> torch.Tensor:
        freq  = self.fourier_freq.abs() + 0.01
        h_sin = torch.sin(h[..., :len(freq)] * freq)
        h_cos = torch.cos(h[..., :len(freq)] * freq)
        return self.fourier_proj(torch.cat([h_sin, h_cos], dim=-1))

    def effective_radius(self, h: torch.Tensor) -> torch.Tensor:
        """R_eff(h) = R_base + δR(h) · event_prob(h)"""
        event_prob = self.event_detector(h)
        delta_r    = self.delta_r_net(h).clamp(max=self.max_expansion)
        return self.R_base + delta_r * event_prob

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        L  = self.linear(h)
        N  = self.nonlinear(h)
        F_ = self._fourier_skip(h) if self.use_fourier_skip else 0.0

        R_eff    = self.effective_radius(h)
        norm_sq  = (h ** 2).sum(dim=-1, keepdim=True)
        attractor = -self.alpha * (norm_sq - R_eff ** 2) * h

        return L + N + F_ + attractor

    def get_diagnostics(self, h: torch.Tensor) -> dict:
        event_prob = self.event_detector(h)
        delta_r    = self.delta_r_net(h).clamp(max=self.max_expansion)
        R_eff      = self.R_base + delta_r * event_prob
        return {
            'event_prob': event_prob.mean().item(),
            'delta_r':    delta_r.mean().item(),
            'R_eff':      R_eff.mean().item(),
            'R_base':     self.R_base,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# V23 UNIFIED MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SynechismV23(nn.Module):
    """
    SynechismCore v23.0.1 — all patches applied.

    Three ablation flags allow isolating each component's contribution:
        use_elastic_manifold:   ElasticAttractorODE vs v22 AttractorODEFunc
        use_irrational_shutter: rk4+φ-steps vs dopri5+φ-eval-points
        use_laminar_bypass:     chunked bypass vs straight odeint
    """

    def __init__(
        self,
        in_dim:                 int,
        out_dim:                int,
        hidden:                 int   = 128,
        pred_steps:             int   = 20,
        alpha:                  float = 0.1,
        R_base:                 float = 1.0,
        use_ufo_encoder:        bool  = True,
        use_koopman:            bool  = True,
        use_fourier_skip:       bool  = True,
        use_hyperagent:         bool  = False,
        use_elastic_manifold:   bool  = True,
        use_irrational_shutter: bool  = True,
        use_laminar_bypass:     bool  = True,
        system:                 str   = 'default',
        phi_base:               float = PHI,
        laminar_threshold:      float = 0.05,
        laminar_chunk_size:     int   = 10,
        max_R_expansion:        float = 5.0,
        dt_scale:               float = 0.02,
    ):
        super().__init__()
        self.pred_steps  = pred_steps
        self.use_koopman = use_koopman
        self.use_agent   = use_hyperagent
        self.use_elastic = use_elastic_manifold
        self.use_shutter = use_irrational_shutter
        self.use_bypass  = use_laminar_bypass

        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from models import UFOEncoder, GRUEncoder, KoopmanLifting
        from hyperagent import HyperAgent, HyperAgentLoss

        # Encoder
        if use_ufo_encoder:
            _seq_len = 64 if system == 'ks_pde' else 50
            self.encoder = UFOEncoder(in_dim, hidden, seq_len=_seq_len)
        else:
            self.encoder = GRUEncoder(in_dim, hidden)

        self.saga    = nn.Parameter(torch.randn(hidden) * 0.01)
        self.h0_proj = nn.Linear(hidden * 2, hidden)

        if use_koopman:
            self.koopman = KoopmanLifting(hidden)

        # ODE function
        if use_elastic_manifold:
            self.ode_func = ElasticAttractorODE(
                hidden, alpha=alpha, R_base=R_base,
                max_expansion=max_R_expansion,
                use_fourier_skip=use_fourier_skip,
            )
        else:
            from models import AttractorODEFunc
            self.ode_func = AttractorODEFunc(
                hidden, alpha=alpha, R=R_base,
                use_fourier_skip=use_fourier_skip,
            )

        # Integration strategy
        if use_irrational_shutter:
            self.shutter = IrrationalShutter(phi_base=phi_base,
                                              dt_scale=dt_scale)
        else:
            from quantum_lattice import make_lattice
            self._lattice = make_lattice(system, phi_base)

        if use_laminar_bypass:
            self.bypass = LaminarBypass(
                hidden,
                threshold=laminar_threshold,
                chunk_size=laminar_chunk_size,
            )

        # HyperAgent
        if use_hyperagent:
            self.agent         = HyperAgent(hidden)
            self.agent_loss_fn = HyperAgentLoss()

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def encode(self, x: torch.Tensor):
        B = x.shape[0]
        z, mu, logvar = self.encoder(x)
        g  = self.saga.unsqueeze(0).expand(B, -1)
        h0 = torch.tanh(self.h0_proj(torch.cat([z, g], dim=-1)))
        if self.use_koopman:
            h0 = self.koopman(h0)
        return h0, mu, logvar

    def integrate(self, h0: torch.Tensor,
                  pred_steps: Optional[int] = None) -> torch.Tensor:
        if pred_steps is None:
            pred_steps = self.pred_steps

        # Get time evaluation points
        if self.use_shutter:
            t_eval = self.shutter.get_eval_times(pred_steps, device=h0.device)
        else:
            t_eval = self._lattice.get_full_eval_times(pred_steps, h0.device)
            t_eval = t_eval[:pred_steps + 1]

        # Integrate
        if self.use_bypass:
            h_traj = self.bypass.integrate(
                self.ode_func, h0, t_eval, pred_steps
            )
        else:
            if self.use_shutter:
                # IrrationalShutter requires rk4 (fixed-step solver)
                h_traj = odeint(self.ode_func, h0, t_eval, method='rk4')[1:]
            else:
                try:
                    h_traj = odeint(self.ode_func, h0, t_eval,
                                    method='dopri5', rtol=1e-3, atol=1e-4)[1:]
                except Exception:
                    h_traj = odeint(self.ode_func, h0, t_eval, method='rk4')[1:]

        return h_traj

    def forward(self, x: torch.Tensor,
                pred_steps: Optional[int] = None) -> Tuple:
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

    def get_integration_diagnostics(self) -> dict:
        diags = {}
        if self.use_bypass:
            diags['laminar_fraction'] = self.bypass.laminar_fraction
        return diags


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY: ablation variants
# ═══════════════════════════════════════════════════════════════════════════════

def make_v23(variant: str, in_dim: int, out_dim: int,
             hidden: int = 128, pred_steps: int = 20,
             system: str = 'default', **kwargs) -> SynechismV23:
    """
    Factory for v23 ablation variants. Each isolates one component.

    v22_baseline  — all v23 flags off (reproduces v22 integration behavior)
    shutter_only  — IrrationalShutter only (tests Claim 4 φ-vs-solver)
    elastic_only  — ElasticManifold only (tests Robotics recovery)
    bypass_only   — LaminarBypass only (measures efficiency)
    v23_full      — all three components
    v23_hybrid    — all three + HyperAgent (full discontinuous model)
    """
    base = dict(in_dim=in_dim, out_dim=out_dim, hidden=hidden,
                pred_steps=pred_steps, system=system, **kwargs)

    configs = {
        'v22_baseline':  dict(use_elastic_manifold=False,
                              use_irrational_shutter=False,
                              use_laminar_bypass=False,
                              use_hyperagent=False),
        'shutter_only':  dict(use_elastic_manifold=False,
                              use_irrational_shutter=True,
                              use_laminar_bypass=False,
                              use_hyperagent=False),
        'elastic_only':  dict(use_elastic_manifold=True,
                              use_irrational_shutter=False,
                              use_laminar_bypass=False,
                              use_hyperagent=False),
        'bypass_only':   dict(use_elastic_manifold=False,
                              use_irrational_shutter=False,
                              use_laminar_bypass=True,
                              use_hyperagent=False),
        'v23_full':      dict(use_elastic_manifold=True,
                              use_irrational_shutter=True,
                              use_laminar_bypass=True,
                              use_hyperagent=False),
        'v23_hybrid':    dict(use_elastic_manifold=True,
                              use_irrational_shutter=True,
                              use_laminar_bypass=True,
                              use_hyperagent=True),
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Options: {list(configs.keys())}")

    return SynechismV23(**base, **configs[variant])
