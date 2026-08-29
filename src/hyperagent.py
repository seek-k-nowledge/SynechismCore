"""
SynechismCore v19.0 — HyperAgent Module
========================================
HyperAgent is the discrete jump mechanism inside the hybrid ODE extension.

Mathematical role in the architecture:
    h_{t+1} = ODE_integrate(h_t, dt) + E(h_t) * [J(h_t) + C(h_t)]

Where:
    E(h_t) = event detector     <- HyperAgent.detect_event()
    J(h_t) = jump function      <- HyperAgent.compute_jump()
    C(h_t) = smooth correction  <- HyperAgent.compute_correction()

What HyperAgent solves:
    The base Synechism ODE uses attractor stabilization (-alpha*(||h||^2-R^2)*h)
    which constrains trajectories to a sphere of radius R. This is great for
    smooth continuous systems (KS PDE: 1.43x win) but actively harmful for
    systems with instantaneous discontinuities (robotics: 0.52x loss) because
    the physically correct trajectory must reach large amplitudes.

    HyperAgent detects when a discontinuity is about to occur and applies a
    discrete correction OUTSIDE the attractor constraint, allowing the system
    to handle both smooth and discontinuous dynamics.

Why "Darwin-Gödel" framing is misleading and what we actually build instead:
    The architecture doesn't "rewrite itself." What it does is learn to detect
    when the ODE's continuous assumptions break down and apply a learned
    discrete correction. This is well-defined, implementable, and honest.

Author: Paul E. Harris IV — SynechismCore v19.0
"""

import torch
import torch.nn as nn
import numpy as np


class EventDetector(nn.Module):
    """
    Detects whether the current hidden state is near a discontinuity.

    Output: scalar in [0, 1]
        ~0 = smooth dynamics, use ODE only
        ~1 = discontinuity detected, activate jump correction

    Architecture: small MLP with sigmoid output.
    Trained jointly with the rest of the model via the hybrid loss.
    """
    def __init__(self, hidden: int, bottleneck: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck // 2),
            nn.ReLU(),
            nn.Linear(bottleneck // 2, 1),
            nn.Sigmoid(),
        )
        # Initialize toward 0 (assume smooth by default)
        nn.init.constant_(self.net[-2].bias, -3.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: hidden state (B, hidden)
        Returns:
            event probability (B, 1) in [0, 1]
        """
        return self.net(h)


class JumpFunction(nn.Module):
    """
    Computes a discrete correction for instantaneous state transitions.

    This is the J(h_t) term: a learned mapping from pre-discontinuity
    hidden state to a correction vector.

    NOT constrained by the attractor term — it can output vectors of any
    magnitude, allowing the system to represent high-amplitude transitions.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden),
        )
        # Near-zero init: start with small corrections
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: hidden state (B, hidden)
        Returns:
            jump correction (B, hidden)
        """
        return self.net(h)


class SmoothCorrection(nn.Module):
    """
    Computes a smooth correction term C(h_t).

    This handles near-discontinuities — events that are rapid but not
    truly instantaneous. Examples: fast actuator deceleration, rapid
    regime transitions in finance.

    Unlike JumpFunction, this uses spectral norm to keep corrections
    bounded and smooth.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class HyperAgent(nn.Module):
    """
    Full HyperAgent module: event detection + jump + smooth correction.

    Implements the hybrid extension from the Synechism paper (Section 3.6):
        correction = E(h) * [J(h) + C(h)]

    Usage:
        agent = HyperAgent(hidden=128)
        correction = agent(h)               # Get full correction
        event_prob = agent.event_prob(h)    # Get event probability only

    The correction is added to the ODE-integrated hidden state:
        h_next = ode_integrate(h, dt) + agent(h)
    """
    def __init__(self, hidden: int, event_bottleneck: int = 32):
        super().__init__()
        self.detector   = EventDetector(hidden, event_bottleneck)
        self.jump       = JumpFunction(hidden)
        self.correction = SmoothCorrection(hidden)

    def event_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Returns event probability in [0, 1]. Shape: (B, 1)"""
        return self.detector(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute full hybrid correction.

        Args:
            h: hidden state (B, hidden)
        Returns:
            correction vector (B, hidden)
            When event_prob ~ 0: correction ~ 0 (pure ODE)
            When event_prob ~ 1: correction = J(h) + C(h)
        """
        e = self.detector(h)           # (B, 1)
        j = self.jump(h)               # (B, hidden)
        c = self.correction(h)         # (B, hidden)
        return e * (j + c)             # (B, hidden)

    def forward_with_diagnostics(self, h: torch.Tensor) -> dict:
        """
        Same as forward() but returns diagnostic information.
        Useful for understanding when the agent activates.
        """
        e = self.detector(h)
        j = self.jump(h)
        c = self.correction(h)
        correction = e * (j + c)
        return {
            'correction':   correction,
            'event_prob':   e,
            'jump_mag':     j.norm(dim=-1, keepdim=True),
            'smooth_mag':   c.norm(dim=-1, keepdim=True),
            'total_mag':    correction.norm(dim=-1, keepdim=True),
        }


class HyperAgentLoss(nn.Module):
    """
    Additional loss terms for training HyperAgent effectively.

    Without these, the event detector tends to either:
    (a) Always output 0 (never activates, agent is useless)
    (b) Always output 1 (always activates, destabilizes smooth dynamics)

    Two regularization terms prevent this:
    1. Sparsity: penalize high event_prob to encourage sparse activation
    2. Magnitude: penalize large corrections on smooth regions
    """
    def __init__(self, sparsity_weight: float = 0.01,
                 magnitude_weight: float = 0.001):
        super().__init__()
        self.sparsity_w  = sparsity_weight
        self.magnitude_w = magnitude_weight

    def forward(self, event_prob: torch.Tensor,
                correction: torch.Tensor) -> torch.Tensor:
        # Sparsity: event_prob should be mostly 0
        sparsity_loss = event_prob.mean()

        # Magnitude: corrections should be small (bounded by event gate)
        magnitude_loss = correction.norm(dim=-1).mean()

        return (self.sparsity_w * sparsity_loss +
                self.magnitude_w * magnitude_loss)
