"""
SynechismCore v20.0 — Quantum Lattice Module
=============================================
Quasi-random low-discrepancy sequences for ODE integration time points.

The core insight: naive uniform time sampling creates resonance artifacts
on fractal/chaotic attractors because uniform grids align with the attractor's
intrinsic frequencies. Low-discrepancy sequences (especially those based on
the golden ratio) are maximally aperiodic — they provably avoid these artifacts.

This is sometimes called a "quantum lattice" because the same mathematical
structure underlies quantum Monte Carlo integration and quantum error correction
lattice codes: the goal in all cases is to distribute sample points with
maximum uniformity and minimum correlation structure.

Three sequence types implemented:
    1. PhiLattice    — Weyl sequence: t_k = (k * phi) mod 1
                       Default. Optimal for 1D chaotic attractors (d < 3).
    2. FibonacciLattice — 2D generalization using consecutive Fibonacci ratios.
                          Better for high-dimensional systems (Lorenz-96, KS PDE).
    3. HaltonLattice — Multi-base Halton sequence for D-dimensional systems.
                       Most general. Slower to compute but most uniform.

HyEvo optimizes phi_base to find the best value for each specific system.
Default phi = 1.6180339... (golden ratio) is near-optimal for most cases.

Author: Paul E. Harris IV — SynechismCore v20.0
"""

import torch
import numpy as np
from typing import Optional

PHI  = (1.0 + 5.0 ** 0.5) / 2.0   # 1.6180339887...
PHI2 = PHI ** 2                     # 2.6180339887...


# ── Helper: van der Corput sequence ───────────────────────────────────────────

def van_der_corput(n: int, base: int) -> np.ndarray:
    """Van der Corput sequence in given base — building block for Halton."""
    seq = np.zeros(n)
    for i in range(n):
        x, denom, num = i + 1, 1, 0
        while x > 0:
            denom *= base
            num   += (x % base) / denom
            x     //= base
        seq[i] = num
    return seq


# ══════════════════════════════════════════════════════════════════════════════
# 1. PhiLattice — Golden Ratio Weyl Sequence
# ══════════════════════════════════════════════════════════════════════════════

class PhiLattice:
    """
    Weyl equidistribution sequence: t_k = (k * phi_base) mod 1.

    phi_base = golden ratio (default) gives the most uniform 1D distribution
    achievable with a linear recurrence (Weyl equidistribution theorem, 1916).

    Why phi and not e, sqrt(2), etc.?
    - phi has the slowest-converging continued fraction: [1; 1, 1, 1, ...]
    - This means it is the "most irrational" real number
    - A sequence based on phi avoids alignment with ANY rational frequency
    - This is exactly what we need to avoid resonance with chaotic attractors

    HyEvo can find a system-specific phi_base if the default is suboptimal.
    """
    def __init__(self, phi_base: float = PHI):
        self.phi_base = phi_base

    def get_times(self, n_steps: int, device=None) -> torch.Tensor:
        """
        Returns n_steps time points in (0, 1], sorted.
        Prepend 0 yourself if you want t=0 included.
        """
        pts = torch.FloatTensor(
            [(k * self.phi_base) % 1.0 for k in range(1, n_steps + 2)]
        )
        pts, _ = pts.sort()
        if device is not None:
            pts = pts.to(device)
        return pts

    def get_full_eval_times(self, n_steps: int, device=None) -> torch.Tensor:
        """Returns [0] + sorted phi-times — ready for torchdiffeq odeint."""
        pts = self.get_times(n_steps, device)
        zero = torch.zeros(1, device=pts.device)
        return torch.cat([zero, pts])


# ══════════════════════════════════════════════════════════════════════════════
# 2. FibonacciLattice — 2D Golden Ratio Lattice
# ══════════════════════════════════════════════════════════════════════════════

class FibonacciLattice:
    """
    2D Fibonacci lattice using two consecutive golden-ratio-derived bases.

    For high-dimensional systems (KS PDE: 64D, Weather L96: 40D), the 1D
    phi sequence isn't enough — we need coverage across multiple time scales.

    The Fibonacci lattice uses (phi, phi^2) as the two base dimensions,
    projecting down to 1D by taking a weighted sum. This achieves better
    coverage of the [0,1]^2 unit square than two independent phi sequences.

    Use this for systems with dim > 10.
    """
    def __init__(self, w1: float = 1.0, w2: float = 0.5):
        """
        w1, w2: weights for the two dimensions.
        Default (1.0, 0.5) gives a good 1D projection.
        """
        self.w1 = w1
        self.w2 = w2

    def get_times(self, n_steps: int, device=None) -> torch.Tensor:
        k = np.arange(1, n_steps + 2)
        d1 = (k * PHI)  % 1.0
        d2 = (k * PHI2) % 1.0
        pts = (self.w1 * d1 + self.w2 * d2) / (self.w1 + self.w2)
        pts = np.sort(pts)
        t = torch.FloatTensor(pts)
        if device is not None:
            t = t.to(device)
        return t

    def get_full_eval_times(self, n_steps: int, device=None) -> torch.Tensor:
        pts = self.get_times(n_steps, device)
        zero = torch.zeros(1, device=pts.device)
        return torch.cat([zero, pts])


# ══════════════════════════════════════════════════════════════════════════════
# 3. HaltonLattice — Multi-dimensional Quasi-Random Sequence
# ══════════════════════════════════════════════════════════════════════════════

class HaltonLattice:
    """
    Halton sequence: base-2 and base-3 van der Corput sequences combined.

    This is the most general low-discrepancy approach. It achieves O(log(N)^d / N)
    discrepancy vs O(1/sqrt(N)) for random sampling — meaning many fewer
    integration points are needed for the same accuracy.

    Computationally heavier than PhiLattice but strictly more uniform for d >= 2.
    Use for Lorenz-96 and KS PDE where spatial correlations matter most.
    """
    def __init__(self, base1: int = 2, base2: int = 3,
                 w1: float = 0.7, w2: float = 0.3):
        self.base1 = base1
        self.base2 = base2
        self.w1 = w1
        self.w2 = w2

    def get_times(self, n_steps: int, device=None) -> torch.Tensor:
        n = n_steps + 1
        d1 = van_der_corput(n, self.base1)
        d2 = van_der_corput(n, self.base2)
        pts = np.sort((self.w1 * d1 + self.w2 * d2) / (self.w1 + self.w2))
        # Ensure all values are in (0, 1]
        pts = np.clip(pts, 1e-6, 1.0)
        t = torch.FloatTensor(pts)
        if device is not None:
            t = t.to(device)
        return t

    def get_full_eval_times(self, n_steps: int, device=None) -> torch.Tensor:
        pts = self.get_times(n_steps, device)
        zero = torch.zeros(1, device=pts.device)
        return torch.cat([zero, pts])


# ══════════════════════════════════════════════════════════════════════════════
# Factory — picks best lattice for each system
# ══════════════════════════════════════════════════════════════════════════════

def make_lattice(system: str = 'default',
                 phi_base: Optional[float] = None) -> PhiLattice:
    """
    Factory that picks the best lattice type for each dynamical system.

    Args:
        system: one of 'lorenz', 'ks_pde', 'finance', 'weather', 'robotics', 'default'
        phi_base: override phi_base (from HyEvo if available)

    Returns:
        A lattice object with .get_full_eval_times(n_steps, device) method.
    """
    # High-dimensional spatial systems → Fibonacci or Halton
    if system in ('ks_pde', 'weather'):
        return FibonacciLattice()

    # 1D / low-D chaotic systems → PhiLattice with optional HyEvo base
    base = phi_base if phi_base is not None else PHI
    return PhiLattice(phi_base=base)


# ── Discrepancy diagnostic ─────────────────────────────────────────────────

def star_discrepancy(pts: np.ndarray) -> float:
    """
    Compute the star discrepancy D* of a sequence in [0,1].
    D* measures how far the empirical distribution is from uniform.
    Lower = more uniform.

    phi lattice D* ~ log(N)/N
    random D*     ~ 1/sqrt(N)
    uniform grid  ~ 1/(2N)    (best possible but has resonance)
    """
    n = len(pts)
    pts_sorted = np.sort(pts)
    upper = max((i+1)/n - pts_sorted[i] for i in range(n))
    lower = max(pts_sorted[i] - i/n     for i in range(n))
    return max(upper, lower)


def compare_lattices(n_steps: int = 50) -> dict:
    """
    Compare discrepancy of all three lattice types vs uniform and random.
    Returns a dict of {name: D*}.
    """
    phi  = PhiLattice()
    fib  = FibonacciLattice()
    hal  = HaltonLattice()

    uniform = np.linspace(1/n_steps, 1.0, n_steps)
    random  = np.sort(np.random.rand(n_steps))

    return {
        'phi_lattice':       star_discrepancy(phi.get_times(n_steps).numpy()),
        'fibonacci_lattice': star_discrepancy(fib.get_times(n_steps).numpy()),
        'halton_lattice':    star_discrepancy(hal.get_times(n_steps).numpy()),
        'uniform_grid':      star_discrepancy(uniform),
        'random':            star_discrepancy(random),
    }


if __name__ == '__main__':
    print("Quantum Lattice Discrepancy Comparison (n=50 steps)")
    print("-" * 50)
    results = compare_lattices(50)
    for name, d in sorted(results.items(), key=lambda x: x[1]):
        bar = '█' * int(d * 500)
        print(f"  {name:<22} D*={d:.4f}  {bar}")
    print("\nLower D* = more uniform = less resonance risk")
    print(f"\nDefault phi_base = {PHI:.10f} (golden ratio)")
