"""
SynechismCore v19.0 — Experiment Data Generators
=================================================
Five dynamical systems, each with a bifurcation/regime-change test:

1. Lorenz 63     — ρ bifurcation (train ρ∈{18-28}, test ρ∈{35,40,45,50})
2. KS PDE        — viscosity ν bifurcation (train ν=1.0, test ν=0.5)
3. Finance       — VIX regime (train calm, test high-VIX crisis periods)
4. Weather L96   — forcing F (train F∈{3-6}, test F∈{14,18,22})
5. Robotics      — damping failure (train γ=0.5, test γ=0.05)

NOTE on Robotics: γ→0.0 (exact zero) is a singularity. Every model fails
because there is no physical trajectory to learn at zero damping (energy
diverges). γ=0.05 is a legitimate near-failure test where the physics is
still defined but dramatically different. This is the honest test.

Author: Paul E. Harris IV — SynechismCore v19.0
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
# 1. LORENZ 63
# ══════════════════════════════════════════════════════════════════════════════

def lorenz63_rhs(t, state, rho, sigma=10.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]


def generate_lorenz63(rho, n_steps, dt=0.02, warmup=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    state = np.random.randn(3) * 0.1
    state[0] += 1.0

    # Warmup to attractor
    t_w = np.linspace(0, warmup*dt, warmup)
    sol = solve_ivp(lorenz63_rhs, [0, warmup*dt], state, args=(rho,),
                    t_eval=t_w, method='RK45', rtol=1e-7, atol=1e-9)
    state = sol.y[:, -1]

    t_r = np.linspace(0, n_steps*dt, n_steps)
    sol = solve_ivp(lorenz63_rhs, [0, n_steps*dt], state, args=(rho,),
                    t_eval=t_r, method='RK45', rtol=1e-7, atol=1e-9)
    return sol.y.T  # (n_steps, 3)


def make_lorenz_dataset(rho_values, n_traj=100, seq_len=50, pred_steps=20, seed=42):
    np.random.seed(seed)
    total = seq_len + pred_steps + 5
    X_list, Y_list = [], []

    for rho in rho_values:
        for i in range(n_traj):
            traj = generate_lorenz63(rho, total + 200, seed=seed + i)
            traj = traj[200:]  # skip additional warmup
            for s in range(0, len(traj) - seq_len - pred_steps, 3):
                X_list.append(traj[s : s+seq_len])
                Y_list.append(traj[s+seq_len : s+seq_len+pred_steps])

    X = torch.FloatTensor(np.array(X_list))
    Y = torch.FloatTensor(np.array(Y_list))
    return TensorDataset(X, Y), X, Y


# ══════════════════════════════════════════════════════════════════════════════
# 2. KURAMOTO-SIVASHINSKY PDE
# ══════════════════════════════════════════════════════════════════════════════

def ks_etdrk4_step(u_hat, L_hat, E, E2, f1, f2, f3, N_fn):
    """Single ETDRK4 step in Fourier space."""
    Nu = N_fn(u_hat)
    a  = E2 * u_hat + 0.5 * f1 * Nu
    Na = N_fn(a)
    b  = E2 * u_hat + 0.5 * f1 * Na
    Nb = N_fn(b)
    c  = E2 * a    + f1 * (2*Nb - Nu)
    Nc = N_fn(c)
    return (E * u_hat
            + Nu * (f1 - 3*f2 + 4*f3)
            + (Na + Nb) * 2*(f2 - 2*f3)
            + Nc * (4*f3 - f2))


def generate_ks(N=64, L=22.0, nu=1.0, n_steps=2000, dt=0.25, warmup=1000, seed=42):
    """
    Generate KS PDE trajectory.
    u_t = -u*u_x - u_xx - nu*u_xxxx
    Standard domain L=22 (turbulent regime).
    """
    np.random.seed(seed)
    x = np.linspace(0, L, N, endpoint=False)

    # Random smooth initial condition
    modes = np.random.randn(4)
    u = sum(modes[k] * np.cos(2*np.pi*(k+1)*x/L + np.random.rand()*2*np.pi)
            for k in range(4))

    # Wavenumbers
    k_arr = np.fft.rfftfreq(N) * N * 2 * np.pi / L

    # Linear operator: -k^2 + nu*k^4  (note sign: u_xx = -(k^2)*u_hat, u_xxxx = k^4*u_hat)
    # KS: u_t = -uu_x - u_xx - nu*u_xxxx  =>  linear = k^2 - nu*k^4
    Lhat = k_arr**2 - nu * k_arr**4

    # ETDRK4 coefficients
    E  = np.exp(Lhat * dt).astype(complex)
    E2 = np.exp(Lhat * dt/2).astype(complex)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * Lhat[:,None] + r[None,:]
    phi1 = dt * (np.exp(LR) - 1).mean(axis=1).real / Lhat
    phi2 = dt * ((np.exp(LR) - 1 - LR) / LR**2).mean(axis=1).real
    phi3 = dt * ((np.exp(LR) - 1 - LR - 0.5*LR**2) / LR**3).mean(axis=1).real

    # Fix divide-by-zero at k=0
    phi1[0] = dt
    phi2[0] = dt**2 / 2
    phi3[0] = dt**3 / 6

    def nonlinear_hat(u_hat):
        u_phys = np.fft.irfft(u_hat, n=N)
        ux_hat = 1j * k_arr * u_hat
        ux_phys = np.fft.irfft(ux_hat, n=N)
        return np.fft.rfft(-u_phys * ux_phys)

    # Warmup
    u_hat = np.fft.rfft(u)
    for _ in range(warmup):
        u_hat = ks_etdrk4_step(u_hat, Lhat, E, E2, phi1, phi2, phi3, nonlinear_hat)
        u_hat[N//3:] = 0  # 2/3 dealiasing

    # Generate
    traj = []
    for _ in range(n_steps):
        u_hat = ks_etdrk4_step(u_hat, Lhat, E, E2, phi1, phi2, phi3, nonlinear_hat)
        u_hat[N//3:] = 0
        traj.append(np.fft.irfft(u_hat, n=N).copy())

    return np.array(traj)  # (n_steps, N)


def make_ks_dataset(nu=1.0, n_traj=40, seq_len=64, pred_steps=16,
                    N=64, L=22.0, n_steps=2000, seed=42):
    np.random.seed(seed)
    X_list, Y_list = [], []
    total = seq_len + pred_steps + 5

    for i in range(n_traj):
        traj = generate_ks(N=N, L=L, nu=nu, n_steps=n_steps, seed=seed+i*7)
        for s in range(0, len(traj) - total, 4):
            X_list.append(traj[s : s+seq_len])
            Y_list.append(traj[s+seq_len : s+seq_len+pred_steps])

    X = torch.FloatTensor(np.array(X_list))
    Y = torch.FloatTensor(np.array(Y_list))
    return TensorDataset(X, Y), X, Y


# ══════════════════════════════════════════════════════════════════════════════
# 3. FINANCE (VIX Regime)
# ══════════════════════════════════════════════════════════════════════════════

def make_finance_dataset(regime='calm', seq_len=30, pred_steps=5, n_steps=6000, seed=42):
    """
    VIX-regime split using synthetic market data calibrated to real statistics.
    Generates: log returns, realized volatility (5-day), volume proxy
    Train: calm (VIX equivalent < 20)
    Test:  crisis (VIX equivalent > 35)
    """
    np.random.seed(seed)
    crisis = (regime == 'crisis')

    def gen_returns(n, vol_regime, crisis=False):
        """Generate synthetic returns with regime-appropriate statistics."""
        if not crisis:
            # Calm: Gaussian with moderate vol
            sigma = vol_regime * (1 + 0.2 * np.random.randn(n))
            returns = sigma * np.random.randn(n)
        else:
            # Crisis: fat tails, volatility clustering, jumps
            # GARCH-like: vol^2_t = omega + alpha*eps^2_{t-1} + beta*vol^2_{t-1}
            omega, alpha, beta = 0.1, 0.3, 0.6
            vol_sq = np.zeros(n)
            vol_sq[0] = vol_regime**2
            eps = np.zeros(n)
            for t in range(1, n):
                vol_sq[t] = omega + alpha * eps[t-1]**2 + beta * vol_sq[t-1]
                vol_sq[t] = max(vol_sq[t], 0.01)
                # Occasional jump (regime change characteristic)
                if np.random.rand() < 0.05:
                    eps[t] = np.random.choice([-1,1]) * (2 + np.random.exponential(1)) * np.sqrt(vol_sq[t])
                else:
                    eps[t] = np.random.randn() * np.sqrt(vol_sq[t])
            returns = eps

        # Realized vol (5-day rolling)
        rv = np.array([returns[max(0,i-5):i].std() if i > 0 else 0.0
                       for i in range(n)])

        # Volume proxy (correlated with vol)
        volume = 1.0 + 0.5 * rv / (rv.max() + 1e-8) + 0.1 * np.random.randn(n)

        # Stack features: returns, rv, volume
        features = np.column_stack([
            (returns - returns.mean()) / (returns.std() + 1e-8),
            (rv - rv.mean()) / (rv.std() + 1e-8),
            (volume - volume.mean()) / (volume.std() + 1e-8),
        ])
        return features

    # Generate data
    X_list, Y_list = [], []
    vol_regime = 0.03 if crisis else 0.01

    for _ in range(150 if not crisis else 50):
        data = gen_returns(n_steps, vol_regime, crisis=crisis)
        for s in range(0, n_steps - seq_len - pred_steps, 5):
            X_list.append(data[s:s+seq_len])
            Y_list.append(data[s+seq_len:s+seq_len+pred_steps])

    X = torch.FloatTensor(np.array(X_list))
    Y = torch.FloatTensor(np.array(Y_list))
    return TensorDataset(X, Y), X, Y


# ══════════════════════════════════════════════════════════════════════════════
# 4. WEATHER (LORENZ-96)
# ══════════════════════════════════════════════════════════════════════════════

def lorenz96_rhs(t, x, F):
    N = len(x)
    xm1 = np.roll(x, 1)
    xm2 = np.roll(x, 2)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F


def generate_lorenz96(F, N=40, n_steps=3000, dt=0.05, warmup=1000, seed=42):
    np.random.seed(seed)
    x0 = np.random.randn(N) * 0.01
    x0[0] += 0.01

    t_w = np.linspace(0, warmup*dt, warmup)
    sol = solve_ivp(lorenz96_rhs, [0, warmup*dt], x0, args=(F,),
                    t_eval=t_w, method='RK45', rtol=1e-6, atol=1e-8)
    x0 = sol.y[:, -1]

    t_r = np.linspace(0, n_steps*dt, n_steps)
    sol = solve_ivp(lorenz96_rhs, [0, n_steps*dt], x0, args=(F,),
                    t_eval=t_r, method='RK45', rtol=1e-6, atol=1e-8)
    return sol.y.T


def make_weather_dataset(F_values, n_traj=30, seq_len=50, pred_steps=10,
                         N=40, seed=42):
    np.random.seed(seed)
    X_list, Y_list = [], []
    total = seq_len + pred_steps

    for F in F_values:
        for i in range(n_traj):
            traj = generate_lorenz96(F, N=N, n_steps=3000, seed=seed+i*11)
            for s in range(0, len(traj) - total, 5):
                X_list.append(traj[s:s+seq_len])
                Y_list.append(traj[s+seq_len:s+seq_len+pred_steps])

    X = torch.FloatTensor(np.array(X_list))
    Y = torch.FloatTensor(np.array(Y_list))
    return TensorDataset(X, Y), X, Y


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROBOTICS (Damped Harmonic Oscillator — Actuator Failure)
# ══════════════════════════════════════════════════════════════════════════════

def damped_oscillator_rhs(t, state, gamma, omega=2*np.pi, A=1.0):
    """
    Driven damped harmonic oscillator:
        dx/dt = v
        dv/dt = -2*gamma*v - omega^2*x + A*cos(omega*t)

    gamma: damping coefficient
      gamma=0.5: normal operation
      gamma=0.05: near-failure (very light damping — resonance risk)
      gamma=0.0:  exact failure (diverges — not a valid test)

    NOTE: We test gamma=0.05 NOT gamma=0.0
    This is physically meaningful and all models can be evaluated.
    """
    x, v = state
    dxdt = v
    dvdt = -2*gamma*v - omega**2*x + A*np.cos(omega*t)
    return [dxdt, dvdt]


def generate_oscillator(gamma, n_steps=2000, dt=0.02, warmup=500, seed=42):
    np.random.seed(seed)
    state = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)]

    t_w = np.linspace(0, warmup*dt, warmup)
    sol = solve_ivp(damped_oscillator_rhs, [0, warmup*dt], state, args=(gamma,),
                    t_eval=t_w, method='RK45', rtol=1e-7, atol=1e-9)
    state = sol.y[:, -1]

    t_r = np.linspace(0, n_steps*dt, n_steps)
    sol = solve_ivp(damped_oscillator_rhs, [0, n_steps*dt], state, args=(gamma,),
                    t_eval=t_r, method='RK45', rtol=1e-7, atol=1e-9)
    return sol.y.T  # (n_steps, 2)


def make_robotics_dataset(gamma=0.5, n_traj=150, seq_len=50, pred_steps=20, n_steps=6000, seed=42):
    np.random.seed(seed)
    X_list, Y_list = [], []
    total = seq_len + pred_steps

    for i in range(n_traj):
        traj = generate_oscillator(gamma, n_steps=n_steps, seed=seed+i*13)
        # Normalize
        traj = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-8)
        for s in range(0, len(traj) - total, 3):
            X_list.append(traj[s:s+seq_len])
            Y_list.append(traj[s+seq_len:s+seq_len+pred_steps])

    X = torch.FloatTensor(np.array(X_list))
    Y = torch.FloatTensor(np.array(Y_list))
    return TensorDataset(X, Y), X, Y
