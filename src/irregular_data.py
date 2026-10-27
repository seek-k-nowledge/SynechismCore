"""
Irregularly sampled Lorenz-63 data for the irregular-timing study.

The system is integrated once with a high-accuracy solver (dense output), then
observed at times drawn from one of four sampling regimes. Every regime has the
same mean gap (mean_dt), so they differ only in how uneven the timing is:

    regular      every gap = mean_dt                      (control)
    jitter       gap = mean_dt * U(0.5, 1.5)              (mildly uneven)
    exponential  gap = mean_dt * Exp(1), floored at 0.1*mean_dt   (very uneven)
    missing      a grid of mean_dt/2 with each point kept with p = 0.5
                 (gaps are multiples of mean_dt/2; half the readings are missing)

Windows: each example is the last `seq_len` observations (values and the gap
before each one), the gaps to the next `pred_len` observation times, and the
values at those times.
"""

import numpy as np
from scipy.integrate import solve_ivp

REGIMES = ('regular', 'jitter', 'exponential', 'missing')
LORENZ63_LYAPUNOV_EXPONENT = 0.906           # largest exponent, rho = 28
LYAPUNOV_TIME = 1.0 / LORENZ63_LYAPUNOV_EXPONENT   # ~1.104 time units


def lorenz63_rhs(t, s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def sample_gaps(n, regime, mean_dt, rng):
    """Return n gaps (>0) whose mean is mean_dt, for the given regime."""
    if regime == 'regular':
        return np.full(n, mean_dt)
    if regime == 'jitter':
        return mean_dt * rng.uniform(0.5, 1.5, n)
    if regime == 'exponential':
        g = mean_dt * rng.exponential(1.0, n)
        return np.maximum(g, 0.1 * mean_dt)
    if regime == 'missing':
        # grid of mean_dt/2, keep each point with probability 0.5 -> geometric gaps
        k = rng.geometric(0.5, n)            # 1, 2, 3, ... grid steps
        return k * (mean_dt / 2.0)
    raise ValueError(f"unknown regime '{regime}', choose from {REGIMES}")


def generate_irregular_lorenz(n_obs, regime, seed, mean_dt=0.02, rho=28.0,
                              warmup_time=40.0):
    """
    Integrate Lorenz-63 accurately and observe it at irregular times.

    Returns (times, values): times (n_obs,) starting at 0, values (n_obs, 3).
    """
    rng = np.random.default_rng(seed)
    s0 = rng.normal(0.0, 1.0, 3) + np.array([1.0, 1.0, 20.0])
    warm = solve_ivp(lorenz63_rhs, (0.0, warmup_time), s0, args=(10.0, rho),
                     method='DOP853', rtol=1e-10, atol=1e-12)
    s_start = warm.y[:, -1]

    gaps = sample_gaps(n_obs - 1, regime, mean_dt, rng)
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    sol = solve_ivp(lorenz63_rhs, (0.0, times[-1]), s_start, args=(10.0, rho),
                    method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    values = sol.sol(times).T
    return times, values.astype(np.float64)


def make_windows(times, values, seq_len=50, pred_len=20, stride=3,
                 start=0, end=None):
    """
    Cut (history, future) windows from one irregular trajectory.

    Returns dict of float32 arrays:
        x_hist   (N, seq_len, 3)    values of the last seq_len observations
        dt_hist  (N, seq_len, 1)    gap before each history observation
        dt_fut   (N, pred_len)      gaps between consecutive future times
                                    (the first is from the last history time)
        y_fut    (N, pred_len, 3)   values at the future times
    """
    end = len(times) if end is None else end
    gaps = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    xs, dh, df, ys = [], [], [], []
    for s in range(max(start, 1), end - seq_len - pred_len + 1, stride):
        h = slice(s, s + seq_len)
        f = slice(s + seq_len, s + seq_len + pred_len)
        xs.append(values[h])
        dh.append(gaps[h, None])
        df.append(gaps[f])
        ys.append(values[f])
    return {
        'x_hist':  np.asarray(xs, np.float32),
        'dt_hist': np.asarray(dh, np.float32),
        'dt_fut':  np.asarray(df, np.float32),
        'y_fut':   np.asarray(ys, np.float32),
    }
