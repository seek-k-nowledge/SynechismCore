"""
Irregular-timing study: does a continuous-time latent ODE keep chaotic forecasts
accurate longer than Transformer, LSTM and Mamba forecasters when observations
arrive at uneven times?

For each seed and each sampling regime (regular, jitter, exponential, missing):
  1. Generate an irregularly sampled Lorenz-63 training trajectory and a separate
     test trajectory (src/irregular_data.py).
  2. Train every model on (50 past observations + their gaps) -> (values at the
     next 20 observation times, given those times).
  3. From many start points on the test trajectory, roll each model forward on its
     own predictions (the future observation TIMES are known, the values are not),
     and record the valid prediction time (VPT): the elapsed physical time until the
     normalized error first exceeds a threshold, in Lyapunov times. A run that never
     crosses within the horizon is censored at the horizon.

Usage (GPU):
  python -u run_irregular.py --seeds 0 1 2 3 4 2>&1 | tee run_log_irregular.txt
Smoke test:
  python -u run_irregular.py --seeds 0 --epochs 3 --n-starts 4 --horizon-lt 2 --smoke
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from irregular_data import (REGIMES, LYAPUNOV_TIME, generate_irregular_lorenz,  # noqa: E402
                            make_windows)
from irregular_models import MODEL_NAMES, make_model                             # noqa: E402

LR = {'synechism': 1e-3, 'synechism_nophi': 1e-3, 'v22_baseline': 1e-3,
      'transformer': 6e-4, 'lstm': 1e-3, 'mamba': 5e-4}
THRESHOLDS = (0.3, 0.4)
SEQ_LEN, PRED_LEN = 50, 20


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_t(arr, device):
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def run_model(model, x, dh, df):
    out = model(x, dh, df)
    return out if isinstance(out, tuple) else (out, None, None, None)


def train(model, name, data, epochs, device, batch_size=64, beta=5e-4, warmup=30):
    opt = torch.optim.Adam(model.parameters(), lr=LR[name])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    X, DH, DF, Y = (to_t(data[k], device) for k in ('x_hist', 'dt_hist', 'dt_fut', 'y_fut'))
    n = X.shape[0]
    model.train()
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred, mu, logvar, h0 = run_model(model, X[idx], DH[idx], DF[idx])
            loss = nn.functional.mse_loss(pred, Y[idx])
            if getattr(model, 'is_latent_ode', False):
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + beta * min(1.0, ep / max(1, warmup)) * kl
                loss = loss + model.extra_loss(h0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        losses.append(total / n)
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      [{name}] epoch {ep + 1:>3}/{epochs}  loss {total / n:.5f}", flush=True)
    model.eval()
    return losses


@torch.no_grad()
def rollout_vpt(model, times, vals, starts, horizon, sigma, device):
    """
    Roll the model forward from each start index on its own predictions.
    Returns VPT (in Lyapunov times) per threshold, censored flags, and the
    per-observation error curve (elapsed time, error) for every start.
    """
    gaps = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    n_need = max(int(np.searchsorted(times, times[s - 1] + horizon, side='right') - s + 1)
                 for s in starts)
    n_chunks = math.ceil(n_need / PRED_LEN)
    hist_x = to_t(np.stack([vals[s - SEQ_LEN:s] for s in starts]), device)
    hist_dt = to_t(np.stack([gaps[s - SEQ_LEN:s, None] for s in starts]), device)
    preds = []
    for c in range(n_chunks):
        idx0 = [s + c * PRED_LEN for s in starts]
        dfut = to_t(np.stack([gaps[i:i + PRED_LEN] for i in idx0]), device)
        p, *_ = run_model(model, hist_x, hist_dt, dfut)
        preds.append(p)
        hist_x = torch.cat([hist_x[:, PRED_LEN:], p], dim=1)
        hist_dt = torch.cat([hist_dt[:, PRED_LEN:], dfut.unsqueeze(-1)], dim=1)
    preds = torch.cat(preds, dim=1).cpu().numpy()            # (n_starts, n_chunks*K, 3)

    out = {f'vpt_{th}': [] for th in THRESHOLDS}
    out.update({f'censored_{th}': [] for th in THRESHOLDS})
    curves = []
    for j, s in enumerate(starts):
        L = preds.shape[1]
        truth = vals[s:s + L]
        elapsed = times[s:s + L] - times[s - 1]
        keep = elapsed <= horizon
        err = np.sqrt(np.mean(((preds[j, :len(truth)] - truth) / sigma) ** 2, axis=-1))
        err = np.where(np.isfinite(err), err, np.inf)[keep]
        el = elapsed[keep]
        curves.append(np.stack([el, err]).astype(np.float32))
        for th in THRESHOLDS:
            hit = np.nonzero(err > th)[0]
            if hit.size:
                out[f'vpt_{th}'].append(float(el[hit[0]] / LYAPUNOV_TIME))
                out[f'censored_{th}'].append(False)
            else:
                out[f'vpt_{th}'].append(float(horizon / LYAPUNOV_TIME))
                out[f'censored_{th}'].append(True)
    return out, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--models', nargs='+', default=list(MODEL_NAMES))
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--n-train-obs', type=int, default=6000)
    ap.add_argument('--n-starts', type=int, default=30)
    ap.add_argument('--horizon-lt', type=float, default=10.0,
                    help='rollout horizon in Lyapunov times')
    ap.add_argument('--out', default='results/irregular')
    ap.add_argument('--smoke', action='store_true', help='tag output as a smoke test')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    horizon = args.horizon_lt * LYAPUNOV_TIME
    tag = 'smoke' if args.smoke else 'main'
    os.makedirs(os.path.join(args.out, 'curves'), exist_ok=True)
    print(f"Device: {device}  seeds={args.seeds}  regimes={args.regimes}\n"
          f"models={args.models}  epochs={args.epochs}  starts={args.n_starts}  "
          f"horizon={args.horizon_lt} Lyapunov times", flush=True)

    results = []
    t_start = time.time()
    for seed in args.seeds:
        for ri, regime in enumerate(args.regimes):
            print(f"\n=== seed {seed} | regime {regime} ===", flush=True)
            tr_t, tr_v = generate_irregular_lorenz(args.n_train_obs, regime, seed=1000 * seed + ri)
            n_test = int(args.n_starts * 60 + 3 * horizon / 0.02 + SEQ_LEN + 100)
            te_t, te_v = generate_irregular_lorenz(n_test, regime, seed=1000 * seed + ri + 500)
            mu, sd = tr_v.mean(0), tr_v.std(0)
            tr_n, te_n = (tr_v - mu) / sd, (te_v - mu) / sd
            sigma = tr_n.std(0)
            data = make_windows(tr_t, tr_n, SEQ_LEN, PRED_LEN, stride=3)
            last_start = int(np.searchsorted(te_t, te_t[-1] - 1.6 * horizon)) - 1
            starts = np.linspace(SEQ_LEN + 1, last_start, args.n_starts).astype(int).tolist()
            for name in args.models:
                set_seed(seed)
                model = make_model(name, SEQ_LEN, PRED_LEN).to(device)
                t0 = time.time()
                losses = train(model, name, data, args.epochs, device)
                vpt, curves = rollout_vpt(model, te_t, te_n, starts, horizon, sigma, device)
                np.savez_compressed(os.path.join(args.out, 'curves',
                                                 f'{tag}_{regime}_{name}_seed{seed}.npz'),
                                    *curves)
                med = {th: float(np.median(vpt[f'vpt_{th}'])) for th in THRESHOLDS}
                ncen = {th: int(sum(vpt[f'censored_{th}'])) for th in THRESHOLDS}
                print(f"    {name:16s} median VPT(0.4) {med[0.4]:5.2f} LT  "
                      f"VPT(0.3) {med[0.3]:5.2f} LT  censored {ncen[0.4]}/{len(starts)}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
                results.append({'seed': seed, 'regime': regime, 'model': name,
                                'n_params': sum(p.numel() for p in model.parameters()),
                                'train_loss': losses, **vpt})
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            with open(os.path.join(args.out, f'{tag}_results.json'), 'w') as f:
                json.dump({'args': vars(args), 'lyapunov_time': LYAPUNOV_TIME,
                           'results': results}, f)

    print("\n=== Summary: median VPT(0.4) in Lyapunov times, pooled over seeds and starts ===")
    print(f"  {'model':16s}" + ''.join(f"{r:>13s}" for r in args.regimes))
    for name in args.models:
        row = []
        for r in args.regimes:
            v = [x for res in results if res['model'] == name and res['regime'] == r
                 for x in res['vpt_0.4']]
            row.append(f"{np.median(v):13.2f}" if v else f"{'-':>13s}")
        print(f"  {name:16s}" + ''.join(row))
    print(f"\nSaved: {os.path.join(args.out, tag + '_results.json')}  "
          f"total time {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
