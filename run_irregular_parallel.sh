#!/usr/bin/env bash
# Full irregular-timing run on one GPU: one worker per timing regime, run side by side
# (the latent ODE is limited by kernel-launch latency, so 4 jobs share an A100 well).
# Usage: bash run_irregular_parallel.sh [--shutdown]
set -u
cd "$(dirname "$0")"
SEEDS="0 1 2 3 4"
MODELS="synechism synechism_nophi v22_baseline transformer lstm mamba"
mkdir -p results/irregular
for r in regular jitter exponential missing; do
  screen -dmS irr_$r bash -c "python3 -W ignore -u run_irregular.py --seeds $SEEDS --regimes $r --models $MODELS --out results/irregular/$r > results/irregular/log_$r.txt 2>&1; echo EXIT \$? >> results/irregular/log_$r.txt"
done
if [ "${1:-}" = "--shutdown" ]; then
  screen -dmS irr_shutdown bash -c 'sleep 60; while screen -ls | grep -q "irr_[rjem]"; do sleep 60; done; date > results/irregular/ALL_DONE.txt; sleep 120; sudo -n shutdown -h now'
fi
echo "started: $(screen -ls | grep -c irr_) screens"
