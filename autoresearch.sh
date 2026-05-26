#!/bin/bash
# Auto-research benchmark script.
# Runs the LP solver benchmark on Google Colab and outputs the METRIC line.
# The agent modifies this script if the benchmark command changes.

set -euo pipefail

cd "$(dirname "$0")"

# ── Run the benchmark on Colab ──
# The colab_exec.py manages a persistent T4 GPU session.
# We run the colab_benchmark.py in --benchmark-only mode (pipeline only, no tests).
python3 colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && python3 -c \"
import sys, os, time, json
sys.path.insert(0, '.')
import numpy as np
from src.numpy_solver import optimize_laminate
from src.lp_functions import compute_lp_rmse
from src.test_cases import LP_VIQUERAT

# Stable benchmark: Viquerat 12-layer, 30 starts, 3 repeats
rng = np.random.default_rng(42)
N = 12
n_starts = 30
n_repeats = 3
times = []

for rep in range(n_repeats):
    rand_lams = rng.random((n_starts, N), dtype=np.float32) * np.pi - np.pi / 2
    t0 = time.perf_counter()
    opt, losses = optimize_laminate(rand_lams, LP_VIQUERAT)
    dt = time.perf_counter() - t0
    times.append(dt)

mean_time = sum(times) / len(times)
best_rmse = float(np.min(losses))

# Output structured metric line
print(f'METRIC solve_time={mean_time:.4f}')
print(f'METRIC best_rmse={best_rmse:.2e}')
# Additional diagnostics (not primary metrics)
print(f'METRIC worst_rmse={float(np.max(losses)):.2e}')
print(f'METRIC median_rmse={float(np.median(losses)):.2e}')
\""
