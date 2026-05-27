#!/usr/bin/env python3
"""48-layer Sprengholz discovery on GPU."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.gpu_lp import create_gpu_lp_solver, batch_lp_gpu
from src.lp_functions import _z2_z3, compute_lp_rmse
from src.test_cases import LP_SPRENGHOLZ_48
from src.numba_solver import optimize_laminate_numba

N = 48
Z2, Z3 = _z2_z3(N)

print("Creating 48-layer GPU LP solver...", flush=True)
dev, mod = create_gpu_lp_solver(N, Z2, Z3)
print("  device OK", flush=True)

# GPU-accelerated 48-layer discovery
target_lp = LP_SPRENGHOLZ_48
rmse_threshold = 2e-2  # From previous experiments

from scipy.stats.qmc import Sobol
sampler = Sobol(d=N, scramble=True, seed=42)

found = set()
t_start = time.perf_counter()

# Phase 1: GPU batch evaluation of Sobol starts
max_starts = 50000
top_k = 2000
batch_size = 10000

all_starts = []
all_losses = []
starts_used = 0

while starts_used < max_starts:
    remaining = min(batch_size, max_starts - starts_used)
    new_lams = sampler.random(remaining).astype(np.float32) * np.pi - np.pi / 2
    starts_used += remaining
    
    lps = batch_lp_gpu(dev, mod, new_lams, N)
    losses = np.sum((lps - target_lp) ** 2, axis=1)
    
    all_starts.append(new_lams)
    all_losses.append(losses)

all_starts = np.vstack(all_starts)
all_losses = np.concatenate(all_losses)
t_gpu = time.perf_counter() - t_start
print(f"GPU phase: {t_gpu:.2f}s, {starts_used} starts evaluated", flush=True)

# Phase 2: CPU iRprop on top-K
top_indices = np.argsort(all_losses)[:top_k]
t_irprop_start = time.perf_counter()

for i, idx in enumerate(top_indices):
    lam = all_starts[idx:idx+1]
    opt, losses = optimize_laminate_numba(
        lam, target_lp, n_coarse_fine=2, irprop_grad_tol=1e-4
    )
    best_loss = float(losses[0])
    
    if best_loss < rmse_threshold:
        key = tuple(round(a, 1) for a in opt[0])
        found.add(key)
    
    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_start
        print(f"  [{i+1}/{top_k}] {len(found)} unique solutions, {elapsed:.1f}s", flush=True)

t_total = time.perf_counter() - t_start
print(f"\nTOTAL: {t_total:.2f}s, {len(found)} unique solutions from {starts_used} starts (top {top_k} refined)", flush=True)
print(f"METRIC sprengholz_solutions={len(found)} sprengholz_starts={top_k}", flush=True)