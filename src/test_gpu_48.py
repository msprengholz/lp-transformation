#!/usr/bin/env python3
"""Test 48-layer GPU LP computation."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.gpu_lp import create_gpu_lp_solver, batch_lp_gpu
from src.lp_functions import _z2_z3, _norm_factors
from src.numpy_fast import get_lp_batch

# 48-layer test
N = 48
Z2, Z3 = _z2_z3(N)

print("Creating 48-layer GPU LP solver...", flush=True)
dev, mod = create_gpu_lp_solver(N, Z2, Z3)
print("  device OK", flush=True)

# Correctness test
M = 100
lams48 = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2

# GPU
t0 = time.perf_counter()
gpu_lp = batch_lp_gpu(dev, mod, lams48, N)
t_gpu = time.perf_counter() - t0

# CPU (numpy)
from src.lp_functions import get_lp
cpu_lp = np.zeros((M, 12))
t0 = time.perf_counter()
for i in range(M):
    cpu_lp[i] = get_lp(lams48[i])
t_cpu = time.perf_counter() - t0

max_err = np.max(np.abs(gpu_lp - cpu_lp))
print(f"\nCorrectness: max error = {max_err:.2e}", flush=True)
print(f"GPU: {t_gpu*1000:.1f}ms for {M} laminates", flush=True)
print(f"CPU: {t_cpu*1000:.1f}ms for {M} laminates", flush=True)
print(f"Speedup: {t_cpu/t_gpu:.1f}x", flush=True)

# Large batch benchmark
print("\n--- 48-layer GPU batch LP speed ---", flush=True)
for M in [100, 1000, 10000, 50000]:
    lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
    # warmup
    _ = batch_lp_gpu(dev, mod, lams[:10], N)
    t0 = time.perf_counter()
    for _ in range(5):
        _ = batch_lp_gpu(dev, mod, lams, N)
    t = (time.perf_counter() - t0) / 5
    print(f"  M={M:>6d}: {t*1000:.1f}ms ({M/t/1e6:.2f}M lam/s)", flush=True)