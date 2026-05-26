#!/usr/bin/env python3
"""Confirm RMSE filter active for both numpy and numba solvers."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.test_cases import LP_VIQUERAT
from src.lp_functions import compute_lp_rmse

N = 12
lp_t = LP_VIQUERAT
n = 200

# --- Numpy original ---
print("--- Numpy original ---")
from src.numpy_solver import optimize_laminate
rng = np.random.default_rng(42)
s = (rng.random((n, N), dtype=np.float32) * np.pi - np.pi / 2)
t0 = time.perf_counter()
o, l = optimize_laminate(s, lp_t, n_coarse_fine=3)
t = time.perf_counter() - t0
v = sum(1 for x in l if x < 1e-3)
print("  %.1fs  best=%.2e  valid(RMSE<1e-3)=%d" % (t, l.min(), v))

# --- Numpy fast ---
print("--- Numpy fast ---")
from src.numpy_fast import optimize_fast, sobol_starts
s2 = sobol_starts(n, N)
t0 = time.perf_counter()
o2, l2 = optimize_fast(s2, lp_t)
t = time.perf_counter() - t0
v2 = sum(1 for x in l2 if x < 1e-3)
print("  %.1fs  best=%.2e  valid(RMSE<1e-3)=%d" % (t, l2.min(), v2))

# --- Numba ---
print("--- Numba ---")
try:
    from src.numba_solver import optimize_laminate_numba
    rng3 = np.random.default_rng(42)
    s3 = (rng3.random((n, N), dtype=np.float32) * np.pi - np.pi / 2)
    t0 = time.perf_counter()
    o3, l3 = optimize_laminate_numba(s3, lp_t)
    t = time.perf_counter() - t0
    v3 = sum(1 for x in l3 if x < 1e-3)
    print("  %.1fs  best=%.2e  valid(RMSE<1e-3)=%d" % (t, l3.min(), v3))
except ImportError:
    print("  numba not available")
