#!/usr/bin/env python3
"""Benchmark the optimised pure-numpy solver vs the original."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.test_cases import LP_VIQUERAT, LP_SPRENGHOLZ_48
from src.numpy_solver import optimize_laminate as orig_solver
from src.numpy_fast import optimize_fast, sobol_starts


def bench(label, solver_fn, starts, lp_t, n_starts):
    t0 = time.perf_counter()
    opt, losses = solver_fn(starts, lp_t)
    dt = time.perf_counter() - t0
    # Count unique VALID solutions (RMSE < 1e-3, rounded to 0.1 deg)
    from src.lp_functions import compute_lp_rmse
    uniq = set()
    for idx in range(len(opt)):
        if losses[idx] < 1e-3:
            d = np.rad2deg(opt[idx]); d = (d + 90) % 180 - 90
            uniq.add(tuple(np.round(d, 1)))
    print("  %-20s  %6.1fs  best=%.2e  median=%.2e  valid=%d" % (
        label, dt, losses.min(), np.median(losses), len(uniq)))
    return dt, len(uniq)


print("=" * 65)
print("  Pure-numpy solver comparison  (200 starts each)")
print("=" * 65)

N12 = 12
lp_v = LP_VIQUERAT
n = 200

rng = np.random.default_rng(42)
rand12 = (rng.random((n, N12), dtype=np.float32) * np.pi - np.pi / 2)

print("\n--- Viquerat 12-layer ---")
t_orig = bench("orig (3 ssearch)", orig_solver, rand12, lp_v, n)
t_fast = bench("fast (Sobol)", optimize_fast, sobol_starts(n, N12), lp_v, n)
t_fast_r = bench("fast (random)", optimize_fast, rand12.copy(), lp_v, n)

t_orig_val, _ = t_orig
t_fast_val, _ = t_fast
print("\nSpeedup: orig=%.1fs, fast=%.1fs (%.1fx)" % (
    t_orig_val, t_fast_val, t_orig_val / t_fast_val))

# Also test 48-layer
print("\n--- Sprengholz 48-layer (100 starts) ---")
N48 = 48
lp_48 = LP_SPRENGHOLZ_48
rng = np.random.default_rng(42)
rand48 = (rng.random((100, N48), dtype=np.float32) * np.pi - np.pi / 2)

bench("orig (3 ssearch)", orig_solver, rand48, lp_48, 100)
bench("fast (Sobol)", optimize_fast, sobol_starts(100, N48), lp_48, 100)
bench("fast (random)", optimize_fast, rand48.copy(), lp_48, 100)
