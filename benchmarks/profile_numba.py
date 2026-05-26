#!/usr/bin/env python3
"""Profile the numba solver to find remaining bottlenecks."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.numba_solver import optimize_laminate_numba, _get_lp_numba, _ssearch_numba, _irpropm_numba, _prepare_arrays
from src.test_cases import LP_VIQUERAT

rng = np.random.default_rng(42)
N = 12
Z2, Z3, invN, N2, N3 = _prepare_arrays(N)
lp_t = LP_VIQUERAT

# Warmup
lam = rng.random(N, dtype=np.float32) * np.pi - np.pi / 2
_get_lp_numba(lam, Z2, Z3, invN, N2, N3)

# 1. Micro: get_lp
times = []
for _ in range(1000):
    lam = rng.random(N, dtype=np.float32) * np.pi - np.pi / 2
    t0 = time.perf_counter_ns()
    lp = _get_lp_numba(lam, Z2, Z3, invN, N2, N3)
    times.append(time.perf_counter_ns() - t0)
arr = np.array(times, dtype=np.float64)
print("get_lp:       %.1f us (%.1f-%.1f)" % (arr.mean()/1000, arr.min()/1000, arr.max()/1000))

# 2. Micro: ssearch (1 call, 10 deg)
times = []
lam = rng.random(N, dtype=np.float32) * np.pi - np.pi / 2
for _ in range(20):
    lam2 = lam.copy()
    t0 = time.perf_counter_ns()
    _ssearch_numba(lam2, lp_t, np.float32(np.deg2rad(10.0)), 18, Z2, Z3, invN, N2, N3)
    times.append(time.perf_counter_ns() - t0)
arr = np.array(times[5:], dtype=np.float64)
print("ssearch(10):  %.1f ms (%.1f-%.1f)" % (arr.mean()/1e6, arr.min()/1e6, arr.max()/1e6))

# 3. Micro: iRprop (500 iters)
lam = rng.random(N, dtype=np.float32) * np.pi - np.pi / 2
t0 = time.perf_counter_ns()
_irpropm_numba(lam, lp_t, 500, Z2, Z3, invN, N2, N3, 0.1, 1e-8, 0.3, 1.2, 0.5, 1e-6)
dt = time.perf_counter_ns() - t0
print("iRprop(500):  %.2f ms" % (dt / 1e6))
