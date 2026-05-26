#!/usr/bin/env python3
"""
Quick test for the numba solver on Colab.
Runs correctness checks and a tiny benchmark.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.numba_solver import optimize_laminate_numba
from src.test_cases import LP_VIQUERAT
from src.lp_functions import compute_lp_rmse

print("Numba solver test")
print("  numba available:", end=" ")
try:
    import numba
    print("yes (%s)" % numba.__version__)
except ImportError:
    print("NO")
    sys.exit(1)

# Warmup JIT
rng = np.random.default_rng(42)
lam = rng.random(12, dtype=np.float32) * np.pi - np.pi / 2
t0 = time.perf_counter()
opt, losses = optimize_laminate_numba(lam.reshape(1, -1), LP_VIQUERAT, irprop_iters=10)
dt = time.perf_counter() - t0
print("  JIT warmup: %.3fs" % dt)
print("  Warmup loss: %.2e" % losses[0])

# Short benchmark
n_starts = 10
rand_lams = rng.random((n_starts, 12), dtype=np.float32) * np.pi - np.pi / 2
t0 = time.perf_counter()
opt, losses = optimize_laminate_numba(rand_lams, LP_VIQUERAT)
dt = time.perf_counter() - t0
print("  %d starts: %.3fs (%.4fs/start)" % (n_starts, dt, dt/n_starts))
print("  Best loss: %.2e" % losses.min())
print("  Median loss: %.2e" % np.median(losses))

# Check correctness
from src.numpy_solver import optimize_laminate as opt_np
rng = np.random.default_rng(42)
lam_np = rng.random((n_starts, 12), dtype=np.float32) * np.pi - np.pi / 2
opt_np_result, losses_np = opt_np(lam_np.copy(), LP_VIQUERAT)
opt_nb_result, losses_nb = optimize_laminate_numba(lam_np.copy(), LP_VIQUERAT)

loss_diff = abs(losses_np.min() - losses_nb.min())
print("  Loss diff (numpy vs numba): %.2e" % loss_diff)

if loss_diff < 0.01:
    print("  Correctness: PASS")
    print("METRIC numba_ok=1")
else:
    print("  Correctness: FAIL")
    print("METRIC numba_ok=0")
