#!/usr/bin/env python3
"""
Quick test for the numba solver on Colab.
All prints are flushed for real-time visibility via colab-cli.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("Numba solver test", flush=True)

print("  importing numba...", end=" ", flush=True)
try:
    import numba
    print("yes (%s)" % numba.__version__, flush=True)
except ImportError:
    print("NO", flush=True)
    sys.exit(1)

print("  importing solver modules...", flush=True)
import numpy as np
from src.numba_solver import optimize_laminate_numba
from src.test_cases import LP_VIQUERAT
from src.lp_functions import compute_lp_rmse
print("  imports done", flush=True)

# Warmup JIT
print("  JIT warmup (10 iters)...", flush=True)
rng = np.random.default_rng(42)
lam = rng.random(12, dtype=np.float32) * np.pi - np.pi / 2
t0 = time.perf_counter()
opt, losses = optimize_laminate_numba(lam.reshape(1, -1), LP_VIQUERAT, irprop_iters=10)
dt = time.perf_counter() - t0
print("  JIT warmup: %.3fs" % dt, flush=True)

# Short benchmark
print("  running %d-start benchmark..." % 10, flush=True)
n_starts = 10
rand_lams = rng.random((n_starts, 12), dtype=np.float32) * np.pi - np.pi / 2
t0 = time.perf_counter()
opt, losses = optimize_laminate_numba(rand_lams, LP_VIQUERAT)
dt = time.perf_counter() - t0
print("  %d starts: %.3fs (%.4fs/start)" % (n_starts, dt, dt/n_starts), flush=True)
print("  Best loss: %.2e" % losses.min(), flush=True)

# Correctness check vs numpy
print("  checking correctness vs numpy...", flush=True)
from src.numpy_solver import optimize_laminate as opt_np
rng = np.random.default_rng(42)
lam_np = rng.random((n_starts, 12), dtype=np.float32) * np.pi - np.pi / 2
opt_np_result, losses_np = opt_np(lam_np.copy(), LP_VIQUERAT)
opt_nb_result, losses_nb = optimize_laminate_numba(lam_np.copy(), LP_VIQUERAT)

loss_diff = abs(float(losses_np.min()) - float(losses_nb.min()))
print("  Loss diff (numpy vs numba): %.2e" % loss_diff, flush=True)

if loss_diff < 0.01:
    print("  Correctness: PASS", flush=True)
    print("METRIC numba_ok=1", flush=True)
else:
    print("  Correctness: FAIL", flush=True)
    print("METRIC numba_ok=0", flush=True)
