#!/usr/bin/env python3
"""Benchmark CuPy GPU batch LP vs numpy."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cupy as cp

from src.lp_functions import _z2_z3, _norm_factors
from src.numpy_fast import get_lp_batch

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

# CuPy batch LP
def get_lp_batch_gpu(lams_cp):
    M = lams_cp.shape[0]
    lam2 = lams_cp * 2
    lam4 = lams_cp * 4
    cos2 = cp.cos(lam2)
    sin2 = cp.sin(lam2)
    cos4 = cp.cos(lam4)
    sin4 = cp.sin(lam4)

    lp = cp.empty((M, 12), dtype=cp.float32)
    Z2_cp = cp.asarray(Z2)
    Z3_cp = cp.asarray(Z3)

    lp[:, 0] = cp.sum(cos2, axis=1) * invN
    lp[:, 1] = cp.sum(sin2, axis=1) * invN
    lp[:, 2] = cp.sum(cos4, axis=1) * invN
    lp[:, 3] = cp.sum(sin4, axis=1) * invN
    lp[:, 4] = cos2 @ Z2_cp * N2
    lp[:, 5] = sin2 @ Z2_cp * N2
    lp[:, 6] = cos4 @ Z2_cp * N2
    lp[:, 7] = sin4 @ Z2_cp * N2
    lp[:, 8] = cos2 @ Z3_cp * N3
    lp[:, 9] = sin2 @ Z3_cp * N3
    lp[:, 10] = cos4 @ Z3_cp * N3
    lp[:, 11] = sin4 @ Z3_cp * N3
    return cp.asnumpy(lp)


batch_sizes = [1, 10, 100, 1000, 10000, 100000]
print("Batch LP benchmark (12 layers)")
print("batch_size  numpy(ms)  cupy(ms)  speedup")
print("-" * 50)

for M in batch_sizes:
    lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
    lams_cp = cp.asarray(lams)

    # Warmup
    _ = get_lp_batch_gpu(lams_cp)
    _ = get_lp_batch(lams)

    # Numpy timed
    t0 = time.perf_counter()
    for _ in range(20):
        lp_np = get_lp_batch(lams)
    t_np = (time.perf_counter() - t0) / 20

    # CuPy timed (includes GPU sync)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        lp_cp = get_lp_batch_gpu(lams_cp)
    cp.cuda.Stream.null.synchronize()
    t_cp = (time.perf_counter() - t0) / 20

    speedup = t_np / t_cp if t_cp > 0 else float('inf')
    print("  %7d  %8.3f  %8.3f  %7.1fx" % (M, t_np*1000, t_cp*1000, speedup))

    # Verify correctness
    if M <= 1000:
        err = np.max(np.abs(lp_np - lp_cp))
        if err > 1e-4:
            print("  ERROR: max diff =", err)
