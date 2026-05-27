#!/usr/bin/env python3
"""Benchmark SlangPy GPU LP solver vs numpy.

Verified correct: max error ~8.9e-08 (float32 precision).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors
from src.numpy_fast import get_lp_batch

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

z2_str = ', '.join(str(int(round(z))) for z in Z2)
z3_str = ', '.join(str(int(round(z))) for z in Z3)

SLANG_LP_SOURCE = f"""
float[{12}] batch_lp(int call_id, float* lams_flat, int M) {{
    int m = call_id;
    float r[12] = (float[12])0;
    if (m >= M) return r;

    float invN = 1.0 / {N}.0;
    float N2 = 2.0 / ({N}.0 * {N}.0);
    float N3 = 4.0 / ({N}.0 * {N}.0 * {N}.0);

    float z2[{N}] = {{{z2_str}}};
    float z3[{N}] = {{{z3_str}}};

    float sum_cos2 = 0.0; float sum_sin2 = 0.0;
    float sum_cos4 = 0.0; float sum_sin4 = 0.0;
    float dc2z2 = 0.0; float ds2z2 = 0.0;
    float dc4z2 = 0.0; float ds4z2 = 0.0;
    float dc2z3 = 0.0; float ds2z3 = 0.0;
    float dc4z3 = 0.0; float ds4z3 = 0.0;

    for (int i = 0; i < {N}; i++) {{
        float lam = lams_flat[m * {N} + i];
        float c2 = cos(lam * 2.0); float s2 = sin(lam * 2.0);
        float c4 = cos(lam * 4.0); float s4 = sin(lam * 4.0);

        sum_cos2 += c2; sum_sin2 += s2;
        sum_cos4 += c4; sum_sin4 += s4;

        float z2i = z2[i]; float z3i = z3[i];
        dc2z2 += c2 * z2i; ds2z2 += s2 * z2i;
        dc4z2 += c4 * z2i; ds4z2 += s4 * z2i;
        dc2z3 += c2 * z3i; ds2z3 += s2 * z3i;
        dc4z3 += c4 * z3i; ds4z3 += s4 * z3i;
    }}

    r[0] = sum_cos2 * invN; r[1] = sum_sin2 * invN;
    r[2] = sum_cos4 * invN; r[3] = sum_sin4 * invN;
    r[4] = dc2z2 * N2; r[5] = ds2z2 * N2;
    r[6] = dc4z2 * N2; r[7] = ds4z2 * N2;
    r[8] = dc2z3 * N3; r[9] = ds2z3 * N3;
    r[10] = dc4z3 * N3; r[11] = ds4z3 * N3;
    return r;
}}
"""

print("Creating CUDA device...", flush=True)
dev = sl.create_device(type=sl.DeviceType.cuda)
print("  device OK", flush=True)

print("Loading Slang module...", flush=True)
mod = sl.Module.load_from_source(dev, "batch_lp", SLANG_LP_SOURCE)
print("  module loaded", flush=True)

# Correctness verification
M = 100
lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
lams_flat = lams.flatten()
lams_tensor = Tensor.from_numpy(dev, lams_flat)

result_gpu = mod.batch_lp(grid(shape=(M,)), lams_tensor.storage.device_address, int(M), _result="numpy")
lp_np = get_lp_batch(lams)
gpu_2d = result_gpu.reshape(M, 12)
max_err = np.max(np.abs(gpu_2d - lp_np))
print(f"\nCorrectness: max error = {max_err:.2e} (PASS)" if max_err < 1e-4 else f"\nCorrectness: max error = {max_err:.2e} (FAIL)", flush=True)

# Benchmark
print("\nBenchmark (batch LP computation):", flush=True)
print("  M        numpy(ms)  gpu(ms)    speedup", flush=True)
print("  " + "-" * 50, flush=True)

batch_sizes = [1, 10, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
for M in batch_sizes:
    lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
    lams_flat = lams.flatten()
    lams_tensor = Tensor.from_numpy(dev, lams_flat)

    # Warmup GPU
    _ = mod.batch_lp(grid(shape=(M,)), lams_tensor.storage.device_address, int(M), _result="numpy")

    # GPU timed
    t0 = time.perf_counter()
    for _ in range(20):
        result_gpu = mod.batch_lp(grid(shape=(M,)), lams_tensor.storage.device_address, int(M), _result="numpy")
    t_gpu = (time.perf_counter() - t0) / 20

    # Numpy timed
    reps = max(1, 5 if M <= 10000 else 2 if M <= 100000 else 1)
    t0 = time.perf_counter()
    for _ in range(reps):
        lp_np = get_lp_batch(lams)
    t_np = (time.perf_counter() - t0) / reps

    speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
    print(f"  {M:8d}  {t_np*1000:8.3f}  {t_gpu*1000:8.3f}  {speedup:8.1f}x", flush=True)