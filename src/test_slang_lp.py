#!/usr/bin/env python3
"""Test SlangPy GPU LP solver — return value pattern.

SlangPy's grid dispatch works when functions RETURN a value,
not when they write to buffers via void. So we rewrite batch_lp
to return a struct containing 12 floats.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors
from src.numpy_fast import get_lp_batch


def generate_slang_lp(N, Z2, Z3):
    """Generate Slang shader source for LP computation — return struct pattern."""
    z2_str = ', '.join(str(int(round(z))) for z in Z2)
    z3_str = ', '.join(str(int(round(z))) for z in Z3)

    return f"""
struct LPResult {{
    float v0; float v1; float v2; float v3;
    float v4; float v5; float v6; float v7;
    float v8; float v9; float v10; float v11;
}};

LPResult batch_lp(int call_id, float* lams_flat, int M) {{
    int m = call_id;
    LPResult r;
    if (m >= M) {{
        r.v0 = 0; r.v1 = 0; r.v2 = 0; r.v3 = 0;
        r.v4 = 0; r.v5 = 0; r.v6 = 0; r.v7 = 0;
        r.v8 = 0; r.v9 = 0; r.v10 = 0; r.v11 = 0;
        return r;
    }}

    float invN = 1.0 / {N}.0;
    float N2 = 2.0 / {N}.0;
    float N3 = 2.0 / {N}.0;

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

    r.v0 = sum_cos2 * invN; r.v1 = sum_sin2 * invN;
    r.v2 = sum_cos4 * invN; r.v3 = sum_sin4 * invN;
    r.v4 = dc2z2 * N2; r.v5 = ds2z2 * N2;
    r.v6 = dc4z2 * N2; r.v7 = ds4z2 * N2;
    r.v8 = dc2z3 * N3; r.v9 = ds2z3 * N3;
    r.v10 = dc4z3 * N3; r.v11 = ds4z3 * N3;
    return r;
}}
"""


print("Creating CUDA device...", flush=True)
dev = sl.create_device(type=sl.DeviceType.cuda)
print("  device OK", flush=True)

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

lp_source = generate_slang_lp(N, Z2, Z3)
print("\nLoading Slang module...", flush=True)
mod = sl.Module.load_from_source(dev, "batch_lp", lp_source)
print("  module loaded", flush=True)

# Small correctness test
M = 10
lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
lams_flat = lams.flatten()
lams_tensor = Tensor.from_numpy(dev, lams_flat)

print("\nCorrectness test (M=10)...", flush=True)
result = mod.batch_lp(grid(shape=(M,)), lams_tensor.storage.device_address, int(M), _result="numpy")
print("  result type:", type(result), flush=True)
if isinstance(result, np.ndarray):
    print("  result shape:", result.shape, flush=True)
    lp_np = get_lp_batch(lams)
    result_2d = result.reshape(M, 12) if result.ndim == 2 else result
    if result.ndim > 2:
        result_2d = result.reshape(M, 12)
    max_err = np.max(np.abs(result_2d - lp_np))
    print(f"  max error vs numpy: {max_err:.2e}", flush=True)
    print("  PASS" if max_err < 1e-4 else "  FAIL", flush=True)
else:
    print("  result:", result, flush=True)

# Benchmark
print("\nBenchmarking...", flush=True)
batch_sizes = [1, 10, 100, 1000, 10000, 50000, 100000]
print("  M        numpy(ms)  gpu(ms)    speedup", flush=True)
print("  " + "-" * 50, flush=True)

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
    t0 = time.perf_counter()
    for _ in range(20):
        lp_np = get_lp_batch(lams)
    t_np = (time.perf_counter() - t0) / 20

    speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
    print(f"  {M:8d}  {t_np*1000:8.3f}  {t_gpu*1000:8.3f}  {speedup:8.1f}x", flush=True)