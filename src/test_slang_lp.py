#!/usr/bin/env python3
"""Test SlangPy GPU LP solver using correct call_id + grid pattern.

Key insight from SlangPy tests:
- Slang function takes `int call_id` as a regular parameter
- Python call uses `grid(shape=(N,))` to dispatch N GPU threads
- Each thread gets its own call_id (0..N-1)
- Data passed via raw pointers (device_address) or Tensor
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors
from src.numpy_fast import get_lp_batch

def generate_slang_lp(N, Z2, Z3):
    """Generate Slang shader source for LP computation."""
    z2_str = ', '.join(str(int(round(z))) for z in Z2)
    z3_str = ', '.join(str(int(round(z))) for z in Z3)

    return f"""
void batch_lp(int call_id, float* lams_flat, float* result_flat, int M) {{
    int m = call_id;
    if (m >= M) return;

    float invN = 1.0 / {N}.0;
    float N2 = 2.0 / {N}.0;
    float N3 = 2.0 / {N}.0;

    // Z2 and Z3 vectors
    float z2[{N}] = {{{z2_str}}};
    float z3[{N}] = {{{z3_str}}};

    // Accumulators
    float sum_cos2 = 0.0;
    float sum_sin2 = 0.0;
    float sum_cos4 = 0.0;
    float sum_sin4 = 0.0;
    float dot_cos2_z2 = 0.0;
    float dot_sin2_z2 = 0.0;
    float dot_cos4_z2 = 0.0;
    float dot_sin4_z2 = 0.0;
    float dot_cos2_z3 = 0.0;
    float dot_sin2_z3 = 0.0;
    float dot_cos4_z3 = 0.0;
    float dot_sin4_z3 = 0.0;

    for (int i = 0; i < {N}; i++) {{
        float lam = lams_flat[m * {N} + i];
        float c2 = cos(lam * 2.0);
        float s2 = sin(lam * 2.0);
        float c4 = cos(lam * 4.0);
        float s4 = sin(lam * 4.0);

        sum_cos2 += c2;
        sum_sin2 += s2;
        sum_cos4 += c4;
        sum_sin4 += s4;

        dot_cos2_z2 += c2 * z2[i];
        dot_sin2_z2 += s2 * z2[i];
        dot_cos4_z2 += c4 * z2[i];
        dot_sin4_z2 += s4 * z2[i];
        dot_cos2_z3 += c2 * z3[i];
        dot_sin2_z3 += s2 * z3[i];
        dot_cos4_z3 += c4 * z3[i];
        dot_sin4_z3 += s4 * z3[i];
    }}

    result_flat[m * 12 + 0] = sum_cos2 * invN;
    result_flat[m * 12 + 1] = sum_sin2 * invN;
    result_flat[m * 12 + 2] = sum_cos4 * invN;
    result_flat[m * 12 + 3] = sum_sin4 * invN;
    result_flat[m * 12 + 4] = dot_cos2_z2 * N2;
    result_flat[m * 12 + 5] = dot_sin2_z2 * N2;
    result_flat[m * 12 + 6] = dot_cos4_z2 * N2;
    result_flat[m * 12 + 7] = dot_sin4_z2 * N2;
    result_flat[m * 12 + 8] = dot_cos2_z3 * N3;
    result_flat[m * 12 + 9] = dot_sin2_z3 * N3;
    result_flat[m * 12 + 10] = dot_cos4_z3 * N3;
    result_flat[m * 12 + 11] = dot_sin4_z3 * N3;
}}
"""


print("Creating CUDA device...", flush=True)
dev = sl.create_device(type=sl.DeviceType.cuda)
print("  device OK", flush=True)

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

# Generate and load Slang module
lp_source = generate_slang_lp(N, Z2, Z3)
print("\nLoading Slang module...", flush=True)
mod = sl.Module.load_from_source(dev, "batch_lp", lp_source)
print("  module loaded", flush=True)

# Test with small batch first
M = 10
lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
lams_flat = lams.flatten()
result_flat = np.zeros(M * 12, dtype=np.float32)

# Create GPU tensors
lams_tensor = Tensor.from_numpy(dev, lams_flat)
result_tensor = Tensor.from_numpy(dev, result_flat)

print("\nTesting batch_lp GPU...", flush=True)
result = mod.batch_lp(
    grid(shape=(M,)),
    lams_tensor.storage.device_address,
    result_tensor.storage.device_address,
    int(M),
    _result="numpy"
)

# Compare with numpy
lp_np = get_lp_batch(lams)
result_2d = result.reshape(M, 12)

print("  GPU result shape:", result.shape if hasattr(result, 'shape') else type(result), flush=True)
print("  numpy result shape:", lp_np.shape, flush=True)

if isinstance(result, np.ndarray):
    max_err = np.max(np.abs(result_2d - lp_np))
    print(f"  max error: {max_err:.2e}", flush=True)
    print("  PASS" if max_err < 1e-4 else "  FAIL", flush=True)
else:
    print("  Result type:", type(result), flush=True)
    print("  Need to adjust result handling", flush=True)

# Benchmark
print("\nBenchmarking...", flush=True)
batch_sizes = [1, 10, 100, 1000, 10000, 50000]
print("  M      numpy(ms)  gpu(ms)   speedup", flush=True)
print("  " + "-" * 45, flush=True)

for M in batch_sizes:
    lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
    lams_flat = lams.flatten()

    lams_tensor = Tensor.from_numpy(dev, lams_flat)
    result_tensor = Tensor.from_numpy(dev, np.zeros(M * 12, dtype=np.float32))

    # Warmup GPU
    _ = mod.batch_lp(grid(shape=(M,)), lams_tensor.storage.device_address, result_tensor.storage.device_address, int(M), _result="numpy")

    # GPU timed
    t0 = time.perf_counter()
    for _ in range(20):
        result_gpu = mod.batch_lp(grid(shape=(M,)), lams_tensor.storage.device_address, result_tensor.storage.device_address, int(M), _result="numpy")
    t_gpu = (time.perf_counter() - t0) / 20

    # Numpy timed
    t0 = time.perf_counter()
    for _ in range(20):
        lp_np = get_lp_batch(lams)
    t_np = (time.perf_counter() - t0) / 20

    speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
    print(f"  {M:6d}  {t_np*1000:8.3f}  {t_gpu*1000:8.3f}  {speedup:7.1f}x", flush=True)