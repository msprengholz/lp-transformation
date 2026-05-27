#!/usr/bin/env python3
"""GPU solver vs CPU solver benchmark for Viquerat discovery.

Benchmark the full pipeline: ssearch + iRprop, comparing:
  1. CPU (numba) baseline
  2. GPU (SlangPy) accelerated LP + CPU ssearch/iRprop
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors
from src.numba_solver import optimize_laminate_numba

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

def get_lp_gpu(dev, mod, lams):
    """Compute LP for one laminate on GPU."""
    M = lams.shape[0] if lams.ndim == 2 else 1
    if lams.ndim == 1:
        lams = lams.reshape(1, -1)
    lams_flat = lams.flatten().astype(np.float32)
    lams_tensor = Tensor.from_numpy(dev, lams_flat)
    result = mod.batch_lp(
        grid(shape=(M,)),
        lams_tensor.storage.device_address,
        int(M),
        _result="numpy"
    )
    return result.reshape(M, 12)


def get_lp_gpu_batch(dev, mod, lams_batch):
    """Compute LP for batch of laminates on GPU."""
    M = lams_batch.shape[0]
    lams_flat = lams_batch.flatten().astype(np.float32)
    lams_tensor = Tensor.from_numpy(dev, lams_flat)
    result = mod.batch_lp(
        grid(shape=(M,)),
        lams_tensor.storage.device_address,
        int(M),
        _result="numpy"
    )
    return result.reshape(M, 12)


if __name__ == "__main__":
    print("SlangPy GPU LP benchmark vs CPU", flush=True)
    print("=" * 60, flush=True)

    dev = sl.create_device(type=sl.DeviceType.cuda)
    mod = sl.Module.load_from_source(dev, "batch_lp", SLANG_LP_SOURCE)
    print("GPU device initialized", flush=True)

    # Test 1: Pure GPU batch LP speed
    print("\n--- GPU Batch LP Speed ---", flush=True)
    for M in [100, 1000, 10000, 100000]:
        lams = np.random.random((M, 12)).astype(np.float32) * np.pi - np.pi / 2
        _ = get_lp_gpu_batch(dev, mod, lams[:10])  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            _ = get_lp_gpu_batch(dev, mod, lams)
        t = (time.perf_counter() - t0) / 5
        print(f"  M={M:>8d}: {t*1000:.1f}ms ({M/t/1e6:.2f}M lam/s)", flush=True)

    # Test 2: CPU vs GPU per-call LP (single laminate)
    print("\n--- Single-laminate LP: CPU vs GPU ---", flush=True)
    from src.numpy_fast import get_lp_batch
    test_lams = np.random.random((1, 12)).astype(np.float32) * np.pi - np.pi / 2
    
    # CPU warmup
    for _ in range(100):
        _ = get_lp_batch(test_lams)
    t0 = time.perf_counter()
    for _ in range(10000):
        _ = get_lp_batch(test_lams)
    t_cpu = (time.perf_counter() - t0) / 10000
    
    # GPU warmup
    test_lam_1d = test_lams[0]
    _ = get_lp_gpu(dev, mod, test_lam_1d)
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = get_lp_gpu(dev, mod, test_lam_1d)
    t_gpu = (time.perf_counter() - t0) / 1000
    
    print(f"  CPU: {t_cpu*1e6:.0f} µs/call", flush=True)
    print(f"  GPU: {t_gpu*1e6:.0f} µs/call", flush=True)
    print(f"  CPU/GPU ratio: {t_cpu/t_gpu:.1f}x", flush=True)

    # Test 3: Full Viquerat benchmark
    print("\n--- Viquerat Discovery Benchmark ---", flush=True)
    from src.test_cases import viquerat_problems
    problems = viquerat_problems()
    print(f"  {len(problems)} Viquerat problems", flush=True)
    
    # CPU baseline
    rng = np.random.RandomState(42)
    t0 = time.perf_counter()
    cpu_found = 0
    for name, lp_t in problems:
        rand_lams = (rng.random((155, 12)).astype(np.float32) * np.pi - np.pi / 2)
        opt_lams, losses = optimize_laminate_numba(rand_lams, lp_t, n_coarse_fine=1, max_iter=100, grad_tol=1e-4)
        if losses[0] < 1e-6:
            cpu_found += 1
    t_cpu_total = time.perf_counter() - t0
    print(f"  CPU: {t_cpu_total:.2f}s, {cpu_found}/{len(problems)} found", flush=True)

    print(f"\nMETRIC viquerat_discovery_time={t_cpu_total:.2f}", flush=True)