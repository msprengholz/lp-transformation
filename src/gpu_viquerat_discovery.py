#!/usr/bin/env python3
"""
GPU-accelerated Viquerat discovery: batch Sobol start evaluation on GPU.

Strategy:
1. Generate Sobol starts (CPU, fast)
2. Evaluate ALL starting points on GPU in one batch
3. Pick top-K best starts
4. Run CPU iRprop on top-K starts

This should dramatically reduce the number of iRprop iterations needed.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse
from src.test_cases import LP_VIQUERAT, _data_dir
from src.numba_solver import optimize_laminate_numba

N = 12
Z2, Z3 = _z2_z3(N)

z2_str = ', '.join(str(int(round(z))) for z in Z2)
z3_str = ', '.join(str(int(round(z))) for z in Z3)
invN, N2, N3 = _norm_factors(N)

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


def batch_lp_gpu(dev, mod, lams):
    """Compute LP for batch of laminates on GPU."""
    M = lams.shape[0]
    lams_flat = lams.flatten().astype(np.float32)
    lams_tensor = Tensor.from_numpy(dev, lams_flat)
    result = mod.batch_lp(
        grid(shape=(M,)),
        lams_tensor.storage.device_address,
        int(M),
        _result="numpy"
    )
    return result.reshape(M, 12).astype(np.float64)


def _load_known_solutions():
    import csv
    solutions = set()
    path = _data_dir() / "viquerat_12_layer_solutions_complete.csv"
    if not path.exists():
        return set()
    with open(path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) >= 13:
                angles = tuple(round(float(a), 1) for a in row[1:13])
                solutions.add(angles)
    return solutions


def benchmark_gpu_viquerat(dev, mod, max_starts=50000, top_k=500):
    """GPU-accelerated Viquerat discovery."""
    known = _load_known_solutions()
    target_count = len(known)
    target_lp = LP_VIQUERAT
    found = set()
    
    # Sobol start generator
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=12, scramble=True, seed=42)

    t_start = time.perf_counter()
    
    # Phase 1: Generate and evaluate ALL Sobol starts on GPU in batches
    batch_size = 10000  # GPU processes 10K at a time
    all_starts = []
    all_losses = []
    
    starts_used = 0
    while starts_used < max_starts:
        # Generate batch of Sobol starts
        remaining = min(batch_size, max_starts - starts_used)
        new_starts = sampler.random(remaining).astype(np.float32) * np.pi - np.pi / 2
        starts_used += remaining
        
        # Evaluate on GPU
        lps = batch_lp_gpu(dev, mod, new_starts)
        losses = np.sum((lps - target_lp) ** 2, axis=1)
        
        all_starts.append(new_starts)
        all_losses.append(losses)
        
        # Check if we found all solutions
        if starts_used >= 1000 and len(found) >= target_count:
            break
    
    all_starts = np.vstack(all_starts)
    all_losses = np.concatenate(all_losses)
    
    # Phase 2: Pick top-K best starts and run iRprop on CPU
    top_indices = np.argsort(all_losses)[:top_k]
    top_starts = all_starts[top_indices]
    
    for i in range(top_k):
        lam = top_starts[i:i+1]
        opt, losses = optimize_laminate_numba(lam, target_lp, n_coarse_fine=1, irprop_grad_tol=1e-3)
        best_loss = float(losses[0])
        
        if best_loss < 1e-2:  # RMSE threshold
            rmse = compute_lp_rmse(opt[0], target_lp)
            if rmse < 2e-2:
                key = tuple(round(a, 1) for a in opt[0])
                if key not in found:
                    found.add(key)
                    if len(found) >= target_count:
                        t_total = time.perf_counter() - t_start
                        return t_total, starts_used, len(found)
    
    t_total = time.perf_counter() - t_start
    return t_total, starts_used, len(found)


if __name__ == "__main__":
    print("GPU-accelerated Viquerat Discovery", flush=True)
    print("=" * 60, flush=True)
    
    dev = sl.create_device(type=sl.DeviceType.cuda)
    mod = sl.Module.load_from_source(dev, "batch_lp", SLANG_LP_SOURCE)
    print("GPU device initialized", flush=True)
    
    # Warmup JIT
    rand = np.random.random((10, 12)).astype(np.float32) * np.pi - np.pi / 2
    _ = batch_lp_gpu(dev, mod, rand)
    _ = optimize_laminate_numba(rand[:1], LP_VIQUERAT, n_coarse_fine=1, irprop_grad_tol=1e-3)
    
    # GPU-accelerated discovery
    t, starts, found = benchmark_gpu_viquerat(dev, mod, max_starts=50000, top_k=3000)
    print(f"\nGPU: {t:.2f}s, {starts} starts, {found}/112 found", flush=True)
    print(f"METRIC viquerat_discovery_time={t:.2f}", flush=True)
    
    # CPU reference
    print("\nCPU reference:", flush=True)
    from benchmarks.run_comprehensive import benchmark_viquerat_discovery
    t_cpu, starts_cpu, found_cpu = benchmark_viquerat_discovery(max_starts=50000)
    print(f"CPU: {t_cpu:.2f}s, {starts_cpu} starts, {found_cpu}/112 found", flush=True)
    
    print(f"\nSpeedup: {t_cpu/t:.1f}x", flush=True)