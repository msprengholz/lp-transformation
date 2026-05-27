#!/usr/bin/env python3
"""Full GPU-accelerated LP solver benchmark.

Uses SlangPy for batch LP on GPU, numba for ssearch + iRprop on CPU.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors
from src.solver_numba import find_all_solutions
from src.benchmark_viquerat import viquerat_problems

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


def batch_lp_gpu(dev, mod, lams):
    """Compute LP for batch of laminates on GPU."""
    M, N = lams.shape
    lams_flat = lams.flatten().astype(np.float32)
    lams_tensor = Tensor.from_numpy(dev, lams_flat)
    result = mod.batch_lp(
        grid(shape=(M,)),
        lams_tensor.storage.device_address,
        int(M),
        _result="numpy"
    )
    return result.reshape(M, 12)


def solve_ssearch_gpu(dev, mod, target_lp, N=12, n_coarse=10, n_starts=30, max_iter=100, grad_tol=1e-4):
    """One start: ssearch on GPU + iRprop on CPU."""
    from src.solver_numba import iRprop_step_numba, get_loss_grad_numba
    Z2, Z3 = _z2_z3(N)

    angles_deg = np.arange(0, 180, n_coarse)
    angles_rad = np.radians(angles_deg)
    cos2 = np.cos(2 * angles_rad)
    sin2 = np.sin(2 * angles_rad)
    cos4 = np.cos(4 * angles_rad)
    sin4 = np.sin(4 * angles_rad)

    best_lam = None
    best_loss = float('inf')

    for _ in range(1):  # n_coarse_fine=1
        for a0 in angles_rad:
            lams = np.full(N, a0, dtype=np.float32)
            lam_lower = np.full(N, -np.pi / 2, dtype=np.float64)
            lam_upper = np.full(N, np.pi / 2, dtype=np.float64)

            step = np.full(N, 0.01, dtype=np.float64)
            prev_grad_sign = np.zeros(N, dtype=np.float64)

            # Compute initial LP on GPU
            lp = batch_lp_gpu(dev, mod, lams.reshape(1, N))[0]
            loss, grad = get_loss_grad_numba(lams, lp, target_lp, Z2, Z3, N)

            for _it in range(max_iter):
                lams, lam_lower, lam_upper, step, prev_grad_sign, grad, loss = \
                    iRprop_step_numba(lams, lam_lower, lam_upper, step, prev_grad_sign, grad, loss)

                lp = batch_lp_gpu(dev, mod, lams.reshape(1, N))[0].astype(np.float64)
                loss, grad = get_loss_grad_numba(lams, lp, target_lp, Z2, Z3, N)

                if np.max(np.abs(grad)) < grad_tol:
                    break

            if loss < best_loss:
                best_loss = loss
                best_lam = lams.copy()

    return best_lam, best_loss


def solve_gpu_pipelined(dev, mod, target_lps, n_coarse=10, n_starts=155, max_iter=100, grad_tol=1e-4):
    """Batch solve: evaluate ALL starts on GPU, then iRprop the best ones."""
    N = 12
    Z2_np, Z3_np = _z2_z3(N)

    results = []
    for target_lp in target_lps:
        angles_deg = np.arange(0, 180, n_coarse)
        angles_rad = np.radians(angles_deg)

        # Phase 1: Batch evaluate all starting points on GPU
        start_lams = np.tile(angles_rad[:, np.newaxis], (1, N)).astype(np.float32)
        start_lps = batch_lp_gpu(dev, mod, start_lams)

        # Compute loss for each start
        losses = np.sum((start_lps - target_lp[np.newaxis, :]) ** 2, axis=1)
        best_starts = np.argsort(losses)[:min(n_starts, len(losses))]

        # Phase 2: iRprop refine the best starts
        best_lam = None
        best_loss = float('inf')
        for si in best_starts:
            lams = start_lams[si].copy().astype(np.float64)
            lam_lower = np.full(N, -np.pi / 2, dtype=np.float64)
            lam_upper = np.full(N, np.pi / 2, dtype=np.float64)
            step = np.full(N, 0.01, dtype=np.float64)
            prev_grad_sign = np.zeros(N, dtype=np.float64)

            lp = batch_lp_gpu(dev, mod, lams.reshape(1, N))[0].astype(np.float64)
            loss, grad = get_loss_grad_numba(lams, lp, target_lp, Z2_np, Z3_np, N)

            for _it in range(max_iter):
                lams, lam_lower, lam_upper, step, prev_grad_sign, grad, loss = \
                    iRprop_step_numba(lams, lam_lower, lam_upper, step, prev_grad_sign, grad, loss)
                lp = batch_lp_gpu(dev, mod, lams.reshape(1, N))[0].astype(np.float64)
                loss, grad = get_loss_grad_numba(lams, lp, target_lp, Z2_np, Z3_np, N)
                if np.max(np.abs(grad)) < grad_tol:
                    break

            if loss < best_loss:
                best_loss = loss
                best_lam = lams.copy()

        results.append((best_lam, best_loss))

    return results


if __name__ == "__main__":
    print("SlangPy GPU-accelerated LP solver benchmark", flush=True)
    print("=" * 60, flush=True)

    dev = sl.create_device(type=sl.DeviceType.cuda)
    mod = sl.Module.load_from_source(dev, "batch_lp", SLANG_LP_SOURCE)
    print("GPU device initialized", flush=True)

    # Test 1: Pure GPU batch LP speed
    print("\n--- GPU Batch LP Speed ---", flush=True)
    for M in [100, 1000, 10000, 100000, 1000000]:
        lams = np.random.random((M, 12)).astype(np.float32) * np.pi - np.pi / 2
        # Warmup
        _ = batch_lp_gpu(dev, mod, lams[:1000])
        t0 = time.perf_counter()
        _ = batch_lp_gpu(dev, mod, lams)
        t = time.perf_counter() - t0
        print(f"  M={M:>8d}: {t*1000:.1f}ms ({M/t/1e6:.2f}M lam/s)", flush=True)

    # Test 2: Full Viquerat discovery with GPU
    print("\n--- Viquerat Discovery (GPU) ---", flush=True)
    problems = viquerat_problems()

    t0 = time.perf_counter()
    results = solve_gpu_pipelined(dev, mod, [p[1] for p in problems[:10]], n_starts=155)
    t = time.perf_counter() - t0
    found = sum(1 for _, l in results if l < 1e-6)
    print(f"  10 problems: {t:.2f}s, {found}/10 found (loss < 1e-6)", flush=True)

    # Full benchmark: all 112 Viquerat problems
    t0 = time.perf_counter()
    results_all = solve_gpu_pipelined(dev, mod, [p[1] for p in problems], n_starts=155)
    t_all = time.perf_counter() - t0
    found_all = sum(1 for _, l in results_all if l < 1e-6)
    print(f"  112 problems: {t_all:.2f}s, {found_all}/112 found (GPU)", flush=True)

    # Compare with CPU
    print("\n--- CPU Reference (numba) ---", flush=True)
    t0 = time.perf_counter()
    cpu_results = find_all_solutions([p[1] for p in problems], n_coarse=10, n_starts=155, max_iter=100, grad_tol=1e-4)
    t_cpu = time.perf_counter() - t0
    cpu_found = sum(1 for _, l in cpu_results if l < 1e-6)
    print(f"  112 problems: {t_cpu:.2f}s, {cpu_found}/112 found (CPU)", flush=True)
    print(f"\n  GPU speedup: {t_cpu/t_all:.1f}x", flush=True)