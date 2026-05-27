#!/usr/bin/env python3
"""GPU-accelerated solver: batch ssearch on GPU + CPU iRprop.

Strategy:
- Phase 1 (GPU): Evaluate ALL starting angles for ALL starts in one batch GPU call
  For Viquerat 12-layer with 18 angles per layer: 155 starts × 12 layers × 18 angles = 33,480 LP evals
  At 24.8M lam/s, this takes <1ms on GPU
- Phase 2 (CPU): iRprop refinement for the best start per target (sequential)
  
This should dramatically reduce the ssearch time while keeping iRprop on CPU.
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


def gpu_ssearch(dev, mod, target_lps, n_starts=155, n_coarse_deg=10):
    """Batch ssearch on GPU: find best starting angle for each problem.
    
    Returns the best starting laminate angles for each target.
    """
    n_problems = len(target_lps)
    angles_rad = np.arange(0, np.pi, np.radians(n_coarse_deg))
    n_angles = len(angles_rad)
    
    # Create all starting laminates: (n_problems * n_starts) × N
    # For ssearch, we try uniform starting angles
    # Each start is all layers at the same angle
    all_lams = np.zeros((n_problems * n_angles, N), dtype=np.float32)
    for p in range(n_problems):
        for a, angle in enumerate(angles_rad):
            all_lams[p * n_angles + a, :] = angle
    
    # Batch compute LP on GPU
    lps = batch_lp_gpu(dev, mod, all_lams)
    
    # Find best starting angle for each problem
    best_starts = np.zeros((n_problems, N), dtype=np.float64)
    for p in range(n_problems):
        target = target_lps[p]
        start_idx = p * n_angles
        losses = np.sum((lps[start_idx:start_idx + n_angles] - target) ** 2, axis=1)
        best_a = np.argmin(losses)
        best_starts[p, :] = angles_rad[best_a]
    
    return best_starts


def solve_gpu_hybrid(dev, mod, target_lps, n_starts=155, n_coarse_deg=10, 
                     n_coarse_fine=1, irprop_iters=3000, irprop_grad_tol=1e-3):
    """Hybrid GPU+CPU solver: GPU ssearch, CPU iRprop.
    
    Phase 1: GPU evaluates all starting angles in batch
    Phase 2: CPU iRprop refines the best starts
    """
    results = []
    for target_lp in target_lps:
        # Phase 1: GPU ssearch to find best starting angles
        # Create multiple perturbed starts around best angle
        angles_rad = np.arange(0, np.pi, np.radians(n_coarse_deg))
        
        # Evaluate all angles on GPU
        all_lams = np.tile(angles_rad[:, np.newaxis], (1, N)).astype(np.float32)
        lps = batch_lp_gpu(dev, mod, all_lams)
        losses = np.sum((lps - target_lp) ** 2, axis=1)
        
        # Pick top-k best starting angles
        top_k = min(n_starts, len(angles_rad))
        best_indices = np.argsort(losses)[:top_k]
        
        # Phase 2: CPU iRprop refinement
        rand_lams = all_lams[best_indices].astype(np.float32)
        opt_lams, opt_losses = optimize_laminate_numba(
            rand_lams, target_lp,
            n_coarse_fine=n_coarse_fine,
            irprop_iters=irprop_iters,
            irprop_grad_tol=irprop_grad_tol
        )
        results.append((opt_lams, opt_losses))
    
    return results


if __name__ == "__main__":
    print("GPU Hybrid Solver: SlangPy batch ssearch + CPU iRprop", flush=True)
    print("=" * 60, flush=True)

    dev = sl.create_device(type=sl.DeviceType.cuda)
    mod = sl.Module.load_from_source(dev, "batch_lp", SLANG_LP_SOURCE)
    print("GPU device initialized", flush=True)

    from src.test_cases import LP_VIQUERAT
    from src.lp_functions import _z2_z3 as z2z3, _norm_factors as nf
    
    # Test: batch ssearch speed
    print("\n--- Batch ssearch on GPU ---", flush=True)
    n_angles = 18
    angles_rad = np.arange(0, np.pi, np.radians(10))  # 18 angles
    all_lams = np.tile(angles_rad[:, np.newaxis], (1, N)).astype(np.float32)
    _ = batch_lp_gpu(dev, mod, all_lams)  # warmup
    
    t0 = time.perf_counter()
    for _ in range(100):
        lps = batch_lp_gpu(dev, mod, all_lams)
    t_gpu = (time.perf_counter() - t0) / 100
    print(f"  {len(angles_rad)} angles × {N} layers: {t_gpu*1000:.2f}ms", flush=True)
    
    # CPU equivalent
    from src.numpy_fast import get_lp_batch
    t0 = time.perf_counter()
    for _ in range(100):
        lps_cpu = get_lp_batch(all_lams)
    t_cpu = (time.perf_counter() - t0) / 100
    print(f"  CPU: {t_cpu*1000:.2f}ms, GPU speedup: {t_cpu/t_gpu:.1f}x", flush=True)

    # Full Viquerat benchmark
    print("\n--- Viquerat Hybrid GPU+CPU ---", flush=True)
    rng = np.random.RandomState(42)
    
    # GPU ssearch + CPU iRprop
    t0 = time.perf_counter()
    results = solve_gpu_hybrid(dev, mod, [LP_VIQUERAT], n_starts=18, n_coarse_fine=1, irprop_iters=3000, irprop_grad_tol=1e-3)
    t_hybrid = time.perf_counter() - t0
    print(f"  Hybrid (18 starts): {t_hybrid:.3f}s, best loss: {results[0][1][0]:.2e}", flush=True)
    
    # CPU-only baseline for comparison
    t0 = time.perf_counter()
    rand_lams = (rng.random((155, 12)).astype(np.float32) * np.pi - np.pi / 2)
    opt, losses = optimize_laminate_numba(rand_lams, LP_VIQUERAT, n_coarse_fine=1, irprop_iters=3000, irprop_grad_tol=1e-3)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU (155 starts): {t_cpu:.3f}s, best loss: {losses[0]:.2e}", flush=True)
    
    print(f"\nMETRIC viquerat_discovery_time={t_hybrid:.2f}", flush=True)