#!/usr/bin/env python3
"""
GPU iRprop v2: Separate input/output buffers, proper readback.

The kernel takes input buffers (angles, step_sizes, prev_grad) and writes
to output buffers (out_angles, out_steps, out_pgrad). Loss is returned per start.
After each iteration, outputs become the next iteration's inputs.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse
from src.test_cases import LP_VIQUERAT

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

z2_str = ', '.join(str(int(round(z))) for z in Z2)
z3_str = ', '.join(str(int(round(z))) for z in Z3)

SLANG_IRPROP_STEP2 = f"""
float irprop_step(int call_id,
                  float* in_angles,     // M * N
                  float* in_steps,      // M * N
                  float* in_pgrad,      // M * N
                  float* target_lp,      // 12
                  float* out_angles,    // M * N
                  float* out_steps,     // M * N
                  float* out_pgrad,     // M * N
                  int M) {{
    int m = call_id;
    if (m >= M) return 0.0;
    
    float invN_val = 1.0 / {N}.0;
    float N2_val = 2.0 / ({N}.0 * {N}.0);
    float N3_val = 4.0 / ({N}.0 * {N}.0 * {N}.0);
    float z2[{N}] = {{{z2_str}}};
    float z3[{N}] = {{{z3_str}}};
    float n_p = 1.2;
    float n_m = 0.5;
    float s_max = 0.5;
    float s_min_val = 0.00001;
    
    float cos2[{N}], sin2a[{N}], cos4[{N}], sin4a[{N}];
    float c2s = 0.0, s2s = 0.0, c4s = 0.0, s4s = 0.0;
    float dc2z2 = 0.0, ds2z2 = 0.0, dc4z2 = 0.0, ds4z2 = 0.0;
    float dc2z3 = 0.0, ds2z3 = 0.0, dc4z3 = 0.0, ds4z3 = 0.0;
    
    for (int i = 0; i < {N}; i++) {{
        float lam = in_angles[m * {N} + i];
        cos2[i] = cos(lam * 2.0); sin2a[i] = sin(lam * 2.0);
        cos4[i] = cos(lam * 4.0); sin4a[i] = sin(lam * 4.0);
        c2s += cos2[i]; s2s += sin2a[i]; c4s += cos4[i]; s4s += sin4a[i];
        float z2i = z2[i]; float z3i = z3[i];
        dc2z2 += cos2[i] * z2i; ds2z2 += sin2a[i] * z2i;
        dc4z2 += cos4[i] * z2i; ds4z2 += sin4a[i] * z2i;
        dc2z3 += cos2[i] * z3i; ds2z3 += sin2a[i] * z3i;
        dc4z3 += cos4[i] * z3i; ds4z3 += sin4a[i] * z3i;
    }}
    
    float lp[12];
    lp[0] = c2s * invN_val;  lp[1] = s2s * invN_val;
    lp[2] = c4s * invN_val;  lp[3] = s4s * invN_val;
    lp[4] = dc2z2 * N2_val;  lp[5] = ds2z2 * N2_val;
    lp[6] = dc4z2 * N2_val;  lp[7] = ds4z2 * N2_val;
    lp[8] = dc2z3 * N3_val;  lp[9] = ds2z3 * N3_val;
    lp[10] = dc4z3 * N3_val; lp[11] = ds4z3 * N3_val;
    
    float loss = 0.0;
    for (int j = 0; j < 12; j++) {{
        float diff = target_lp[j] - lp[j];
        loss += diff * diff;
    }}
    
    for (int k = 0; k < {N}; k++) {{
        float grad_k = (
            -2.0 * sin2a[k] * (target_lp[0] - lp[0]) +
             2.0 * cos2[k] * (target_lp[1] - lp[1]) +
            -2.0 * sin4a[k] * (target_lp[2] - lp[2]) +
             2.0 * cos4[k] * (target_lp[3] - lp[3]) +
            -2.0 * sin2a[k] * z2[k] * (target_lp[4] - lp[4]) +
             2.0 * cos2[k] * z2[k] * (target_lp[5] - lp[5]) +
            -2.0 * sin4a[k] * z2[k] * (target_lp[6] - lp[6]) +
             2.0 * cos4[k] * z2[k] * (target_lp[7] - lp[7]) +
            -2.0 * sin2a[k] * z3[k] * (target_lp[8] - lp[8]) +
             2.0 * cos2[k] * z3[k] * (target_lp[9] - lp[9]) +
            -2.0 * sin4a[k] * z3[k] * (target_lp[10] - lp[10]) +
             2.0 * cos4[k] * z3[k] * (target_lp[11] - lp[11])
        );
        // grad_k is d_loss/d_lam_k (uphill direction)
        // Convert to -2*val matching numba convention
        grad_k = -grad_k * 2.0;
        
        float ss = in_steps[m * {N} + k];
        float pg = in_pgrad[m * {N} + k];
        
        float new_ss = ss;
        float new_pg = grad_k;
        
        if (pg * grad_k > 0.0) {{
            new_ss = min(ss * n_p, s_max);
        }} else if (pg * grad_k < 0.0) {{
            new_ss = max(ss * n_m, s_min_val);
            new_pg = 0.0;
        }}
        
        float new_angle = in_angles[m * {N} + k] - sign(new_pg) * new_ss;
        new_angle = max(-1.57079632679, min(1.57079632679, new_angle));
        
        out_angles[m * {N} + k] = new_angle;
        out_steps[m * {N} + k] = new_ss;
        out_pgrad[m * {N} + k] = new_pg;
    }}
    
    return loss;
}}
"""


def gpu_batch_irprop(dev, mod, lams_init, target_lp, max_iter=100):
    """Run batch iRprop on GPU. All M starts processed in parallel."""
    M, N_layers = lams_init.shape
    MS = M * N_layers
    
    angles = lams_init.flatten().astype(np.float32).copy()
    step_sizes = np.full(MS, 0.01, dtype=np.float32)
    prev_grad = np.zeros(MS, dtype=np.float32)
    target = target_lp.astype(np.float32)
    
    best_angles = angles.copy()
    best_losses = np.full(M, 1e10, dtype=np.float32)
    
    for iteration in range(max_iter):
        in_a = Tensor.from_numpy(dev, angles)
        in_s = Tensor.from_numpy(dev, step_sizes)
        in_g = Tensor.from_numpy(dev, prev_grad)
        tgt = Tensor.from_numpy(dev, target)
        out_a = Tensor.from_numpy(dev, np.empty(MS, dtype=np.float32))
        out_s = Tensor.from_numpy(dev, np.empty(MS, dtype=np.float32))
        out_g = Tensor.from_numpy(dev, np.empty(MS, dtype=np.float32))
        
        losses = mod.irprop_step(
            grid(shape=(M,)),
            in_a.storage.device_address,
            in_s.storage.device_address,
            in_g.storage.device_address,
            tgt.storage.device_address,
            out_a.storage.device_address,
            out_s.storage.device_address,
            out_g.storage.device_address,
            int(M),
            _result="numpy"
        )
        
        # Read back outputs
        new_angles = out_a.storage.to_numpy().view(np.float32)[:MS].copy()
        new_steps = out_s.storage.to_numpy().view(np.float32)[:MS].copy()
        new_pgrad = out_g.storage.to_numpy().view(np.float32)[:MS].copy()
        
        # Update best
        for m in range(M):
            if losses[m] < best_losses[m]:
                best_losses[m] = losses[m]
                best_angles[m*N_layers:(m+1)*N_layers] = new_angles[m*N_layers:(m+1)*N_layers]
        
        # Swap for next iteration
        angles = new_angles
        step_sizes = new_steps
        prev_grad = new_pgrad
    
    return best_angles.reshape(M, N_layers), best_losses


if __name__ == "__main__":
    print("GPU iRprop v2: separate I/O buffers", flush=True)
    print("=" * 60, flush=True)
    
    dev = sl.create_device(type=sl.DeviceType.cuda)
    mod = sl.Module.load_from_source(dev, "irprop_step2", SLANG_IRPROP_STEP2)
    print("Module loaded!", flush=True)
    
    # Correctness test
    M = 10
    rng = np.random.RandomState(42)
    lams = rng.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
    target = LP_VIQUERAT
    
    print(f"\nCorrectness test: {M} starts, 100 iters...", flush=True)
    t0 = time.perf_counter()
    best_angles, best_losses = gpu_batch_irprop(dev, mod, lams, target, max_iter=100)
    t_gpu = time.perf_counter() - t0
    print(f"  GPU time: {t_gpu:.3f}s", flush=True)
    print(f"  Best loss: {np.min(best_losses):.6f}", flush=True)
    print(f"  Median loss: {np.median(best_losses):.6f}", flush=True)
    
    # CPU comparison
    from src.numba_solver import optimize_laminate_numba
    t0 = time.perf_counter()
    opt, losses = optimize_laminate_numba(lams[:1].copy(), target, n_coarse_fine=1, irprop_grad_tol=1e-3)
    t_cpu = time.perf_counter() - t0
    print(f"\n  CPU time: {t_cpu*1000:.1f}ms, CPU loss: {losses[0]:.6f}", flush=True)
    print(f"  GPU angle[0]: {best_angles[0]}", flush=True)
    print(f"  CPU angle[0]: {opt[0]}", flush=True)
    print(f"  GPU RMSE: {compute_lp_rmse(best_angles[0].astype(np.float32), target):.6f}", flush=True)
    print(f"  CPU RMSE: {compute_lp_rmse(opt[0], target):.6f}", flush=True)
    
    # Batch benchmark
    print("\n--- GPU iRprop batch benchmark ---", flush=True)
    for M in [100, 500, 1000, 2000, 5000, 10000]:
        lams = rng.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
        t0 = time.perf_counter()
        best_a, best_l = gpu_batch_irprop(dev, mod, lams, target, max_iter=50)
        t = time.perf_counter() - t0
        print(f"  M={M:6d}: {t:.3f}s ({M/t:.0f} starts/s), best loss: {np.min(best_l):.4f}", flush=True)