#!/usr/bin/env python3
"""
GPU iRprop: Batch iRprop on GPU using SlangPy.

The kernel does one iRprop step per dispatch:
- Input: angles (M, N), step_sizes (M, N), prev_grad (M, N), target_lp (12)
- Output: updated angles, step_sizes, prev_grad, loss per start

We iterate from Python, dispatching the kernel M times per iteration.
All M starts are processed in parallel on the GPU.
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

# iRprop step kernel. Processes M starts in parallel.
# Each start has: angles[N], step_sizes[N], prev_grad[N], target_lp[12]
# Output: updated angles, step_sizes, prev_grad, and loss
SLANG_IRPROP_STEP = f"""
void irprop_step(int call_id,
                 float* angles,     // M * N
                 float* step_sizes,  // M * N
                 float* prev_grad,   // M * N
                 float* target_lp,   // 12
                 float* out_angles,  // M * N
                 float* out_steps,   // M * N
                 float* out_pgrad,   // M * N
                 float* out_loss,    // M
                 int M) {{
    int m = call_id;
    if (m >= M) return;
    
    float invN = 1.0 / {N}.0;
    float N2 = 2.0 / ({N}.0 * {N}.0);
    float N3 = 4.0 / ({N}.0 * {N}.0 * {N}.0);
    float z2[{N}] = {{{z2_str}}};
    float z3[{N}] = {{{z3_str}}};
    float n_p = 1.2;
    float n_m = 0.5;
    float s_max = 0.5;
    float s_min_val = 0.00001;
    
    // Compute trig
    float cos2[{N}], sin2a[{N}], cos4[{N}], sin4a[{N}];
    float c2s = 0.0, s2s = 0.0, c4s = 0.0, s4s = 0.0;
    float dc2z2 = 0.0, ds2z2 = 0.0, dc4z2 = 0.0, ds4z2 = 0.0;
    float dc2z3 = 0.0, ds2z3 = 0.0, dc4z3 = 0.0, ds4z3 = 0.0;
    
    for (int i = 0; i < {N}; i++) {{
        float lam = angles[m * {N} + i];
        cos2[i] = cos(lam * 2.0); sin2a[i] = sin(lam * 2.0);
        cos4[i] = cos(lam * 4.0); sin4a[i] = sin(lam * 4.0);
        c2s += cos2[i]; s2s += sin2a[i]; c4s += cos4[i]; s4s += sin4a[i];
        float z2i = z2[i]; float z3i = z3[i];
        dc2z2 += cos2[i] * z2i; ds2z2 += sin2a[i] * z2i;
        dc4z2 += cos4[i] * z2i; ds4z2 += sin4a[i] * z2i;
        dc2z3 += cos2[i] * z3i; ds2z3 += sin2a[i] * z3i;
        dc4z3 += cos4[i] * z3i; ds4z3 += sin4a[i] * z3i;
    }}
    
    // Compute LP
    float lp[12];
    lp[0] = c2s * invN;  lp[1] = s2s * invN;
    lp[2] = c4s * invN;  lp[3] = s4s * invN;
    lp[4] = dc2z2 * N2;  lp[5] = ds2z2 * N2;
    lp[6] = dc4z2 * N2;  lp[7] = ds4z2 * N2;
    lp[8] = dc2z3 * N3;  lp[9] = ds2z3 * N3;
    lp[10] = dc4z3 * N3; lp[11] = ds4z3 * N3;
    
    // Compute loss
    float loss = 0.0;
    for (int j = 0; j < 12; j++) {{
        float diff = target_lp[j] - lp[j];
        loss += diff * diff;
    }}
    out_loss[m] = loss;
    
    // iRprop step for each layer
    for (int k = 0; k < {N}; k++) {{
        float grad_k = -2.0 * (
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
        grad_k *= -2.0;
        
        float ss = step_sizes[m * {N} + k];
        float pg = prev_grad[m * {N} + k];
        
        float new_ss = ss;
        float new_pg = grad_k;
        
        if (pg * grad_k > 0.0) {{
            new_ss = min(ss * n_p, s_max);
        }} else if (pg * grad_k < 0.0) {{
            new_ss = max(ss * n_m, s_min_val);
            new_pg = 0.0;
        }}
        
        float new_angle = angles[m * {N} + k] - sign(new_pg) * new_ss;
        new_angle = max(-1.57079632679, min(1.57079632679, new_angle));
        
        out_angles[m * {N} + k] = new_angle;
        out_steps[m * {N} + k] = new_ss;
        out_pgrad[m * {N} + k] = new_pg;
    }}
}}
"""


def gpu_batch_irprop(dev, mod, lams_init, target_lp, max_iter=100, grad_tol=1e-3):
    """Run batch iRprop on GPU. All M starts processed in parallel."""
    M, N_layers = lams_init.shape
    
    # State arrays
    angles = lams_init.flatten().astype(np.float32).copy()
    step_sizes = np.full(M * N_layers, 0.01, dtype=np.float32)
    prev_grad = np.zeros(M * N_layers, dtype=np.float32)
    target = target_lp.astype(np.float32)
    
    # Best tracking
    best_angles = angles.copy()
    best_losses = np.full(M, 1e10, dtype=np.float32)
    
    for iteration in range(max_iter):
        # Allocate output buffers
        out_angles = np.empty_like(angles)
        out_steps = np.empty_like(step_sizes)
        out_pgrad = np.empty_like(prev_grad)
        out_loss = np.empty(M, dtype=np.float32)
        
        # Create tensors and dispatch
        a_t = Tensor.from_numpy(dev, angles)
        s_t = Tensor.from_numpy(dev, step_sizes)
        g_t = Tensor.from_numpy(dev, prev_grad)
        tgt_t = Tensor.from_numpy(dev, target)
        oa_t = Tensor.from_numpy(dev, out_angles)
        os_t = Tensor.from_numpy(dev, out_steps)
        og_t = Tensor.from_numpy(dev, out_pgrad)
        ol_t = Tensor.from_numpy(dev, out_loss)
        
        mod.irprop_step(
            grid(shape=(M,)),
            a_t.storage.device_address,
            s_t.storage.device_address,
            g_t.storage.device_address,
            tgt_t.storage.device_address,
            oa_t.storage.device_address,
            os_t.storage.device_address,
            og_t.storage.device_address,
            ol_t.storage.device_address,
            int(M),
            _result="numpy"
        )
        
        # Read back outputs
        angles = dev.queue.read_buffer(oa_t.storage).cast('f').tolist()
        # Hmm, this won't work easily. Let me try a different approach.
        # The irprop_step is void, so _result="numpy" won't give us output.
        # We need to read back buffers manually.
        pass
    
    return best_angles.reshape(M, N_layers), best_losses


if __name__ == "__main__":
    print("GPU iRprop solver", flush=True)
    print("=" * 50, flush=True)
    
    dev = sl.create_device(type=sl.DeviceType.cuda)
    print("Loading Slang module...", flush=True)
    
    try:
        mod = sl.Module.load_from_source(dev, "irprop_step", SLANG_IRPROP_STEP)
        print("  Module loaded successfully!", flush=True)
    except Exception as e:
        print(f"  Module load error: {e}", flush=True)
        sys.exit(1)
    
    # Test: Just dispatch one step
    M = 10
    lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2
    target = LP_VIQUERAT.astype(np.float32)
    
    print(f"\nTesting iRprop step with {M} starts...", flush=True)
    angles = lams.flatten().copy()
    step_sizes = np.full(M * N, 0.01, dtype=np.float32)
    prev_grad = np.zeros(M * N, dtype=np.float32)
    out_angles = np.empty_like(angles)
    out_steps = np.empty_like(step_sizes)
    out_pgrad = np.empty_like(prev_grad)
    out_loss = np.empty(M, dtype=np.float32)
    
    try:
        result = mod.irprop_step(
            grid(shape=(M,)),
            Tensor.from_numpy(dev, angles).storage.device_address,
            Tensor.from_numpy(dev, step_sizes).storage.device_address,
            Tensor.from_numpy(dev, prev_grad).storage.device_address,
            Tensor.from_numpy(dev, target).storage.device_address,
            Tensor.from_numpy(dev, out_angles).storage.device_address,
            Tensor.from_numpy(dev, out_steps).storage.device_address,
            Tensor.from_numpy(dev, out_pgrad).storage.device_address,
            Tensor.from_numpy(dev, out_loss).storage.device_address,
            int(M),
            _result="numpy"
        )
        print(f"  Result: {result}", flush=True)
        print(f"  Loss[0]: {out_loss[0]:.6f}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)