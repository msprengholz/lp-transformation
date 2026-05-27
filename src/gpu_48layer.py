#!/usr/bin/env python3
"""48-layer Sprengholz discovery with full GPU pipeline (Sobol + LP filter + iRprop)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse
from src.test_cases import LP_SPRENGHOLZ_48
from src.gpu_lp import batch_lp_gpu
from src.gpu_irprop import SLANG_IRPROP_STEP2, gpu_batch_irprop

N = 48
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

z2_str = ', '.join(str(int(round(z))) for z in Z2)
z3_str = ', '.join(str(int(round(z))) for z in Z3)

SLANG_LP_48 = f"""
float[12] batch_lp(int call_id, float* lams_flat, int M) {{
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

SLANG_IRPROP_48 = f"""
float irprop_step(int call_id,
                  float* in_angles,
                  float* in_steps,
                  float* in_pgrad,
                  float* target_lp,
                  float* out_angles,
                  float* out_steps,
                  float* out_pgrad,
                  int M) {{
    int m = call_id;
    if (m >= M) return 0.0;
    float invN_val = 1.0 / {N}.0;
    float N2_val = 2.0 / ({N}.0 * {N}.0);
    float N3_val = 4.0 / ({N}.0 * {N}.0 * {N}.0);
    float z2[{N}] = {{{z2_str}}};
    float z3[{N}] = {{{z3_str}}};
    float n_p = 1.2; float n_m = 0.5;
    float s_max = 0.5; float s_min_val = 0.00001;

    float cos2[{N}], sin2a[{N}], cos4[{N}], sin4a[{N}];
    float c2s=0,s2s=0,c4s=0,s4s=0;
    float dc2z2=0,ds2z2=0,dc4z2=0,ds4z2=0;
    float dc2z3=0,ds2z3=0,dc4z3=0,ds4z3=0;
    for (int i = 0; i < {N}; i++) {{
        float lam = in_angles[m * {N} + i];
        cos2[i] = cos(lam*2.0); sin2a[i] = sin(lam*2.0);
        cos4[i] = cos(lam*4.0); sin4a[i] = sin(lam*4.0);
        c2s+=cos2[i]; s2s+=sin2a[i]; c4s+=cos4[i]; s4s+=sin4a[i];
        float z2i=z2[i], z3i=z3[i];
        dc2z2+=cos2[i]*z2i; ds2z2+=sin2a[i]*z2i;
        dc4z2+=cos4[i]*z2i; ds4z2+=sin4a[i]*z2i;
        dc2z3+=cos2[i]*z3i; ds2z3+=sin2a[i]*z3i;
        dc4z3+=cos4[i]*z3i; ds4z3+=sin4a[i]*z3i;
    }}
    float lp[12];
    lp[0]=c2s*invN_val; lp[1]=s2s*invN_val;
    lp[2]=c4s*invN_val; lp[3]=s4s*invN_val;
    lp[4]=dc2z2*N2_val; lp[5]=ds2z2*N2_val;
    lp[6]=dc4z2*N2_val; lp[7]=ds4z2*N2_val;
    lp[8]=dc2z3*N3_val; lp[9]=ds2z3*N3_val;
    lp[10]=dc4z3*N3_val; lp[11]=ds4z3*N3_val;
    float loss=0; for(int j=0;j<12;j++){{float d=target_lp[j]-lp[j]; loss+=d*d;}}
    for (int k = 0; k < {N}; k++) {{
        float gk = (-2.0*sin2a[k]*(target_lp[0]-lp[0]) + 2.0*cos2[k]*(target_lp[1]-lp[1])
            -2.0*sin4a[k]*(target_lp[2]-lp[2]) + 2.0*cos4[k]*(target_lp[3]-lp[3])
            -2.0*sin2a[k]*z2[k]*(target_lp[4]-lp[4]) + 2.0*cos2[k]*z2[k]*(target_lp[5]-lp[5])
            -2.0*sin4a[k]*z2[k]*(target_lp[6]-lp[6]) + 2.0*cos4[k]*z2[k]*(target_lp[7]-lp[7])
            -2.0*sin2a[k]*z3[k]*(target_lp[8]-lp[8]) + 2.0*cos2[k]*z3[k]*(target_lp[9]-lp[9])
            -2.0*sin4a[k]*z3[k]*(target_lp[10]-lp[10]) + 2.0*cos4[k]*z3[k]*(target_lp[11]-lp[11]));
        gk = -gk * 2.0;
        float ss = in_steps[m*{N}+k]; float pg = in_pgrad[m*{N}+k];
        float ns=ss, ng=gk;
        if(pg*gk>0)ns=min(ss*n_p,s_max);
        else if(pg*gk<0){{ns=max(ss*n_m,s_min_val); ng=0.0;}}
        float na = in_angles[m*{N}+k] - sign(ng)*ns;
        na = max(-1.57079632679, min(1.57079632679, na));
        out_angles[m*{N}+k]=na; out_steps[m*{N}+k]=ns; out_pgrad[m*{N}+k]=ng;
    }}
    return loss;
}}
"""


def _round_key(lam, decimals=1):
    return tuple(round(float(a), decimals) for a in lam)


if __name__ == "__main__":
    print("48-layer Sprengholz GPU Discovery (Sobol + LP filter + iRprop)", flush=True)
    print("=" * 70, flush=True)
    
    dev = sl.create_device(type=sl.DeviceType.cuda)
    mod_lp = sl.Module.load_from_source(dev, "batch_lp_48", SLANG_LP_48)
    mod_irp = sl.Module.load_from_source(dev, "irprop_48", SLANG_IRPROP_48)
    print("GPU devices initialized", flush=True)
    
    target_lp = LP_SPRENGHOLZ_48
    rmse_threshold = 2e-2
    time_limit = 60.0
    
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=N, scramble=True, seed=42)
    
    found = set()
    completed = 0
    t_start = time.perf_counter()
    t_end = t_start + time_limit
    
    # Process in batches
    batch_size = 5000  # Sobol starts per batch
    
    # Warmup
    warmup = np.random.random((100, N)).astype(np.float32) * np.pi - np.pi / 2
    _ = batch_lp_gpu(dev, mod_lp, warmup, N)
    
    total_starts = 0
    
    while time.perf_counter() < t_end:
        # Generate Sobol starts
        remaining_time = t_end - time.perf_counter()
        if remaining_time < 0.5:
            break
        
        new_starts = sampler.random(batch_size).astype(np.float32) * np.pi - np.pi / 2
        total_starts += batch_size
        
        # GPU batch LP filter
        lps = batch_lp_gpu(dev, mod_lp, new_starts, N)
        losses = np.sum((lps - target_lp) ** 2, axis=1)
        
        # Select top-K
        top_k = min(500, len(losses))
        top_indices = np.argsort(losses)[:top_k]
        best_starts = new_starts[top_indices]
        
        # GPU iRprop refinement
        t_irp = time.perf_counter()
        best_angles, best_losses = gpu_batch_irprop(dev, mod_irp, best_starts, target_lp, max_iter=50)
        t_irp_elapsed = time.perf_counter() - t_irp
        
        # Collect solutions
        for i in range(top_k):
            if best_losses[i] < rmse_threshold:
                key = _round_key(best_angles[i])
                found.add(key)
                completed += 1
        
        elapsed = time.perf_counter() - t_start
        print(f"  [{elapsed:.0f}s] {len(found)} unique, {total_starts} starts, irprop {t_irp_elapsed:.2f}s", flush=True)
    
    t_total = time.perf_counter() - t_start
    print(f"\nRESULT: {len(found)} unique solutions in {t_total:.1f}s ({total_starts} starts)", flush=True)
    
    # CPU reference
    print("\n--- CPU Reference (60s) ---", flush=True)
    from benchmarks.run_comprehensive import benchmark_sprengholz_48
    t0 = time.perf_counter()
    cpu_found, cpu_completed = benchmark_sprengholz_48(time_limit=60.0)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU: {len(cpu_found)} unique solutions in {t_cpu:.1f}s ({cpu_completed} starts)", flush=True)
    print(f"\nMETRIC sprengholz_solutions={len(found)} sprengholz_starts={total_starts}", flush=True)