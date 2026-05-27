#!/usr/bin/env python3
"""
General GPU LP solver using SlangPy with configurable N.
"""
import numpy as np
import slangpy as sl
from slangpy import grid, Tensor


def generate_slang_lp_shader(N, Z2, Z3):
    """Generate Slang LP shader for N layers with given Z2/Z3 vectors."""
    invN, N2, N3 = 1.0/N, 2.0/(N*N), 4.0/(N*N*N)
    z2_str = ', '.join(str(int(round(z))) for z in Z2)
    z3_str = ', '.join(str(int(round(z))) for z in Z3)
    
    return f"""
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


def create_gpu_lp_solver(N, Z2=None, Z3=None):
    """Create a SlangPy GPU LP solver for N layers."""
    from src.lp_functions import _z2_z3, _norm_factors
    
    if Z2 is None or Z3 is None:
        Z2, Z3 = _z2_z3(N)
    
    dev = sl.create_device(type=sl.DeviceType.cuda)
    source = generate_slang_lp_shader(N, Z2, Z3)
    mod = sl.Module.load_from_source(dev, f"batch_lp_{N}", source)
    
    return dev, mod


def batch_lp_gpu(dev, mod, lams, N):
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