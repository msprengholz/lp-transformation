#!/usr/bin/env python3
"""Debug SlangPy GPU LP correctness — test with known input."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors
from src.numpy_fast import get_lp_batch

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)

z2_str = ', '.join(str(int(round(z))) for z in Z2)
z3_str = ', '.join(str(int(round(z))) for z in Z3)

# Start with simplest possible kernel: single laminate, return 12 floats
shader_simple = f"""
float[{12}] batch_lp(int call_id, float* lams_flat, int M) {{
    int m = call_id;
    float r[12] = (float[12])0;
    if (m >= M) return r;

    float invN = 1.0 / {N}.0;
    float N2 = 2.0 / ({N}.0 * {N}.0);
    float N3 = 2.0 / ({N}.0 * {N}.0 * {N}.0);

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

print("Creating CUDA device...", flush=True)
dev = sl.create_device(type=sl.DeviceType.cuda)

# Test 1: Verify with zero angles (should give known values)
print("\n--- Test 1: Zero angles (lam=0) ---", flush=True)
mod = sl.Module.load_from_source(dev, "test1", shader_simple)

# All angles = 0: cos(0)=1, sin(0)=0
# lp should be: [0.083, 0, 0.083, 0, 0, 0, 0, 0, ...]
lams_zero = np.zeros((1, N), dtype=np.float32).flatten()
lams_tensor = Tensor.from_numpy(dev, lams_zero)
result = mod.batch_lp(grid(shape=(1,)), lams_tensor.storage.device_address, int(1), _result="numpy")
print("  GPU result:", result, flush=True)

# Expected: numpy reference
lp_ref = get_lp_batch(np.zeros((1, N), dtype=np.float32))
print("  numpy ref: ", lp_ref, flush=True)

# Test 2: Known angles
print("\n--- Test 2: Known angles ---", flush=True)
np.random.seed(42)
lams_test = np.random.random((5, N)).astype(np.float32) * np.pi - np.pi / 2
lams_flat = lams_test.flatten()
lams_tensor = Tensor.from_numpy(dev, lams_flat)

for i in range(5):
    result = mod.batch_lp(grid(shape=(5,)), lams_tensor.storage.device_address, int(5), _result="numpy")
lp_ref = get_lp_batch(lams_test)

result_2d = result.reshape(5, 12) if result.ndim != 2 else result
print("  GPU[0]:", result_2d[0], flush=True)
print("  np [0]: ", lp_ref[0], flush=True)
print("  diff[0]:", np.abs(result_2d[0] - lp_ref[0]), flush=True)
print("  max error:", np.max(np.abs(result_2d - lp_ref)), flush=True)

# Test 3: Print individual values for debugging
print("\n--- Test 3: Debug individual threads ---", flush=True)
lams_single = np.array([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2]], dtype=np.float32)
result_s = mod.batch_lp(grid(shape=(1,)), Tensor.from_numpy(dev, lams_single.flatten()).storage.device_address, int(1), _result="numpy")
result_np = get_lp_batch(lams_single)
print("  GPU: ", result_s, flush=True)
print("  numpy:", result_np, flush=True)
print("  diff: ", np.abs(result_s - result_np), flush=True)