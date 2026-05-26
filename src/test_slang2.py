#!/usr/bin/env python3
"""Debug SlangPy shader compilation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl

print("Creating CUDA device...", flush=True)
dev = sl.create_device(type=sl.DeviceType.cuda)
print("  device:", dev.info.adapter_name, flush=True)

# Simplest kernel
src1 = """
[AutoPy]
float add(float a, float b) { return a + b; }
"""
print("Loading simple module...", flush=True)
mod1 = sl.Module.load_from_source(dev, "simple", src1)
v = mod1.add(np.float32(2.0), np.float32(3.0))
print("  add(2, 3) =", v, flush=True)

# Array test
src2 = """
[AutoPy]
float arr_sum(float[] arr, int N) {
    float s = 0.0;
    for (int i = 0; i < N; i++) s += arr[i];
    return s;
}
"""
print("Loading array module...", flush=True)
mod2 = sl.Module.load_from_source(dev, "arr_sum", src2)
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
v2 = mod2.arr_sum(arr, np.int32(5))
print("  sum([1,2,3,4,5]) =", v2, flush=True)

print("All tests passed!", flush=True)
