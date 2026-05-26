#!/usr/bin/env python3
"""Test SlangPy call_id() and batch kernel dispatch."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl

dev = sl.create_device(type=sl.DeviceType.cuda)
print("device:", dev.info.adapter_name, flush=True)

# Test call_id() — each thread should get a unique index
src = """
[AutoPy]
void fill_indices(float[] result, int M) {
    int idx = int(call_id().x);
    if (idx < M) result[idx] = float(idx);
}
"""
mod = sl.Module.load_from_source(dev, "indices", src)

M = 16
result = [0.0] * M
mod.fill_indices(result, int(M))
print("indices:", result[:10], "...", flush=True)
print("All OK!" if result[:5] == [0.0, 1.0, 2.0, 3.0, 4.0] else "FAIL!", flush=True)
