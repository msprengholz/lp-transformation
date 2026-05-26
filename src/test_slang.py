#!/usr/bin/env python3
"""Test basic SlangPy functionality on Colab GPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl

print("slangpy version:", sl.__version__)

# Test: basic Slang module via load_from_source
print("\n--- Test: basic ---")
source = """
[AutoPy]
float add_scalar(float a, float b) { return a + b; }

[AutoPy]
float4 add4(float4 a, float4 b) { return a + b; }
"""
mod = sl.Module.load_from_source(source)

r1 = mod.add_scalar(np.float32(2.0), np.float32(3.0))
print("  add_scalar 2+3 =", r1)

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
r2 = mod.add4(a, b)
print("  add4 [1,2,3,4]+[5,6,7,8] =", r2)

print("\nAll tests passed!")
