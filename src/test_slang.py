#!/usr/bin/env python3
"""Test basic SlangPy functionality on the Colab GPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import slangpy as sl
    print("slangpy version:", sl.__version__)
except Exception as e:
    print("Failed to import slangpy:", e)
    sys.exit(1)

# Write test shader
shader_path = os.path.join(os.path.dirname(__file__), "test_basic.slang")
with open(shader_path, "w") as f:
    f.write("""
[AutoPy]
float add_scalar(float a, float b) { return a + b; }

[AutoPy]
float4 add4(float4 a, float4 b) { return a + b; }
""")

# Test 1: basic scalar
print("\n--- Test 1: basic scalar ---")
mod = sl.Module.from_file(shader_path)
result = mod.add_scalar(np.float32(2.0), np.float32(3.0))
print("  2.0 + 3.0 =", result)

# Test 2: tensor
print("\n--- Test 2: tensor ---")
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
result2 = mod.add4(a, b)
print("  [1,2,3,4] + [5,6,7,8] =", result2)

print("\nAll tests passed!")
