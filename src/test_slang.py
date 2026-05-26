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

# Test 1: basic scalar operation
print("\n--- Test 1: basic scalar ---")
mod = sl.Module.from_string("""
[AutoPy]
float add_scalar(float a, float b) { return a + b; }
""")
result = mod.add_scalar(np.float32(2.0), np.float32(3.0))
print("  2.0 + 3.0 =", result)

# Test 2: tensor operation
print("\n--- Test 2: tensor element-wise ---")
mod2 = sl.Module.from_string("""
[AutoPy]
float4 add4(float4 a, float4 b) { return a + b; }
""")
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
result2 = mod2.add4(a, b)
print("  [1,2,3,4] + [5,6,7,8] =", result2)

# Test 3: array reduction (sum)
print("\n--- Test 3: array sum ---")
mod3 = sl.Module.from_string("""
[AutoPy]
float sum_array(float[] arr, int N) {
    float s = 0.0;
    for (int i = 0; i < N; i++) s += arr[i];
    return s;
}
""")
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
result3 = mod3.sum_array(arr, np.int32(5))
print("  sum([1,2,3,4,5]) =", result3)

# Test 4: cos/sin (trig needed for LP computation)
print("\n--- Test 4: trig ---")
mod4 = sl.Module.from_string("""
[AutoPy]
float2 trig(float x) { return float2(cos(x), sin(x)); }
""")
result4 = mod4.trig(np.float32(0.5))
print("  cos(0.5), sin(0.5) =", result4)

print("\nAll basic tests passed!")
