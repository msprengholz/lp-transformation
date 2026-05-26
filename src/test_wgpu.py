#!/usr/bin/env python3
"""Test wgpu GPU compute on Colab."""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import wgpu

# Get GPU adapter and device
print("Enumerating adapters...", flush=True)
adapters = wgpu.request_adapter_sync(canvas=None, power_preference="high-performance")
print("  adapter:", adapters, flush=True)

# Try to get a device
try:
    device = wgpu.utils.get_default_device()
    print("  device:", type(device).__name__, flush=True)
except Exception as e:
    print("  error creating device:", e, flush=True)
    # Try creating device manually
    adapter = wgpu.request_adapter_sync(canvas=None, power_preference="high-performance")
    device = adapter.request_device_sync(required_limits={})
    print("  manual device:", type(device).__name__, flush=True)

# WGSL compute shader: add two buffers element-wise
shader_source = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&a)) {
        result[i] = a[i] + b[i];
    }
}
"""

# Create data
n = 16
a = np.array([float(i) for i in range(n)], dtype=np.float32)
b = np.array([float(n - i) for i in range(n)], dtype=np.float32)

# Create buffers
buf_a = device.create_buffer_with_data(data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE)
buf_b = device.create_buffer_with_data(data=b.tobytes(), usage=wgpu.BufferUsage.STORAGE)
buf_result = device.create_buffer(size=a.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

# Create shader module
shader = device.create_shader_module(code=shader_source)

# Create compute pipeline
pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={"module": shader, "entry_point": "main"},
)

# Bind buffers
bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": a.nbytes}},
        {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": b.nbytes}},
        {"binding": 2, "resource": {"buffer": buf_result, "offset": 0, "size": a.nbytes}},
    ],
)

# Dispatch compute
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
compute_pass.dispatch_workgroups(n // 64 + 1, 1, 1)
compute_pass.end()

# Read back result
buf_read = device.create_buffer(size=a.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
command_encoder.copy_buffer_to_buffer(buf_result, 0, buf_read, 0, a.nbytes)
device.queue.submit([command_encoder.finish()])

data = device.queue.read_buffer(buf_read).cast("f").tolist()
print("  a:", a.tolist(), flush=True)
print("  b:", b.tolist(), flush=True)
print("  a+b:", data[:n], flush=True)

expected = [a[i] + b[i] for i in range(n)]
print("  expected:", expected, flush=True)
print("  match:", data[:n] == expected, flush=True)

print("wgpu GPU compute WORKS!" if data[:n] == expected else "FAIL", flush=True)
