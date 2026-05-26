#!/usr/bin/env python3
"""Test wgpu GPU compute on Colab."""
import sys, os, struct, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wgpu
import numpy as np
from wgpu.utils import get_default_device

device = get_default_device()
print("device:", type(device).__name__, flush=True)
print("adapter:", device.adapter.info.description if hasattr(device, "adapter") else "unknown", flush=True)

# WGSL compute shader
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

n = 16
a = np.array([float(i) for i in range(n)], dtype=np.float32)
b = np.array([float(n - i) for i in range(n)], dtype=np.float32)

buf_a = device.create_buffer_with_data(data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE)
buf_b = device.create_buffer_with_data(data=b.tobytes(), usage=wgpu.BufferUsage.STORAGE)
buf_result = device.create_buffer(size=a.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

shader = device.create_shader_module(code=shader_source)

pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={"module": shader, "entry_point": "main"},
)

bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": a.nbytes}},
        {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": b.nbytes}},
        {"binding": 2, "resource": {"buffer": buf_result, "offset": 0, "size": a.nbytes}},
    ],
)

cmd = device.create_command_encoder()
pass_ = cmd.begin_compute_pass()
pass_.set_pipeline(pipeline)
pass_.set_bind_group(0, bind_group, [], 0, 999999)
pass_.dispatch_workgroups(n // 64 + 1, 1, 1)
pass_.end()

buf_read = device.create_buffer(size=a.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
cmd.copy_buffer_to_buffer(buf_result, 0, buf_read, 0, a.nbytes)
device.queue.submit([cmd.finish()])

# Read back using map
data = device.queue.read_buffer(buf_read).cast("f").tolist()
print("a+b:", data[:n], flush=True)
expected = [a[i] + b[i] for i in range(n)]
print("match:", data[:n] == expected, flush=True)

print("GPU WORKS!" if data[:n] == expected else "FAIL", flush=True)
