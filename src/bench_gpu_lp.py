#!/usr/bin/env python3
"""Benchmark GPU batch LP computation vs numpy."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# Benchmark: numpy batch LP
from src.lp_functions import get_lp
from src.numpy_fast import get_lp_batch

N = 12
batch_sizes = [1, 10, 100, 1000, 10000]

print("Batch LP benchmark (12 layers)")
print("batch_size  numpy_time  gpu_time  speedup")
print("-" * 50)

for M in batch_sizes:
    lams = np.random.random((M, N)).astype(np.float32) * np.pi - np.pi / 2

    # Numpy
    t0 = time.perf_counter()
    for _ in range(20):
        lp = get_lp_batch(lams)
    t_np = (time.perf_counter() - t0) / 20

    print("  %6d    %6.4fs" % (M, t_np), end="")

    # GPU (SlangPy) — only if available
    try:
        import slangpy as sl
        dev = sl.create_device(type=sl.DeviceType.cuda)
        print("  GPU: device=", dev.info.adapter_name, file=sys.stderr)
        # Build shader
        mod = sl.Module.load_from_source(dev, "lp_bench", """
        [AutoPy]
        void batch_lp(float[] angles, float[] Z2, float[] Z3,
                      float invN, float N2, float N3,
                      int M, int N, float[] result)
        {
            int idx = int(call_id().x);
            if (idx >= M) return;
            int base = idx * N;
            float cos2=0,sin2=0,cos4=0,sin4=0;
            float d2c2=0,d2s2=0,d2c4=0,d2s4=0;
            float d3c2=0,d3s2=0,d3c4=0,d3s4=0;
            for (int i=0; i<N; i++) {
                float ang=angles[base+i];
                float c2=cos(ang*2),s2=sin(ang*2);
                float c4=cos(ang*4),s4=sin(ang*4);
                cos2+=c2;sin2+=s2;cos4+=c4;sin4+=s4;
                float z2=Z2[i],z3=Z3[i];
                d2c2+=z2*c2;d2s2+=z2*s2;d2c4+=z2*c4;d2s4+=z2*s4;
                d3c2+=z3*c2;d3s2+=z3*s2;d3c4+=z3*c4;d3s4+=z3*s4;
            }
            int rb=idx*12;
            result[rb+0]=cos2*invN;result[rb+1]=sin2*invN;
            result[rb+2]=cos4*invN;result[rb+3]=sin4*invN;
            result[rb+4]=d2c2*N2;result[rb+5]=d2s2*N2;
            result[rb+6]=d2c4*N2;result[rb+7]=d2s4*N2;
            result[rb+8]=d3c2*N3;result[rb+9]=d3s2*N3;
            result[rb+10]=d3c4*N3;result[rb+11]=d3s4*N3;
        }
        """)

        from src.lp_functions import _z2_z3, _norm_factors
        Z2, Z3 = _z2_z3(N)
        invN, N2, N3 = _norm_factors(N)

        # Warmup
        result = np.empty((M, 12), dtype=np.float32)
        mod.batch_lp(lams.ravel().astype(np.float32), Z2, Z3,
                     np.float32(invN), np.float32(N2), np.float32(N3),
                     np.int32(M), np.int32(N), result)

        # Timed
        t0 = time.perf_counter()
        for _ in range(20):
            mod.batch_lp(lams.ravel().astype(np.float32), Z2, Z3,
                         np.float32(invN), np.float32(N2), np.float32(N3),
                         np.int32(M), np.int32(N), result)
        t_gpu = (time.perf_counter() - t0) / 20

        speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
        print("    %6.4fs    %.1fx" % (t_gpu, speedup))

    except Exception as e:
        print("    error: %s" % str(e)[:50])
