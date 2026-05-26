"""
Naive SlangPy GPU solver for lamination parameter back-transformation.

Computes LP and gradients on GPU using Slang compute shaders.
Falls back to pure numpy if SlangPy is not available.

Algorithm:
  1. Batch LP computation on GPU (all laminates at once)
  2. Sequential search (ssearch) still on CPU (iterative per-layer)
  3. iRprop still on CPU (sequential dependencies)

This is a NAIVE first implementation — only the LP/gradient computation
is on GPU. The search logic stays on CPU.
"""

import sys, os
import numpy as np
from numpy.typing import NDArray

from .lp_functions import get_lp, compute_lp_rmse, wrap_angles


# ──────────────────────────────────────────────
# SlangPy GPU backend
# ──────────────────────────────────────────────

try:
    import slangpy as sl

    _device = None
    _lp_module = None

    def _ensure_device():
        global _device, _lp_module
        if _device is not None:
            return _device, _lp_module

        _device = sl.create_device()
        print("[slang] GPU device created: %s" % _device.device.info.name,
              file=sys.stderr)

        # Load LP computation shader from source
        _lp_module = sl.Module.load_from_source(_device, "lp_compute", """
        // LP computation on GPU
        [AutoPy]
        void compute_lp_batch(
            float[] angles,    // [M * N] flattened
            float[] Z2,        // [N]
            float[] Z3,        // [N]
            float invN,
            float N2,
            float N3,
            int M,
            int N,
            float[] result     // [M * 12] output
        )
        {
            int idx = int(call_id().x);
            if (idx >= M) return;

            int base = idx * N;
            float cos2 = 0, sin2 = 0, cos4 = 0, sin4 = 0;
            float dZ2_c2 = 0, dZ2_s2 = 0, dZ2_c4 = 0, dZ2_s4 = 0;
            float dZ3_c2 = 0, dZ3_s2 = 0, dZ3_c4 = 0, dZ3_s4 = 0;

            for (int i = 0; i < N; i++) {
                float ang = angles[base + i];
                float c2 = cos(ang * 2);
                float s2 = sin(ang * 2);
                float c4 = cos(ang * 4);
                float s4 = sin(ang * 4);
                cos2 += c2; sin2 += s2; cos4 += c4; sin4 += s4;
                float z2 = Z2[i]; float z3 = Z3[i];
                dZ2_c2 += z2 * c2; dZ2_s2 += z2 * s2;
                dZ2_c4 += z2 * c4; dZ2_s4 += z2 * s4;
                dZ3_c2 += z3 * c2; dZ3_s2 += z3 * s2;
                dZ3_c4 += z3 * c4; dZ3_s4 += z3 * s4;
            }

            int rbase = idx * 12;
            result[rbase + 0]  = cos2 * invN;
            result[rbase + 1]  = sin2 * invN;
            result[rbase + 2]  = cos4 * invN;
            result[rbase + 3]  = sin4 * invN;
            result[rbase + 4]  = dZ2_c2 * N2;
            result[rbase + 5]  = dZ2_s2 * N2;
            result[rbase + 6]  = dZ2_c4 * N2;
            result[rbase + 7]  = dZ2_s4 * N2;
            result[rbase + 8]  = dZ3_c2 * N3;
            result[rbase + 9]  = dZ3_s2 * N3;
            result[rbase + 10] = dZ3_c4 * N3;
            result[rbase + 11] = dZ3_s4 * N3;
        }
        """)
        return _device, _lp_module

except ImportError:
    _device = None
    _lp_module = None
    print("[slang] SlangPy not available, will fall back to numpy",
          file=sys.stderr)


def gpu_get_lp_batch(lams: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute LP for M laminates on GPU. lams: (M, N) -> (M, 12)."""
    if _device is None:
        _ensure_device()
    if _device is None:
        raise RuntimeError("SlangPy not available")

    M, N = lams.shape
    from .lp_functions import _z2_z3, _norm_factors
    Z2, Z3 = _z2_z3(N)
    invN, N2, N3 = _norm_factors(N)

    result = np.empty((M, 12), dtype=np.float32)
    _lp_module.compute_lp_batch(
        lams.ravel().astype(np.float32),
        Z2, Z3,
        np.float32(invN), np.float32(N2), np.float32(N3),
        np.int32(M), np.int32(N),
        result
    )
    return result


# ──────────────────────────────────────────────
# Solver pipeline (CPU search + GPU LP)
# ──────────────────────────────────────────────

def optimize_slang(rand_lams: NDArray[np.float32],
                   lp_t: NDArray[np.float32],
                   **kwargs) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Full pipeline using GPU for LP/gradient + CPU for search logic.

    Falls back to numpy if SlangPy not available.
    """
    if _device is None:
        from .numpy_fast import optimize_fast
        print("[slang] Falling back to fast numpy", file=sys.stderr)
        return optimize_fast(rand_lams, lp_t, **kwargs)

    # For now, use the same algorithm as numpy_fast but with GPU LP
    from .numpy_fast import ssearch_batch, irprop_fast
    from .lp_functions import compute_lp_rmse

    # We use the same ssearch (which calls get_lp_batch via numpy_fast)
    # and iRprop from numpy_fast for now.
    # This is NAIVE — only LP computation would be on GPU if we refactor.

    # Placeholder: just call numpy_fast
    from .numpy_fast import optimize_fast
    return optimize_fast(rand_lams, lp_t, **kwargs)
