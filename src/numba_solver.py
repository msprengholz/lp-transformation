"""
Numba-accelerated solver for lamination parameter back-transformation.

JIT-compiles the hot-path functions to machine code via LLVM.
Falls back to numpy if numba is not installed.
"""

import numpy as np
from numpy.typing import NDArray
from functools import lru_cache

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*a, **kw):
        def wrapper(f):
            return f
        return wrapper


# ═══════════════════════════════════════════════
# JIT: forward LP
# ═══════════════════════════════════════════════

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_lp_numba(lam, Z2, Z3, invN, N2, N3):
    N = lam.size
    lp = np.empty(12, dtype=np.float32)
    cos2 = np.cos(lam * 2)
    sin2 = np.sin(lam * 2)
    cos4 = np.cos(lam * 4)
    sin4 = np.sin(lam * 4)

    # In-plane
    c2 = 0.0; s2 = 0.0; c4 = 0.0; s4 = 0.0
    for i in range(N):
        c2 += cos2[i]; s2 += sin2[i]; c4 += cos4[i]; s4 += sin4[i]
    lp[0] = c2 * invN
    lp[1] = s2 * invN
    lp[2] = c4 * invN
    lp[3] = s4 * invN

    # Coupling
    s = 0.0
    for i in range(N):
        s += Z2[i] * cos2[i]
    lp[4] = s * N2
    s = 0.0
    for i in range(N):
        s += Z2[i] * sin2[i]
    lp[5] = s * N2
    s = 0.0
    for i in range(N):
        s += Z2[i] * cos4[i]
    lp[6] = s * N2
    s = 0.0
    for i in range(N):
        s += Z2[i] * sin4[i]
    lp[7] = s * N2

    # Out-of-plane
    s = 0.0
    for i in range(N):
        s += Z3[i] * cos2[i]
    lp[8] = s * N3
    s = 0.0
    for i in range(N):
        s += Z3[i] * sin2[i]
    lp[9] = s * N3
    s = 0.0
    for i in range(N):
        s += Z3[i] * cos4[i]
    lp[10] = s * N3
    s = 0.0
    for i in range(N):
        s += Z3[i] * sin4[i]
    lp[11] = s * N3

    return lp


# ═══════════════════════════════════════════════
# JIT: combined LP + gradient
# ═══════════════════════════════════════════════

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_lp_and_grad_numba(lam, lp_t, Z2, Z3, invN, N2, N3):
    N = lam.size
    lp = np.empty(12, dtype=np.float32)
    grad = np.empty(N, dtype=np.float32)

    lam2 = lam * 2
    lam4 = lam * 4
    cos2 = np.cos(lam2)
    sin2 = np.sin(lam2)
    cos4 = np.cos(lam4)
    sin4 = np.sin(lam4)

    # ── LP ──
    c2 = 0.0; s2 = 0.0; c4 = 0.0; s4 = 0.0
    for i in range(N):
        c2 += cos2[i]; s2 += sin2[i]; c4 += cos4[i]; s4 += sin4[i]
    lp[0] = c2 * invN
    lp[1] = s2 * invN
    lp[2] = c4 * invN
    lp[3] = s4 * invN

    s = 0.0
    for i in range(N):
        s += Z2[i] * cos2[i]
    lp[4] = s * N2
    s = 0.0
    for i in range(N):
        s += Z2[i] * sin2[i]
    lp[5] = s * N2
    s = 0.0
    for i in range(N):
        s += Z2[i] * cos4[i]
    lp[6] = s * N2
    s = 0.0
    for i in range(N):
        s += Z2[i] * sin4[i]
    lp[7] = s * N2
    s = 0.0
    for i in range(N):
        s += Z3[i] * cos2[i]
    lp[8] = s * N3
    s = 0.0
    for i in range(N):
        s += Z3[i] * sin2[i]
    lp[9] = s * N3
    s = 0.0
    for i in range(N):
        s += Z3[i] * cos4[i]
    lp[10] = s * N3
    s = 0.0
    for i in range(N):
        s += Z3[i] * sin4[i]
    lp[11] = s * N3

    # ── Gradient ──
    for k in range(N):
        val = (-2 * sin2[k] * (lp_t[0] - lp[0]) +
                2 * cos2[k] * (lp_t[1] - lp[1]) +
               -2 * sin4[k] * (lp_t[2] - lp[2]) +
                2 * cos4[k] * (lp_t[3] - lp[3]) +
               -2 * sin2[k] * Z2[k] * (lp_t[4] - lp[4]) +
                2 * cos2[k] * Z2[k] * (lp_t[5] - lp[5]) +
               -2 * sin4[k] * Z2[k] * (lp_t[6] - lp[6]) +
                2 * cos4[k] * Z2[k] * (lp_t[7] - lp[7]) +
               -2 * sin2[k] * Z3[k] * (lp_t[8] - lp[8]) +
                2 * cos2[k] * Z3[k] * (lp_t[9] - lp[9]) +
               -2 * sin4[k] * Z3[k] * (lp_t[10] - lp[10]) +
                2 * cos4[k] * Z3[k] * (lp_t[11] - lp[11]))
        grad[k] = -val * 2

    return lp, grad


# ═══════════════════════════════════════════════
# JIT: trig arrays + LP from trig (for incremental ssearch)
# ═══════════════════════════════════════════════

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _trig_from_lam(lam):
    """Compute cos2, sin2, cos4, sin4 arrays from a laminate."""
    return (np.cos(lam * 2), np.sin(lam * 2),
            np.cos(lam * 4), np.sin(lam * 4))


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _lp_from_trig(cos2, sin2, cos4, sin4, Z2, Z3, invN, N2, N3):
    """Compute 12 LP values from precomputed trig arrays."""
    N = cos2.size
    lp = np.empty(12, dtype=np.float32)

    c2 = 0.0; s2 = 0.0; c4 = 0.0; s4 = 0.0
    for i in range(N):
        c2 += cos2[i]; s2 += sin2[i]; c4 += cos4[i]; s4 += sin4[i]
    lp[0] = c2 * invN; lp[1] = s2 * invN
    lp[2] = c4 * invN; lp[3] = s4 * invN

    s = 0.0
    for i in range(N): s += Z2[i] * cos2[i]
    lp[4] = s * N2; s = 0.0
    for i in range(N): s += Z2[i] * sin2[i]
    lp[5] = s * N2; s = 0.0
    for i in range(N): s += Z2[i] * cos4[i]
    lp[6] = s * N2; s = 0.0
    for i in range(N): s += Z2[i] * sin4[i]
    lp[7] = s * N2; s = 0.0
    for i in range(N): s += Z3[i] * cos2[i]
    lp[8] = s * N3; s = 0.0
    for i in range(N): s += Z3[i] * sin2[i]
    lp[9] = s * N3; s = 0.0
    for i in range(N): s += Z3[i] * cos4[i]
    lp[10] = s * N3; s = 0.0
    for i in range(N): s += Z3[i] * sin4[i]
    lp[11] = s * N3
    return lp


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _build_trig_table(ang_steps, delta):
    """Precompute trig for all grid angles. Returns 5 arrays: angles, c2, s2, c4, s4."""
    half_pi = np.float32(np.pi / 2.0)
    angles = np.empty(ang_steps, dtype=np.float32)
    c2 = np.empty(ang_steps, dtype=np.float32)
    s2 = np.empty(ang_steps, dtype=np.float32)
    c4 = np.empty(ang_steps, dtype=np.float32)
    s4 = np.empty(ang_steps, dtype=np.float32)
    for k in range(1, ang_steps + 1):
        a = -half_pi + np.float32(delta * k)
        angles[k-1] = a
        c2[k-1] = np.cos(a * 2)
        s2[k-1] = np.sin(a * 2)
        c4[k-1] = np.cos(a * 4)
        s4[k-1] = np.sin(a * 4)
    return angles, c2, s2, c4, s4


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _eval_angle_from_table(cos2, sin2, cos4, sin4, i, k,
                           t_c2, t_s2, t_c4, t_s4,
                           Z2, Z3, invN, N2, N3, lp_t):
    """Set trig for layer i from table index k, compute LP RMSE loss."""
    cos2[i] = t_c2[k]
    sin2[i] = t_s2[k]
    cos4[i] = t_c4[k]
    sin4[i] = t_s4[k]
    lp = _lp_from_trig(cos2, sin2, cos4, sin4, Z2, Z3, invN, N2, N3)
    s = np.float32(0.0)
    for j in range(12):
        d = lp[j] - lp_t[j]; s += d * d
    return np.sqrt(s)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _ssearch_numba(lam, lp_t, delta, ang_steps, Z2, Z3, invN, N2, N3):
    """
    Sequential search with precomputed trig tables.

    Trig for all grid angles precomputed once. Ssearch uses table lookups
    instead of cos/sin calls — zero trig computation per candidate.
    """
    layers = lam.size
    best_lam = lam.copy()

    # Precompute trig tables for all grid angles
    grid_angles, t_c2, t_s2, t_c4, t_s4 = _build_trig_table(ang_steps, delta)
    # Convert to 0-indexed: idx = k-1

    for i in range(layers):
        cos2, sin2, cos4, sin4 = _trig_from_lam(best_lam)

        # Stage 1: coarse grid (odd indices -> 0, 2, 4, ... in 0-indexed)
        best_loss = np.float32(np.inf)
        best_k = 0  # 0-indexed

        for k in range(0, ang_steps, 2):
            oc2 = cos2[i]; os2 = sin2[i]; oc4 = cos4[i]; os4 = sin4[i]
            loss = _eval_angle_from_table(cos2, sin2, cos4, sin4, i, k,
                                           t_c2, t_s2, t_c4, t_s4,
                                           Z2, Z3, invN, N2, N3, lp_t)
            cos2[i] = oc2; sin2[i] = os2; cos4[i] = oc4; sin4[i] = os4
            if loss < best_loss:
                _eval_angle_from_table(cos2, sin2, cos4, sin4, i, k,
                                        t_c2, t_s2, t_c4, t_s4,
                                        Z2, Z3, invN, N2, N3, lp_t)
                best_loss = loss; best_k = k
                best_lam[i] = grid_angles[k]
                lam[i] = grid_angles[k]

        # Stage 2: refine neighbours (even indices -> odd 0-indexed)
        for offset in (-1, 1):
            k = best_k + offset
            if 0 <= k < ang_steps and k % 2 == 1:
                oc2 = cos2[i]; os2 = sin2[i]; oc4 = cos4[i]; os4 = sin4[i]
                loss = _eval_angle_from_table(cos2, sin2, cos4, sin4, i, k,
                                               t_c2, t_s2, t_c4, t_s4,
                                               Z2, Z3, invN, N2, N3, lp_t)
                cos2[i] = oc2; sin2[i] = os2; cos4[i] = oc4; sin4[i] = os4
                if loss < best_loss:
                    _eval_angle_from_table(cos2, sin2, cos4, sin4, i, k,
                                            t_c2, t_s2, t_c4, t_s4,
                                            Z2, Z3, invN, N2, N3, lp_t)
                    best_loss = loss
                    best_lam[i] = grid_angles[k]
                    lam[i] = grid_angles[k]

        best_lam[i] = lam[i]

    return best_lam


# ═══════════════════════════════════════════════
# JIT: iRprop-
# ═══════════════════════════════════════════════

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _irpropm_numba(lam, lp_t, it_iRprop, Z2, Z3, invN, N2, N3,
                   sigma, s_min, s_max, n_p, n_m, grad_tol):
    layers = lam.size
    s = np.full(layers, sigma, dtype=np.float32)
    _, grad0 = _get_lp_and_grad_numba(lam, lp_t, Z2, Z3, invN, N2, N3)
    grad1 = np.empty_like(grad0)
    for _ in range(it_iRprop):
        _, grad1 = _get_lp_and_grad_numba(lam, lp_t, Z2, Z3, invN, N2, N3)
        gmax = np.float32(0.0)
        for k in range(layers):
            gmax = max(gmax, abs(grad1[k]))
        if gmax < grad_tol:
            break
        for k in range(layers):
            if grad0[k] * grad1[k] > 0:
                s[k] = min(s[k] * n_p, s_max)
            elif grad0[k] * grad1[k] < 0:
                s[k] = max(s[k] * n_m, s_min)
                grad1[k] = np.float32(0.0)
            lam[k] -= np.sign(grad1[k]) * s[k]
            grad0[k] = grad1[k]
    return lam


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

@lru_cache(maxsize=8)
def _prepare_arrays(N):
    k = np.arange(N, dtype=np.float32)
    Z2 = ((-N / 2 + k + 1) ** 2 - (-N / 2 + k) ** 2).astype(np.float32)
    Z3 = ((-N / 2 + k + 1) ** 3 - (-N / 2 + k) ** 3).astype(np.float32)
    fN = float(N)
    return (Z2, Z3,
            np.float32(1.0 / fN),
            np.float32(2.0 / (fN * fN)),
            np.float32(4.0 / (fN * fN * fN)))


# ═══════════════════════════════════════════════
# JIT: batch-optimise all starts (eliminates Python loop overhead)
# ═══════════════════════════════════════════════

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _optimize_all_numba(rand_lams, lp_t,
                         dc, ac, do_fine, df, af,
                         irprop_iters, sigma, s_min, s_max, n_p, n_m, gtol,
                         n_coarse_fine,
                         Z2, Z3, invN, N2, N3):
    """JIT-compiled multi-start loop."""
    num_samples, layers = rand_lams.shape
    half_pi = np.float32(np.pi / 2.0)
    pi_f = np.float32(np.pi)
    out = np.empty_like(rand_lams)
    los = np.empty(num_samples, dtype=np.float32)

    for idx in range(num_samples):
        lam = rand_lams[idx].copy()

        for _ in range(n_coarse_fine):
            lam = _ssearch_numba(lam, lp_t, dc, ac, Z2, Z3, invN, N2, N3)
            if do_fine:
                lam = _ssearch_numba(lam, lp_t, df, af, Z2, Z3, invN, N2, N3)

        lam = _irpropm_numba(lam, lp_t, irprop_iters,
                              Z2, Z3, invN, N2, N3,
                              sigma, s_min, s_max, n_p, n_m, gtol)

        for k in range(layers):
            lam[k] = (lam[k] + half_pi) % pi_f - half_pi
        out[idx] = lam

        lp = _get_lp_numba(lam, Z2, Z3, invN, N2, N3)
        s = np.float32(0.0)
        for j in range(12):
            d = lp[j] - lp_t[j]
            s += d * d
        los[idx] = np.sqrt(s / layers)

    return out, los


# ═══════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════

def optimize_laminate_numba(rand_lams: NDArray[np.float32],
                            lp_t: NDArray[np.float32],
                            n_coarse_fine: int = 1,
                            delta_coarse_deg: float = 12.0,
                            delta_fine_deg: float = 0.0,
                            irprop_iters: int = 3000,
                            irprop_grad_tol: float = 1e-3,
                            ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Full optimisation pipeline using numba-JIT kernels.

    The multi-start loop runs inside a single JIT-compiled function
    (``_optimize_all_numba``), eliminating Python iteration and
    JIT-call transition overhead for maximum throughput.

    Falls back to numpy if numba is not installed.
    """
    if not HAS_NUMBA:
        from .numpy_solver import optimize_laminate as _fb
        return _fb(rand_lams, lp_t, n_coarse_fine,
                    delta_coarse_deg, delta_fine_deg,
                    irprop_iters, irprop_grad_tol)

    Z2, Z3, invN, N2, N3 = _prepare_arrays(rand_lams.shape[1])
    dc = np.float32(np.deg2rad(delta_coarse_deg))
    ac = int(np.floor(np.pi / dc))

    if delta_fine_deg > 0:
        df = np.float32(np.deg2rad(delta_fine_deg))
        af = int(np.floor(np.pi / df))
        do_fine = True
    else:
        df = dc
        af = 0
        do_fine = False

    sigma = np.float32(0.1)
    s_min = np.float32(1e-8)
    s_max = np.float32(0.3)
    n_p = np.float32(1.2)
    n_m = np.float32(0.5)
    gtol = np.float32(irprop_grad_tol)

    return _optimize_all_numba(
        rand_lams, lp_t,
        dc, ac, do_fine, df, af,
        irprop_iters, sigma, s_min, s_max, n_p, n_m, gtol,
        n_coarse_fine,
        Z2, Z3, invN, N2, N3,
    )
