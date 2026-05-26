"""
Numba-accelerated solver for lamination parameter back-transformation.

JIT-compiles the hot-path functions (get_lp, get_loss_grad, ssearch, iRpropm)
to machine code via LLVM, eliminating Python loop overhead.

Usage:
    from src.numba_solver import optimize_laminate_numba
    opt_lams, losses = optimize_laminate_numba(rand_lams, lp_t)
"""

import numpy as np
from numpy.typing import NDArray

# Numba is optional — fall back to numpy if not available
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define no-op decorator so the module can at least be parsed
    def jit(*a, **kw):
        def wrapper(f):
            return f
        return wrapper
    prange = range


# ──────────────────────────────────────────────
# Core JIT functions
# ──────────────────────────────────────────────

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_lp_numba(lam, Z2, Z3, invN, N2, N3):
    """
    Numba-JIT version of get_lp.

    Parameters are pre-computed arrays/scalars (no lru_cache inside JIT).
    """
    N = lam.size
    lp = np.zeros(12, dtype=np.float32)

    cos2 = np.cos(lam * 2)
    sin2 = np.sin(lam * 2)
    cos4 = np.cos(lam * 4)
    sin4 = np.sin(lam * 4)

    # In-plane
    lp[0] = np.sum(cos2) * invN
    lp[1] = np.sum(sin2) * invN
    lp[2] = np.sum(cos4) * invN
    lp[3] = np.sum(sin4) * invN

    # Coupling (use manual dot products since numba handles them)
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * cos2[i]
    lp[4] = s2 * N2
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * sin2[i]
    lp[5] = s2 * N2
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * cos4[i]
    lp[6] = s2 * N2
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * sin4[i]
    lp[7] = s2 * N2

    # Out-of-plane
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * cos2[i]
    lp[8] = s2 * N3
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * sin2[i]
    lp[9] = s2 * N3
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * cos4[i]
    lp[10] = s2 * N3
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * sin4[i]
    lp[11] = s2 * N3

    return lp


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_lp_and_grad_numba(lam, lp_t, Z2, Z3, invN, N2, N3):
    """
    Numba-JIT combined LP + gradient, avoiding redundant trig.

    Returns (lp, grad) as two arrays.
    """
    N = lam.size
    lp = np.zeros(12, dtype=np.float32)
    grad = np.zeros(N, dtype=np.float32)

    lam2 = lam * 2
    lam4 = lam * 4
    cos2 = np.cos(lam2)
    sin2 = np.sin(lam2)
    cos4 = np.cos(lam4)
    sin4 = np.sin(lam4)

    # ── LP ──
    s2 = 0.0
    for i in range(N):
        s2 += cos2[i]
    lp[0] = s2 * invN
    s2 = 0.0
    for i in range(N):
        s2 += sin2[i]
    lp[1] = s2 * invN
    s2 = 0.0
    for i in range(N):
        s2 += cos4[i]
    lp[2] = s2 * invN
    s2 = 0.0
    for i in range(N):
        s2 += sin4[i]
    lp[3] = s2 * invN

    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * cos2[i]
    lp[4] = s2 * N2
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * sin2[i]
    lp[5] = s2 * N2
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * cos4[i]
    lp[6] = s2 * N2
    s2 = 0.0
    for i in range(N):
        s2 += Z2[i] * sin4[i]
    lp[7] = s2 * N2

    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * cos2[i]
    lp[8] = s2 * N3
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * sin2[i]
    lp[9] = s2 * N3
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * cos4[i]
    lp[10] = s2 * N3
    s2 = 0.0
    for i in range(N):
        s2 += Z3[i] * sin4[i]
    lp[11] = s2 * N3

    # ── Gradient (descent direction) ──
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


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _ssearch_numba(lam, lp_t, delta, ang_steps, Z2, Z3, invN, N2, N3):
    """
    Numba-JIT sequential coordinate search.

    Returns the best-found laminate.
    """
    layers = lam.size
    best_lam = lam.copy()

    for i in range(layers):
        best_loss = np.float32('inf')
        for k in range(1, ang_steps + 1):
            best_lam[i] = np.float32(-np.pi / 2.0 + delta * k)
            # Compute LP for this candidate
            lp = _get_lp_numba(best_lam, Z2, Z3, invN, N2, N3)
            loss = np.float32(0.0)
            for j in range(12):
                diff = lp[j] - lp_t[j]
                loss += diff * diff
            loss = np.sqrt(loss)
            if loss < best_loss:
                lam[i] = best_lam[i]
                best_loss = loss
        best_lam[i] = lam[i]

    return best_lam


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _irpropm_numba(lam, lp_t, it_iRprop, Z2, Z3, invN, N2, N3,
                   sigma, s_min, s_max, n_p, n_m, grad_tol):
    """
    Numba-JIT iRprop- with gradient-norm early stopping.
    """
    layers = lam.size
    s = np.full(layers, sigma, dtype=np.float32)

    lp, grad0 = _get_lp_and_grad_numba(lam, lp_t, Z2, Z3, invN, N2, N3)
    grad1 = np.empty_like(grad0)

    for _ in range(it_iRprop):
        lp, grad1 = _get_lp_and_grad_numba(lam, lp_t, Z2, Z3, invN, N2, N3)

        # Gradient-norm convergence
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


# ──────────────────────────────────────────────
# Helpers: precompute layer geometry
# ──────────────────────────────────────────────

def _prepare_arrays(N):
    """Return (Z2, Z3, invN, N2, N3) for a given layer count."""
    k = np.arange(N, dtype=np.float32)
    Z2 = ((-N / 2 + k + 1) ** 2 - (-N / 2 + k) ** 2).astype(np.float32)
    Z3 = ((-N / 2 + k + 1) ** 3 - (-N / 2 + k) ** 3).astype(np.float32)
    fN = float(N)
    invN = np.float32(1.0 / fN)
    N2 = np.float32(2.0 / (fN * fN))
    N3 = np.float32(4.0 / (fN * fN * fN))
    return Z2, Z3, invN, N2, N3


# ──────────────────────────────────────────────
# Public API — drop-in replacement for numpy_solver
# ──────────────────────────────────────────────

def optimize_laminate_numba(rand_lams: NDArray[np.float32],
                            lp_t: NDArray[np.float32],
                            n_coarse_fine: int = 3,
                            delta_coarse_deg: float = 10.0,
                            delta_fine_deg: float = 5.0,
                            irprop_iters: int = 3000,
                            irprop_grad_tol: float = 1e-6,
                            ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Full optimisation pipeline using numba-JIT kernels.

    Same interface as ``numpy_solver.optimize_laminate``.
    Falls back to numpy if numba is not installed.
    """
    if not HAS_NUMBA:
        from .numpy_solver import optimize_laminate as _fallback
        return _fallback(rand_lams, lp_t, n_coarse_fine,
                          delta_coarse_deg, delta_fine_deg,
                          irprop_iters, irprop_grad_tol)

    num_samples, layers = rand_lams.shape
    Z2, Z3, invN, N2, N3 = _prepare_arrays(layers)

    delta_coarse = np.float32(np.deg2rad(delta_coarse_deg))
    delta_fine = np.float32(np.deg2rad(delta_fine_deg))
    ang_steps_coarse = int(np.floor(np.pi / delta_coarse))
    ang_steps_fine = int(np.floor(np.pi / delta_fine))

    optimised_lams = np.zeros_like(rand_lams)
    losses = np.zeros(num_samples, dtype=np.float32)

    for idx in range(num_samples):
        lam = rand_lams[idx].copy()

        # Coarse-to-fine grid search (JIT)
        for _ in range(n_coarse_fine):
            lam = _ssearch_numba(lam, lp_t, delta_coarse, ang_steps_coarse,
                                  Z2, Z3, invN, N2, N3)
            lam = _ssearch_numba(lam, lp_t, delta_fine, ang_steps_fine,
                                  Z2, Z3, invN, N2, N3)

        # iRprop refinement (JIT)
        lam = _irpropm_numba(lam, lp_t, irprop_iters,
                              Z2, Z3, invN, N2, N3,
                              0.1, 1e-8, 0.3, 1.2, 0.5, irprop_grad_tol)

        # Wrap to [-π/2, π/2]
        for k in range(layers):
            lam[k] = (lam[k] + np.float32(np.pi / 2)) % np.float32(np.pi) - np.float32(np.pi / 2)

        optimised_lams[idx] = lam

        # RMSE loss
        lp = _get_lp_numba(lam, Z2, Z3, invN, N2, N3)
        s = np.float32(0.0)
        for j in range(12):
            d = lp[j] - lp_t[j]
            s += d * d
        losses[idx] = np.sqrt(s / layers)

    return optimised_lams, losses
