"""
Core lamination parameter functions.

Forward:  given ply angles → 12 lamination parameters (A, B, D matrices)
Backward: given target LPs + initial guess → optimised ply angles

All functions operate on float32 for consistency.
"""

import numpy as np
from numpy.typing import NDArray
from functools import lru_cache


# ──────────────────────────────────────────────
# Cached layer-geometry arrays (Z2, Z3)
# These depend only on N (layer count), not on the
# actual angle values — compute once and reuse.
# ──────────────────────────────────────────────

@lru_cache(maxsize=32)
def _z2_z3(N: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Precompute Z2 and Z3 arrays for a given layer count N."""
    k = np.arange(N, dtype=np.float32)
    Z2 = ((-N / 2 + k + 1) ** 2 - (-N / 2 + k) ** 2).astype(np.float32)
    Z3 = ((-N / 2 + k + 1) ** 3 - (-N / 2 + k) ** 3).astype(np.float32)
    return Z2, Z3


@lru_cache(maxsize=32)
def _norm_factors(N: int) -> tuple[float, float, float]:
    """Normalisation factors: 1/N, 2/N², 4/N³."""
    fN = float(N)
    return 1.0 / fN, 2.0 / (fN * fN), 4.0 / (fN * fN * fN)


# ──────────────────────────────────────────────
# Low-level: compute trig arrays + LP from trig
# ──────────────────────────────────────────────

def _trig_arrays(lam: NDArray[np.float32
                               ]) -> tuple[NDArray[np.float32],
                                           NDArray[np.float32],
                                           NDArray[np.float32],
                                           NDArray[np.float32]]:
    """Compute cos2, sin2, cos4, sin4 from angle vector (no .astype needed)."""
    lam2 = lam * 2
    lam4 = lam * 4
    return (np.cos(lam2, dtype=np.float32),
            np.sin(lam2, dtype=np.float32),
            np.cos(lam4, dtype=np.float32),
            np.sin(lam4, dtype=np.float32))


def _lp_from_trig(cos2, sin2, cos4, sin4, Z2, Z3,
                  invN, N2, N3) -> NDArray[np.float32]:
    """Build the 12 LP components from precomputed trig and Z arrays."""
    lp = np.empty(12, dtype=np.float32)
    lp[0] = np.sum(cos2) * invN
    lp[1] = np.sum(sin2) * invN
    lp[2] = np.sum(cos4) * invN
    lp[3] = np.sum(sin4) * invN
    lp[4] = np.dot(Z2, cos2) * N2
    lp[5] = np.dot(Z2, sin2) * N2
    lp[6] = np.dot(Z2, cos4) * N2
    lp[7] = np.dot(Z2, sin4) * N2
    lp[8] = np.dot(Z3, cos2) * N3
    lp[9] = np.dot(Z3, sin2) * N3
    lp[10] = np.dot(Z3, cos4) * N3
    lp[11] = np.dot(Z3, sin4) * N3
    return lp


# ──────────────────────────────────────────────
# Forward computation  (single laminate)
# ──────────────────────────────────────────────

def get_lp(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute 12 lamination parameters from a laminate angle vector.

    Parameters
    ----------
    lam : (N,) float32 array — ply angles in radians, range [-π/2, π/2].

    Returns
    -------
    lp : (12,) float32 array — lamination parameters.
    """
    N = lam.size
    Z2, Z3 = _z2_z3(N)
    invN, N2, N3 = _norm_factors(N)
    cos2, sin2, cos4, sin4 = _trig_arrays(lam)
    return _lp_from_trig(cos2, sin2, cos4, sin4, Z2, Z3, invN, N2, N3)


# ──────────────────────────────────────────────
# Batch LP computation (vectorised over laminates)
# ──────────────────────────────────────────────

def calc_lp_array(lams: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute LP for multiple laminates at once.

    Parameters
    ----------
    lams : (M, N) float32 — M laminates, N layers each

    Returns
    -------
    lps : (M, 12) float32 — LP vectors
    """
    M = lams.shape[0]
    out = np.empty((M, 12), dtype=np.float32)
    for i in range(M):
        out[i] = get_lp(lams[i])
    return out


# ──────────────────────────────────────────────
# Combined LP + Gradient  (avoids redundant trig)
# ──────────────────────────────────────────────

def get_lp_and_grad(lam: NDArray[np.float32],
                    lp_t: NDArray[np.float32],
                    lp_out: NDArray[np.float32] | None = None
                    ) -> NDArray[np.float32]:
    """
    Compute LP and loss gradient in a single pass, reusing trig arrays.

    Parameters
    ----------
    lam    : (N,) float32 — current ply angles
    lp_t   : (12,) float32 — target lamination parameters
    lp_out : (12,) float32 or None — if provided, LP is stored here

    Returns
    -------
    grad : (N,) float32 — ∂loss/∂θ_i  (descent direction)
    """
    N = lam.size
    Z2, Z3 = _z2_z3(N)
    invN, N2, N3 = _norm_factors(N)
    cos2, sin2, cos4, sin4 = _trig_arrays(lam)

    # ── Forward LP ──
    lp = _lp_from_trig(cos2, sin2, cos4, sin4, Z2, Z3, invN, N2, N3)
    if lp_out is not None:
        lp_out[:] = lp

    # ── Gradient of 0.5 * ||lp - lp_t||^2  (descent direction) ──
    # lp_d = lp_t - lp  →  grad = -(lp - lp_t)·J = lp_d·J
    # With the original convention: return negative of the SSE gradient.
    lp_d = lp_t - lp
    grad = np.zeros(N, dtype=np.float32)
    grad += -2 * sin2 * lp_d[0]
    grad += 2 * cos2 * lp_d[1]
    grad += -2 * sin4 * lp_d[2]
    grad += 2 * cos4 * lp_d[3]
    grad += -2 * sin2 * Z2 * lp_d[4]
    grad += 2 * cos2 * Z2 * lp_d[5]
    grad += -2 * sin4 * Z2 * lp_d[6]
    grad += 2 * cos4 * Z2 * lp_d[7]
    grad += -2 * sin2 * Z3 * lp_d[8]
    grad += 2 * cos2 * Z3 * lp_d[9]
    grad += -2 * sin4 * Z3 * lp_d[10]
    grad += 2 * cos4 * Z3 * lp_d[11]

    return -grad * 2


def get_loss_grad(lam: NDArray[np.float32],
                  lp_t: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Gradient of the LP-matching loss w.r.t. each ply angle.

    For performance, call ``get_lp_and_grad`` when LP is also needed.
    """
    return get_lp_and_grad(lam, lp_t)


# ──────────────────────────────────────────────
# Analytic LP Jacobian  (∂ξⱼ / ∂θᵢ)
# ──────────────────────────────────────────────

def get_lp_jac(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Jacobian of the 12 LP outputs w.r.t. each ply angle.

    Shape (N, 12) where J[i, j] = ∂ξⱼ / ∂θᵢ.
    """
    N = lam.size
    Z2, Z3 = _z2_z3(N)
    invN, N2, N3 = _norm_factors(N)
    cos2, sin2, cos4, sin4 = _trig_arrays(lam)

    grad = np.empty((N, 12), dtype=np.float32)
    grad[:, 0] = -sin2 * 2 * invN
    grad[:, 1] = cos2 * 2 * invN
    grad[:, 2] = -sin4 * 4 * invN
    grad[:, 3] = cos4 * 4 * invN
    grad[:, 4] = -sin2 * 2 * Z2 * N2
    grad[:, 5] = cos2 * 2 * Z2 * N2
    grad[:, 6] = -sin4 * 4 * Z2 * N2
    grad[:, 7] = cos4 * 4 * Z2 * N2
    grad[:, 8] = -sin2 * 2 * Z3 * N3
    grad[:, 9] = cos2 * 2 * Z3 * N3
    grad[:, 10] = -sin4 * 4 * Z3 * N3
    grad[:, 11] = cos4 * 4 * Z3 * N3
    return grad


# ──────────────────────────────────────────────
# Loss computation (RMSE of LP residual)
# ──────────────────────────────────────────────

def compute_lp_rmse(lam: NDArray[np.float32],
                    lp_t: NDArray[np.float32]) -> float:
    """Root-mean-square error of LP mismatch, normalised by layer count."""
    lp_d = get_lp(lam) - lp_t
    return float(np.sqrt(np.sum(lp_d ** 2) / lam.size))


def compute_angle_deviation(lam_opt: NDArray[np.float32],
                            lam_true: NDArray[np.float32]) -> NDArray[np.float32]:
    """Minimum angular deviation accounting for periodicity [0, π)."""
    diff = np.abs(lam_opt - lam_true)
    return np.minimum(diff, np.pi - diff)


# ──────────────────────────────────────────────
# Test-problem generators
# ──────────────────────────────────────────────

def make_random_laminate(n_layers: int, rng: np.random.Generator | None = None
                         ) -> NDArray[np.float32]:
    """Generate a random laminate with angles in [-π/2, π/2]."""
    if rng is None:
        rng = np.random.default_rng()
    de = 11.25
    ang_steps = int(np.floor(180 / de))
    steps = rng.integers(0, ang_steps, size=n_layers).astype(np.float32)
    return np.deg2rad(steps * de - (90 - de)).astype(np.float32)


def make_target_lp_from_laminate(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    return get_lp(lam)


def wrap_angles(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    """Wrap angles into [-π/2, π/2]."""
    return (lam + np.pi / 2) % np.pi - np.pi / 2
