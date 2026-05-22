"""
Core lamination parameter functions.

Forward:  given ply angles → 12 lamination parameters (A, B, D matrices)
Backward: given target LPs + initial guess → optimized ply angles
"""

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────
# Forward computation
# ──────────────────────────────────────────────

def get_lp(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute 12 lamination parameters from a laminate angle vector.

    Parameters
    ----------
    lam : (N,) float32 array
        Ply angles in radians, range [-π/2, π/2].

    Returns
    -------
    lp : (12,) float32 array
        Lamination parameters [A0..A3, B0..B3, D0..D3].
    """
    N = lam.size
    lp = np.zeros(12, dtype=np.float32)

    lam2 = lam * 2
    lam4 = lam * 4
    cos2 = np.cos(lam2).astype(np.float32)
    cos4 = np.cos(lam4).astype(np.float32)
    sin2 = np.sin(lam2).astype(np.float32)
    sin4 = np.sin(lam4).astype(np.float32)

    # Z arrays for bending coupling
    k = np.arange(N, dtype=np.float32)
    Z2 = ((-N / 2 + k + 1) ** 2 - (-N / 2 + k) ** 2).astype(np.float32)
    Z3 = ((-N / 2 + k + 1) ** 3 - (-N / 2 + k) ** 3).astype(np.float32)

    # In-plane (A)
    lp[0] = np.sum(cos2) / N
    lp[1] = np.sum(sin2) / N
    lp[2] = np.sum(cos4) / N
    lp[3] = np.sum(sin4) / N

    # Coupling (B)
    N2 = 2 / (N ** 2)
    lp[4] = np.dot(Z2, cos2) * N2
    lp[5] = np.dot(Z2, sin2) * N2
    lp[6] = np.dot(Z2, cos4) * N2
    lp[7] = np.dot(Z2, sin4) * N2

    # Out-of-plane (D)
    N3 = 4 / (N ** 3)
    lp[8] = np.dot(Z3, cos2) * N3
    lp[9] = np.dot(Z3, sin2) * N3
    lp[10] = np.dot(Z3, cos4) * N3
    lp[11] = np.dot(Z3, sin4) * N3

    return lp


# ──────────────────────────────────────────────
# Gradient of the LP loss
# ──────────────────────────────────────────────

def get_loss_grad(lam: NDArray[np.float32],
                  lp_t: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Gradient of the LP-matching loss w.r.t. each ply angle.

    loss = 0.5 * Σ (lp_k(lam) - lp_t_k)²

    Parameters
    ----------
    lam  : (N,) float32 — current ply angles
    lp_t : (12,) float32 — target lamination parameters

    Returns
    -------
    grad : (N,) float32 — ∂loss/∂lam_i
    """
    N = lam.size

    lam2 = lam * 2
    lam4 = lam * 4
    cos2 = np.cos(lam2)
    cos4 = np.cos(lam4)
    sin2 = np.sin(lam2)
    sin4 = np.sin(lam4)

    k = np.arange(N, dtype=np.float32)
    Z2 = ((-N / 2 + k + 1) ** 2 - (-N / 2 + k) ** 2).astype(np.float32)
    Z3 = ((-N / 2 + k + 1) ** 3 - (-N / 2 + k) ** 3).astype(np.float32)

    lp_d = lp_t - get_lp(lam)

    # Precompute factors
    N2 = 2.0 / (N ** 2)
    N3 = 4.0 / (N ** 3)

    # Vectorized gradient
    grad = np.zeros(N, dtype=np.float32)
    grad += -2 * sin2 * lp_d[0] + 2 * cos2 * lp_d[1]
    grad += -2 * sin4 * lp_d[2] + 2 * cos4 * lp_d[3]
    grad += (-2 * sin2 * Z2 * lp_d[4] + 2 * cos2 * Z2 * lp_d[5])
    grad += (-2 * sin4 * Z2 * lp_d[6] + 2 * cos4 * Z2 * lp_d[7])
    grad += (-2 * sin2 * Z3 * lp_d[8] + 2 * cos2 * Z3 * lp_d[9])
    grad += (-2 * sin4 * Z3 * lp_d[10] + 2 * cos4 * Z3 * lp_d[11])

    return grad * 2


# ──────────────────────────────────────────────
# Loss computation (RMSE of LP residual)
# ──────────────────────────────────────────────

def compute_lp_rmse(lam: NDArray[np.float32],
                    lp_t: NDArray[np.float32]) -> float:
    """Root-mean-square error of LP mismatch, normalized by layer count."""
    lp_d = get_lp(lam) - lp_t
    return float(np.sqrt(np.sum(lp_d ** 2) / lam.size))


def compute_angle_deviation(lam_opt: NDArray[np.float32],
                            lam_true: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Minimum angular deviation accounting for periodicity [0, π).

    Returns per-angle deviations in radians.
    """
    diff = np.abs(lam_opt - lam_true)
    # Account for π-periodicity: angles in [-π/2, π/2] are equivalent under ±π
    return np.minimum(diff, np.pi - diff)


# ──────────────────────────────────────────────
# Test-problem generators
# ──────────────────────────────────────────────

def make_random_laminate(n_layers: int, rng: np.random.Generator | None = None
                         ) -> NDArray[np.float32]:
    """
    Generate a random laminate with angles in [-π/2, π/2].

    Uses discrete steps of 11.25° (matching the original ssearch grid).
    """
    if rng is None:
        rng = np.random.default_rng()
    de = 11.25  # degrees
    ang_steps = int(np.floor(180 / de))
    steps = rng.integers(0, ang_steps, size=n_layers).astype(np.float32)
    lam = np.deg2rad(steps * de - (90 - de))
    return lam.astype(np.float32)


def make_target_lp_from_laminate(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute the target LP from a known laminate (for self-consistency tests)."""
    return get_lp(lam)


def wrap_angles(lam: NDArray[np.float32]) -> NDArray[np.float32]:
    """Wrap angles into [-π/2, π/2]."""
    return (lam + np.pi / 2) % np.pi - np.pi / 2
