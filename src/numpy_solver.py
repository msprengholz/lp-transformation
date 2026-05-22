"""
Numpy-based solver for lamination parameter back-transformation.

Implements:
  - Sequential search (ssearch): coarse-to-fine grid search per layer
  - iRprop-: improved resilient backpropagation local optimization
  - optimize_laminate: combined pipeline (3× coarse→fine search + iRprop)
"""

import numpy as np
from numpy.typing import NDArray

from .lp_functions import get_lp, get_loss_grad, compute_lp_rmse


# ──────────────────────────────────────────────
# Sequential search (global, per-layer grid)
# ──────────────────────────────────────────────

def ssearch(lam: NDArray[np.float32], delta: float,
            lp_t: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Sequential coordinate search over a uniform grid of step *delta*.

    For each layer, evaluates all angle candidates on [-π/2, π/2) with
    spacing *delta* and keeps the one that minimises the LP RMSE.

    Parameters
    ----------
    lam   : (N,) current angle vector (will be modified in place)
    delta : angular grid spacing in radians
    lp_t  : (12,) target lamination parameters

    Returns
    -------
    lam_best : (N,) best-found angles
    """
    lam_best = lam.copy()
    layers = lam.size
    ang_steps = int(np.floor(np.pi / delta))

    for i in range(layers):
        best_loss = float('inf')
        for k in range(1, ang_steps + 1):
            lam[i] = -np.pi / 2.0 + delta * k
            loss = np.sqrt(np.sum((get_lp(lam) - lp_t) ** 2))
            if loss < best_loss:
                lam_best[i] = lam[i]
                best_loss = loss
        lam[i] = lam_best[i]

    return lam_best


# ──────────────────────────────────────────────
# iRprop- local optimisation
# ──────────────────────────────────────────────

def iRpropm(lam: NDArray[np.float32], lp_t: NDArray[np.float32],
            it_iRprop: int = 3000,
            sigma: float = 0.1,
            s_min: float = 0.0,
            s_max: float = 0.3,
            n_p: float = 1.2,
            n_m: float = 0.5) -> NDArray[np.float32]:
    """
    Improved Rprop- (resilient backpropagation with weight-backtracking).

    Parameters
    ----------
    lam      : (N,) initial ply angles
    lp_t     : (12,) target lamination parameters
    it_iRprop : number of iterations
    sigma    : initial step size
    s_min    : minimum step size
    s_max    : maximum step size
    n_p      : step increase factor (when sign persists)
    n_m      : step decrease factor (when sign changes)

    Returns
    -------
    lam : (N,) optimised ply angles
    """
    layers = lam.size
    s = np.full(layers, sigma, dtype=np.float32)

    grad0 = get_loss_grad(lam, lp_t)
    grad1 = np.zeros(layers, dtype=np.float32)

    for _ in range(it_iRprop):
        grad1 = get_loss_grad(lam, lp_t)

        for k in range(layers):
            if grad0[k] * grad1[k] > 0:
                s[k] = min(s[k] * n_p, s_max)
            elif grad0[k] * grad1[k] < 0:
                s[k] = max(s[k] * n_m, s_min)
                grad1[k] = 0        # weight-backtracking
            lam[k] -= np.sign(grad1[k]) * s[k]
            grad0[k] = grad1[k]

    return lam


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────

def optimize_laminate(rand_lams: NDArray[np.float32],
                      lp_t: NDArray[np.float32]) -> tuple[NDArray[np.float32],
                                                          NDArray[np.float32]]:
    """
    Full optimisation pipeline for multiple starting guesses.

    1. 3× coarse-to-fine grid search (10°, then 5°)
    2. iRprop- refinement
    3. Angle wrap to [-π/2, π/2]

    Parameters
    ----------
    rand_lams : (M, N) float32 — M random starting laminates, N layers each
    lp_t      : (12,) float32 — target lamination parameters

    Returns
    -------
    optimised_lams : (M, N) — final ply angles
    losses         : (M,)    — LP RMSE for each result
    """
    num_samples, layers = rand_lams.shape
    optimised_lams = np.zeros_like(rand_lams)
    losses = np.zeros(num_samples, dtype=np.float32)

    for idx in range(num_samples):
        lam = rand_lams[idx].copy()

        # Coarse-to-fine global search
        for _ in range(3):
            lam = ssearch(lam, np.deg2rad(10.0), lp_t)
            lam = ssearch(lam, np.deg2rad(5.0), lp_t)

        # Local refinement
        lam = iRpropm(lam, lp_t)

        # Wrap
        lam = (lam + np.pi / 2) % np.pi - np.pi / 2

        optimised_lams[idx] = lam
        losses[idx] = compute_lp_rmse(lam, lp_t)

    return optimised_lams, losses
