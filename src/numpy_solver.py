"""
Numpy-based solver for lamination parameter back-transformation.

Implements:
  - Sequential search (ssearch): coarse-to-fine grid search per layer
  - iRprop-: improved resilient backpropagation local optimization
  - optimize_laminate: combined pipeline (3× coarse→fine search + iRprop)
"""

import numpy as np
from numpy.typing import NDArray

from .lp_functions import get_lp, get_lp_and_grad, compute_lp_rmse


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
            s_min: float = 1e-8,
            s_max: float = 0.3,
            n_p: float = 1.2,
            n_m: float = 0.5,
            grad_tol: float = 1e-6) -> NDArray[np.float32]:
    """
    Improved Rprop- (resilient backpropagation with weight-backtracking).

    Uses the combined ``get_lp_and_grad`` to avoid redundant trig
    computation.  Early-stops when the max gradient component falls
    below *grad_tol* (indicating a stationary point has been reached).

    Parameters
    ----------
    lam       : (N,) initial ply angles
    lp_t      : (12,) target lamination parameters
    it_iRprop : maximum number of iterations
    sigma     : initial step size
    s_min     : minimum step size
    s_max     : maximum step size
    n_p       : step increase factor (when sign persists)
    n_m       : step decrease factor (when sign changes)
    grad_tol  : stop when max |grad| < this threshold

    Returns
    -------
    lam : (N,) optimised ply angles
    """
    layers = lam.size
    s = np.full(layers, sigma, dtype=np.float32)

    grad0 = get_lp_and_grad(lam, lp_t)
    grad1 = np.empty_like(grad0)

    for _ in range(it_iRprop):
        grad1 = get_lp_and_grad(lam, lp_t)

        # Check convergence via gradient norm
        if np.max(np.abs(grad1)) < grad_tol:
            break

        for k in range(layers):
            if grad0[k] * grad1[k] > 0:
                s[k] = min(s[k] * n_p, s_max)
            elif grad0[k] * grad1[k] < 0:
                s[k] = max(s[k] * n_m, s_min)
                grad1[k] = 0.0  # weight-backtracking
            lam[k] -= np.sign(grad1[k]) * s[k]
            grad0[k] = grad1[k]

    return lam


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────

def optimize_laminate(rand_lams: NDArray[np.float32],
                      lp_t: NDArray[np.float32],
                      n_coarse_fine: int = 1,
                      delta_coarse_deg: float = 10.0,
                      delta_fine_deg: float = 5.0,
                      irprop_iters: int = 3000,
                      irprop_grad_tol: float = 1e-6,
                      ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Full optimisation pipeline for multiple starting guesses.

    1. *n_coarse_fine* rounds of coarse-to-fine grid search
    2. iRprop- refinement
    3. Angle wrap to [-π/2, π/2]

    Parameters
    ----------
    rand_lams        : (M, N) float32 — M random starting laminates
    lp_t             : (12,) float32 — target lamination parameters
    n_coarse_fine    : rounds of coarse→fine search
    delta_coarse_deg : coarse grid spacing (degrees)
    delta_fine_deg   : fine grid spacing (degrees)
    irprop_iters     : max iRprop iterations
    irprop_grad_tol  : iRprop gradient-norm convergence threshold

    Returns
    -------
    optimised_lams : (M, N) — final ply angles
    losses         : (M,)    — LP RMSE for each result
    """
    num_samples, layers = rand_lams.shape
    optimised_lams = np.zeros_like(rand_lams)
    losses = np.zeros(num_samples, dtype=np.float32)

    delta_coarse = np.deg2rad(delta_coarse_deg)
    delta_fine = np.deg2rad(delta_fine_deg)

    for idx in range(num_samples):
        lam = rand_lams[idx].copy()

        # Coarse-to-fine global search
        for _ in range(n_coarse_fine):
            lam = ssearch(lam, delta_coarse, lp_t)
            lam = ssearch(lam, delta_fine, lp_t)

        # Local refinement with gradient-norm convergence
        lam = iRpropm(lam, lp_t, it_iRprop=irprop_iters,
                      grad_tol=irprop_grad_tol)

        # Wrap to [-π/2, π/2]
        lam = (lam + np.pi / 2) % np.pi - np.pi / 2

        optimised_lams[idx] = lam
        losses[idx] = compute_lp_rmse(lam, lp_t)

    return optimised_lams, losses
