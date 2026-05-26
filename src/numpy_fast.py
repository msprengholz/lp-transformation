"""
Optimised pure-numpy solver with vectorised batch LP computation.

Key speedups over the original numpy solver:
  1. **Vectorised get_lp_batch** — LP for (M, N) laminates in one go,
     using numpy array ops instead of Python loops
  2. **Batch ssearch** — evaluate ALL angle candidates for a layer in
     a single vectorised call instead of one-at-a-time
  3. **Two-stage search** — only evaluate every other angle, then refine
  4. **Sobol starting points** — space-filling quasi-random starts
  5. **Relaxed grad_tol=1e-3** — same quality, faster convergence
"""

import math
import time
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache

from .lp_functions import _z2_z3, _norm_factors, wrap_angles


# ═══════════════════════════════════════════════
# Vectorised LP computation  (M laminates at once)
# ═══════════════════════════════════════════════

@lru_cache(maxsize=32)
def _batch_arrays(N: int):
    """Precompute Z2, Z3, invN, N2, N3 for a given N (cached)."""
    Z2, Z3 = _z2_z3(N)
    invN, N2, N3 = _norm_factors(N)
    return Z2, Z3, invN, N2, N3


def get_lp_batch(lams: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Vectorised LP for M laminates at once.

    lams : (M, N) float32
    returns : (M, 12) float32
    """
    M, N = lams.shape
    Z2, Z3, invN, N2, N3 = _batch_arrays(N)

    lam2 = lams * 2          # (M, N)
    lam4 = lams * 4
    cos2 = np.cos(lam2)      # (M, N)
    sin2 = np.sin(lam2)
    cos4 = np.cos(lam4)
    sin4 = np.sin(lam4)

    lp = np.empty((M, 12), dtype=np.float32)

    # In-plane — sum over layers
    lp[:, 0] = np.sum(cos2, axis=1) * invN
    lp[:, 1] = np.sum(sin2, axis=1) * invN
    lp[:, 2] = np.sum(cos4, axis=1) * invN
    lp[:, 3] = np.sum(sin4, axis=1) * invN

    # Coupling — dot products with Z2 (vectorised across M)
    lp[:, 4] = np.dot(cos2, Z2) * N2   # (M,) = (M,N) @ (N,)
    lp[:, 5] = np.dot(sin2, Z2) * N2
    lp[:, 6] = np.dot(cos4, Z2) * N2
    lp[:, 7] = np.dot(sin4, Z2) * N2

    # Out-of-plane — dot products with Z3
    lp[:, 8] = np.dot(cos2, Z3) * N3
    lp[:, 9] = np.dot(sin2, Z3) * N3
    lp[:, 10] = np.dot(cos4, Z3) * N3
    lp[:, 11] = np.dot(sin4, Z3) * N3

    return lp


def compute_rmse_batch(lps: NDArray[np.float32],
                       lp_t: NDArray[np.float32]) -> NDArray[np.float32]:
    """RMSE for each of M LP vectors against target lp_t."""
    diff = lps - lp_t           # (M, 12)
    return np.sqrt(np.mean(diff ** 2, axis=1))  # (M,)


# ═══════════════════════════════════════════════
# Sobol starting point generator
# ═══════════════════════════════════════════════

def sobol_starts(n: int, dim: int, seed: int = 42) -> NDArray[np.float32]:
    """Generate n Sobol quasi-random starts in [-pi/2, pi/2]^dim."""
    try:
        from scipy.stats.qmc import Sobol
        sobol = Sobol(d=dim, scramble=True, seed=seed)
        pts = sobol.random(n=n)
    except ImportError:
        rng = np.random.default_rng(seed)
        pts = rng.random((n, dim))
    return (pts * np.pi - np.pi / 2).astype(np.float32)


# ═══════════════════════════════════════════════
# Two-stage ssearch (batch mode)
# ═══════════════════════════════════════════════

def ssearch_batch(lam: NDArray[np.float32], lp_t: NDArray[np.float32],
                  delta_deg: float = 10.0) -> NDArray[np.float32]:
    """
    Sequential coordinate search in batch mode.

    For each layer, evaluates all angle candidates in ONE batch LP call,
    then picks the best.  Uses two-stage evaluation:
      Stage 1: every other angle (coarse grid)
      Stage 2: neighbours of the best coarse angle

    Parameters
    ----------
    lam       : (N,) current angles
    lp_t      : (12,) target LP
    delta_deg : grid spacing (degrees)
    """
    layers = lam.size
    delta = np.deg2rad(delta_deg)
    ang_steps = int(np.floor(np.pi / delta))
    half_pi = np.float32(np.pi / 2.0)

    best_lam = lam.copy()

    for i in range(layers):
        # Stage 1: coarse grid (odd indices)
        k_coarse = np.arange(1, ang_steps + 1, 2)
        n_coarse = len(k_coarse)

        candidates = np.tile(best_lam, (n_coarse, 1))  # (n_coarse, layers)
        candidates[:, i] = -half_pi + delta * k_coarse

        lps = get_lp_batch(candidates)
        losses = compute_rmse_batch(lps, lp_t)
        best_idx = int(np.argmin(losses))
        best_k = k_coarse[best_idx]

        # Stage 2: refine neighbours (even indices around best_k)
        k_fine = []
        for offset in (-1, 1):
            k = best_k + offset
            if 1 <= k <= ang_steps and k % 2 == 0:
                k_fine.append(k)

        if k_fine:
            candidates_fine = np.tile(best_lam, (len(k_fine), 1))
            candidates_fine[:, i] = -half_pi + delta * np.array(k_fine)
            lps_fine = get_lp_batch(candidates_fine)
            losses_fine = compute_rmse_batch(lps_fine, lp_t)
            fine_best = int(np.argmin(losses_fine))
            if losses_fine[fine_best] < losses[best_idx]:
                best_k = k_fine[fine_best]

        best_lam[i] = -half_pi + delta * best_k

    return best_lam


# ═══════════════════════════════════════════════
# iRprop- (same as numpy_solver but with better defaults)
# ═══════════════════════════════════════════════

def irprop_fast(lam: NDArray[np.float32], lp_t: NDArray[np.float32],
                iters: int = 3000, grad_tol: float = 1e-3,
                sigma: float = 0.1) -> NDArray[np.float32]:
    """
    iRprop- with faster defaults (grad_tol=1e-3 instead of 1e-6).
    """
    from .numpy_solver import iRpropm
    return iRpropm(lam, lp_t, it_iRprop=iters, sigma=sigma,
                   grad_tol=grad_tol)


# ═══════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════

def optimize_fast(rand_lams: NDArray[np.float32],
                  lp_t: NDArray[np.float32],
                  n_rounds: int = 1,
                  delta_deg: float = 10.0,
                  irprop_iters: int = 3000,
                  irprop_grad_tol: float = 1e-3,
                  ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Fast pure-numpy optimisation pipeline.

    1. Two-stage batch ssearch (coarse grid + neighbour refine)
    2. iRprop- with relaxed gradient tolerance
    3. Angle wrap

    Parameters
    ----------
    rand_lams : (M, N) — M random starting laminates
    lp_t      : (12,) — target lamination parameters
    n_rounds  : rounds of ssearch (1 is usually enough with grad_tol=1e-3)
    """
    M, N = rand_lams.shape
    out = np.empty_like(rand_lams)
    los = np.empty(M, dtype=np.float32)

    for idx in range(M):
        lam = rand_lams[idx].copy()
        for _ in range(n_rounds):
            lam = ssearch_batch(lam, lp_t, delta_deg)
        lam = irprop_fast(lam, lp_t, irprop_iters, irprop_grad_tol)
        lam = wrap_angles(lam)
        out[idx] = lam
        los[idx] = compute_rmse_batch(get_lp_batch(lam[None, :]), lp_t)[0]

    return out, los


# ═══════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    from src.test_cases import LP_VIQUERAT

    N = 12
    lp_t = LP_VIQUERAT
    n_starts = 200

    print("Optimised numpy solver benchmark")
    print("=" * 45)

    # Test 1: Sobol starts
    starts = sobol_starts(n_starts, N)
    t0 = time.perf_counter()
    opt, losses = optimize_fast(starts, lp_t)
    dt = time.perf_counter() - t0
    print("  Sobol starts: %.1fs, best=%.2e, median=%.2e" % (
        dt, losses.min(), np.median(losses)))

    # Test 2: Random starts
    rng = np.random.default_rng(42)
    starts_r = (rng.random((n_starts, N), dtype=np.float32)
                * np.pi - np.pi / 2)
    t0 = time.perf_counter()
    opt_r, losses_r = optimize_fast(starts_r, lp_t)
    dt = time.perf_counter() - t0
    print("  Random starts: %.1fs, best=%.2e, median=%.2e" % (
        dt, losses_r.min(), np.median(losses_r)))
