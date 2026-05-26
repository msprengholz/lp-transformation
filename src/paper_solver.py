"""
Numpy implementation of the paper's original search algorithm.

Key algorithmic differences from our current solver:
  1. **Finer ssearch grid** (3° instead of 10°) — more thorough per-layer search
  2. **Line search** along the gradient direction (nearsearch) — explores diagonal
     moves in angle space instead of one axis at a time
  3. **Multiple alternating rounds** of coordinate descent + line search
  4. Each algorithm step is implemented as a standalone numpy function,
     making it easy to experiment with different hybrid approaches.
"""

import time
import numpy as np
from numpy.typing import NDArray
from .lp_functions import get_lp, get_lp_and_grad, compute_lp_rmse, get_loss_grad
from .numpy_solver import iRpropm


# ──────────────────────────────────────────────
# Batch LP computation (for matrix-of-candidates)
# ──────────────────────────────────────────────

def calc_lp_array(lams: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute LP for multiple laminates.  lams: (M, N) -> lps: (M, 12)."""
    M = lams.shape[0]
    out = np.empty((M, 12), dtype=np.float32)
    for i in range(M):
        out[i] = get_lp(lams[i])
    return out


# ──────────────────────────────────────────────
# 1. Coordinate descent (fine grid, paper's ssearch)
# ──────────────────────────────────────────────

def ssearch_paper(lam: NDArray[np.float32], lp_t: NDArray[np.float32],
                  delta_deg: float = 3.0, iterations: int = 1
                  ) -> NDArray[np.float32]:
    """
    Coordinate descent from the paper.

    For each layer, evaluates ALL candidate angles on a uniform grid,
    batch-computes LPs for all candidates, picks the best.

    Uses the paper's batch approach: create a matrix where each row is
    the current laminate with layer i set to one candidate angle, then
    compute LP for ALL rows at once via calc_lp_array.

    Parameters
    ----------
    lam        : (N,) current angles
    lp_t       : (12,) target LP
    delta_deg  : grid spacing in degrees
    iterations : how many times to sweep through all layers
    """
    layers = lam.size
    delta = np.deg2rad(delta_deg)
    ang_steps = int(np.floor(np.pi / delta))

    # Precompute all candidate angles
    angles = np.array([-np.pi / 2 + delta * k for k in range(1, ang_steps + 1)],
                      dtype=np.float32)

    for _ in range(iterations):
        for i in range(layers):
            # Create (ang_steps, layers) matrix
            candidates = np.tile(lam, (ang_steps, 1))  # each row = current lam
            candidates[:, i] = angles  # vary layer i

            # Batch-compute LP for all candidates
            lps = calc_lp_array(candidates)

            # Find best
            losses = np.sqrt(np.sum((lps - lp_t) ** 2, axis=1))
            best_idx = int(np.argmin(losses))
            lam[i] = candidates[best_idx, i]

    return lam


# ──────────────────────────────────────────────
# 2. Line search along gradient direction (paper's nearsearch)
# ──────────────────────────────────────────────

def nearsearch(lam: NDArray[np.float32], lp_t: NDArray[np.float32],
               delta_deg: float = 1.0, sl: float = 1.0, steps: int = 3,
               iterations: int = 1) -> NDArray[np.float32]:
    """
    Line search along the (negative) gradient direction.

    Computes a search direction via finite-difference gradient estimation
    (coarse but effective).  Then evaluates points along this direction
    at varying distances and picks the best.

    This explores DIAGONAL moves (multiple layers at once) which
    coordinate descent cannot do.

    Parameters
    ----------
    lam       : (N,) current angles
    lp_t      : (12,) target LP
    delta_deg : finite-difference step (degrees) for gradient estimation
    sl        : search length divisor (1 = one axis length)
    steps     : number of search steps to evaluate
    iterations: how many times to repeat
    """
    layers = lam.size
    delta = np.deg2rad(delta_deg)

    for _ in range(iterations):
        lp0 = get_lp(lam)
        lam_deg = np.rad2deg(lam)

        # Finite-difference gradient estimation
        # Create laminates where each angle is nudged by +delta
        modlams = np.tile(lam_deg, (layers, 1))
        np.fill_diagonal(modlams, (np.diag(modlams) + delta_deg + 90) % 180 - 90)

        # LP at nudged positions
        modlps = calc_lp_array(np.deg2rad(modlams))

        # Distance from target at current point and nudged points
        dist = np.sqrt(np.sum((lp0 - lp_t) ** 2))
        dists = np.sqrt(np.sum((modlps - lp_t) ** 2, axis=1))

        # Search direction = how much each angle move reduces distance
        sd = dist - dists
        sd_norm = np.linalg.norm(sd)
        if sd_norm > 0:
            sd = sd / sd_norm

        # Search along direction at varying step sizes
        search_steps = np.arange(steps, 180 // max(sl, 1), steps, dtype=np.float32)
        if len(search_steps) == 0:
            continue

        n_steps = len(search_steps)
        search_lams = np.tile(lam_deg, (n_steps, 1))
        search_lams += sd[np.newaxis, :] * search_steps[:, np.newaxis]
        search_lams = (search_lams + 90) % 180 - 90  # wrap

        # Evaluate
        search_lps = calc_lp_array(np.deg2rad(search_lams))
        search_dists = np.sqrt(np.sum((search_lps - lp_t) ** 2, axis=1))
        best_idx = int(np.argmin(search_dists))

        if search_dists[best_idx] < dist:
            lam[:] = np.deg2rad(search_lams[best_idx])

    return lam


# ──────────────────────────────────────────────
# 3. Optimisation loop (paper's search_layup)
# ──────────────────────────────────────────────

def search_layup_paper(lps: NDArray[np.float32], layers: int,
                       eps: float = 0.03,
                       error: float = 1e-6,
                       it_comb: int = 100,
                       it_sub: int = 3,
                       it_s: int = 3,
                       it_n: int = 1,
                       delta_doe: float = 11.25,
                       delta_s: float = 3.0,
                       delta_n: float = 1.0,
                       l_n: float = 1.0,
                       steps_n: int = 3,
                       doe: bool = False,
                       irprop_iters: int = 500,
                       ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Full optimisation loop matching the paper's Algorithm 1.

    1. Generate starting points (DOE or random)
    2. For each start: coordinate descent (ssearch) + line search (nearsearch)
    3. If RMSE < eps, run iRprop refinement
    4. Collect unique solutions (RMSE < error)

    Parameters
    ----------
    lps        : (12,) target lamination parameters
    layers     : number of plies
    eps        : threshold to accept for refinement (RMSE)
    error      : threshold to accept as solved (RMSE)
    it_comb    : number of random starting points
    it_sub     : rounds of (ssearch + nearsearch) per start
    it_s       : ssearch iterations per round
    it_n       : nearsearch iterations per round
    delta_doe  : grid spacing for DOE start points (degrees)
    delta_s    : ssearch grid spacing (degrees)
    delta_n    : finite-difference step for nearsearch (degrees)
    l_n        : search direction length divisor
    steps_n    : nearsearch evaluation steps
    doe        : use LHS (True) or random (False) starting points
    irprop_iters: iRprop iterations for refinement
    """
    solutions = []
    timestamps = []
    stime = time.time()

    modangles = np.deg2rad(np.arange(-90 + delta_s, 90 + delta_s, delta_s))
    angles_s = int(np.floor(180 / delta_s))
    de = delta_doe
    ang = int(np.floor(180 / de))

    # Generate starting points
    if doe:
        # Latin Hypercube Sampling
        try:
            from pyDOE import lhs
            lams = np.deg2rad(np.rint(lhs(layers, it_comb) * (ang - 1))
                              * de - (90 - de)).astype(np.float32)
        except ImportError:
            doe = False  # fall back to random

    if not doe:
        rng = np.random.default_rng(42)
        lams = rng.random((it_comb, layers), dtype=np.float32)
        lams = np.deg2rad(np.rint(lams * (ang - 1)) * de - (90 - de))

    for k in range(it_comb):
        lam = lams[k].copy()

        # Combined search: alternating ssearch + nearsearch
        for _ in range(it_sub):
            lam = ssearch_paper(lam, lps, delta_s, it_s)
            lam = nearsearch(lam, lps, delta_n, l_n, steps_n, it_n)

        # Check if close enough for refinement
        err = compute_lp_rmse(lam, lps)
        if err < eps:
            lam = iRpropm(lam, lps, it_iRprop=irprop_iters, grad_tol=1e-6)
            err = compute_lp_rmse(lam, lps)

            if err < error:
                sol = tuple(np.round(np.rad2deg(lam), 1))
                if sol not in solutions:
                    solutions.append(sol)
                    timestamps.append(time.time() - stime)

    return np.array(solutions), np.array(timestamps)


# ──────────────────────────────────────────────
# Comparative benchmark
# ──────────────────────────────────────────────

def benchmark_algorithms():
    """Compare ssearch+iRprop vs paper's ssearch+nearsearch+iRprop."""
    import time
    from src.test_cases import LP_VIQUERAT, _data_dir

    N = 12
    lp_t = LP_VIQUERAT
    rng = np.random.default_rng(42)
    n_starts = 200

    print("=" * 60)
    print("Algorithm comparison: Viquerat 12-layer")
    print("=" * 60)

    # Our current algorithm: ssearch(10°) + ssearch(5°) + iRprop
    from .numpy_solver import optimize_laminate
    rand_lams = rng.random((n_starts, N), dtype=np.float32) * np.pi - np.pi / 2

    t0 = time.time()
    opt, losses = optimize_laminate(rand_lams, lp_t, n_coarse_fine=3,
                                     delta_coarse_deg=10.0, delta_fine_deg=5.0,
                                     irprop_iters=3000, irprop_grad_tol=1e-6)
    dt = time.time() - t0
    found = set()
    for o in opt:
        deg = np.round(np.rad2deg(o), 1)
        deg = (deg + 90) % 180 - 90
        found.add(tuple(deg))
    print(f"\nOur solver (ssearch10+5 + iRprop):")
    print(f"  {n_starts} starts in {dt:.1f}s, {len(found)} unique solutions")
    print(f"  best={losses.min():.2e}, median={np.median(losses):.2e}")

    # Paper: ssearch(3°) + nearsearch + iRprop
    solutions, timestamps = search_layup_paper(
        lp_t, N, it_comb=n_starts, it_sub=3, it_s=3, it_n=1,
        delta_s=3.0, delta_n=1.0, steps_n=3,
        eps=0.03, error=1e-6, irprop_iters=500
    )
    print(f"\nPaper algorithm (ssearch3 + nearsearch + iRprop):")
    print(f"  {n_starts} starts, {len(solutions)} unique solutions")


if __name__ == "__main__":
    benchmark_algorithms()
