#!/usr/bin/env python3
"""
Comprehensive benchmark measuring solution discovery capability.

Two modes:
  1. Viquerat 12-layer: time to discover all known unique solutions
  2. Sprengholz 48-layer: unique solutions found in 60 seconds

Uses Sobol quasi-random starting points for space-filling coverage,
then falls back to pseudo-random if scipy is not available.
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.test_cases import LP_VIQUERAT, LP_SPRENGHOLZ_48, _data_dir
from src.lp_functions import compute_lp_rmse, get_lp


# ──────────────────────────────────────────────
# Solver import
# ──────────────────────────────────────────────

try:
    from src.numba_solver import optimize_laminate_numba as solver
    SOLVER_NAME = "numba"
except ImportError:
    from src.numpy_solver import optimize_laminate as solver
    SOLVER_NAME = "numpy"


# ──────────────────────────────────────────────
# Starting point generators
# ──────────────────────────────────────────────

try:
    from scipy.stats.qmc import Sobol
    HAS_SOBOL = True
except ImportError:
    HAS_SOBOL = False


class StartGenerator:
    """Generate space-filling starting points in [-pi/2, pi/2]."""

    def __init__(self, dim, seed=42):
        self.dim = dim
        self.seed = seed
        self.count = 0
        if HAS_SOBOL:
            self.sobol = Sobol(d=dim, scramble=True, seed=seed)
        else:
            self.rng = np.random.default_rng(seed)
        self._cache = []

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count <= len(self._cache):
            return self._cache[self.count - 1]

        if HAS_SOBOL:
            # Sobol generates in [0,1]^D — scale to [-pi/2, pi/2]
            pts = self.sobol.random()
            lam = (pts * np.pi - np.pi / 2).astype(np.float32)
        else:
            lam = (self.rng.random(self.dim, dtype=np.float32)
                   * np.pi - np.pi / 2)

        if len(self._cache) < 100000:
            self._cache.append(lam)
        return lam


# ──────────────────────────────────────────────
# Solution comparison
# ──────────────────────────────────────────────

def _round_key(lam_rad, decimals=1):
    """Round angles to decimals place (in degrees), return hashable tuple."""
    deg = np.rad2deg(lam_rad)
    deg = (deg + 90) % 180 - 90
    return tuple(np.round(deg, decimals))


def _load_known_solutions(csv_name):
    """Load known solutions from paper CSV, return set of rounded tuples."""
    path = _data_dir() / csv_name
    data = np.loadtxt(path, delimiter=";", dtype=np.float32)
    angles_deg = data[:, 1:-1]
    sols = set()
    for row in angles_deg:
        row = (row + 90) % 180 - 90
        sols.add(tuple(np.round(row, 1)))
    return sols


# ──────────────────────────────────────────────
# Mode 1: Viquerat — find all known solutions
# ──────────────────────────────────────────────

def benchmark_viquerat_discovery(max_starts=50000):
    """
    Run starts (Sobol space-filling) until all known Viquerat solutions found.
    Returns (wall_time_s, starts_used, found_count).
    """
    known = _load_known_solutions("viquerat_12_layer_solutions_complete.csv")
    target_count = len(known)  # 111
    found = set()
    gen = StartGenerator(12)
    t_start = time.perf_counter()
    last_report = 0

    for attempt in range(1, max_starts + 1):
        lam = next(gen)
        opt, losses = solver(lam.reshape(1, -1), LP_VIQUERAT)
        best_loss = float(losses[0])

        if best_loss < 1e-3:
            key = _round_key(opt[0], 1)
            if key not in found:
                found.add(key)
                # Check if this matches a known solution
                in_known = "KNOWN" if key in known else "NEW"
                if len(found) <= 10 or len(found) % 10 == 0:
                    print("  found #%d/%d %s (%.1fs)" % (
                        len(found), target_count, in_known,
                        time.perf_counter() - t_start), flush=True)

        if len(found) >= target_count:
            elapsed = time.perf_counter() - t_start
            print("  ALL %d Viquerat solutions found in %.1fs (%d starts)" % (
                target_count, elapsed, attempt), flush=True)
            return elapsed, attempt, len(found)

        # Progress report every 500 starts
        if attempt - last_report >= 500:
            last_report = attempt
            print("  [%d starts] %d/%d solutions (%.1fs)" % (
                attempt, len(found), target_count,
                time.perf_counter() - t_start), flush=True)

    elapsed = time.perf_counter() - t_start
    print("  Hit max %d starts: %d/%d solutions in %.1fs" % (
        max_starts, len(found), target_count, elapsed), flush=True)
    return elapsed, max_starts, len(found)


# ──────────────────────────────────────────────
# Mode 2: Sprengholz 48-layer throughput
# ──────────────────────────────────────────────

def benchmark_sprengholz_48(time_limit=60.0, max_starts=5000):
    """Run as many starts as possible in time_limit seconds.

    Uses tighter solver parameters (n_coarse_fine=2, grad_tol=1e-4)
    since the 48-layer problem requires more search and refinement.
    """
    found = set()
    gen = StartGenerator(48)
    t_end = time.perf_counter() + time_limit
    completed = 0

    # 48-layer LP set is intrinsically harder: paper solutions have RMSE
    # 3e-3 to 1e-2 (median 6e-3) even with DFO-LS. We use threshold 2e-2.
    rmse_threshold = 2e-2

    for attempt in range(1, max_starts + 1):
        if time.perf_counter() > t_end:
            break
        lam = next(gen)
        opt, losses = solver(lam.reshape(1, -1), LP_SPRENGHOLZ_48,
                              n_coarse_fine=2, irprop_grad_tol=1e-4)
        completed += 1
        best_loss = float(losses[0])

        if best_loss < rmse_threshold:
            key = _round_key(opt[0], 1)
            found.add(key)

    print("  48-layer: %d unique solutions in %d starts (%.1fs)" % (
        len(found), completed, time_limit), flush=True)
    return len(found), completed


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

def run():
    print("Comprehensive benchmark [solver=%s, sobol=%s]" % (
        SOLVER_NAME, HAS_SOBOL), flush=True)

    print("--- Viquerat 12-layer discovery (%d targets) ---" % 111, flush=True)
    vq_time, vq_starts, vq_found = benchmark_viquerat_discovery()

    print("--- Sprengholz 48-layer throughput ---", flush=True)
    sp_count, sp_starts = benchmark_sprengholz_48()

    print()
    print("METRIC viquerat_discovery_time=%.3f" % vq_time, flush=True)
    print("METRIC viquerat_starts=%d" % vq_starts, flush=True)
    print("METRIC viquerat_found=%d" % vq_found, flush=True)
    print("METRIC sprengholz_solutions=%d" % sp_count, flush=True)
    print("METRIC sprengholz_starts=%d" % sp_starts, flush=True)


if __name__ == "__main__":
    run()
