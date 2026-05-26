#!/usr/bin/env python3
"""
Comprehensive benchmark measuring solution discovery capability.

Two modes:
  1. Viquerat 12-layer:  time to discover all 10 known unique solutions
  2. Sprengholz 48-layer: unique solutions found in 60 seconds

A solution is considered "found" when the solver produces angles
that, rounded to 0.1°, match a known valid solution or produce
RMSE < 1e-3 with angle rounding to 0.1°.
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.test_cases import LP_VIQUERAT, LP_SPRENGHOLZ_48
from src.lp_functions import compute_lp_rmse, get_lp


# ──────────────────────────────────────────────
# Imports — try numba first, fall back to numpy
# ──────────────────────────────────────────────

try:
    from src.numba_solver import optimize_laminate_numba as solver
    SOLVER_NAME = "numba"
except ImportError:
    from src.numpy_solver import optimize_laminate as solver
    SOLVER_NAME = "numpy"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _round_angles(lam_rad, decimals=1):
    """Round angles to given decimal places (in degrees) for comparison."""
    deg = np.rad2deg(lam_rad)
    deg = (deg + 90) % 180 - 90  # wrap to [-90, 90)
    return tuple(np.round(deg, decimals))


def _load_known_solutions(csv_name):
    """Load known solutions from paper CSV, return set of rounded tuples."""
    from src.test_cases import _data_dir
    path = _data_dir() / csv_name
    data = np.loadtxt(path, delimiter=";", dtype=np.float32)
    angles_deg = data[:, 1:-1]
    solutions = set()
    for row in angles_deg:
        # Already in degrees in the CSV
        row = (row + 90) % 180 - 90  # wrap to [-90, 90)
        solutions.add(tuple(np.round(row, 1)))
    return solutions


# ──────────────────────────────────────────────
# Mode 1: Viquerat solution discovery time
# ──────────────────────────────────────────────

def benchmark_viquerat_discovery(max_starts=5000):
    """
    Run random starts until all 10 known Viquerat solutions are found.
    Returns (wall_time_s, starts_used, found_count).
    """
    known = _load_known_solutions("viquerat_12_layer_solutions.csv")
    target_count = len(known)  # 10
    found = set()
    found_details = {}  # rounded_tuple -> (rmse, time_found)

    rng = np.random.default_rng(42)
    N = 12
    t_start = time.perf_counter()

    for attempt in range(1, max_starts + 1):
        lam = rng.random(N, dtype=np.float32) * np.pi - np.pi / 2
        opt, losses = solver(lam.reshape(1, -1), LP_VIQUERAT)
        opt_lam = opt[0]
        best_loss = float(losses[0])

        key = _round_angles(opt_lam, 1)
        if best_loss < 1e-3 and key not in found:
            found.add(key)
            found_details[key] = (best_loss, time.perf_counter() - t_start)

        if len(found) >= target_count:
            elapsed = time.perf_counter() - t_start
            print("  Found %d/%d Viquerat solutions in %.1fs (%d starts)" % (
                len(found), target_count, elapsed, attempt), flush=True)
            return elapsed, attempt, len(found)

        if attempt % 50 == 0 and len(found) > 0:
            print("  [%d starts] found %d/%d solutions (%.1fs)" % (
                attempt, len(found), target_count,
                time.perf_counter() - t_start), flush=True)

    elapsed = time.perf_counter() - t_start
    print("  Hit max %d starts: found %d/%d solutions in %.1fs" % (
        max_starts, len(found), target_count, elapsed), flush=True)
    return elapsed, max_starts, len(found)


# ──────────────────────────────────────────────
# Mode 2: Sprengholz 48-layer — solutions/min
# ──────────────────────────────────────────────

def benchmark_sprengholz_48(time_limit=60.0, max_starts=2000):
    """
    Run as many starts as possible in `time_limit` seconds.
    Count unique solutions (RMSE < 1e-3, angles rounded to 0.1°).
    Returns (found_count, starts_completed).
    """
    found = set()
    rng = np.random.default_rng(42)
    N = 48
    t_end = time.perf_counter() + time_limit
    completed = 0

    for attempt in range(1, max_starts + 1):
        if time.perf_counter() > t_end:
            break
        lam = rng.random(N, dtype=np.float32) * np.pi - np.pi / 2
        opt, losses = solver(lam.reshape(1, -1), LP_SPRENGHOLZ_48)
        opt_lam = opt[0]
        best_loss = float(losses[0])
        completed += 1

        if best_loss < 1e-3:
            key = _round_angles(opt_lam, 1)
            found.add(key)

    elapsed = time_limit
    print("  48-layer: %d unique solutions in %d starts (%.1fs)" % (
        len(found), completed, elapsed), flush=True)
    return len(found), completed


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

def run():
    print("Comprehensive benchmark [solver=%s]" % SOLVER_NAME, flush=True)

    # Viquerat discovery time
    print("--- Viquerat 12-layer discovery ---", flush=True)
    vq_time, vq_starts, vq_found = benchmark_viquerat_discovery()

    # Sprengholz 48-layer throughput
    print("--- Sprengholz 48-layer throughput ---", flush=True)
    sp_count, sp_starts = benchmark_sprengholz_48()

    # Output metrics
    print()
    print("METRIC viquerat_discovery_time=%.3f" % vq_time, flush=True)
    print("METRIC viquerat_starts=%d" % vq_starts, flush=True)
    print("METRIC viquerat_found=%d" % vq_found, flush=True)
    print("METRIC sprengholz_solutions=%d" % sp_count, flush=True)
    print("METRIC sprengholz_starts=%d" % sp_starts, flush=True)


if __name__ == "__main__":
    run()
