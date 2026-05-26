#!/usr/bin/env python3
"""
Quality gate for autoresearch: verify solver still converges.

Run after every benchmark.  Checks both the numba solver (if available)
and the numpy solver against the Viquerat LP set with strict RMSE
thresholds.  Exits non-zero if quality has degraded.

Called by autoresearch.checks.sh on Colab.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.test_cases import LP_VIQUERAT

N = 12
N_STARTS = 30
BEST_RMSE_MAX = 1e-3      # best start must achieve at least this
MEDIAN_RMSE_MAX = 0.15     # at least half of starts should converge well


def check_solver(name, solver_fn):
    rng = np.random.default_rng(42)
    rand_lams = rng.random((N_STARTS, N), dtype=np.float32)
    rand_lams = rand_lams * np.pi - np.pi / 2

    opt, losses = solver_fn(rand_lams, LP_VIQUERAT)
    best = float(losses.min())
    median = float(np.median(losses))
    worst = float(losses.max())

    print("  %-5s  best=%.2e  median=%.2e  worst=%.2e" % (
        name, best, median, worst))

    failures = []
    if best > BEST_RMSE_MAX:
        failures.append("best RMSE %.2e > %.1e" % (best, BEST_RMSE_MAX))
    if median > MEDIAN_RMSE_MAX:
        failures.append("median RMSE %.2e > %.2f" % (median, MEDIAN_RMSE_MAX))
    if worst > 0.5:
        failures.append("worst RMSE %.2e > 0.5 (multiple starts diverging)" % worst)

    if failures:
        print("  FAIL: " + "; ".join(failures))
        return False
    return True


def run():
    print("Quality gate: solver convergence check")
    all_ok = True

    # NumPy solver (always available)
    from src.numpy_solver import optimize_laminate as opt_np
    if check_solver("numpy", opt_np):
        print("  numpy: OK")
    else:
        all_ok = False

    # Numba solver (if installed)
    try:
        from src.numba_solver import optimize_laminate_numba as opt_nb
        if check_solver("numba", opt_nb):
            print("  numba: OK")
        else:
            all_ok = False
    except ImportError:
        print("  numba: not available (skipping)")

    if all_ok:
        print("All quality gates passed.")
    else:
        print("Solver quality degraded — rejecting experiment.")
        sys.exit(1)


if __name__ == "__main__":
    run()
