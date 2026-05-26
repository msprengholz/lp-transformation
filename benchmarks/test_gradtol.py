#!/usr/bin/env python3
"""Test iRprop gradient tolerance effect on speed vs quality."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.numba_solver import optimize_laminate_numba
from src.test_cases import LP_VIQUERAT


def run():
    rng = np.random.default_rng(42)
    lam = rng.random((30, 12), dtype=np.float32) * np.pi - np.pi / 2

    print("grad_tol  solve_time  best_rmse   median_rmse")
    for tol in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]:
        rng = np.random.default_rng(42)
        lam_copy = lam.copy()
        t0 = time.perf_counter()
        opt, losses = optimize_laminate_numba(lam_copy, LP_VIQUERAT, irprop_grad_tol=tol)
        dt = time.perf_counter() - t0
        print("  %.0e    %.4fs    %.2e     %.2e" % (
            tol, dt, losses.min(), np.median(losses)))
        # Quality check
        if losses.min() > 1e-3 or np.median(losses) > 0.15:
            print("  -> FAILS quality gate")


if __name__ == "__main__":
    run()
