#!/usr/bin/env python3
"""
Stable benchmark for the pi-autoresearch loop.

Outputs METRIC lines in the format expected by pi-autoresearch.
Designed to be run on the Colab VM.

Primary metric: solve_time (mean of 3 repeats x 30 starts, Viquerat 12-layer)
Secondary metrics: best_rmse, worst_rmse, median_rmse (monitoring)
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.numpy_solver import optimize_laminate
from src.test_cases import LP_VIQUERAT


def run():
    rng = np.random.default_rng(42)
    N = 12
    n_starts = 30
    n_repeats = 3

    times = []
    all_losses = []

    for rep in range(n_repeats):
        rand_lams = rng.random((n_starts, N), dtype=np.float32)
        rand_lams = rand_lams * np.pi - np.pi / 2

        t0 = time.perf_counter()
        opt, losses = optimize_laminate(rand_lams, LP_VIQUERAT)
        dt = time.perf_counter() - t0

        times.append(dt)
        all_losses.extend(losses.tolist())

    mean_time = sum(times) / len(times)
    best = float(np.min(all_losses))
    worst = float(np.max(all_losses))
    median = float(np.median(all_losses))

    # Primary metric — this is what we optimize
    print(f"METRIC solve_time={mean_time:.4f}")

    # Secondary metrics (monitoring only, no keep/discard)
    print(f"METRIC best_rmse={best:.2e}")
    print(f"METRIC worst_rmse={worst:.2e}")
    print(f"METRIC median_rmse={median:.2e}")


if __name__ == "__main__":
    run()
