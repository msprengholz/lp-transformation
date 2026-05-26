#!/usr/bin/env python3
"""
Stable benchmark for the pi-autoresearch loop.

Outputs METRIC lines in the format expected by pi-autoresearch.
Designed to be run on the Colab VM.

Usage:
    python benchmarks/run_benchmark.py                       # numpy (baseline)
    python benchmarks/run_benchmark.py --solver numba        # numba JIT
    python benchmarks/run_benchmark.py --solver numba --n-starts 50  # custom

Primary metric: solve_time (mean of 3 repeats x n_starts, Viquerat 12)
Secondary: best_rmse, worst_rmse, median_rmse (monitoring only)
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.test_cases import LP_VIQUERAT


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["numpy", "numba"], default="numpy",
                        help="Solver backend to benchmark")
    parser.add_argument("--n-starts", type=int, default=30,
                        help="Random starting points per repeat")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Number of repeats for timing stability")
    args = parser.parse_args()

    if args.solver == "numpy":
        from src.numpy_solver import optimize_laminate
    elif args.solver == "numba":
        try:
            from src.numba_solver import optimize_laminate_numba as optimize_laminate
        except ImportError:
            print("numba not available, falling back to numpy", file=sys.stderr)
            from src.numpy_solver import optimize_laminate

    rng = np.random.default_rng(42)
    N = 12
    n_starts = args.n_starts
    n_repeats = args.n_repeats

    # Warmup JIT (first run compiles)
    warmup_lam = rng.random((5, N), dtype=np.float32) * np.pi - np.pi / 2
    _ = optimize_laminate(warmup_lam, LP_VIQUERAT, irprop_iters=10)

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

    # Primary metric
    print(f"METRIC solve_time={mean_time:.4f}")

    # Secondary metrics (monitoring)
    print(f"METRIC best_rmse={best:.2e}")
    print(f"METRIC worst_rmse={worst:.2e}")
    print(f"METRIC median_rmse={median:.2e}")


if __name__ == "__main__":
    run()
