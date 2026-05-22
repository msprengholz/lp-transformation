#!/usr/bin/env python3
"""
Unified benchmark harness for LP back-transformation solvers.

Usage:
    python -m benchmarks.benchmark_runner --solver numpy --n-layers 12,24 --samples 5

Output:
    - Prints a summary table to stdout
    - Writes JSONL results to benchmarks/results/<timestamp>_<solver>.jsonl
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lp_functions import (compute_lp_rmse, compute_angle_deviation,
                                make_random_laminate, make_target_lp_from_laminate,
                                wrap_angles)
from src.utils import (BenchmarkResult, generate_test_problems, timed,
                        save_results, get_rng)


# ──────────────────────────────────────────────
# Solver registry
# ──────────────────────────────────────────────

def _solve_numpy(lam_init: NDArray[np.float32],
                 lp_t: NDArray[np.float32]) -> NDArray[np.float32]:
    """Run the full numpy optimisation pipeline."""
    from src.numpy_solver import optimize_laminate
    # optimize_laminate expects (M, N) input
    rand_lams = lam_init.reshape(1, -1)
    opt_lams, losses = optimize_laminate(rand_lams, lp_t)
    return opt_lams[0]


SOLVERS = {
    "numpy": _solve_numpy,
    # Future:
    # "numba": _solve_numba,
    # "slang": _solve_slang,
}


def solve_and_evaluate(solver_name: str, lam_true: NDArray[np.float32],
                       lp_t: NDArray[np.float32],
                       n_starts: int = 20) -> BenchmarkResult:
    """
    Run a solver on one problem and return a detailed result.

    Uses multiple random starts and picks the best.
    """
    rng = get_rng()
    N = lam_true.size
    solve_fn = SOLVERS[solver_name]

    best_loss = float('inf')
    best_lam = None
    t_total = 0.0

    for _ in range(n_starts):
        lam_init = (rng.random(N).astype(np.float32) * np.pi - np.pi / 2)

        dt, lam_opt = timed(solve_fn, lam_init, lp_t)
        t_total += dt

        loss = compute_lp_rmse(lam_opt, lp_t)
        if loss < best_loss:
            best_loss = loss
            best_lam = lam_opt

    dev = compute_angle_deviation(best_lam, lam_true)
    mean_dev_deg = float(np.rad2deg(np.mean(dev)))
    max_dev_deg = float(np.rad2deg(np.max(dev)))

    return BenchmarkResult(
        solver=solver_name,
        n_layers=N,
        problem_label=f"N={N}",
        lp_rmse=float(best_loss),
        max_angle_dev_deg=max_dev_deg,
        mean_angle_dev_deg=mean_dev_deg,
        time_s=t_total,
        converged=best_loss < 5e-5,
    )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LP solver benchmark")
    parser.add_argument("--solver", choices=list(SOLVERS.keys()) + ["all"],
                        default="all", help="Solver to benchmark")
    parser.add_argument("--n-layers", type=str, default="2,4,8,12,16,24,32,48",
                        help="Comma-separated layer counts")
    parser.add_argument("--samples", type=int, default=5,
                        help="Problems per layer count")
    parser.add_argument("--n-starts", type=int, default=20,
                        help="Random starts per problem")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: auto-generated)")
    args = parser.parse_args()

    n_layers = [int(s) for s in args.n_layers.split(",")]
    solvers = list(SOLVERS.keys()) if args.solver == "all" else [args.solver]

    problems = generate_test_problems(
        n_layer_list=n_layers,
        problems_per_size=args.samples,
    )

    all_results: list[BenchmarkResult] = []

    print(f"Benchmarking solvers: {solvers}")
    print(f"Layer counts: {n_layers}, {args.samples} problems each")
    print(f"{'Solver':<10} {'N':<5} {'LP RMSE':<12} {'Max dev °':<12} "
          f"{'Mean dev °':<12} {'Time (s)':<10} {'Converged':<10}")
    print("-" * 75)

    for solver in solvers:
        for prob in problems:
            result = solve_and_evaluate(solver, prob.lam_true, prob.lp_t,
                                        n_starts=args.n_starts)
            all_results.append(result)
            print(f"{result.solver:<10} {result.n_layers:<5} "
                  f"{result.lp_rmse:<12.2e} {result.max_angle_dev_deg:<12.4f} "
                  f"{result.mean_angle_dev_deg:<12.4f} "
                  f"{result.time_s:<10.2f} "
                  f"{'✓' if result.converged else '✗':<10}")

    # Save results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(out_dir / f"{ts}_benchmark.jsonl")

    save_results(all_results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
