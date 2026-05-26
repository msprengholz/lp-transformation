#!/usr/bin/env python3
"""
Standalone benchmark for Colab: profile and test the numpy solver.

Usage: python3 benchmarks/colab_benchmark.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.numpy_solver import optimize_laminate
from src.lp_functions import get_lp, get_lp_and_grad, get_loss_grad
from src.test_cases import LP_VIQUERAT, LP_SPRENGHOLZ_48


def micro_benchmark():
    """Per-call micro-benchmarks of core functions."""
    rng = np.random.default_rng(42)
    lam = rng.random(12, dtype=np.float32) * np.pi - np.pi / 2
    lp_t = LP_VIQUERAT.copy()
    lp_out = np.empty(12, dtype=np.float32)

    for name, fn, args in [
        ("get_lp",           get_lp,         (lam,)),
        ("get_loss_grad",    get_loss_grad,  (lam, lp_t)),
        ("get_lp_and_grad",  get_lp_and_grad,(lam, lp_t, lp_out)),
    ]:
        # Warmup
        for _ in range(200):
            fn(*args)
        times = []
        for _ in range(5000):
            t0 = time.perf_counter_ns()
            fn(*args)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
        times = np.array(times, dtype=np.float64) / 1000  # µs
        print(f"  {name:20s}  {times.mean():7.1f} us  ({times.min():.1f}–{times.max():.1f})")


def pipeline_benchmark(label, lp_target, n_layers, n_starts):
    """End-to-end solver benchmark."""
    rng = np.random.default_rng(42)
    rand_lams = rng.random((n_starts, n_layers), dtype=np.float32)
    rand_lams = rand_lams * np.pi - np.pi / 2

    t0 = time.time()
    opt, losses = optimize_laminate(rand_lams, lp_target)
    dt = time.time() - t0

    best = float(losses.min())
    print(f"  {label:20s}  {dt:6.1f}s total  {dt/n_starts:.3f}s/start  "
          f"best RMSE {best:.2e}")

    for th in [1e-6, 1e-4, 1e-2, 0.1, 0.3]:
        n = int(np.sum(losses < th))
        if n > 0:
            print(f"    RMSE < {th:.0e}:  {n}/{n_starts} ({n*100//n_starts}%)")


def run_tests():
    """Run the pytest suite."""
    import subprocess
    r = subprocess.run(
        ["python", "-m", "pytest",
         "tests/test_lp_functions.py",
         "tests/test_paper_validation.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, timeout=300
    )
    print(r.stdout)
    if r.returncode != 0:
        print("Failures detected")
    return r.returncode


if __name__ == "__main__":
    print("=" * 65)
    print("  LP Transformation — Colab Benchmark")
    print("=" * 65)

    print("\n[1/4] Micro-benchmarks (per-call timing)")
    micro_benchmark()

    print("\n[2/4] Pipeline benchmarks")
    pipeline_benchmark("Viquerat 12", LP_VIQUERAT, 12, 50)
    pipeline_benchmark("Sprengholz 48", LP_SPRENGHOLZ_48, 48, 20)

    print("\n[3/4] Correctness tests")
    code = run_tests()

    print("\n[4/4] Summary")
    print(f"  Tests: {'PASS' if code == 0 else 'FAIL'}")
    print("  Done.")
    sys.exit(code)
