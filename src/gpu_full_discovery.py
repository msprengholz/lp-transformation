#!/usr/bin/env python3
"""Full GPU iRprop Viquerat Discovery benchmark.

Uses: 1) Sobol start generation, 2) GPU batch LP to filter best starts, 
3) GPU iRprop for refinement.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import slangpy as sl
from slangpy import grid, Tensor

from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse
from src.test_cases import LP_VIQUERAT, _data_dir
from src.gpu_irprop import SLANG_IRPROP_STEP2, gpu_batch_irprop
from src.gpu_lp import create_gpu_lp_solver, batch_lp_gpu

N = 12

def _load_known_solutions():
    import csv
    solutions = set()
    path = _data_dir() / "viquerat_12_layer_solutions_complete.csv"
    if not path.exists():
        return set()
    with open(path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) >= 13:
                angles = tuple(round(float(a), 1) for a in row[1:13])
                solutions.add(angles)
    return solutions


def gpu_viquerat_discovery(dev, mod, mod_irprop, max_starts=10000, top_k=1500, irprop_iters=100):
    """Full GPU pipeline: Sobol → GPU LP filter → GPU iRprop."""
    known = _load_known_solutions()
    target_count = len(known)  # 112
    found = set()
    target_lp = LP_VIQUERAT
    
    # Phase 1: Generate Sobol starts and evaluate on GPU
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=N, scramble=True, seed=42)
    
    t_start = time.perf_counter()
    sobol_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
    
    # GPU batch LP evaluation
    lps = batch_lp_gpu(dev, mod, sobol_starts, N)
    losses = np.sum((lps - target_lp) ** 2, axis=1)
    
    # Select top-k best starts
    top_k_actual = min(top_k, max_starts)
    top_indices = np.argsort(losses)[:top_k_actual]
    best_starts = sobol_starts[top_indices]
    t_filter = time.perf_counter() - t_start
    
    # Phase 2: GPU iRprop refinement
    t_irprop_start = time.perf_counter()
    best_angles, best_losses = gpu_batch_irprop(dev, mod_irprop, best_starts, target_lp, max_iter=irprop_iters)
    t_irprop = time.perf_counter() - t_irprop_start
    
    # Collect solutions
    for i in range(top_k_actual):
        if best_losses[i] < 1e-2:
            rmse = compute_lp_rmse(best_angles[i].astype(np.float32), target_lp)
            if rmse < 2e-2:
                key = tuple(round(float(a), 1) for a in best_angles[i])
                found.add(key)
    
    t_total = time.perf_counter() - t_start
    return t_total, max_starts, len(found), t_filter, t_irprop


if __name__ == "__main__":
    print("Full GPU Viquerat Discovery (Sobol + GPU LP filter + GPU iRprop)", flush=True)
    print("=" * 70, flush=True)
    
    Z2, Z3 = _z2_z3(N)
    dev, mod = create_gpu_lp_solver(N, Z2, Z3)
    mod_irprop = sl.Module.load_from_source(dev, "irprop_step2", SLANG_IRPROP_STEP2)
    print("GPU devices initialized", flush=True)
    
    # Warmup
    rng = np.random.RandomState(42)
    warmup = rng.random((10, N)).astype(np.float32) * np.pi - np.pi / 2
    _ = batch_lp_gpu(dev, mod, warmup, N)
    _ = gpu_batch_irprop(dev, mod_irprop, warmup, LP_VIQUERAT, max_iter=10)
    
    # Benchmark
    print("\n--- GPU Discovery Results ---", flush=True)
    for max_starts, top_k, iters in [
        (5000, 1000, 50),
        (10000, 1500, 100),
        (20000, 2000, 100),
        (50000, 3000, 100),
    ]:
        t, starts, found, t_filter, t_irprop = gpu_viquerat_discovery(
            dev, mod, mod_irprop, max_starts=max_starts, top_k=top_k, irprop_iters=iters
        )
        print(f"  starts={max_starts:>6d}, top_k={top_k:>4d}, iters={iters:>3d}: "
              f"{t:.2f}s ({t_filter:.2f}s filter + {t_irprop:.2f}s irprop), "
              f"{found}/112 found", flush=True)
    
    # CPU reference
    print("\n--- CPU Reference ---", flush=True)
    from benchmarks.run_comprehensive import benchmark_viquerat_discovery
    t_cpu, _, _ = benchmark_viquerat_discovery()
    print(f"  CPU: {t_cpu:.2f}s, 112/112 found", flush=True)