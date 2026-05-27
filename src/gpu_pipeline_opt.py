#!/usr/bin/env python3
"""
Experiment: Optimize GPU pipeline parameters.

Key questions:
1. How few GPU iRprop iterations are needed after Sobol+LP filtering?
2. Does adding CPU ssearch before GPU iRprop help?
3. What's the optimal top_k for a given number of Sobol starts?
4. Different max_starts values and their effect on discovery time.
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
from src.numba_solver import optimize_laminate_numba

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


def _round_key(lam_rad, decimals=1):
    deg = np.rad2deg(lam_rad)
    deg = (deg + 90) % 180 - 90
    return tuple(np.round(deg, decimals))


def experiment_irprop_iterations(dev, mod, mod_irprop, max_starts=10000, top_k=1500):
    """Test how many GPU iRprop iterations are needed."""
    known = _load_known_solutions()
    target_count = len(known)
    target = LP_VIQUERAT
    
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=N, scramble=True, seed=42)
    
    # Generate Sobol starts
    sobol_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
    
    # GPU LP filter
    lps = batch_lp_gpu(dev, mod, sobol_starts, N)
    losses = np.sum((lps - target) ** 2, axis=1)
    top_indices = np.argsort(losses)[:top_k]
    best_starts = sobol_starts[top_indices]
    
    print(f"\nSobol starts: {max_starts}, top_k: {top_k}", flush=True)
    print(f"{'Iters':<8} {'Found':<8} {'Time(ms)':<12} {'RMSE_best':<12}", flush=True)
    
    # Run iRprop with increasing iteration counts on the SAME starts
    # We need to reinitialize each time
    for n_iters in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
        t0 = time.perf_counter()
        best_angles, best_losses = gpu_batch_irprop(dev, mod_irprop, best_starts.copy(), target, max_iter=n_iters)
        t = time.perf_counter() - t0
        
        found = set()
        best_rmse = float('inf')
        for i in range(top_k):
            if best_losses[i] < 1e-2:
                rmse = compute_lp_rmse(best_angles[i].astype(np.float32), target)
                if rmse < 2e-2:
                    key = _round_key(best_angles[i])
                    found.add(key)
                best_rmse = min(best_rmse, rmse)
        
        print(f"{n_iters:<8d} {len(found):<8d} {t*1000:<12.1f} {best_rmse:<12.6f}", flush=True)


def experiment_ssearch_then_gpu(dev, mod, mod_irprop, max_starts=10000, top_k=1500):
    """Test CPU ssearch before GPU iRprop."""
    known = _load_known_solutions()
    target = LP_VIQUERAT
    
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=N, scramble=True, seed=42)
    
    sobol_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
    
    # GPU LP filter
    lps = batch_lp_gpu(dev, mod, sobol_starts, N)
    losses = np.sum((lps - target) ** 2, axis=1)
    top_indices = np.argsort(losses)[:top_k]
    best_starts = sobol_starts[top_indices]
    
    # CPU ssearch on top-K starts
    print(f"\nCPU ssearch + GPU iRprop (top_k={top_k})", flush=True)
    print(f"{'ssearch':<10} {'irprop':<10} {'Found':<8} {'Time(ms)':<12}", flush=True)
    
    for n_cf in [0, 1, 2]:
        for n_iters in [10, 30, 50]:
            t0 = time.perf_counter()
            
            # CPU ssearch
            if n_cf > 0:
                ssearched = best_starts.copy()
                for i in range(top_k):
                    opt, losses_i = optimize_laminate_numba(
                        ssearched[i:i+1], target, 
                        n_coarse_fine=n_cf,
                        irprop_grad_tol=1.0,  # skip iRprop, just ssearch
                        irprop_iters=0)
                    ssearched[i] = opt[0]
                # GPU iRprop on ssearched starts
                best_angles, best_losses = gpu_batch_irprop(dev, mod_irprop, ssearched, target, max_iter=n_iters)
            else:
                # No ssearch, just GPU iRprop
                best_angles, best_losses = gpu_batch_irprop(dev, mod_irprop, best_starts.copy(), target, max_iter=n_iters)
            
            t = time.perf_counter() - t0
            
            found = set()
            for i in range(top_k):
                if best_losses[i] < 1e-2:
                    rmse = compute_lp_rmse(best_angles[i].astype(np.float32), target)
                    if rmse < 2e-2:
                        key = _round_key(best_angles[i])
                        found.add(key)
            
            print(f"{n_cf:<10d} {n_iters:<10d} {len(found):<8d} {t*1000:<12.1f}", flush=True)


def experiment_optimal_pipeline(dev, mod, mod_irprop):
    """Find optimal pipeline configuration for minimum Viquerat discovery time."""
    known = _load_known_solutions()
    target_count = len(known)
    target = LP_VIQUERAT
    
    print("\n=== Full Pipeline Optimization ===", flush=True)
    print(f"{'starts':<8} {'top_k':<8} {'iters':<8} {'Found':<8} {'Time(ms)':<12} {'Found/s':<10}", flush=True)
    
    from scipy.stats.qmc import Sobol
    
    best_config = None
    best_found_per_sec = 0
    
    for max_starts in [3000, 5000, 10000, 20000]:
        sampler = Sobol(d=N, scramble=True, seed=42)
        sobol_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
        
        lps = batch_lp_gpu(dev, mod, sobol_starts, N)
        losses = np.sum((lps - target) ** 2, axis=1)
        
        for top_k in [300, 500, 1000, 1500, 2000]:
            if top_k > max_starts:
                continue
            top_indices = np.argsort(losses)[:top_k]
            best_starts = sobol_starts[top_indices]
            
            for n_iters in [20, 50, 100]:
                t0 = time.perf_counter()
                best_angles, best_losses = gpu_batch_irprop(dev, mod_irprop, best_starts.copy(), target, max_iter=n_iters)
                t = time.perf_counter() - t0
                
                found = set()
                for i in range(top_k):
                    if best_losses[i] < 1e-2:
                        rmse = compute_lp_rmse(best_angles[i].astype(np.float32), target)
                        if rmse < 2e-2:
                            key = _round_key(best_angles[i])
                            found.add(key)
                
                fps = len(found) / t if t > 0 else 0
                print(f"{max_starts:<8d} {top_k:<8d} {n_iters:<8d} {len(found):<8d} {t*1000:<12.1f} {fps:<10.1f}", flush=True)
                
                # Track best found/s that finds all solutions
                if len(found) >= target_count and (best_config is None or t < best_config[1]):
                    best_config = (f"starts={max_starts}, top_k={top_k}, iters={n_iters}", t, len(found))
    
    if best_config:
        print(f"\nBest config: {best_config[0]}, time={best_config[1]*1000:.1f}ms, found={best_config[2]}", flush=True)
    else:
        print("\nNo config found all solutions — need more starts/top_k", flush=True)


if __name__ == "__main__":
    print("GPU Pipeline Parameter Optimization", flush=True)
    print("=" * 70, flush=True)
    
    Z2, Z3 = _z2_z3(N)
    dev, mod = create_gpu_lp_solver(N, Z2, Z3)
    mod_irprop = sl.Module.load_from_source(dev, "irprop_step2", SLANG_IRPROP_STEP2)
    print("GPU initialized", flush=True)
    
    # Warmup
    warmup = np.random.random((100, N)).astype(np.float32) * np.pi - np.pi / 2
    _ = batch_lp_gpu(dev, mod, warmup, N)
    _ = gpu_batch_irprop(dev, mod_irprop, warmup, LP_VIQUERAT, max_iter=10)
    _ = optimize_laminate_numba(warmup[:1], LP_VIQUERAT, n_coarse_fine=1, irprop_grad_tol=1e-3)
    
    # Experiment 1: iRprop iteration count
    print("\n--- Experiment 1: iRprop iteration count ---", flush=True)
    experiment_irprop_iterations(dev, mod, mod_irprop, max_starts=10000, top_k=1500)
    
    # Experiment 2: CPU ssearch + GPU iRprop
    print("\n--- Experiment 2: CPU ssearch + GPU iRprop ---", flush=True)
    experiment_ssearch_then_gpu(dev, mod, mod_irprop, max_starts=10000, top_k=1500)
    
    # Experiment 3: Optimal pipeline
    print("\n--- Experiment 3: Full pipeline optimization ---", flush=True)
    experiment_optimal_pipeline(dev, mod, mod_irprop)