#!/usr/bin/env python3
"""
GPU pipeline optimization: find optimal iRprop iteration count.

Key hypothesis: After Sobol+LP filtering, most starts are already close to a basin.
Fewer iRprop iterations may suffice, dramatically reducing GPU time.

This script runs on Colab with GPU.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse
from src.test_cases import LP_VIQUERAT, _data_dir
from src.numba_solver import optimize_laminate_numba

N = 12
Z2, Z3 = _z2_z3(N)
target = LP_VIQUERAT


def _round_key(lam_rad, decimals=1):
    deg = np.rad2deg(lam_rad)
    deg = (deg + 90) % 180 - 90
    return tuple(np.round(deg, decimals))


def _load_known_solutions():
    solutions = set()
    path = _data_dir() / "viquerat_12_layer_solutions_complete.csv"
    if not path.exists():
        return set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 13:
                angles = tuple(round(float(a), 1) for a in parts[1:13])
                solutions.add(angles)
    return solutions


def experiment_irprop_iters_with_ssearch():
    """Test how many iRprop iterations are needed after ssearch."""
    known = _load_known_solutions()
    target_count = len(known)
    
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=N, scramble=True, seed=42)
    
    print("=== iRprop iteration count after ssearch + Sobol starts ===", flush=True)
    print(f"Known solutions: {target_count}", flush=True)
    
    for max_starts in [5000, 10000, 20000]:
        # Generate Sobol starts
        starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
        
        for n_cf in [1, 2]:  # n_coarse_fine: ssearch rounds
            for n_iters in [5, 10, 15, 20, 30, 50, 100, 300, 1000, 3000]:
                t0 = time.perf_counter()
                
                found = set()
                for i in range(max_starts):
                    opt, losses = optimize_laminate_numba(
                        starts[i:i+1], target,
                        n_coarse_fine=n_cf,
                        irprop_grad_tol=1e-8 if n_iters > 100 else 1e-3,
                        irprop_iters=n_iters)
                    best_loss = float(losses[0])
                    if best_loss < 1e-2:
                        rmse = compute_lp_rmse(opt[0], target)
                        if rmse < 2e-2:
                            key = _round_key(opt[0])
                            found.add(key)
                
                t = time.perf_counter() - t0
                fps = len(found) / t if t > 0 else 0
                
                print(f"  starts={max_starts:>6d}, n_cf={n_cf}, iters={n_iters:>5d}: "
                      f"{len(found):>3d}/{target_count}, {t:.2f}s, {fps:.1f} sol/s", flush=True)
                
                if len(found) >= target_count:
                    break
    
    print("\n=== Done ===", flush=True)


def experiment_gpu_pipeline_params():
    """Test GPU pipeline parameters. Only runs if slangpy is available."""
    try:
        import slangpy as sl
        from slangpy import grid, Tensor
        from src.gpu_lp import create_gpu_lp_solver, batch_lp_gpu
        from src.gpu_irprop import SLANG_IRPROP_STEP2, gpu_batch_irprop
    except ImportError:
        print("slangpy not available — skipping GPU experiments", flush=True)
        return
    
    known = _load_known_solutions()
    target_count = len(known)
    
    Z2, Z3 = _z2_z3(N)
    dev, mod = create_gpu_lp_solver(N, Z2, Z3)
    mod_irprop = sl.Module.load_from_source(dev, "irprop_step2", SLANG_IRPROP_STEP2)
    
    # Warmup
    warmup = np.random.random((100, N)).astype(np.float32) * np.pi - np.pi / 2
    _ = batch_lp_gpu(dev, mod, warmup, N)
    _ = gpu_batch_irprop(dev, mod_irprop, warmup, LP_VIQUERAT, max_iter=10)
    
    print("\n=== GPU Pipeline Parameters ===", flush=True)
    print(f"{'starts':<8} {'top_k':<8} {'iters':<8} {'Found':<8} {'Time(ms)':<12} {'Found/s':<10}", flush=True)
    
    from scipy.stats.qmc import Sobol
    
    for max_starts in [5000, 10000, 20000, 50000]:
        sampler = Sobol(d=N, scramble=True, seed=42)
        sobol_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
        
        # GPU LP filter
        lps = batch_lp_gpu(dev, mod, sobol_starts, N)
        losses = np.sum((lps - target) ** 2, axis=1)
        
        for top_k in [500, 1000, 1500, 2000, 3000]:
            if top_k > max_starts:
                continue
            top_indices = np.argsort(losses)[:top_k]
            best_starts = sobol_starts[top_indices]
            
            for n_iters in [10, 20, 50, 100]:
                t0 = time.perf_counter()
                best_angles, best_losses = gpu_batch_irprop(
                    dev, mod_irprop, best_starts.copy(), target, max_iter=n_iters)
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
    
    print("\n=== GPU Done ===", flush=True)


if __name__ == "__main__":
    # Run CPU experiments (always available)
    experiment_irprop_iters_with_ssearch()
    
    # Run GPU experiments (only on Colab)
    experiment_gpu_pipeline_params()