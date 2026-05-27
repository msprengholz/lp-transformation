#!/usr/bin/env python3
"""
Profile iRprop: how many iterations does each start actually use?
This tells us if we can reduce the iteration limit.
Uses numpy solver (fallback) when numba is not available.
"""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse, get_lp
from src.test_cases import LP_VIQUERAT

N = 12


def irprop_numpy(lam, lp_t, Z2, Z3, invN, N2, N3, max_iters=3000, grad_tol=1e-3,
                 sigma=0.1, n_plus=1.2, n_minus=0.5, s_min=1e-8, s_max=0.3):
    """Pure numpy iRprop implementation."""
    lam = lam.astype(np.float64).copy()
    step = np.full(N, sigma, dtype=np.float64)
    prev_grad = np.zeros(N, dtype=np.float64)
    iters_used = 0
    
    for it in range(max_iters):
        cos2 = np.cos(2*lam); sin2 = np.sin(2*lam)
        cos4 = np.cos(4*lam); sin4 = np.sin(4*lam)
        lp = np.array([np.sum(cos2)*invN, np.sum(sin2)*invN,
            np.sum(cos4)*invN, np.sum(sin4)*invN,
            Z2@cos2*N2, Z2@sin2*N2, Z2@cos4*N2, Z2@sin4*N2,
            Z3@cos2*N3, Z3@sin2*N3, Z3@cos4*N3, Z3@sin4*N3])
        res = lp_t - lp
        
        # Gradient
        grad = np.zeros(N)
        for k in range(N):
            val = (-2*sin2[k]*res[0] + 2*cos2[k]*res[1]
                   -2*sin4[k]*res[2] + 2*cos4[k]*res[3]
                   -2*sin2[k]*Z2[k]*res[4] + 2*cos2[k]*Z2[k]*res[5]
                   -2*sin4[k]*Z2[k]*res[6] + 2*cos4[k]*Z2[k]*res[7]
                   -2*sin2[k]*Z3[k]*res[8] + 2*cos2[k]*Z3[k]*res[9]
                   -2*sin4[k]*Z3[k]*res[10] + 2*cos4[k]*Z3[k]*res[11])
            grad[k] = -val * 2  # downhill
        
        gmax = np.max(np.abs(grad))
        if gmax < grad_tol:
            break
        iters_used = it + 1
        
        # iRprop update
        for k in range(N):
            if prev_grad[k] * grad[k] > 0:
                step[k] = min(step[k] * n_plus, s_max)
            elif prev_grad[k] * grad[k] < 0:
                step[k] = max(step[k] * n_minus, s_min)
                grad[k] = 0.0
            lam[k] -= np.sign(grad[k]) * step[k]
            lam[k] = np.clip(lam[k], -np.pi/2, np.pi/2)
            prev_grad[k] = grad[k]
    
    return lam, iters_used


def ssearch_numpy(lam, lp_t, Z2, Z3, invN, N2, N3, delta_deg=10.0):
    """Perform sequential angle search (one pass)."""
    lam = lam.copy()
    delta = np.deg2rad(delta_deg)
    n_angles = int(np.pi / delta) + 1
    angles = np.linspace(-np.pi/2, np.pi/2, n_angles)
    best_lam = lam.copy()
    
    for i in range(N):
        best_loss = float('inf')
        best_angle = lam[i]
        for angle in angles:
            trial = best_lam.copy()
            trial[i] = angle
            lp = get_lp(trial.astype(np.float32))
            loss = np.sum((lp_t - lp)**2)
            if loss < best_loss:
                best_loss = loss
                best_angle = angle
        best_lam[i] = best_angle
    
    return best_lam


def profile_viquerat():
    """Profile iRprop iteration needs on Viquerat 12-layer."""
    target = LP_VIQUERAT.astype(np.float64)
    Z2_ = Z2.astype(np.float64)
    Z3_ = Z3.astype(np.float64)
    invN_ = float(invN)
    N2_ = float(N2)
    N3_ = float(N3)
    
    rng = np.random.RandomState(42)
    n_starts = 200
    
    print("Profiling iRprop iteration needs on Viquerat 12-layer", flush=True)
    print("=" * 70, flush=True)
    
    # Generate starts
    starts = rng.random((n_starts, N)).astype(np.float64) * np.pi - np.pi / 2
    
    # Phase 1: ssearch + iRprop with different iteration limits
    print(f"\n{'Config':<40} {'Time(s)':<10} {'<1e-3':<8} {'<1e-2':<8} {'Best':<12}", flush=True)
    print("-" * 85, flush=True)
    
    configs = [
        # (n_cf, grad_tol, max_iters, label)
        (1, 1e-3, 100, "ssearch + iRprop (100 iters)"),
        (1, 1e-3, 50, "ssearch + iRprop (50 iters)"),
        (1, 1e-3, 20, "ssearch + iRprop (20 iters)"),
        (1, 1e-2, 100, "ssearch + iRprop (tol=1e-2, 100 iters)"),
        (1, 1e-3, 10, "ssearch + iRprop (10 iters)"),
        (2, 1e-3, 100, "2x ssearch + iRprop (100 iters)"),
    ]
    
    for n_cf, grad_tol, max_iters, label in configs:
        results = []
        t0 = time.perf_counter()
        for i in range(n_starts):
            lam = starts[i].copy()
            for _ in range(n_cf):
                lam = ssearch_numpy(lam, target, Z2_, Z3_, invN_, N2_, N3_, delta_deg=10.0)
            lam, iters = irprop_numpy(lam, target, Z2_, Z3_, invN_, N2_, N3_,
                                       max_iters=max_iters, grad_tol=grad_tol)
            rmse = compute_lp_rmse(lam.astype(np.float32), LP_VIQUERAT)
            results.append((rmse, iters))
        
        results = np.array([(r[0], r[1]) for r in results])
        rmses = results[:, 0]
        iters = results[:, 1]
        t = time.perf_counter() - t0
        
        n_1e3 = int(np.sum(rmses < 1e-3))
        n_1e2 = int(np.sum(rmses < 1e-2))
        print(f"{label:<40} {t:<10.2f} {n_1e3:<8d} {n_1e2:<8d} {np.min(rmses):<12.2e}", flush=True)
        print(f"  Iters: mean={iters.mean():.1f}, median={np.median(iters):.0f}, "
              f"p90={np.percentile(iters, 90):.0f}, max={iters.max()}", flush=True)


N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)


if __name__ == "__main__":
    profile_viquerat()