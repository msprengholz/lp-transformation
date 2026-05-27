#!/usr/bin/env python3
"""
Investigate analytical convergence for LP back-transformation.

Compare Newton's method (exact diagonal Hessian) vs iRprop vs coordinate descent.
"""
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lp_functions import _z2_z3, _norm_factors, get_lp, compute_lp_rmse
from src.test_cases import LP_VIQUERAT

N = 12
Z2, Z3 = _z2_z3(N)
invN, N2, N3 = _norm_factors(N)


def irprop_manual(lam, lp_t, max_iter=100, sigma=0.01, grad_tol=1e-3, n_plus=1.2, n_minus=0.5):
    """Manual iRprop implementation for fair comparison."""
    lam = lam.astype(np.float64).copy()
    step = np.full(N, sigma, dtype=np.float64)
    prev_grad = np.zeros(N, dtype=np.float64)
    best_lam = lam.copy()
    best_loss = 1e10
    
    for it in range(max_iter):
        cos2 = np.cos(2*lam); sin2 = np.sin(2*lam)
        cos4 = np.cos(4*lam); sin4 = np.sin(4*lam)
        lp = np.array([np.sum(cos2)*invN, np.sum(sin2)*invN,
            np.sum(cos4)*invN, np.sum(sin4)*invN,
            Z2@cos2*N2, Z2@sin2*N2, Z2@cos4*N2, Z2@sin4*N2,
            Z3@cos2*N3, Z3@sin2*N3, Z3@cos4*N3, Z3@sin4*N3])
        res = lp_t - lp
        loss = np.sum(res**2)
        
        if loss < best_loss:
            best_loss = loss
            best_lam = lam.copy()
        
        # Gradient
        grad = np.zeros(N)
        for k in range(N):
            val = (-2*sin2[k]*res[0] + 2*cos2[k]*res[1]
                   -2*sin4[k]*res[2] + 2*cos4[k]*res[3]
                   -2*sin2[k]*Z2[k]*res[4] + 2*cos2[k]*Z2[k]*res[5]
                   -2*sin4[k]*Z2[k]*res[6] + 2*cos4[k]*Z2[k]*res[7]
                   -2*sin2[k]*Z3[k]*res[8] + 2*cos2[k]*Z3[k]*res[9]
                   -2*sin4[k]*Z3[k]*res[10] + 2*cos4[k]*Z3[k]*res[11])
            grad[k] = -val * 2
        
        if np.max(np.abs(grad)) < grad_tol:
            break
        
        # iRprop step update
        for k in range(N):
            if prev_grad[k] * grad[k] > 0:
                step[k] = min(step[k] * n_plus, 0.5)
            elif prev_grad[k] * grad[k] < 0:
                step[k] = max(step[k] * n_minus, 1e-5)
                grad[k] = 0.0
            lam[k] -= np.sign(grad[k]) * step[k]
            lam[k] = np.clip(lam[k], -np.pi/2, np.pi/2)
            prev_grad[k] = grad[k]
    
    return best_lam, best_loss


def newton_diagonal(lam, lp_t, max_iter=20, grad_tol=1e-8, step_clip=0.3):
    """Newton's method with exact per-angle diagonal Hessian.
    
    For each angle k, compute exact gradient and Hessian diagonal,
    then take a Newton step: Δλ = -grad / H_diag.
    """
    lam = lam.astype(np.float64).copy()
    lp_t = lp_t.astype(np.float64)
    best_lam = lam.copy()
    best_loss = 1e10
    losses = []
    
    for it in range(max_iter):
        cos2 = np.cos(2*lam); sin2 = np.sin(2*lam)
        cos4 = np.cos(4*lam); sin4 = np.sin(4*lam)
        lp = np.array([np.sum(cos2)*invN, np.sum(sin2)*invN,
            np.sum(cos4)*invN, np.sum(sin4)*invN,
            Z2@cos2*N2, Z2@sin2*N2, Z2@cos4*N2, Z2@sin4*N2,
            Z3@cos2*N3, Z3@sin2*N3, Z3@cos4*N3, Z3@sin4*N3])
        res = lp_t - lp
        loss = np.sum(res**2)
        losses.append(loss)
        
        if loss < best_loss:
            best_loss = loss
            best_lam = lam.copy()
        
        if loss < 1e-10:
            break
        
        gmax = 0.0
        for k in range(N):
            # Gradient (same as numba)
            val = (-2*sin2[k]*res[0] + 2*cos2[k]*res[1]
                   -2*sin4[k]*res[2] + 2*cos4[k]*res[3]
                   -2*sin2[k]*Z2[k]*res[4] + 2*cos2[k]*Z2[k]*res[5]
                   -2*sin4[k]*Z2[k]*res[6] + 2*cos4[k]*Z2[k]*res[7]
                   -2*sin2[k]*Z3[k]*res[8] + 2*cos2[k]*Z3[k]*res[9]
                   -2*sin4[k]*Z3[k]*res[10] + 2*cos4[k]*Z3[k]*res[11])
            grad_k = -val * 2  # downhill gradient
            gmax = max(gmax, abs(grad_k))
            
            # Diagonal of exact Hessian
            dLP = np.array([
                -2*sin2[k]*invN, 2*cos2[k]*invN,
                -2*sin4[k]*invN, 2*cos4[k]*invN,
                -2*sin2[k]*Z2[k]*N2, 2*cos2[k]*Z2[k]*N2,
                -2*sin4[k]*Z2[k]*N2, 2*cos4[k]*Z2[k]*N2,
                -2*sin2[k]*Z3[k]*N3, 2*cos2[k]*Z3[k]*N3,
                -2*sin4[k]*Z3[k]*N3, 2*cos4[k]*Z3[k]*N3])
            d2LP = np.array([
                -4*cos2[k]*invN, -4*sin2[k]*invN,
                -16*cos4[k]*invN, -16*sin4[k]*invN,
                -4*cos2[k]*Z2[k]*N2, -4*sin2[k]*Z2[k]*N2,
                -16*cos4[k]*Z2[k]*N2, -16*sin4[k]*Z2[k]*N2,
                -4*cos2[k]*Z3[k]*N3, -4*sin2[k]*Z3[k]*N3,
                -16*cos4[k]*Z3[k]*N3, -16*sin4[k]*Z3[k]*N3])
            
            h_kk = 2.0 * (np.dot(dLP, dLP) - np.dot(res, d2LP))
            
            if abs(h_kk) > 1e-12:
                step = np.clip(grad_k / h_kk, -step_clip, step_clip)
                lam[k] -= step
                lam[k] = np.clip(lam[k], -np.pi/2, np.pi/2)
        
        if gmax < grad_tol:
            break
    
    return best_lam.astype(np.float32), best_loss, losses


def newton_full(lam, lp_t, max_iter=20, grad_tol=1e-8, step_clip=0.3):
    """Full Newton's method with complete Hessian.
    
    Solves H * delta = -g for the full N×N Hessian.
    Much better than diagonal because it accounts for angle-angle coupling.
    """
    lam = lam.astype(np.float64).copy()
    lp_t = lp_t.astype(np.float64)
    best_lam = lam.copy()
    best_loss = 1e10
    losses = []
    
    for it in range(max_iter):
        cos2 = np.cos(2*lam); sin2 = np.sin(2*lam)
        cos4 = np.cos(4*lam); sin4 = np.sin(4*lam)
        lp = np.array([np.sum(cos2)*invN, np.sum(sin2)*invN,
            np.sum(cos4)*invN, np.sum(sin4)*invN,
            Z2@cos2*N2, Z2@sin2*N2, Z2@cos4*N2, Z2@sin4*N2,
            Z3@cos2*N3, Z3@sin2*N3, Z3@cos4*N3, Z3@sin4*N3])
        res = lp_t - lp
        loss = np.sum(res**2)
        losses.append(loss)
        
        if loss < best_loss:
            best_loss = loss
            best_lam = lam.copy()
        
        if loss < 1e-10:
            break
        
        # Compute gradient
        grad = np.zeros(N)
        for k in range(N):
            val = (-2*sin2[k]*res[0] + 2*cos2[k]*res[1]
                   -2*sin4[k]*res[2] + 2*cos4[k]*res[3]
                   -2*sin2[k]*Z2[k]*res[4] + 2*cos2[k]*Z2[k]*res[5]
                   -2*sin4[k]*Z2[k]*res[6] + 2*cos4[k]*Z2[k]*res[7]
                   -2*sin2[k]*Z3[k]*res[8] + 2*cos2[k]*Z3[k]*res[9]
                   -2*sin4[k]*Z3[k]*res[10] + 2*cos4[k]*Z3[k]*res[11])
            grad[k] = -val * 2  # downhill
        
        # Compute full Hessian
        H = np.zeros((N, N))
        for k in range(N):
            dLPk = np.array([
                -2*sin2[k]*invN, 2*cos2[k]*invN,
                -2*sin4[k]*invN, 2*cos4[k]*invN,
                -2*sin2[k]*Z2[k]*N2, 2*cos2[k]*Z2[k]*N2,
                -2*sin4[k]*Z2[k]*N2, 2*cos4[k]*Z2[k]*N2,
                -2*sin2[k]*Z3[k]*N3, 2*cos2[k]*Z3[k]*N3,
                -2*sin4[k]*Z3[k]*N3, 2*cos4[k]*Z3[k]*N3])
            
            d2LPk = np.array([
                -4*cos2[k]*invN, -4*sin2[k]*invN,
                -16*cos4[k]*invN, -16*sin4[k]*invN,
                -4*cos2[k]*Z2[k]*N2, -4*sin2[k]*Z2[k]*N2,
                -16*cos4[k]*Z2[k]*N2, -16*sin4[k]*Z2[k]*N2,
                -4*cos2[k]*Z3[k]*N3, -4*sin2[k]*Z3[k]*N3,
                -16*cos4[k]*Z3[k]*N3, -16*sin4[k]*Z3[k]*N3])
            
            # Diagonal
            H[k,k] = 2.0 * (np.dot(dLPk, dLPk) - np.dot(res, d2LPk))
            
            # Off-diagonal: H[k,l] = 2 * dLP_k · dLP_l
            for l in range(k+1, N):
                dLPl = np.array([
                    -2*sin2[l]*invN, 2*cos2[l]*invN,
                    -2*sin4[l]*invN, 2*cos4[l]*invN,
                    -2*sin2[l]*Z2[l]*N2, 2*cos2[l]*Z2[l]*N2,
                    -2*sin4[l]*Z2[l]*N2, 2*cos4[l]*Z2[l]*N2,
                    -2*sin2[l]*Z3[l]*N3, 2*cos2[l]*Z3[l]*N3,
                    -2*sin4[l]*Z3[l]*N3, 2*cos4[l]*Z3[l]*N3])
                H[k,l] = 2.0 * np.dot(dLPk, dLPl)
                H[l,k] = H[k,l]
        
        # Newton step: Δ = H⁻¹ g
        try:
            delta = np.linalg.solve(H, grad)
            # Clip step for stability
            max_step = np.max(np.abs(delta))
            if max_step > step_clip:
                delta *= step_clip / max_step
            lam -= delta
            lam = np.clip(lam, -np.pi/2, np.pi/2)
        except np.linalg.LinAlgError:
            # Singular Hessian, fall back to diagonal Newton
            h_diag = np.array([H[k,k] for k in range(N)])
            for k in range(N):
                if abs(h_diag[k]) > 1e-12:
                    step = np.clip(grad[k] / h_diag[k], -step_clip, step_clip)
                    lam[k] -= step
                    lam[k] = np.clip(lam[k], -np.pi/2, np.pi/2)
        
        if np.max(np.abs(grad)) < grad_tol:
            break
    
    return best_lam.astype(np.float32), best_loss, losses


if __name__ == "__main__":
    target = LP_VIQUERAT
    rng = np.random.RandomState(42)
    
    print("Analytical convergence investigation", flush=True)
    print("=" * 70, flush=True)
    
    n_starts = 50
    
    print(f"\n{'Method':<20} {'Best RMSE':<12} {'Median':<12} {'<1e-3':<8} {'<1e-6':<8} {'Time(ms)':<10}", flush=True)
    print("-" * 75, flush=True)
    
    for name, solver_fn in [
        ("iRprop", lambda lam: irprop_manual(lam, target)),
        ("Newton-diag", lambda lam: newton_diagonal(lam, target)),
        ("Newton-full", lambda lam: newton_full(lam, target)),
    ]:
        results = []
        times = []
        for i in range(n_starts):
            lam = rng.random(N).astype(np.float32) * np.pi - np.pi/2
            t0 = time.perf_counter()
            best_lam, best_loss = solver_fn(lam)[:2]
            t = time.perf_counter() - t0
            rmse = compute_lp_rmse(best_lam, target)
            results.append(rmse)
            times.append(t * 1000)
        
        results = np.array(results)
        print(f"{name:<20} {np.min(results):<12.2e} {np.median(results):<12.2e} "
              f"{np.sum(results<1e-3):<8d} {np.sum(results<1e-6):<8d} {np.mean(times):<10.1f}", flush=True)
    
    # Per-iteration convergence comparison
    print("\n--- Per-iteration loss (single start) ---", flush=True)
    lam0 = rng.random(N).astype(np.float32) * np.pi - np.pi/2
    
    ir_lam, ir_loss = irprop_manual(lam0, target, max_iter=50)
    nd_lam, nd_loss, nd_losses = newton_diagonal(lam0, target, max_iter=50)
    nf_lam, nf_loss, nf_losses = newton_full(lam0, target, max_iter=50)
    
    losses_ir = [ir_loss] * 50  # iRprop doesn't track per-iter loss easily
    losses_nd = nd_losses
    losses_nf = nf_losses
    
    print(f"{'Iter':<6} {'iRprop':<15} {'Newton-diag':<15} {'Newton-full':<15}", flush=True)
    for i in range(min(30, len(losses_ir), len(losses_nd), len(losses_nf))):
        if i % 3 == 0 or i < 10:
            ir = losses_ir[i] if i < len(losses_ir) else float('inf')
            nd = losses_nd[i] if i < len(losses_nd) else float('inf')
            nf = losses_nf[i] if i < len(losses_nf) else float('inf')
            print(f"{i:<6} {ir:<15.6e} {nd:<15.6e} {nf:<15.6e}", flush=True)
    
    print(f"\niRprop: {len(losses_ir)} iters, final loss: {losses_ir[-1]:.2e}" if losses_ir else "")
    print(f"Newton-diag: {len(losses_nd)} iters, final loss: {losses_nd[-1]:.2e}" if losses_nd else "")
    print(f"Newton-full: {len(losses_nf)} iters, final loss: {losses_nf[-1]:.2e}" if losses_nf else "")