#!/usr/bin/env python3
"""Investigate why 48-layer solver can't converge below RMSE 1e-3."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.numba_solver import optimize_laminate_numba as solver
from src.numpy_solver import optimize_laminate as solver_np
from src.test_cases import LP_SPRENGHOLZ_48
from src.lp_functions import compute_lp_rmse

rng = np.random.default_rng(42)

print("48-layer RMSE investigation")
print("=" * 50)

# Test various parameter combinations
for label, params in [
    ("numba default", {}),
    ("numba n_cf=2, gt=1e-4", {"n_coarse_fine": 2, "irprop_grad_tol": 1e-4}),
    ("numba n_cf=3, gt=1e-5", {"n_coarse_fine": 3, "irprop_grad_tol": 1e-5}),
    ("numpy default", {"solver": "numpy"}),
]:
    use_np = params.pop("solver", None) == "numpy"
    fn = solver_np if use_np else solver
    
    losses = []
    rng = np.random.default_rng(42)
    for i in range(30):
        lam = (rng.random(48, dtype=np.float32) * np.pi - np.pi / 2)
        if use_np:
            opt, ls = fn(lam.reshape(1, -1), LP_SPRENGHOLZ_48)
        else:
            opt, ls = fn(lam.reshape(1, -1), LP_SPRENGHOLZ_48, **params)
        losses.append(float(ls[0]))
    
    losses = np.array(losses)
    print("\n  %s:" % label)
    print("    min: %.4e  median: %.4e" % (losses.min(), np.median(losses)))
    print("    <1e-3: %d/30  <1e-2: %d/30  <0.1: %d/30" % (
        np.sum(losses < 1e-3), np.sum(losses < 1e-2), np.sum(losses < 0.1)))
