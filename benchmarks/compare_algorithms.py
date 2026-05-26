#!/usr/bin/env python3
"""Compare solution discovery of different algorithms on Viquerat LP set."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.test_cases import LP_VIQUERAT
from src.lp_functions import compute_lp_rmse


def _round_and_wrap(lam):
    deg = np.round(np.rad2deg(lam), 1)
    return tuple(((deg + 90) % 180 - 90).astype(np.float32))


def count_unique(opt_lams, threshold=1e-3):
    found = set()
    for o in opt_lams:
        loss = compute_lp_rmse(o, LP_VIQUERAT)
        if loss < threshold:
            found.add(_round_and_wrap(o))
    return len(found)


print("=" * 65)
print("  Algorithm comparison: Viquerat 12-layer")
print("=" * 65)

N = 12
lp_t = LP_VIQUERAT
rng = np.random.default_rng(42)
n_starts = 500

# ---------- 1. Our current solver: coarse ssearch + iRprop ----------
from src.numpy_solver import optimize_laminate

print("\n[1] Our solver: 10 deg ssearch + 5 deg fine + iRprop")
rand_lams = rng.random((n_starts, N), dtype=np.float32) * np.pi - np.pi / 2
t0 = time.perf_counter()
opt, losses = optimize_laminate(rand_lams, lp_t, n_coarse_fine=3,
                                 delta_coarse_deg=10.0, delta_fine_deg=5.0,
                                 irprop_iters=3000, irprop_grad_tol=1e-6)
dt = time.perf_counter() - t0
n1 = count_unique(opt)
print("  %d starts, %.1fs, %d unique solutions" % (n_starts, dt, n1))
print("  best=%.2e, median=%.2e" % (losses.min(), np.median(losses)))

# ---------- 2. Paper: fine ssearch(3 deg) + nearsearch + iRprop ----------
from src.paper_solver import search_layup_paper

print("\n[2] Paper: 3 deg ssearch + nearsearch + iRprop")
t0 = time.perf_counter()
sols, ts = search_layup_paper(lp_t, N, it_comb=n_starts, it_sub=3,
                               it_s=3, it_n=1, delta_s=3.0,
                               eps=0.03, error=1e-6, irprop_iters=500)
dt = time.perf_counter() - t0
print("  %d starts, %.1fs, %d unique solutions" % (n_starts, dt, len(sols)))

# ---------- 3. Just fine ssearch(3 deg) + iRprop (no nearsearch) ----------
print("\n[3] 3 deg ssearch + iRprop (no nearsearch)")
from src.paper_solver import ssearch_paper
from src.numpy_solver import iRpropm
rng2 = np.random.default_rng(42)
found = set()
t0 = time.perf_counter()
for i in range(n_starts):
    lam = (rng2.random(N, dtype=np.float32) * np.pi - np.pi / 2)
    lam = ssearch_paper(lam, lp_t, delta_deg=3.0, iterations=3)
    lam = iRpropm(lam, lp_t, it_iRprop=500, grad_tol=1e-6)
    loss = compute_lp_rmse(lam, lp_t)
    if loss < 1e-6:
        found.add(_round_and_wrap(lam))
dt = time.perf_counter() - t0
print("  %d starts, %.1fs, %d unique solutions" % (n_starts, dt, len(found)))
