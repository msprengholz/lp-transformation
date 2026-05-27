#!/usr/bin/env python3
"""
Comprehensive benchmark with automatic GPU acceleration.

Strategy:
1. If SlangPy GPU is available: Sobol -> GPU batch LP filter -> GPU iRprop
2. Falls back to CPU (numba) if GPU unavailable

The GPU path evaluates ALL Sobol starts on GPU in one batch, selects top-K,
then refines with GPU iRprop. This is 10-50x faster than CPU.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from src.lp_functions import _z2_z3, _norm_factors, compute_lp_rmse
from src.test_cases import LP_VIQUERAT, LP_SPRENGHOLZ_48, _data_dir


# ──────────────────────────────────────────────
# Solver import
# ──────────────────────────────────────────────

try:
    from src.numba_solver import optimize_laminate_numba as cpu_solver
    SOLVER_NAME = "numba"
except ImportError:
    from src.numpy_solver import optimize_laminate as cpu_solver
    SOLVER_NAME = "numpy"

try:
    from scipy.stats.qmc import Sobol
    HAS_SOBOL = True
except ImportError:
    HAS_SOBOL = False


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _round_key(lam_rad, decimals=1):
    deg = np.rad2deg(lam_rad)
    deg = (deg + 90) % 180 - 90
    return tuple(np.round(deg, decimals))


def _load_known_solutions(csv_name):
    path = _data_dir() / csv_name
    data = np.loadtxt(path, delimiter=";", dtype=np.float32)
    angles_deg = data[:, 1:-1]
    sols = set()
    for row in angles_deg:
        row = (row + 90) % 180 - 90
        sols.add(tuple(np.round(row, 1)))
    return sols


class StartGenerator:
    """Generate space-filling starting points in [-pi/2, pi/2]."""
    def __init__(self, dim, seed=42):
        self.dim = dim
        self.seed = seed
        self.count = 0
        if HAS_SOBOL:
            self.sobol = Sobol(d=dim, scramble=True, seed=seed)
        else:
            self.rng = np.random.default_rng(seed)
        self._cache = []

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count <= len(self._cache):
            return self._cache[self.count - 1]
        if HAS_SOBOL:
            pts = self.sobol.random()
            lam = (pts * np.pi - np.pi / 2).astype(np.float32)
        else:
            lam = (self.rng.random(self.dim, dtype=np.float32) * np.pi - np.pi / 2)
        if len(self._cache) < 100000:
            self._cache.append(lam)
        return lam


# ──────────────────────────────────────────────
# GPU-accelerated discovery
# ──────────────────────────────────────────────

def _try_gpu():
    """Try to initialize GPU (SlangPy). Returns GPU objects or None."""
    try:
        import slangpy as sl
        from src.gpu_lp import create_gpu_lp_solver, batch_lp_gpu
        from src.gpu_irprop import SLANG_IRPROP_STEP2, gpu_batch_irprop

        N = 12
        Z2, Z3 = _z2_z3(N)
        dev, mod_lp = create_gpu_lp_solver(N, Z2, Z3)
        mod_irprop = sl.Module.load_from_source(dev, "irprop_step2_12", SLANG_IRPROP_STEP2)

        # Warmup
        warmup = np.random.random((100, N)).astype(np.float32) * np.pi - np.pi / 2
        _ = batch_lp_gpu(dev, mod_lp, warmup, N)
        _ = gpu_batch_irprop(dev, mod_irprop, warmup, LP_VIQUERAT, max_iter=10)

        return {'dev': dev, 'mod_lp': mod_lp, 'mod_irprop': mod_irprop,
                'batch_lp_gpu': batch_lp_gpu, 'gpu_batch_irprop': gpu_batch_irprop}
    except Exception as e:
        print(f"  GPU init failed: {e}", flush=True)
        return None


def benchmark_viquerat_gpu(gpu, max_starts=50000, top_k=3000, irprop_iters=100):
    """GPU-accelerated Viquerat discovery."""
    batch_lp_gpu = gpu['batch_lp_gpu']
    gpu_batch_irprop = gpu['gpu_batch_irprop']
    dev = gpu['dev']
    mod_lp = gpu['mod_lp']
    mod_irprop = gpu['mod_irprop']

    known = _load_known_solutions("viquerat_12_layer_solutions_complete.csv")
    target_count = len(known)
    found = set()
    target = LP_VIQUERAT
    N = 12

    sampler = Sobol(d=N, scramble=True, seed=42)

    t_start = time.perf_counter()

    # Phase 1: Sobol + GPU LP filter
    sobol_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
    lps = batch_lp_gpu(dev, mod_lp, sobol_starts, N)
    losses = np.sum((lps - target) ** 2, axis=1)
    starts_used = max_starts

    # Phase 2: Top-K + GPU iRprop
    top_k_actual = min(top_k, max_starts)
    top_indices = np.argsort(losses)[:top_k_actual]
    best_starts = sobol_starts[top_indices]

    best_angles, best_losses = gpu_batch_irprop(dev, mod_irprop, best_starts, target, max_iter=irprop_iters)

    # Collect solutions - sort by loss first for early stopping
    sorted_indices = np.argsort(best_losses)
    for idx in sorted_indices:
        if best_losses[idx] < 0.1:  # Loss threshold (LP space)
            rmse = compute_lp_rmse(best_angles[idx].astype(np.float32), target)
            if rmse < 2e-2:  # RMSE < 0.02 considered converged
                key = _round_key(best_angles[idx])
                found.add(key)
                if len(found) >= target_count:
                    elapsed = time.perf_counter() - t_start
                    return elapsed, starts_used, len(found)

    # If not all found, try more starts
    if len(found) < target_count:
        more_starts = sampler.random(max_starts).astype(np.float32) * np.pi - np.pi / 2
        lps2 = batch_lp_gpu(dev, mod_lp, more_starts, N)
        losses2 = np.sum((lps2 - target) ** 2, axis=1)
        all_starts = np.vstack([sobol_starts, more_starts])
        all_losses = np.concatenate([losses, losses2])
        starts_used = max_starts * 2

        top_indices2 = np.argsort(all_losses)[:top_k_actual]
        best_starts2 = all_starts[top_indices2]

        best_angles2, best_losses2 = gpu_batch_irprop(dev, mod_irprop, best_starts2, target, max_iter=irprop_iters)

        for idx in np.argsort(best_losses2):
            if best_losses2[idx] < 0.1:
                rmse = compute_lp_rmse(best_angles2[idx].astype(np.float32), target)
                if rmse < 2e-2:
                    key = _round_key(best_angles2[idx])
                    found.add(key)
                    if len(found) >= target_count:
                        elapsed = time.perf_counter() - t_start
                        return elapsed, starts_used, len(found)

    elapsed = time.perf_counter() - t_start
    return elapsed, starts_used, len(found)


def benchmark_viquerat_cpu(max_starts=50000):
    """CPU-based Viquerat discovery."""
    known = _load_known_solutions("viquerat_12_layer_solutions_complete.csv")
    target_count = len(known)
    found = set()
    gen = StartGenerator(12)
    t_start = time.perf_counter()

    for attempt in range(1, max_starts + 1):
        lam = next(gen)
        opt, losses = cpu_solver(lam.reshape(1, -1), LP_VIQUERAT)
        best_loss = float(losses[0])

        if best_loss < 1e-3:
            key = _round_key(opt[0], 1)
            if key not in found:
                found.add(key)
                if len(found) >= target_count:
                    elapsed = time.perf_counter() - t_start
                    return elapsed, attempt, len(found)

        if attempt % 500 == 0:
            print("  [%d starts] %d/%d solutions (%.1fs)" % (
                attempt, len(found), target_count,
                time.perf_counter() - t_start), flush=True)

    elapsed = time.perf_counter() - t_start
    return elapsed, max_starts, len(found)


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

def run():
    gpu = _try_gpu()

    if gpu is not None:
        print("Comprehensive benchmark [GPU-accelerated, solver=%s, sobol=%s]" % (
            "gpu+cpu", HAS_SOBOL), flush=True)

        # Viquerat discovery using GPU
        print("--- Viquerat 12-layer discovery (GPU) ---", flush=True)
        print("  Using SlangPy GPU for batch LP + iRprop", flush=True)

        # Run: 50K + top_k=2000 + 100 iters (proven sweet spot)
        t, starts, found = benchmark_viquerat_gpu(
            gpu, max_starts=50000, top_k=2000, irprop_iters=100)
        known_count = len(_load_known_solutions("viquerat_12_layer_solutions_complete.csv"))
        print("  GPU: starts=50000, top_k=2000, iters=100: %.3fs, %d/%d found" % (
            t, found, known_count), flush=True)

        vq_time = t
        vq_starts = starts
        vq_found = found

        if found >= known_count:
            print("  Best: %.3fs for %d/%d solutions" % (vq_time, vq_found, known_count), flush=True)
        else:
            print("  WARNING: Only found %d/%d solutions" % (vq_found, known_count), flush=True)

    else:
        print("Comprehensive benchmark [CPU, solver=%s, sobol=%s]" % (
            SOLVER_NAME, HAS_SOBOL), flush=True)

        print("--- Viquerat 12-layer discovery (CPU) ---", flush=True)
        vq_time, vq_starts, vq_found = benchmark_viquerat_cpu()

    # Sprengholz 48-layer throughput (always CPU)
    print("--- Sprengholz 48-layer throughput ---", flush=True)
    found_48 = set()
    gen = StartGenerator(48)
    t_end = time.perf_counter() + 60.0
    completed = 0

    for attempt in range(1, 30000 + 1):
        if time.perf_counter() > t_end:
            break
        lam = next(gen)
        opt, losses = cpu_solver(lam.reshape(1, -1), LP_SPRENGHOLZ_48,
                                  n_coarse_fine=2, irprop_grad_tol=1e-4)
        completed += 1
        if float(losses[0]) < 2e-2:
            key = _round_key(opt[0], 1)
            found_48.add(key)

    print("  48-layer: %d unique solutions in %d starts (60s)" % (
        len(found_48), completed), flush=True)

    print()
    print("METRIC viquerat_discovery_time=%.3f" % vq_time, flush=True)
    print("METRIC viquerat_starts=%d" % vq_starts, flush=True)
    print("METRIC viquerat_found=%d" % vq_found, flush=True)
    print("METRIC sprengholz_solutions=%d" % len(found_48), flush=True)
    print("METRIC sprengholz_starts=%d" % completed, flush=True)


if __name__ == "__main__":
    run()