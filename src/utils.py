"""
Shared utilities: seeding, test-problem generation, timing, result helpers.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .lp_functions import (make_random_laminate, make_target_lp_from_laminate,
                            compute_lp_rmse, compute_angle_deviation,
                            wrap_angles)


# ──────────────────────────────────────────────
# Seeding for reproducibility
# ──────────────────────────────────────────────

_SEED = 42


def get_rng(seed: int = _SEED) -> np.random.Generator:
    """Return a seeded RNG for reproducible benchmark data."""
    return np.random.default_rng(seed)


# ──────────────────────────────────────────────
# Benchmark problem specification
# ──────────────────────────────────────────────

@dataclass
class Problem:
    """A single optimisation problem: recover *lam_true* from *lp_t*."""
    lam_true: NDArray[np.float32]
    lp_t: NDArray[np.float32]
    label: str = ""

    @property
    def n_layers(self) -> int:
        return self.lam_true.size


@dataclass
class BenchmarkResult:
    """Result of running one solver on one problem."""
    solver: str
    n_layers: int
    problem_label: str
    lp_rmse: float
    max_angle_dev_deg: float
    mean_angle_dev_deg: float
    time_s: float
    converged: bool


# ──────────────────────────────────────────────
# Problem generators
# ──────────────────────────────────────────────

def generate_test_problems(
    n_layer_list: list[int] | None = None,
    problems_per_size: int = 5,
    seed: int = _SEED,
) -> list[Problem]:
    """
    Generate a reproducible set of test problems.

    For each layer count in *n_layer_list*, creates *problems_per_size*
    random laminates and computes their target LPs.
    """
    if n_layer_list is None:
        n_layer_list = [2, 4, 8, 12, 16, 24, 32, 48]

    rng = get_rng(seed)
    problems: list[Problem] = []

    for n in n_layer_list:
        for i in range(problems_per_size):
            lam_true = make_random_laminate(n, rng)
            lp_t = make_target_lp_from_laminate(lam_true)
            problems.append(Problem(
                lam_true=lam_true,
                lp_t=lp_t,
                label=f"N={n} #{i+1}",
            ))

    return problems


# ──────────────────────────────────────────────
# Timing wrapper
# ──────────────────────────────────────────────

def timed(fn: Callable, *args, **kwargs) -> tuple[float, any]:
    """Return (wall_time_s, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return dt, result


# ──────────────────────────────────────────────
# Result I/O
# ──────────────────────────────────────────────

def save_results(results: list[BenchmarkResult], path: str | Path) -> None:
    """Save benchmark results as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def load_results(path: str | Path) -> list[BenchmarkResult]:
    """Load benchmark results from a JSONL file."""
    path = Path(path)
    results = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            results.append(BenchmarkResult(**data))
    return results
