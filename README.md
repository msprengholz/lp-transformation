# LP Transformation

> **Lamination parameter back-transformation** — recover ply angles from target lamination parameters.

This repository provides a structured framework for optimising the inverse problem: given 12 target lamination parameters (A, B, D matrices), find the stacking sequence of ply angles that produces them.

## Problem

For a laminate with N plies at angles θᵢ, the forward function computes 12 lamination parameters:

```
get_lp(θ) → ξ = [ξ₁ᴬ, ξ₂ᴬ, ξ₃ᴬ, ξ₄ᴬ, ξ₁ᴮ, ξ₂ᴮ, ξ₃ᴮ, ξ₄ᴮ, ξ₁ᴰ, ξ₂ᴰ, ξ₃ᴰ, ξ₄ᴰ]
```

The **back-transformation** is the inverse: given target ξ*, find θ such that `get_lp(θ) ≈ ξ*`.

## Solver algorithm

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | Sequential search (10°) | Coarse global exploration |
| 2 | Sequential search (5°) | Finer grid refinement |
| 3 | iRprop- (3000 iter) | Local gradient-based optimisation |

## Structure

```
├── src/
│   ├── lp_functions.py      # Core math: get_lp, get_loss_grad, compute_lp_rmse
│   ├── numpy_solver.py      # Numpy-based solver (baseline)
│   ├── numba_solver.py      # [TBD] Numba-accelerated solver
│   └── slang_solver.py      # [TBD] SlangPy GPU solver
├── tests/
│   ├── test_lp_functions.py  # Forward computation correctness
│   ├── test_solver_numpy.py  # Self-consistency (can we recover known laminates?)
│   └── test_accuracy.py      # Angle deviation thresholds
├── benchmarks/
│   └── benchmark_runner.py   # Unified timing harness
├── colab/
│   ├── setup_colab.ipynb     # Interactive Colab notebook
│   └── colab_runner.sh       # Headless Colab runner
├── autoresearch.md           # pi-autoresearch session document
├── autoresearch.sh           # Benchmark command for autoresearch loop
└── autoresearch.checks.sh    # Correctness checks
```

## Getting started

```bash
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run benchmarks
python -m benchmarks.benchmark_runner --solver all --samples 5

# Run a single problem
python -c "
from src.lp_functions import get_lp, make_random_laminate
from src.numpy_solver import optimize_laminate
import numpy as np

lp_t = np.array([0.2, -0.05, -0.15, -0.1, 0.4, 0.2, 0.4, 0.25, 0.2, 0.2, -0.05, -0.1], dtype=np.float32)
rand_lams = np.random.uniform(-np.pi/2, np.pi/2, (50, 12)).astype(np.float32)
opt_lams, losses = optimize_laminate(rand_lams, lp_t)
print('Best loss:', losses.min())
"
```

## Roadmap

- [x] Numpy baseline with test suite
- [ ] Numba JIT acceleration (`src/numba_solver.py`)
- [ ] SlangPy GPU implementation (`src/slang_solver.py`)
- [ ] CuPy alternative GPU path
- [ ] Large-layer scaling (N=64, 96, 128)

## Metrics

| Backend | Target | Status |
|---------|--------|--------|
| numpy | Baseline accuracy & speed | ✅ |
| numba | 2-5× speedup | ⏳ |
| slangpy (CUDA) | 10-100× speedup on GPU | ⏳ |

Correctness threshold: angle deviation < 0.1° per layer (scaled by √N).

## pi-autoresearch integration

This repo supports [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) for automated optimisation loops:

```bash
# Start an autoresearch session
/skill:autoresearch-create

# The agent will read autoresearch.md and begin optimising
```

## License

MIT
