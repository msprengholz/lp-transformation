# LP Transformation — Auto-Research Session

## Objective

Optimise the lamination parameter (LP) back-transformation solver — recover ply angles from target LPs — making it **faster** while preserving **accuracy**.

## Metric

- **Name:** `solve_time`
- **Unit:** seconds (wall-clock)
- **Direction:** lower is better
- **Measurement:** `python -m benchmarks.benchmark_runner --solver numpy --n-layers 12 --samples 3 --n-starts 10`

## Constraint (must pass)

After optimisation, the solver must still pass:

```
pytest tests/ -x -q --tb=short -m "not slow"
```

This verifies:
1. `test_lp_functions.py` — forward LP computation is mathematically correct
2. `test_solver_numpy.py` — solver self-consistency (can recover known laminates)
3. `test_accuracy.py` — per-angle deviation < thresholds scaled by layer count

## Baseline

| Solver | N=12 mean time (3 runs) |
|--------|------------------------|
| numpy  | (to be measured)       |

## Files in scope

- `src/numpy_solver.py` — current implementation
- `src/lp_functions.py` — core LP math (do not change the forward `get_lp` signature)
- `src/numba_solver.py` — numba target (create)
- `src/slang_solver.py` — SlangPy GPU target (create)

## What has been tried

*(appended by the agent as experiments run)*

## Ideas to explore

- Numba JIT compilation of hot loops (`get_lp`, `get_loss_grad`, `ssearch`)
- Vectorisation beyond current numpy
- SlangPy compute shader for parallel batch optimisation
- Reduce iRprop iterations with adaptive convergence
- Multi-start parallelisation
