# LP Transformation — Auto-Research Session

## Objective

Optimise the lamination parameter (LP) back-transformation solver — recover ply angles from target LPs — making it **faster** while preserving **accuracy**.

All experiments run on **Google Colab (T4 GPU)** via the colab-cli skill.

## Metrics

| Name | Unit | Direction | Description |
|------|------|-----------|-------------|
| `solve_time` | seconds (wall) | ↓ lower is better | Mean of 3 repeats × 30 random starts, Viquerat 12-layer LP set |
| `best_rmse` | — | ↓ lower is better | Best LP RMSE (monitoring — must not degrade) |

## Constraint (must pass)

After optimisation, `autoresearch.checks.sh` must exit 0. This runs:

```
pytest tests/test_lp_functions.py tests/test_paper_validation.py -x -q --tb=short
```

This verifies:
1. Forward LP computation is mathematically correct
2. Gradient vanishes at the optimum
3. Paper test cases are reproducible (Viquerat + Sprengholz 48)
4. Quality regression gate (solver still converges)

## Baseline

| Commit | Solver | solve_time | best_rmse | Speedup |
|--------|--------|-----------|-----------|---------|
| 050e1b8 | **numba JIT** | **0.186 s** | 3.3e-8 | **69×** |
| a31d618 | numpy (opt) | 12.88 s | 3.2e-8 | 1× |

## Files in scope

- `src/numpy_solver.py` — current implementation (modify to optimise)
- `src/lp_functions.py` — core LP math (do NOT change signatures; internal changes OK)
- `src/numba_solver.py` — numba JIT target (create from numpy baseline)
- `src/slang_solver.py` — SlangPy GPU target (create)

## How experiments work

1. Agent edits code in `src/`
2. `run_experiment` → executes `autoresearch.sh` which:
   - Creates/reuses a persistent Colab GPU session (`lp-autoresearch`)
   - Pulls latest code (git clone/pull)
   - Runs the Viquerat benchmark (30 starts, 3 repeats)
   - Outputs `METRIC solve_time=...` lines
3. `log_experiment` records result, runs checks, keeps or discards

## Guardrails

- **Never** change the tests or test thresholds
- **Never** hardcode benchmark answers
- Keep `get_lp()` mathematically exact — optimise the solver, not the forward function
- Document any accuracy tradeoffs in `asi.description`

## What has been tried

### Run 1-2: Optimized numpy baseline
- Cached Z2/Z3 arrays (lru_cache)
- Combined LP+gradient trig pass
- Gradient-norm early stopping in iRprop
- `solve_time = 12.88s` on Colab T4

### Run 3: Numba JIT solver  ✅  **69× speedup**
- JIT-compiled `get_lp`, `get_lp_and_grad`, `ssearch`, `iRpropm`
- All loop-heavy functions rewritten with explicit loops for numba compat
- Falls back to numpy when numba unavailable
- `solve_time = 0.186s` (69× faster), accuracy preserved (best_rmse 3.3e-8)
- Confidence: 106.8× noise floor

## Ideas to explore

- [done] Cached Z2/Z3 arrays (lru_cache)
- [done] Combined LP+gradient trig pass
- [done] Gradient-norm early stopping in iRprop
- [done] Numba JIT: @jit(nopython=True) on get_lp, get_loss_grad, ssearch, iRpropm
- SlangPy compute shader for batch iRprop across many starts
- Multi-start vectorisation: run all random starts in one batch (prange)
- Reduce coarse-to-fine search rounds (currently 3)
- Adaptive delta in ssearch: start coarse, refine adaptively
- Precompute trig for all possible angles in ssearch grid (numba)
