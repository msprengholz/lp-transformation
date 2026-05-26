# LP Transformation — Auto-Research Session

## Objective

Optimise the lamination parameter (LP) back-transformation solver. The solver must find **all known ply-angle solutions** for the Viquerat LP set, and maximise **unique solutions found per minute** for larger problems.

## Metrics

### Primary: `viquerat_discovery_time` (seconds, ↓ lower is better)
Time to discover all 10 known unique solutions for the Viquerat 12-layer LP set (angles rounded to 0.1°). The paper knows these solutions; the solver must reproduce them.

### Secondary: `sprengholz_solutions_per_min` (count, ↑ higher is better)
Number of unique solutions (RMSE < 1e-3) found for the 48-layer Sprengholz LP set in 60 seconds.

## Constraint (must pass)

After optimisation, `autoresearch.checks.sh` must exit 0. This runs:

```
python benchmarks/run_checks.py
```

Which verifies:
1. Forward LP computation is mathematically correct
2. Gradient vanishes at the optimum
3. Numpy + Numba solvers converge to best_rmse < 1e-3, median < 0.15 on Viquerat

## Baseline

| Commit | Solver | viquerat_discovery_time | sprengholz_solutions/min |
|--------|--------|------------------------|-------------------------|
| (current) | numba JIT | (to be measured) | (to be measured) |

## Files in scope

- `src/numba_solver.py` — numba JIT solver
- `src/numpy_solver.py` — numpy fallback solver
- `src/lp_functions.py` — core LP math (do NOT change signatures)

## How experiments work

1. Agent edits code in `src/`
2. `run_experiment` → executes `autoresearch.sh` which:
   - Creates/reuses a persistent Colab GPU session
   - Runs `benchmarks/run_comprehensive.py` on Colab
   - Outputs METRIC lines
3. `log_experiment` records result, runs checks, keeps or discards

## Guardrails

- **Never** change the tests or test thresholds
- **Never** hardcode benchmark answers
- Keep `get_lp()` mathematically exact — optimise the solver, not the forward function

## Progress summary

### What worked (CPU optimisation, 14 experiments)

| Approach | Gain | Notes |
|----------|------|-------|
| **Numba JIT** on get_lp, ssearch, iRprop | **69×** | Biggest single gain. All hot-path functions JIT-compiled to native. |
| **Skip fine ssearch** (5° grid) | **1.8×** | iRprop refines from 10° coarse grid; fine grid redundant. |
| **Relax grad_tol** (1e-6 → 1e-3) | **3.3×** | iRprop converges to ssearch basin early; tight tolerance wasted. |
| **Two-stage ssearch** (11 evals/layer vs 18) | **1.07×** | Same effective resolution with 39% fewer evaluations. |
| **Incremental trig** in ssearch | **1.05×** | Precompute trig once, update in-place per candidate. |
| **Total CPU speedup** | **740×** (12.9s → 0.017s) | — |

### What didn't work

| Approach | Result | Reason |
|----------|--------|--------|
| ThreadPoolExecutor multi-start | Quality degradation. | Race conditions in JIT calls? |
| `parallel=True` + `prange` | 20× slowdown. | Object-mode fallback, compilation overhead. |
| JIT-compiled multi-start loop | 0% gain. | Python loop overhead already negligible. |
| 15° ssearch grid | Quality gate failed. | Too coarse for reliable convergence. |
| Fixed-step gradient descent | 5× slower, 1000× worse RMSE. | Adaptive steps (Rprop) essential. |

### CPU is saturated
At 740× vs numpy (0.17 ms per solver call), further CPU gains are marginal. Comprehensive benchmark needed to verify real-world solution discovery capability.

## Ideas to explore

- [done] Cached Z2/Z3 arrays
- [done] Combined LP+gradient trig pass
- [done] Gradient-norm early stopping
- [done] Numba JIT
- [done] Skip fine ssearch
- [done] Two-stage ssearch
- [done] Incremental trig
- [done] grad_tol tuning
- SlangPy GPU compute shader
- Systematic enumeration instead of random starts for Viquerat
- Solution-space exploration strategies
