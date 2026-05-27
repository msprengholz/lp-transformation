
## Newton vs iRprop Conclusions (experiment #30)

- **iRprop**: Best RMSE 1.7e-2, median 9.3e-2 over 50 random starts (with ssearch)
- **Newton (diagonal H)**: Best RMSE 1.5e-1, median 3.1e-1 — 8x worse than iRprop
- **Newton (full H)**: Best RMSE 1.4e-1, median 3.0e-1 — 8x worse than iRprop

Newton converges faster per iteration but to MUCH WORSE local minima. The reason is clear:
Newton jumps directly to the bottom of whatever basin it starts in. Without the ssearch exploration
phase, it gets trapped in poor basins. iRprop's advantage comes from ssearch, not from the
gradient descent algorithm.

**Key insight**: The gradient information IS useful, but only for guiding ssearch and for
rapid convergence WITHIN a good basin. The best approach is:
1. **ssearch**: Explore coarse angles to find promising basins
2. **Newton or iRprop**: Converge to the bottom of each basin

The current solver already does this (ssearch + iRprop). Newton could replace iRprop for
faster per-basin convergence, but ssearch is essential.

For GPU, the best approach is:
- Batch ssearch on GPU (find top-K starting basins)
- Batch iRprop on GPU (converge all K in parallel)
- OR: Batch Newton on GPU (converge all K in even fewer iterations)

The GPU approach can process thousands of starts simultaneously, making the per-start cost
irrelevant. The key metric is throughput (starts/sec), not per-start convergence rate.
