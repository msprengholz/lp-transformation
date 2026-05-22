#!/bin/bash
# Auto-research benchmark script
# Runs the solver benchmark and outputs the metric.

set -euo pipefail

cd "$(dirname "$0")"

# Pre-checks: make sure we can import the module
python -c "from src.lp_functions import get_lp; print('import OK')" > /dev/null 2>&1 || {
    echo "FAIL: cannot import lp_functions" >&2
    exit 1
}

# Run benchmark (wall-clock, 3 problems, 10 starts each, N=12 only for speed)
START=$(date +%s.%N)
python -m benchmarks.benchmark_runner \
    --solver numpy \
    --n-layers 12 \
    --samples 3 \
    --n-starts 10 \
    --output /dev/null 2>&1
END=$(date +%s.%N)

ELAPSED=$(echo "$END - $START" | bc)

# Output metric line for pi-autoresearch
echo "METRIC solve_time=$ELAPSED"
