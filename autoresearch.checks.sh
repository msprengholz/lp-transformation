#!/bin/bash
# Correctness checks — run after every benchmark.
# Failures block the "keep" step in the autoresearch loop.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Running correctness checks ==="

# Fast accuracy tests (skip the full self-consistency test suite)
python -m pytest tests/test_lp_functions.py -x -q --tb=short 2>&1

# Self-consistency for key layer counts
python -m pytest tests/test_solver_numpy.py \
    -x -q --tb=short \
    -k "test_self_consistency_N12 or test_self_consistency_N24" 2>&1

# Accuracy baseline
python -m pytest tests/test_accuracy.py -x -q --tb=short 2>&1

echo "=== All checks passed ==="
