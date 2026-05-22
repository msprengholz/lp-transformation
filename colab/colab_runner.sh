#!/bin/bash
# Headless Colab runner — runs the full test + benchmark suite.
# Invoked by the colab-cli skill.

set -euo pipefail

REPO_DIR="${1:-lp-transformation}"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repo..."
    git clone https://github.com/msprengholz/lp-transformation.git "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt
pip install -q numba 2>/dev/null || echo "numba not available"

echo "=== GPU info ==="
nvidia-smi 2>/dev/null || echo "No GPU"

echo "=== Running tests ==="
python -m pytest tests/ -v --tb=short 2>&1 | tail -20

echo "=== Running benchmarks ==="
python -m benchmarks.benchmark_runner --solver all --samples 3

echo "=== Done ==="
