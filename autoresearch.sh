#!/usr/bin/env bash
# Auto-research benchmark — comprehensive solution discovery test.
# Called by run_experiment in the autoresearch loop.

set -euo pipefail
cd "$(dirname "$0")"

uv run --no-project colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && git pull && pip install -q numba scipy slangpy && python benchmarks/run_comprehensive.py" \
  --timeout 600
