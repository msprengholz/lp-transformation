#!/usr/bin/env bash
# Auto-research benchmark — comprehensive solution discovery test.
# Called by run_experiment in the autoresearch loop.

set -euo pipefail
cd "$(dirname "$0")"

uv run --no-project colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && pip install -q numba >/dev/null 2>&1 && python benchmarks/run_comprehensive.py" \
  --timeout 600
