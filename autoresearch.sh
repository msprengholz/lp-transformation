#!/usr/bin/env bash
# Auto-research benchmark — runs on Google Colab GPU.
# Called by run_experiment (Linux/macOS) in the autoresearch loop.

set -euo pipefail
cd "$(dirname "$0")"

uv run --no-project colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && python benchmarks/run_benchmark.py" \
  --timeout 600
