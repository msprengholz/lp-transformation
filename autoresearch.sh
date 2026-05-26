#!/usr/bin/env bash
# Auto-research benchmark — runs on Google Colab GPU.
# Called by run_experiment in the autoresearch loop.
# Delegates to benchmarks/run_benchmark.py on the Colab VM.

set -euo pipefail

cd "$(dirname "$0")"

python colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && python3 benchmarks/run_benchmark.py" \
  --timeout 600
