#!/usr/bin/env bash
# Correctness checks — run on Google Colab after every benchmark.
# Failures block the "keep" step in the autoresearch loop.

set -euo pipefail

cd "$(dirname "$0")"

python colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && pip install -q numba >/dev/null 2>&1 && python benchmarks/run_checks.py 2>&1" \
  --timeout 600
