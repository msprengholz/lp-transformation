#!/bin/bash
# Correctness checks — run after every benchmark on Colab.
# Failures block the "keep" step in the autoresearch loop.

set -euo pipefail

cd "$(dirname "$0")"

python3 colab/colab_exec.py --cmd \
  "cd /content/lp-transformation && python3 -m pytest tests/test_lp_functions.py tests/test_paper_validation.py -x -q --tb=short 2>&1"
