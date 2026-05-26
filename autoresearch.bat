@echo off
REM Run the LP solver benchmark on Google Colab GPU via uv.
REM Called by run_experiment in the autoresearch loop.

cd /d "%~dp0"
uv run --no-project colab\colab_exec.py --cmd "cd /content/lp-transformation && python benchmarks/run_benchmark.py" --timeout 600
