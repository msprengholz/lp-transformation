@echo off
REM Windows batch wrapper for autoresearch.
REM Calls the Colab benchmark directly via Python, avoiding WSL bash issues.

cd /d "%~dp0"
python colab\colab_exec.py --cmd "cd /content/lp-transformation && python benchmarks/run_benchmark.py" --timeout 600
