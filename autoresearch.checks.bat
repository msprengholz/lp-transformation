@echo off
REM Windows batch wrapper for correctness checks on Colab.

cd /d "%~dp0"
python colab\colab_exec.py --cmd "cd /content/lp-transformation && python -m pytest tests/test_lp_functions.py tests/test_paper_validation.py -x -q --tb=short" --timeout 600
