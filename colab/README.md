# Running on Google Colab

This repo is designed to be cloned into a Google Colab instance for GPU-accelerated experiments.

## Option 1: Interactive notebook

Open [`setup_colab.ipynb`](setup_colab.ipynb) in Colab and follow the cells.

## Option 2: Headless Colab (via colab-cli skill)

If you have the pi [colab-cli skill](https://github.com/…) installed:

```
colab create --name lp-experiment --gpu --file colab/colab_runner.sh
```

## What Colab provides

| Backend | Library | Status |
|---------|---------|--------|
| CPU     | numpy   | Baseline |
| CPU JIT | numba   | Implement in `src/numba_solver.py` |
| GPU     | slangpy (CUDA) | Implement in `src/slang_solver.py` |
| GPU     | CuPy    | Alternative GPU path |

## Workflow

1. Clone: `git clone https://github.com/msprengholz/lp-transformation.git`
2. Install deps: `pip install -r requirements.txt`
3. Run tests: `pytest tests/ -v`
4. Run benchmarks: `python -m benchmarks.benchmark_runner --solver all`
5. Implement improvements, repeat.
