# gplearn Baseline

This directory stores the `gplearn` reproduction code for benchmark comparisons.

## What it does

- Reuses the current project benchmark datasets under `benchmark_datasets/`
- Uses the same train/test split format as the main project (`x1`, `x2`, `y`)
- Reports train MSE, test MSE, discovered expression, and runtime
- Supports running one benchmark function or all benchmark functions in batch

## Install

```powershell
pip install gplearn
```

## Run

Run one dataset:

```powershell
python baselines/gplearn/run_gplearn_benchmark.py --function-id 3
```

Run all datasets:

```powershell
python baselines/gplearn/run_gplearn_benchmark.py --all
```

Use a custom config:

```powershell
python baselines/gplearn/run_gplearn_benchmark.py --config baselines/gplearn/config_gplearn.yml --all
```

## Outputs

Results are written to:

- `baselines/gplearn/results/summary.jsonl`
- `baselines/gplearn/results/func{N}/run_summary.json`
- `baselines/gplearn/results/func{N}/generation_history.jsonl`
