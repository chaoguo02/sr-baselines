# PySR Wrapper

Status: implemented as project benchmark wrapper

Source mapping:

- upstream source exists at `baselines/iclrcom/PySR`
- project wrapper entrypoint is `baselines/wrappers/pysr/run.py`

Current behavior:

- reads benchmark metadata from `benchmark_datasets/benchmark_metadata.json`
- reads benchmark CSV files from `benchmark_datasets/`
- uses `x1` only for one-variable tasks and `x1,x2` for two-variable tasks
- writes normalized outputs under `baselines/results/pysr/`

Usage:

```powershell
python baselines/wrappers/pysr/run.py --function-id 3
python baselines/wrappers/pysr/run.py --all
```

Files:

- config: `baselines/wrappers/pysr/config.yml`
- summary: `baselines/results/pysr/summary.jsonl`
- per-function summary: `baselines/results/pysr/funcN/run_summary.json`
- raw equation table: `baselines/results/pysr/funcN/equations.csv`

Known environment requirement:

- the current `gc` environment does not yet provide the runtime dependency
  `juliacall`, so the wrapper is implemented and importable, but full execution
  still requires the PySR runtime stack to be installed.
