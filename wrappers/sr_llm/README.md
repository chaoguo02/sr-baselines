# SR-LLM Wrapper

Status: implemented as project benchmark wrapper

Source mapping:

- upstream source exists at `baselines/external/SR-LLM`
- project wrapper entrypoint is `baselines/wrappers/sr_llm/run.py`
- this wrapper targets the upstream `general_symbolic_regression(...)` interface

Current behavior:

- reads benchmark metadata from `benchmark_datasets/benchmark_metadata.json`
- reads benchmark CSV files from `benchmark_datasets/`
- converts benchmark tasks into the general SR-LLM input interface
- supports both `use_rag: true` and `use_rag: false`
- writes normalized outputs under `baselines/results/sr_llm/`

Usage:

```powershell
python baselines/wrappers/sr_llm/run.py --function-id 3
python baselines/wrappers/sr_llm/run.py --all
python baselines/wrappers/sr_llm/run.py --config baselines/wrappers/sr_llm/config_smoke.yml --function-id 1
```

Files:

- config: `baselines/wrappers/sr_llm/config.yml`
- smoke config: `baselines/wrappers/sr_llm/config_smoke.yml`
- summary: `baselines/results/sr_llm/summary.jsonl`
- per-function summary: `baselines/results/sr_llm/funcN/run_summary.json`

Known environment requirements:

- the active environment still needs the SR-LLM runtime dependencies from
  `baselines/external/SR-LLM/environment.yml`, plus `torch` and `torchvision`
- when `use_rag: true`, `OPENAI_API_KEY` is expected from the repository `.env`
  or the process environment
- the wrapper does not use any upstream hard-coded API key defaults
