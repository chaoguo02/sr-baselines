# LLM-SR Wrapper

This wrapper adapts the upstream `LLM-SR` codebase at
`baselines/external/LLM-SR` to the project's benchmark datasets and result
layout.

## Current Status

Status: mapped and wrapped

What already works:

- benchmark-facing wrapper entrypoint at `baselines/wrappers/llm_sr/run.py`
- project config at `baselines/wrappers/llm_sr/config.yml`
- dynamic generation of benchmark-specific `numpy+BFGS` specification files
- unified result outputs under `baselines/results/llm_sr/`

Current runtime requirements:

- the active environment needs the Python dependencies required by the upstream
  `LLM-SR` project
- API mode needs the `API_KEY` environment variable
- local-model mode needs the upstream local LLM server to be running

## Typical Usage

```powershell
conda run -n gc python baselines\wrappers\llm_sr\run.py --function-id 1
```

The wrapper currently targets the project's 1D and 2D benchmark tasks.
