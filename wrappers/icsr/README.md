# ICSR Wrapper

This wrapper adapts the upstream `In-Context-Symbolic-Regression` codebase at
`baselines/external/In-Context-Symbolic-Regression` to the project's benchmark
datasets and result layout.

## Current Status

Status: mapped and wrapped

What already works:

- benchmark-facing wrapper entrypoint at `baselines/wrappers/icsr/run.py`
- project config at `baselines/wrappers/icsr/config.yml`
- automatic conversion of project CSV datasets into the upstream
  `train_points.npy` and `test_points.npy` format
- direct `OmegaConf` construction in the wrapper, without modifying the
  upstream Hydra config tree
- unified result outputs under `baselines/results/icsr/`

Current runtime requirements:

- the active environment needs the Python dependencies required by the upstream
  `In-Context-Symbolic-Regression` project
- API mode needs the `OPENAI_API_KEY` environment variable

## Typical Usage

```powershell
conda run -n gc python baselines\wrappers\icsr\run.py --function-id 1
```

The wrapper currently targets the project's 1D and 2D benchmark tasks.
