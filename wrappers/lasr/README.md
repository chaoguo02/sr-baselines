# LaSR Wrapper

This wrapper adapts the upstream `LibraryAugmentedSymbolicRegression.jl`
codebase at `baselines/external/LibraryAugmentedSymbolicRegression.jl` to the
project's benchmark datasets and result layout.

## Current Status

Status: mapped and wrapped

What already works:

- benchmark-facing Python entrypoint at `baselines/wrappers/lasr/run.py`
- Julia worker script at `baselines/wrappers/lasr/run_lasr.jl`
- project config at `baselines/wrappers/lasr/config.yml`
- automatic conversion of project CSV datasets into the matrix format consumed
  by the Julia runner
- reuse of the project-local Julia runtime installed for `PySR`
- unified result outputs under `baselines/results/lasr/`

Current runtime requirements:

- the Julia runtime must already exist under `.pysr_runtime/`
- the active environment needs an API key in the configured environment
  variable when `use_llm` is enabled
- the upstream Julia project may still need package instantiation before the
  first real run

## Typical Usage

```powershell
conda run -n gc python baselines\wrappers\lasr\run.py --function-id 1
```

The wrapper currently targets the project's 1D and 2D benchmark tasks.
