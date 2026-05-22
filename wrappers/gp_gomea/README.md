# GP-GOMEA Wrapper

This wrapper adapts the upstream `GP-GOMEA` codebase at
`baselines/external/GP-GOMEA` to the project's benchmark datasets and result
layout.

## Current Status

Status: mapped and wrapped

What already works:

- benchmark-facing wrapper entrypoint at `baselines/wrappers/gp_gomea/run.py`
- project config at `baselines/wrappers/gp_gomea/config.yml`
- unified result outputs under `baselines/results/gp_gomea/`

Current blocker on this machine:

- the upstream Python package depends on a compiled native module
  `gpgomea.so` or equivalent
- the current Windows `gc` environment does not yet have the required C++
  toolchain and libraries to build that module

## Expected Upstream Interface

The wrapper uses the upstream Python API:

```python
from pyGPGOMEA import GPGOMEARegressor
```

## Typical Usage

```powershell
conda run -n gc python baselines\wrappers\gp_gomea\run.py --function-id 1
```

If the native module is not available yet, the wrapper will fail with a clear
runtime message explaining what is missing.
