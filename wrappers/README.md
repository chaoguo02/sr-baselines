# Baseline Wrappers

This directory contains project-owned benchmark wrappers.

Each subdirectory should provide a stable benchmark entrypoint for one target
comparison method, while reusing upstream code from `baselines/external/`,
`baselines/iclrcom/`, or repository-local method code where appropriate.

## Planned wrapper names

- `gplearn`
- `pysr`
- `gp_gomea`
- `llm_sr`
- `icsr`
- `lasr`
- `rag_sr`
- `sr_llm`

## Wrapper policy

- Do not edit upstream third-party code unless strictly necessary.
- Prefer thin adapters that translate our benchmark datasets into the native
  format expected by each method.
- Emit a normalized summary artifact for downstream comparison.
