# External Baselines

This directory stores third-party symbolic regression baselines as external repositories.

## Goal

We keep these methods isolated from the main project codebase:

- do not modify their core implementation unless strictly necessary
- adapt only through wrapper scripts, config files, and dataset conversion layers
- keep all reproduction outputs under `baselines/`

## Current repositories

- `LLM-SR`
- `In-Context-Symbolic-Regression`
- `LibraryAugmentedSymbolicRegression.jl`
- `LLM-Meta-SR` pending, source URL currently unavailable

## Integration rule

For each external method, we will:

1. inspect its required runtime and dependencies
2. identify its minimal runnable entry point
3. map the current benchmark dataset format (`x1`, `x2`, `y`) into the method's expected input
4. add wrappers in this directory only
5. keep experiment outputs under `baselines/<method>/results`
