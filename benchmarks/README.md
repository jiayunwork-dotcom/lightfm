# Incremental-extension benchmarks

These scripts measure the cost of growing a fitted LightFM model together
with its `Dataset` via `Dataset.fit_partial(..., model=model)`.

Run them against the local source tree (no network needed once dependencies
are installed):

```
python -I benchmarks/benchmark_incremental_extension.py
python -I benchmarks/scaling_experiment.py
```

- `benchmark_incremental_extension.py` — per-call extension latency, the
  resident-memory trajectory over many tiny batches, and the cost of failed
  (iterator / allocation) extensions. Captured output:
  `benchmark_results.txt`.
- `scaling_experiment.py` — linear model footprint versus the quadratic
  dense user × item table that is deliberately never allocated, plus the
  copy-cost comparison between one large batch and many two-feature batches.
  Captured output: `scaling_results.txt`.

The design rationale and tradeoff discussion live in
`doc/incremental_extension.md`.
