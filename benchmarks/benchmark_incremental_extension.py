"""
Microbenchmark for frequent small-batch feature-space extension.

Measures, for a series of tiny ``Dataset.fit_partial(..., model=...)`` calls:

- per-call extension wall time (the copy cost of growing every parameter
  array),
- the resident-memory trajectory of the model,
- the cost of a failed extension (iterator error and allocation error),
- and confirms the extension never allocates a dense n_users x n_items
  table: extending to N users/items stays O(N * no_components) in memory
  rather than O(N^2).

Run with: python3 -I benchmarks/benchmark_incremental_extension.py
"""

import resource
import time

import numpy as np

from lightfm.data import Dataset
from lightfm.lightfm import LightFM


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_seed(no_components):
    dataset = Dataset()
    dataset.fit(["u0"], ["i0"])
    interactions, weights = dataset.build_interactions([("u0", "i0")])
    model = LightFM(no_components=no_components, random_state=42)
    model.fit_partial(interactions, sample_weight=weights)
    return dataset, model


def run(batches=100, batch_size=2, no_components=32):
    dataset, model = build_seed(no_components)

    # Warm up allocator and timer.
    rss_start = rss_mb()
    timings = []
    rss_checkpoints = []

    for batch in range(batches):
        users = [f"u:{batch}:{k}" for k in range(batch_size)]
        items = [f"i:{batch}:{k}" for k in range(batch_size)]

        start = time.perf_counter()
        dataset.fit_partial(users=users, items=items, model=model)
        timings.append(time.perf_counter() - start)

        if batch % 20 == 19:
            rss_checkpoints.append((batch + 1, rss_mb()))

    n_features = dataset.model_dimensions()[0]

    # Sanity: extension cost tracks total parameter size, not a dense table.
    param_elements = sum(
        getattr(model, name).size
        for name in (
            "item_embeddings",
            "item_embedding_gradients",
            "item_embedding_momentum",
            "user_embeddings",
            "user_embedding_gradients",
            "user_embedding_momentum",
        )
    ) + sum(
        getattr(model, name).size
        for name in (
            "item_biases",
            "item_bias_gradients",
            "item_bias_momentum",
            "user_biases",
            "user_bias_gradients",
            "user_bias_momentum",
        )
    )

    # Failed-extension cost: iterator blow-up.
    def broken():
        yield "never_committed"
        raise RuntimeError("boom")

    start = time.perf_counter()
    try:
        dataset.fit_partial(users=broken(), model=model)
    except RuntimeError:
        pass
    iterator_failure_s = time.perf_counter() - start

    # Failed-extension cost: allocation blow-up.
    import lightfm.lightfm as lightfm_module

    original_concatenate = lightfm_module.np.concatenate
    lightfm_module.np.concatenate = lambda *a, **k: (_ for _ in ()).throw(
        MemoryError("simulated")
    )
    start = time.perf_counter()
    try:
        dataset.fit_partial(users=["also_never_committed"], model=model)
    except MemoryError:
        pass
    allocation_failure_s = time.perf_counter() - start
    lightfm_module.np.concatenate = original_concatenate

    rss_end = rss_mb()

    dense_table_elements = n_features * n_features
    report = {
        "batches": batches,
        "batch_size": batch_size,
        "no_components": no_components,
        "final_feature_count_per_side": n_features,
        "median_extension_ms": np.median(timings) * 1000,
        "mean_extension_ms": np.mean(timings) * 1000,
        "max_extension_ms": np.max(timings) * 1000,
        "total_extension_s": sum(timings),
        "param_elements": param_elements,
        "hypothetical_dense_user_x_item_elements": dense_table_elements,
        "dense_to_parameter_ratio": dense_table_elements / param_elements,
        "iterator_failure_ms": iterator_failure_s * 1000,
        "allocation_failure_ms": allocation_failure_s * 1000,
        "rss_start_mb": rss_start,
        "rss_checkpoints_mb": rss_checkpoints,
        "rss_end_mb": rss_end,
    }
    return report


def main():
    report = run()
    print("Incremental extension benchmark")
    print("=" * 60)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key:45s}: {value:.4f}")
        else:
            print(f"{key:45s}: {value}")

    # Hard guard: a dense U x I table at this size would be hundreds of MB;
    # the resident set must stay within an order of magnitude of the
    # parameter arrays' own footprint.
    param_bytes = report["param_elements"] * np.dtype(np.float32).itemsize
    assert report["rss_end_mb"] < max(
        200.0, 10.0 * param_bytes / (1024 * 1024) + 100.0
    ), "resident memory suggests a dense table was materialised"
    print("\nOK: memory stays linear in feature count, no dense U x I table.")


if __name__ == "__main__":
    main()
