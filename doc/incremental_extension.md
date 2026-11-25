# Incremental extension of datasets and models

`Dataset.fit_partial(..., model=model)` grows the data mappings and the
LightFM model's parameter tables in **one atomic change**, so a catalogue that
keeps gaining users, items and side features can be learned from continuously
without discarding what the model already knows.

## What changes in a single operation

Each feature in LightFM owns one row in twelve arrays: an embedding, a bias
and, for each of them, the two optimiser state arrays (AdaGrad accumulator and
momentum slot; AdaDelta uses both slots) for both the user and the item side.
Appending `k` features means appending `k` rows to all of them.

The implementation is deliberately split into two phases.

1. **Stage (no mutation).** All four input iterables are consumed and every
   key that is genuinely new is recorded in temporary lists. Nothing in the
   live mappings or the model changes yet, so an iterator that raises halfway
   through leaves no trace. When every key is already known the call returns
   immediately: no mapping change, no allocation, no random draw.
2. **Commit.** The new keys are appended to the four mappings (old indices are
   never reassigned), then `LightFM.extend_dimensions` builds every
   replacement array *before* touching a single model attribute. Only once all
   allocations and random initialisations have succeeded are the attributes
   swapped in; if anything raises, the mapping insertions are rolled back key
   by key and the model's random state is rewound.

Because `FastLightFM`, the Cython object used by training and prediction, is
reconstructed from the model's arrays on every call, replacing the arrays
cannot strand a stale reference inside the native code. `predict` and the
`get_*_representations` helpers therefore always read the same parameters that
training updates.

## Parameter and optimiser state of new rows

New rows reproduce the state a freshly fitted row would have:

| Array                | AdaGrad              | AdaDelta             |
|----------------------|----------------------|----------------------|
| embedding            | random, `(U-0.5)/d`  | random, `(U-0.5)/d`  |
| bias                 | zero                 | zero                 |
| gradient accumulator | one                  | zero                 |
| momentum slot        | zero (unused)        | zero                 |

The identity-feature switches on `Dataset` keep working: with identity
features on, every new user/item id also registers its unique identity
feature; with them off, only explicit side features extend the space.

Old rows are copied element by element and never reinitialised. Recomputing a
score from the pre-extension feature rows yields a bitwise identical result,
and no training happens during extension.

## Failure recovery

Three failure classes are handled without leaving a half-applied extension:

- **Iterator errors** occur in phase 1, before any mutation.
- **Validation errors** (unfitted model, feature space smaller than the
  dataset's, shrinking via `extend_dimensions`, wrong argument type) are raised
  before the commit phase.
- **Allocation errors** inside `extend_dimensions` happen while the new arrays
  are being built; no model attribute has been reassigned, the random generator
  is restored to its pre-call state, and the mapping insertions are undone.

The tests force each of these paths (a generator that raises, a monkeypatched
`np.concatenate`, a mismatched model) and assert byte-identical mappings,
identical array references and an unchanged random state afterwards.

## Memory and copy tradeoffs (measured)

The extension only ever touches arrays sized `O(number of features *
no_components)`. No `n_users x n_items` table is allocated at any point — the
interaction and feature matrices remain COO/CSR throughout. Measured on this
aarch64 Linux / Python 3.11 container (`no_components=32`, float32),
`benchmarks/scaling_experiment.py` reports:

| features per side | parameter MB | dense U×I would be MB | one 100-feature batch | 100 × 2-feature batches |
|-------------------|--------------|-----------------------|-----------------------|-------------------------|
| 1,000             | 0.78         | 3.8                   | 0.54 ms               | 12.7 ms                 |
| 10,000            | 7.78         | 381.5                 | 0.81 ms               | 78.9 ms                 |
| 25,000            | 19.45        | 2,384.2               | 1.60 ms               | 157.8 ms                |
| 50,000            | 38.91        | 9,536.7               | 4.07 ms               | 176.1 ms                |

Three costs follow from the append-and-copy design, and the numbers show how
to manage them.

- **Copy cost is linear in the *current* size, not in the increment.** Every
  batch rebuilds each full array, so a trickle of two-feature batches pays the
  full-table copy over and over (the 100-batch column versus the single
  100-feature batch). Batching arrivals before calling `fit_partial`
  amortises this; at 50k features a single batch still only costs ~4 ms.
- **Resident memory stays linear.** At 50k features the model occupies ~39 MB
  where a dense table would need ~9.5 GB; the resident set grows smoothly with
  feature count (see `benchmark_results.txt`).
- **Failure recovery is cheap.** A failed iterator call costs only the staging
  work (~0.006 ms measured); a failed allocation costs one aborted build
  (~0.13 ms) plus deterministic key-level rollback. There is no snapshot to
  restore because the old arrays are left in place until the commit point and
  are freed by normal reference counting afterwards.

If workloads later require sub-linear per-append copies, the seam to change is
isolated in `LightFM.extend_dimensions`: swapping the contiguous `np.ndarray`
tables for a block-list (or `np.memmap`-backed) representation would remove
the re-copy while keeping the same two-phase commit, at the price of indirect
indexing in the hot training loop. The current design deliberately keeps the
hot loop on flat contiguous arrays and pays copying only on the much rarer
schema-change path.

## Usage

```python
from lightfm import LightFM
from lightfm.data import Dataset

dataset = Dataset()
dataset.fit(users=initial_users, items=initial_items)
interactions, weights = dataset.build_interactions(initial_pairs)

model = LightFM(loss="warp", random_state=42)
model.fit_partial(interactions, sample_weight=weights)

# New catalogue arrivals: mappings and model grow together.
dataset.fit_partial(
    users=new_users,
    items=new_items,
    user_features=new_user_features,
    item_features=new_item_features,
    model=model,
)

# Rebuild feature matrices and keep training; old predictions are unchanged
# until you actually train again.
model.fit_partial(
    *dataset.build_interactions(more_pairs),
    user_features=dataset.build_user_features([]),
    item_features=dataset.build_item_features([]),
)
```

A runnable version lives in `examples/incremental_training.py`, and the
contract is covered by `tests/test_incremental.py`.
