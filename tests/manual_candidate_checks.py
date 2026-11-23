# coding=utf-8
"""
Manual resource / correctness checks for candidate-restricted LightFM.

This script is *not* part of the normal pytest run.  It is intended to be
re-run by hand on Linux / Python 3.11 to check:

1. A large catalogue with a tiny candidate set is handled without ever
   building a dense user x item matrix (memory) and without scoring the whole
   catalogue to rank one user (time).
2. The training code and the metric denominator agree: if training narrows
   the universe, precision's denominator must narrow too.  A deliberately
   broken variant (metric denominator left at ``k``) is shown to be caught.
3. Negative sampling is uniform over the eligible non-positive items, which
   is what the WARP rank estimate assumes.
4. Candidate rows that are almost entirely positive still terminate.

Run with::

    python3 -I tests/manual_candidate_checks.py

(``-I`` keeps the interpreter from importing unrelated modules from the
current directory; the repository itself is added to ``sys.path`` below.)
"""

import gc
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightfm import LightFM  # noqa: E402
from lightfm import evaluation  # noqa: E402
from lightfm._candidates import (  # noqa: E402
    eligible_counts,
    negative_candidates,
    normalize_candidate_items,
    positive_mask,
    to_index_csr,
)


def _max_rss_mb():
    # ru_maxrss is in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _dense_matrix_mb(n_users, n_items, itemsize=4):
    return n_users * n_items * itemsize / (1024.0 ** 2)


def _choose(rng, n_items, size):
    """Draw unique item ids without numpy retaining a 200k-element permutation.

    ``rng.choice(n_items, size, replace=False)`` keeps a view of the full
    permutation as the result's base under some NumPy versions; storing many
    such results would inflate the process RSS with data unrelated to LightFM.
    """

    chosen = set()
    while len(chosen) < size:
        chosen.update(rng.randint(0, n_items, size * 2).tolist())
    return np.array(sorted(chosen)[:size], dtype=np.int64)


def check_large_catalogue_small_candidates():
    print("\n[1] large catalogue, small candidate set")
    n_users, n_items, k = 2000, 200000, 8

    rng = np.random.RandomState(0)
    train_cols = np.concatenate(
        [_choose(rng, n_items, 2) for _ in range(n_users)]
    )
    rows = np.repeat(np.arange(n_users), 2)
    cols = train_cols
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )

    # Candidate rows: k random items each, plus the training positives.
    # Build in COO form to keep construction itself cheap.
    cand_rows = []
    cand_cols = []
    for u in range(n_users):
        chosen = set(_choose(rng, n_items, k).tolist())
        chosen |= set(interactions[u].indices.tolist())
        cand_rows.extend([u] * len(chosen))
        cand_cols.extend(chosen)
    candidates = sp.csr_matrix(
        (
            np.ones(len(cand_rows), dtype=bool),
            (np.array(cand_rows), np.array(cand_cols)),
        ),
        shape=(n_users, n_items),
    )

    dense_mb = _dense_matrix_mb(n_users, n_items)
    print("    dense user x item float32 would be %.1f MB" % dense_mb)
    print("    candidate matrix has %d stored entries" % candidates.nnz)
    assert candidates.nnz < n_users * (k + 2) + 10

    gc.collect()
    baseline = _max_rss_mb()

    model = LightFM(loss="warp", no_components=10, random_state=1, max_sampled=4)
    start = time.time()
    model.fit(interactions, candidate_items=candidates, epochs=2)
    fit_seconds = time.time() - start

    # Build a clean test: one eligible, non-train item per user.
    test_lil = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        forbidden = set(interactions[u].indices.tolist())
        eligible = [int(c) for c in candidates[u].indices if c not in forbidden]
        if eligible:
            test_lil[u, eligible[-1]] = 1.0
    test = test_lil.tocsr()

    start = time.time()
    ranks = model.predict_rank(
        test, train_interactions=interactions, candidate_items=candidates
    )
    rank_seconds = time.time() - start

    gc.collect()
    peak = _max_rss_mb()

    print("    fit time: %.2fs, rank time: %.2fs" % (fit_seconds, rank_seconds))
    print("    peak RSS: %.1f MB (baseline %.1f MB)" % (peak, baseline))
    print("    ranks nnz: %d" % ranks.nnz)

    # A dense feature/score matrix at this size would already be >= dense_mb.
    assert peak - baseline < dense_mb, (
        "memory grew by %.1f MB, close to the %.1f MB dense matrix"
        % (peak - baseline, dense_mb)
    )

    start = time.time()
    precision = evaluation.precision_at_k(
        model,
        test,
        train_interactions=interactions,
        k=5,
        candidate_items=candidates,
    )
    metric_seconds = time.time() - start
    print("    precision_at_k time: %.2fs, mean %.4f" % (metric_seconds, precision.mean()))

    # Denominator must be min(k, effective candidate count), never k when the
    # effective count is smaller.
    effective = eligible_counts(candidates, interactions)
    expected_denominator = np.minimum(5, effective)
    # Recompute the numerator directly from the ranks.
    hits = np.zeros(n_users)
    for u in range(test.shape[0]):
        row = test[u].indices
        for item in row:
            hits[u] += ranks[u, item] < 5
    expected = np.divide(
        hits,
        expected_denominator,
        out=np.zeros(n_users),
        where=expected_denominator > 0,
    )
    assert np.allclose(precision, expected), "precision denominator mismatch"

    return {
        "fit_seconds": fit_seconds,
        "rank_seconds": rank_seconds,
        "metric_seconds": metric_seconds,
        "peak_delta_mb": peak - baseline,
        "dense_mb": dense_mb,
    }


def check_denominator_mismatch_is_caught():
    print("\n[2] denominator mismatch detection")
    n_users, n_items = 4, 40
    rng = np.random.RandomState(1)

    rows = np.repeat(np.arange(n_users), 2)
    cols = np.concatenate([_choose(rng, n_items, 2) for _ in range(n_users)])
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )

    candidates = sp.lil_matrix((n_users, n_items), dtype=bool)
    for u in range(n_users):
        for c in _choose(rng, n_items, 4):
            candidates[u, c] = True
        for c in interactions[u].indices:
            candidates[u, c] = True
    candidates = candidates.tocsr()

    model = LightFM(loss="bpr", no_components=4, random_state=2)
    model.fit(interactions, candidate_items=candidates, epochs=1)

    test = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        forbidden = set(interactions[u].indices.tolist())
        eligible = [int(c) for c in candidates[u].indices if c not in forbidden]
        test[u, eligible[0]] = 1.0
    test = test.tocsr()

    precision = evaluation.precision_at_k(
        model, test, train_interactions=interactions, k=10, candidate_items=candidates
    )

    effective = eligible_counts(candidates, interactions)
    denominators = np.minimum(10, effective)

    ranks = model.predict_rank(
        test, train_interactions=interactions, candidate_items=candidates
    )
    rank_dense = np.asarray(ranks.todense())

    hits = np.array(
        [np.sum(rank_dense[u][test[u].indices] < 10) for u in range(n_users)]
    )
    correct = np.divide(
        hits,
        denominators,
        out=np.zeros(n_users),
        where=denominators > 0,
    )
    broken = hits / 10.0  # the regression: metric still divides by k

    print("    correct precision: %s" % correct)
    print("    broken (divide by k): %s" % broken)
    print("    hits: %s, denominators: %s" % (hits, denominators))

    # Note: denominators can exceed the number of hits and even exceed the
    # number of test positives; precision can exceed 1.0 only because the test
    # set has fewer positives than the effective universe. With one test
    # positive per user the correct value is hits / denominator <= 1.
    assert np.allclose(precision, correct)
    if np.any(denominators != 10):
        # Training has changed the denominator; a divide-by-k bug is visible.
        assert not np.allclose(correct, broken)


def check_sampling_distribution():
    print("\n[3] uniform negative sampling over eligible non-positive items")
    from lightfm._lightfm_fast import CSRIndexMatrix, __test_sample_candidates

    candidates = sp.csr_matrix(
        np.array(
            [[True, True, True, False], [False, True, True, True]],
            dtype=bool,
        )
    )
    interactions = sp.csr_matrix(
        (np.array([1.0, 1.0], dtype=np.float32), ([0, 1], [1, 2])), shape=(2, 4)
    )

    negatives = negative_candidates(candidates, positive_mask(interactions))
    normalized = normalize_candidate_items(negatives, negatives.shape)
    index_matrix = CSRIndexMatrix(to_index_csr(normalized))

    draws = __test_sample_candidates(index_matrix, 0, 60000, 123)
    values, counts = np.unique(draws, return_counts=True)
    print("    user 0 draws: %s" % dict(zip(values.tolist(), counts.tolist())))
    # Item 0 and item 2 only; roughly 50/50.
    assert set(values.tolist()) == {0, 2}
    fractions = counts / counts.sum()
    assert np.all(np.abs(fractions - 0.5) < 0.02)

    # No positive (item 1) or ineligible (item 3) ever sampled.
    assert 1 not in values and 3 not in values


def check_all_positive_candidates_terminate():
    print("\n[4] candidate sets with no negatives terminate")
    n_users, n_items = 50, 30
    rng = np.random.RandomState(2)
    rows = np.repeat(np.arange(n_users), 3)
    cols = np.concatenate([_choose(rng, n_items, 3) for _ in range(n_users)])
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )

    # Candidate set is exactly the training positives: no negatives at all.
    candidates = sp.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n_users, n_items)
    )

    for loss in ("bpr", "warp"):
        model = LightFM(loss=loss, no_components=4, random_state=3, max_sampled=1000)
        start = time.time()
        model.fit(interactions, candidate_items=candidates, epochs=2)
        elapsed = time.time() - start
        assert elapsed < 10.0
        assert np.isfinite(np.sum(model.item_embeddings))
        print("    %s: %.3fs (no hang)" % (loss, elapsed))


def main():
    results = check_large_catalogue_small_candidates()
    check_denominator_mismatch_is_caught()
    check_sampling_distribution()
    check_all_positive_candidates_terminate()
    print("\nAll manual checks passed.")
    print("large-catalogue stats: %r" % (results,))


if __name__ == "__main__":
    main()
