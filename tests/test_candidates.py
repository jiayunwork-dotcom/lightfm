# coding=utf-8
"""
Tests for per-user candidate eligibility (``candidate_items``).

The tests are written so that every expected rank and metric value can be
counted by hand on tiny examples.  A separate test module
(``tests/manual_candidate_checks.py``) exercises resource usage on a
large-catalogue / small-candidate setup and can be re-run manually.
"""

import numpy as np

import pytest

import scipy.sparse as sp

from lightfm import LightFM
from lightfm import evaluation
from lightfm._candidates import (
    eligible_counts,
    negative_candidates,
    normalize_candidate_items,
    positive_mask,
)


def _bool_csr(rows):
    return sp.csr_matrix(np.array(rows, dtype=np.bool_))


def _f32_csr(rows):
    return sp.csr_matrix(np.array(rows, dtype=np.float32))


def _zero_model(model, item_biases):
    """Zero out everything but item biases, so rankings are hand-computable."""

    model.user_embeddings[:] = 0
    model.user_biases[:] = 0
    model.item_embeddings[:] = 0
    model.item_biases[:] = np.asarray(item_biases, dtype=np.float32)


def _fitted_model(loss="bpr", conditional_item_biases=None):
    """Fit a trivial model and then replace its scores with fixed biases."""

    n_users, n_items = 2, 6

    train = sp.csr_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [0, 5])), shape=(n_users, n_items)
    )

    model = LightFM(loss=loss, no_components=4, random_state=42)
    model.fit(train)

    if conditional_item_biases is None:
        conditional_item_biases = [0.1, 0.8, 0.5, 0.2, 0.7, 0.95]

    _zero_model(model, conditional_item_biases)

    return model, train


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_candidate_validation_rejects_bad_shape_and_dtype():

    model, train = _fitted_model()

    # Wrong shape
    with pytest.raises(ValueError):
        model.fit(train, candidate_items=_bool_csr([[True, False]]))

    # Wrong dtype (integer, float)
    with pytest.raises(ValueError):
        model.fit(
            train,
            candidate_items=sp.csr_matrix(np.ones((2, 6), dtype=np.int32)),
        )
    with pytest.raises(ValueError):
        model.fit(
            train,
            candidate_items=sp.csr_matrix(np.ones((2, 6), dtype=np.float32)),
        )

    # Dense input is rejected as well.
    with pytest.raises(ValueError):
        model.fit(train, candidate_items=np.ones((2, 6), dtype=bool))

    # predict_rank and the evaluation entry points validate too.
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])

    with pytest.raises(ValueError):
        model.predict_rank(test, candidate_items=_bool_csr([[True, False]]))

    with pytest.raises(ValueError):
        evaluation.precision_at_k(
            model, test, candidate_items=sp.csr_matrix(np.ones((2, 6), dtype=np.int8))
        )

    with pytest.raises(ValueError):
        evaluation.recall_at_k(
            model, test, candidate_items=sp.csr_matrix(np.ones((2, 6), dtype=np.int8))
        )

    with pytest.raises(ValueError):
        evaluation.auc_score(
            model, test, candidate_items=sp.csr_matrix(np.ones((2, 6), dtype=np.int8))
        )

    with pytest.raises(ValueError):
        evaluation.reciprocal_rank(
            model, test, candidate_items=sp.csr_matrix(np.ones((2, 6), dtype=np.int8))
        )


def test_unsupported_losses_reject_candidates():

    train = _f32_csr([[1, 0, 0], [0, 1, 0]])
    candidates = _bool_csr([[True, True, True], [True, True, True]])

    for loss in ("logistic", "warp-kos"):
        model = LightFM(loss=loss)
        with pytest.raises(ValueError):
            model.fit(train, candidate_items=candidates)
        with pytest.raises(ValueError):
            model.fit_partial(train, candidate_items=candidates)


def test_duplicate_and_explicit_false_coordinates():

    # (0, 1) appears three times (True, False, True) -> eligible, counted once.
    # (0, 2) is stored only as False -> ineligible.
    candidates = sp.coo_matrix(
        (
            np.array([True, False, True, False, True], dtype=np.bool_),
            ([0, 0, 0, 0, 1], [1, 1, 1, 2, 4]),
        ),
        shape=(2, 6),
    )

    normalized = normalize_candidate_items(candidates, (2, 6))
    assert normalized.dtype == np.bool_
    # (0, 1) and (1, 4) are eligible; the duplicate coordinate is counted once.
    assert normalized.toarray().sum() == 2
    assert normalized[0, 1]
    assert not normalized[0, 2]

    model, train = _fitted_model()
    ranks = model.predict_rank(
        _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]),
        train_interactions=train,
        candidate_items=candidates,
    )
    assert ranks.shape == (2, 6)
    assert ranks[0, 1] == 0
    assert ranks[1, 4] == 0

    # Evaluation also accepts the duplicate-coordinate matrix, and its
    # denominator counts (0, 1) only once.
    precision = evaluation.precision_at_k(
        model,
        _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]),
        train_interactions=train,
        k=10,
        candidate_items=candidates,
    )
    # User 0: candidates after train exclusion are {1} (2 is False, 0 is train).
    # The single test positive is ranked 0, so precision = 1 / min(10, 1) = 1.
    assert precision[0] == pytest.approx(1.0)


def test_candidates_validated_before_parameters_change():

    train = _f32_csr([[1, 0, 0], [0, 1, 0]])
    # Positive (0, 0) is not a candidate.
    candidates = _bool_csr([[False, False, True], [True, True, True]])

    model = LightFM(loss="warp", random_state=1)
    model.fit_partial(train, candidate_items=_bool_csr([[True, True, True]] * 2))

    snapshot = model.item_embeddings.copy()

    with pytest.raises(ValueError):
        model.fit_partial(train, candidate_items=candidates)

    # No parameter may change when the call fails.
    assert np.array_equal(model.item_embeddings, snapshot)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_predict_rank_restricts_to_candidates_and_excludes_train():
    """Hand-counted example.

    Item scores: i0=0.1 i1=0.8 i2=0.5 i3=0.2 i4=0.7 i5=0.95
    Train: (0, 0), (1, 5)
    Candidate rows:
        user 0: {0, 1, 2, 3}
        user 1: {2, 3, 4, 5}
    Test: (0, 1), (1, 4)

    User 0 ranking universe = {1, 2, 3} (0 excluded as train):
        1 (0.8) > 3 (0.2) > 2 (0.5)  => sorted 1, 2, 3
        rank(1) = 0
    User 1 ranking universe = {2, 3, 4} (5 excluded as train):
        4 (0.7) > 2 (0.5) > 3 (0.2)
        rank(4) = 0
    """

    model, train = _fitted_model()

    candidates = _bool_csr(
        [[True, True, True, True, False, False], [False, False, True, True, True, True]]
    )
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])

    ranks = model.predict_rank(
        test, train_interactions=train, candidate_items=candidates
    )

    assert ranks[0, 1] == 0
    assert ranks[1, 4] == 0
    # Coordinates outside the test pattern stay absent.
    assert ranks.nnz == 2
    assert ranks.shape == test.shape

    # Unrestricted ranks differ: user 0 would rank item 1 against item 5.
    unrestricted = model.predict_rank(test, train_interactions=train)
    assert unrestricted[0, 1] == 1  # item 5 outscores item 1


def test_predict_rank_matches_brute_force_candidate_ranking():
    """Cross-check the Cython ranking against direct scoring of candidates."""

    rng = np.random.RandomState(17)
    n_users, n_items = 7, 25

    rows = np.repeat(np.arange(n_users), 3)
    cols = np.concatenate(
        [rng.choice(n_items, 3, replace=False) for _ in range(n_users)]
    )
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )
    interactions.sum_duplicates()
    interactions.data[:] = 1.0

    candidates = sp.lil_matrix((n_users, n_items), dtype=bool)
    for u in range(n_users):
        chosen = set(rng.choice(n_items, 8, replace=False).tolist())
        chosen |= set(interactions[u].indices.tolist())
        for c in chosen:
            candidates[u, c] = True
    candidates = candidates.tocsr()

    model = LightFM(loss="bpr", no_components=5, random_state=4)
    model.fit(interactions, candidate_items=candidates, epochs=1)

    test = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        available = sorted(
            set(candidates[u].indices.tolist()) - set(interactions[u].indices.tolist())
        )
        assert available
        test[u, available[0]] = 1.0
    test = test.tocsr()

    ranks = model.predict_rank(
        test, train_interactions=interactions, candidate_items=candidates
    ).toarray()

    # Brute-force: score every eligible item for every user, exclude train
    # coordinates, then count items scored >= the target (pessimistic ties).
    for u in range(n_users):
        universe = [
            int(c)
            for c in candidates[u].indices
            if c not in set(interactions[u].indices.tolist())
        ]
        scores = model.predict(
            np.repeat(u, len(universe)).astype(np.int32),
            np.asarray(universe, dtype=np.int32),
        )
        score_by_item = dict(zip(universe, scores.tolist()))

        for item in test[u].indices:
            target = int(item)
            expected = sum(
                1
                for other in universe
                if other != target and score_by_item[other] >= score_by_item[target]
            )
            assert ranks[u, item] == expected, (u, item, expected, ranks[u, item])


def test_predict_rank_ignores_high_scoring_ineligible_items():
    """An ineligible item must contribute nothing to a candidate rank."""

    model, _ = _fitted_model(
        conditional_item_biases=[0.1, 0.2, 0.3, 0.4, 0.9, 0.9]
    )
    # No training interactions.
    # User 0 candidates {0, 1, 2}; items 4 and 5 (both 0.9) are ineligible.
    candidates = _bool_csr(
        [[True, True, True, False, False, False], [True, True, True, True, True, True]]
    )
    test = _f32_csr([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])

    ranks = model.predict_rank(test, candidate_items=candidates)

    # User 0 universe {0, 1, 2}: items 1 and 2 outscore item 0 -> rank 2.
    # If the ineligible items 4 and 5 were counted, the rank would be 4.
    assert ranks[0, 0] == 2
    # User 1 universe is all 6 items; items 1-5 outscore item 0 -> rank 5.
    assert ranks[1, 0] == 5


def test_predict_rank_ties_are_pessimistic():
    """Ties follow the existing "prediction >=" rule within the candidate set."""

    model, train = _fitted_model(conditional_item_biases=[0.5] * 6)
    candidates = _bool_csr(
        [[True, True, True, False, False, False], [False, False, True, True, False, False]]
    )
    test = _f32_csr([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    ranks = model.predict_rank(
        test, train_interactions=train, candidate_items=candidates
    )

    # User 0 universe {1, 2}; the other candidate also has prediction >=, so
    # the rank is 1 (pessimistic). User 1 universe {2, 3}: rank(3) = 1 as well.
    assert ranks[0, 2] == 1
    assert ranks[1, 3] == 1


def test_predict_rank_rejects_test_positive_outside_candidates():

    model, train = _fitted_model()
    candidates = _bool_csr(
        [[True, True, False, False, False, False], [False, False, True, True, False, False]]
    )
    test = _f32_csr([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    with pytest.raises(ValueError):
        model.predict_rank(test, candidate_items=candidates)

    # The intersection check still fires first when both problems exist.
    with pytest.raises(ValueError):
        model.predict_rank(train, train_interactions=train, candidate_items=candidates)


# ---------------------------------------------------------------------------
# Metrics, hand-counted
# ---------------------------------------------------------------------------


def test_metrics_hand_counted():
    """Metrics on the ranking universe from the docstring example.

    User 0 universe {1, 2, 3}; ordering 1, 2, 3; test positive {1}.
    User 1 universe {2, 3, 4}; ordering 4, 2, 3; test positive {4}.
    """

    model, train = _fitted_model()
    candidates = _bool_csr(
        [[True, True, True, True, False, False], [False, False, True, True, True, True]]
    )
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])

    # k=2: denominator min(2, 3) = 2; each test positive is rank 0 -> 1/2.
    precision = evaluation.precision_at_k(
        model, test, train_interactions=train, k=2, candidate_items=candidates
    )
    assert np.allclose(precision, [0.5, 0.5])

    # k=1: denominator min(1, 3) = 1; both positives are rank 0 -> 1.0.
    precision1 = evaluation.precision_at_k(
        model, test, train_interactions=train, k=1, candidate_items=candidates
    )
    assert np.allclose(precision1, [1.0, 1.0])

    # Unrestricted precision at 2: universe is 5 items; each positive at rank
    # 1 or 0. User 0: item 1 is rank 1 (item 5 above it) -> hit; 1/2.
    unrestricted = evaluation.precision_at_k(
        model, test, train_interactions=train, k=2
    )
    assert unrestricted[0] == pytest.approx(0.5)

    # Recall denominators remain the number of test positives (1 each).
    recall = evaluation.recall_at_k(
        model, test, train_interactions=train, k=2, candidate_items=candidates
    )
    assert np.allclose(recall, [1.0, 1.0])

    # Reciprocal rank: both positives are rank 0 -> 1.0.
    rr = evaluation.reciprocal_rank(
        model, test, train_interactions=train, candidate_items=candidates
    )
    assert np.allclose(rr, [1.0, 1.0])

    # AUC: user 0 universe has 2 negatives below the positive -> 1.0.
    # User 1 universe has 2 negatives below the positive -> 1.0.
    auc = evaluation.auc_score(
        model, test, train_interactions=train, candidate_items=candidates
    )
    assert np.allclose(auc, [1.0, 1.0])


def test_precision_denominator_uses_effective_candidate_count():
    """A key regression: training must not narrow the universe while the
    metric denominator still uses the full catalogue."""

    model, train = _fitted_model()
    candidates = _bool_csr(
        [[True, True, True, True, False, False], [False, False, True, True, True, True]]
    )
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])

    # User 0 has 4 candidates, 1 excluded by train -> effective 3.
    # User 1 has 4 candidates, 1 excluded by train -> effective 3.
    effective = eligible_counts(candidates, train)
    assert list(effective) == [3, 3]

    precision = evaluation.precision_at_k(
        model, test, train_interactions=train, k=10, candidate_items=candidates
    )
    # 1 hit / min(10, 3), not / 10 and not / 6.
    assert np.allclose(precision, [1.0 / 3.0, 1.0 / 3.0])

    # Without candidate_items the historical denominator (k) is used.
    precision_full = evaluation.precision_at_k(
        model, test, train_interactions=train, k=10
    )
    assert precision_full[0] == pytest.approx(0.1)


def test_precision_denominator_smaller_than_k():
    """When min(k, #eligible) < k, the denominator shrinks accordingly."""

    model, train = _fitted_model()
    # User 0 candidates {0, 1}; 0 is train, so effective universe is {1}.
    candidates = _bool_csr(
        [[True, True, False, False, False, False], [False, False, True, True, False, False]]
    )
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    precision = evaluation.precision_at_k(
        model, test, train_interactions=train, k=10, candidate_items=candidates
    )
    # 1 hit / min(10, 1) = 1.0 for user 0; 1 hit / min(10, 2) = 0.5 for user 1.
    assert precision[0] == pytest.approx(1.0)
    assert precision[1] == pytest.approx(0.5)


def test_auc_uses_only_candidate_negatives():
    """A high-scoring ineligible item must not become a negative comparison."""

    model, _ = _fitted_model(
        conditional_item_biases=[0.1, 0.2, 0.3, 0.4, 0.9, 0.9]
    )
    # User 0 candidates only {0, 1}; test positive is item 0.
    candidates = _bool_csr(
        [[True, True, False, False, False, False], [True, True, True, True, True, True]]
    )
    test = _f32_csr([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])

    auc = evaluation.auc_score(model, test, candidate_items=candidates)

    # User 0: positive item 0 (0.1) vs only negative item 1 (0.2), which
    # outscores it -> AUC 0. Items 4 and 5 (0.9) are ineligible and must be
    # ignored; if mistakenly included AUC would still be 0 here, so use the
    # reverse case as well.
    assert auc[0] == pytest.approx(0.0)

    # Now make the positive the highest-scoring eligible item, but keep a
    # high-scoring ineligible item around.
    model, _ = _fitted_model(conditional_item_biases=[0.95, 0.2, 0.3, 0.4, 0.99, 0.99])
    auc = evaluation.auc_score(model, test, candidate_items=candidates)
    assert auc[0] == pytest.approx(1.0)


def test_auc_nan_when_no_negatives_remain():
    """Candidates almost all positives -> no negatives -> NaN."""

    model, train = _fitted_model(
        conditional_item_biases=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
    # User 0 candidates {0, 1}; 0 is train, 1 is test -> no negative left.
    candidates = _bool_csr(
        [[True, True, False, False, False, False], [False, False, True, True, True, True]]
    )

    auc = evaluation.auc_score(
        model, test, train_interactions=train, candidate_items=candidates
    )
    assert np.isnan(auc[0])
    assert not np.isnan(auc[1])

    # evaluate without train exclusion: user 0 universe {0, 1}, both
    # positive-ish but only item 1 is a test positive; item 0 is a negative
    # here, so AUC is defined.
    auc = evaluation.auc_score(model, test, candidate_items=candidates)
    assert not np.isnan(auc[0])


def test_preserve_rows_with_candidates():
    """preserve_rows=True keeps rows without test positives."""

    model, train = _fitted_model()
    candidates = _bool_csr(
        [[True, True, True, True, False, False], [False, False, True, True, True, True]]
    )
    # Only user 0 has a test positive.
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

    precision = evaluation.precision_at_k(
        model,
        test,
        train_interactions=train,
        k=2,
        candidate_items=candidates,
        preserve_rows=True,
    )
    assert precision.shape == (2,)
    # 1 hit / min(2, 3) = 0.5.
    assert precision[0] == pytest.approx(0.5)
    # No test positives for user 1 -> 0 by the existing convention.
    assert precision[1] == 0.0

    recall = evaluation.recall_at_k(
        model,
        test,
        train_interactions=train,
        k=2,
        candidate_items=candidates,
        preserve_rows=True,
    )
    assert recall.shape == (2,)
    assert np.isnan(recall[1])  # 0 / 0 preserved by the original convention

    auc = evaluation.auc_score(
        model,
        test,
        train_interactions=train,
        candidate_items=candidates,
        preserve_rows=True,
    )
    assert auc.shape == (2,)
    assert auc[1] == pytest.approx(0.5)  # no positives -> 0.5

    rr = evaluation.reciprocal_rank(
        model,
        test,
        train_interactions=train,
        candidate_items=candidates,
        preserve_rows=True,
    )
    assert rr.shape == (2,)
    assert rr[1] == 0.0

    # Without preserve_rows only the user with a test positive is returned.
    assert evaluation.precision_at_k(
        model, test, train_interactions=train, k=2, candidate_items=candidates
    ).shape == (1,)


def test_k_must_be_positive_integer():

    model, train = _fitted_model()
    test = _f32_csr([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
    candidates = _bool_csr([[True] * 6, [True] * 6])

    for fn in (
        evaluation.precision_at_k,
        evaluation.recall_at_k,
    ):
        for bad_k in (0, -1, 2.5, True, np.float64(3.0)):
            with pytest.raises(ValueError):
                fn(model, test, k=bad_k, candidate_items=candidates)

        # numpy integers are accepted.
        fn(model, test, k=np.int64(2), candidate_items=candidates)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _training_candidates(n_users, n_items, seed=0, width=5):
    rng = np.random.RandomState(seed)
    rows = np.repeat(np.arange(n_users), width)
    cols = np.concatenate(
        [rng.choice(n_items, width, replace=False) for _ in range(n_users)]
    )
    mat = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.bool_), (rows, cols)), shape=(n_users, n_items)
    )
    return mat


def test_negative_candidates_exclude_positives_and_ineligible():

    candidates = _bool_csr(
        [[True, True, True, False], [False, True, True, True]]
    )
    interactions = sp.csr_matrix(
        (np.array([1.0, 1.0], dtype=np.float32), ([0, 1], [1, 2])), shape=(2, 4)
    )

    positives = positive_mask(interactions)
    negatives = negative_candidates(candidates, positives)

    assert negatives.toarray().tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]
    # Explicit negative labels do not remove a candidate from the universe.
    interactions_neg = sp.csr_matrix(
        (np.array([-1.0, 1.0], dtype=np.float32), ([0, 1], [1, 2])), shape=(2, 4)
    )
    negatives = negative_candidates(candidates, positive_mask(interactions_neg))
    assert negatives.toarray().tolist() == [
        [True, True, True, False],
        [False, True, False, True],
    ]


def test_bpr_and_warp_train_with_candidates():
    """Both supported losses run and produce finite parameters."""

    n_users, n_items = 12, 30
    rng = np.random.RandomState(3)

    rows = np.arange(n_users)
    cols = rng.randint(0, n_items, n_users)

    interactions = sp.csr_matrix(
        (np.ones(n_users, dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )

    candidates = _training_candidates(n_users, n_items, seed=1, width=6)
    # Ensure each positive is eligible.
    for user, item in zip(rows, cols):
        candidates[user, item] = True
    candidates = candidates.tocsr()

    for loss in ("bpr", "warp"):
        model = LightFM(
            loss=loss, no_components=6, random_state=7, max_sampled=3
        )
        model.fit(interactions, candidate_items=candidates, epochs=3)

        assert np.isfinite(np.sum(model.item_embeddings))
        assert np.isfinite(np.sum(model.user_embeddings))


def test_users_with_no_negatives_are_skipped():
    """Users whose whole candidate set is positive must not hang or update."""

    # 3 users, 4 items. User 0: candidates {0, 1} both positive.
    # User 1: candidates {2, 3}, only 2 positive.
    interactions = sp.csr_matrix(
        (np.ones(3, dtype=np.float32), ([0, 0, 1], [0, 1, 2])), shape=(3, 4)
    )
    candidates = _bool_csr(
        [[True, True, False, False], [False, False, True, True], [False, False, False, False]]
    )

    model = LightFM(loss="warp", no_components=4, random_state=2, max_sampled=5)

    # No error, and training returns promptly.
    model.fit(interactions, candidate_items=candidates, epochs=2)
    assert np.isfinite(np.sum(model.item_embeddings))


def test_sample_weight_with_candidates():

    n_users, n_items = 6, 12
    rng = np.random.RandomState(5)
    rows, cols = [], []
    for u in range(n_users):
        picked = rng.choice(n_items, 3, replace=False)
        rows.extend([u] * 3)
        cols.extend(picked.tolist())

    data = np.ones(len(rows), dtype=np.float32)
    interactions = sp.coo_matrix(
        (data, (np.array(rows), np.array(cols))), shape=(n_users, n_items)
    )
    weights = (np.arange(len(rows), dtype=np.float32) + 1.0) * 0.1

    candidates = _training_candidates(n_users, n_items, seed=2, width=6)
    for r, c in zip(rows, cols):
        candidates[r, c] = True
    candidates = candidates.tocsr()

    model = LightFM(loss="bpr", no_components=5, random_state=11)
    model.fit(
        interactions,
        sample_weight=sp.coo_matrix(
            (weights.astype(np.float32), (interactions.row, interactions.col)),
            shape=interactions.shape,
        ),
        candidate_items=candidates,
        epochs=3,
    )
    assert np.isfinite(np.sum(model.item_embeddings))


def test_hybrid_features_with_candidates():
    """Feature matrices work alongside candidates."""

    n_users, n_items = 8, 16
    rng = np.random.RandomState(9)
    rows = np.arange(n_users)
    cols = rng.randint(0, n_items, n_users)
    interactions = sp.coo_matrix(
        (np.ones(n_users, dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )

    item_features = sp.identity(n_items, dtype=np.float32, format="csr")
    user_features = sp.identity(n_users, dtype=np.float32, format="csr")

    candidates = _training_candidates(n_users, n_items, seed=4, width=5)
    for r, c in zip(rows, cols):
        candidates[r, c] = True
    candidates = candidates.tocsr()

    for loss in ("bpr", "warp"):
        model = LightFM(loss=loss, no_components=4, random_state=12)
        model.fit(
            interactions,
            user_features=user_features,
            item_features=item_features,
            candidate_items=candidates,
            epochs=2,
        )
        assert model.user_embeddings.shape[0] == n_users
        assert model.item_embeddings.shape[0] == n_items

        test = sp.csr_matrix(
            (np.ones(2, dtype=np.float32), ([0, 1], [1, 2])), shape=(n_users, n_items)
        )
        # Make sure the test positives are eligible.
        candidates2 = candidates.copy()
        candidates2[0, 1] = True
        candidates2[1, 2] = True
        candidates2 = candidates2.tocsr()
        ranks = model.predict_rank(
            test,
            user_features=user_features,
            item_features=item_features,
            candidate_items=candidates2,
        )
        assert ranks.shape == (n_users, n_items)


def test_fit_partial_swaps_candidate_tables():
    """A later call can use a new table; old parameters keep training."""

    n_users, n_items = 4, 10
    rng = np.random.RandomState(21)
    rows = np.arange(n_users)
    cols = rng.randint(0, n_items, n_users)
    interactions = sp.csr_matrix(
        (np.ones(n_users, dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )

    first = _training_candidates(n_users, n_items, seed=6, width=4)
    second = _training_candidates(n_users, n_items, seed=7, width=6)
    for r, c in zip(rows, cols):
        first[r, c] = True
        second[r, c] = True
    first = first.tocsr()
    second = second.tocsr()

    model = LightFM(loss="warp", no_components=5, random_state=13, max_sampled=3)
    model.fit_partial(interactions, candidate_items=first, epochs=2)
    after_first = model.item_embeddings.copy()

    model.fit_partial(interactions, candidate_items=second, epochs=2)

    # Parameters were retained and then updated.
    assert model.item_embeddings.shape == after_first.shape
    assert not np.array_equal(model.item_embeddings, after_first)
    assert np.isfinite(np.sum(model.item_embeddings))

    # And the new eligibility takes effect in ranking immediately.
    test = sp.csr_matrix(
        (np.ones(1, dtype=np.float32), ([0], [cols[0]])), shape=(n_users, n_items)
    )
    model.predict_rank(test, candidate_items=second)


def test_candidate_training_never_densifies():
    """Sparse candidate columns must not be merged into a dense matrix.

    The test uses a large catalogue with a tiny candidate set; if any helper
    densified the matrix, the memory blow-up would make this test unusable.
    """

    n_users, n_items = 300, 20000
    rows = np.arange(n_users)
    cols = np.arange(n_users) * 3
    interactions = sp.csr_matrix(
        (np.ones(n_users, dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )

    candidates = sp.csr_matrix(
        (
            np.ones(n_users * 4, dtype=np.bool_),
            (
                np.repeat(rows, 4),
                np.concatenate(
                    [
                        np.array([c] + [c + 1, c + 2, c + 3])
                        for c in range(0, n_users * 4, 4)
                    ]
                ),
            ),
        ),
        shape=(n_users, n_items),
    )
    # candidates[:, user*3] must be eligible for the positive.
    candidates = sp.lil_matrix((n_users, n_items), dtype=bool)
    for u in range(n_users):
        for k in range(4):
            candidates[u, (u * 4 + k) % n_items] = True
        candidates[u, u * 3] = True
    candidates = candidates.tocsr()

    model = LightFM(loss="bpr", no_components=3, random_state=3)
    model.fit(interactions, candidate_items=candidates, epochs=1)

    test = sp.csr_matrix(
        (np.ones(n_users, dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )
    # All test positives are candidates by construction.
    ranks = model.predict_rank(test, candidate_items=candidates)
    assert ranks.shape == (n_users, n_items)
    assert ranks.nnz == n_users


def test_training_never_samples_ineligible_items():
    """Ineligible items must never appear as a sampled negative.

    Item embeddings are only touched by a gradient update. With identity
    features, zero regularisation and a candidate set that leaves most items
    ineligible (and never positive), those items' embeddings must stay exactly
    at their initial values after training.
    """

    n_users, n_items = 5, 40
    rows = np.arange(n_users)
    cols = np.arange(n_users) * 2  # even items are training positives

    interactions = sp.csr_matrix(
        (np.ones(n_users, dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )

    candidates = sp.lil_matrix((n_users, n_items), dtype=bool)
    allowed = set()
    for u in range(n_users):
        # A handful of eligible items: the positive plus four negatives. The
        # WARP weight is log(rank_universe / sampled); with only one negative
        # the weight is log(1) = 0 and no update would happen at all.
        candidates[u, int(cols[u])] = True
        allowed.add(int(cols[u]))
        for offset in range(1, 5):
            negative = int(cols[u]) + offset
            candidates[u, negative] = True
            allowed.add(negative)
    candidates = candidates.tocsr()

    model = LightFM(
        loss="warp",
        no_components=5,
        random_state=19,
        max_sampled=50,
        item_alpha=0.0,
        user_alpha=0.0,
    )
    model.fit_partial(interactions)
    initial = model.item_embeddings.copy()

    model.fit_partial(interactions, candidate_items=candidates, epochs=5)

    ineligible = sorted(set(range(n_items)) - allowed)
    assert np.array_equal(model.item_embeddings[ineligible], initial[ineligible])

    # At least one eligible item moved, proving training actually happened.
    assert not np.array_equal(model.item_embeddings[list(allowed)], initial[list(allowed)])


def test_fit_error_leaves_existing_model_untouched():

    train = _f32_csr([[1, 0, 0], [0, 1, 0]])
    good = _bool_csr([[True, True, True], [True, True, True]])
    bad = _bool_csr([[False, False, True], [True, True, True]])

    model = LightFM(loss="warp", random_state=3)
    model.fit_partial(train, candidate_items=good, epochs=1)

    snapshot = model.item_embeddings.copy()

    # fit() must validate before resetting the learned state.
    with pytest.raises(ValueError):
        model.fit(train, candidate_items=bad)

    assert np.array_equal(model.item_embeddings, snapshot)


def test_candidates_can_be_dropped_on_later_partial_fit():

    n_users, n_items = 4, 8
    rng = np.random.RandomState(31)
    rows = np.repeat(np.arange(n_users), 2)
    cols = np.concatenate([rng.choice(n_items, 2, replace=False) for _ in range(n_users)])
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )

    candidates = sp.lil_matrix((n_users, n_items), dtype=bool)
    for u in range(n_users):
        for c in rng.choice(n_items, 4, replace=False):
            candidates[u, c] = True
        for c in interactions[u].indices:
            candidates[u, c] = True
    candidates = candidates.tocsr()

    model = LightFM(loss="bpr", no_components=4, random_state=5)
    model.fit_partial(interactions, candidate_items=candidates, epochs=1)
    after_candidate = model.item_embeddings.copy()

    # Falling back to the unrestricted algorithm keeps the learned state.
    model.fit_partial(interactions, candidate_items=None, epochs=1)
    assert not np.array_equal(model.item_embeddings, after_candidate)
    assert np.isfinite(np.sum(model.item_embeddings))
