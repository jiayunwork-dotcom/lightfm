# coding=utf-8
"""
Helpers for the optional per-user candidate-item eligibility matrix.

The candidate matrix is a SciPy sparse boolean matrix shaped like the
interaction matrix.  A stored ``True`` means the user is eligible to see the
item; an explicitly stored ``False`` is still ineligible.  A coordinate may
appear more than once: in that case the item is eligible if any of its stored
values is ``True``, and the item is counted only once.

The functions here deliberately never densify the matrix: all operations are
sparse and proportional to the number of eligible entries.
"""

import numpy as np

import scipy.sparse as sp


def normalize_candidate_items(candidate_items, shape):
    """Validate and canonicalise a candidate eligibility matrix.

    Parameters
    ----------
    candidate_items : scipy sparse boolean matrix or None
        The eligibility matrix.  ``None`` means "no candidate restriction".
    shape : tuple
        The expected ``(n_users, n_items)`` shape.

    Returns
    -------
    scipy.sparse.csr_matrix (boolean) or None
        A canonical CSR matrix with sorted indices, no duplicates and no
        explicitly stored ``False`` values.  ``None`` is passed through.
    """

    if candidate_items is None:
        return None

    if not sp.issparse(candidate_items):
        raise ValueError(
            "candidate_items must be a scipy sparse boolean matrix, "
            "got {}.".format(type(candidate_items))
        )

    if candidate_items.dtype != np.bool_:
        raise ValueError(
            "candidate_items must have boolean dtype, got {}.".format(
                candidate_items.dtype
            )
        )

    if tuple(candidate_items.shape) != tuple(shape):
        raise ValueError(
            "candidate_items must have the same shape as the interaction "
            "matrix: expected {}, got {}.".format(
                tuple(shape), tuple(candidate_items.shape)
            )
        )

    # Fast path: already canonical.  ``has_canonical_format`` guarantees
    # sorted indices and no duplicate coordinates, and ``data.all()``
    # guarantees there are no explicitly stored False values.
    if (
        candidate_items.format == "csr"
        and candidate_items.has_canonical_format
        and candidate_items.data.all()
    ):
        return candidate_items

    mat = candidate_items.tocsr(copy=True)

    # Collapse duplicate coordinates (Boolean addition saturates to True, so
    # True wins over False) and drop explicitly stored False values.
    mat.sum_duplicates()
    mat.eliminate_zeros()

    if not mat.has_sorted_indices:
        mat.sort_indices()

    return mat


def to_index_csr(mat):
    """Return a CSR matrix whose ``indices``/``indptr`` are ``int32``.

    The Cython extension indexes with C ``int``; SciPy normally uses
    ``int32``, but very large matrices may use ``int64``.  Only rebuild the
    matrix when a conversion is actually needed.
    """

    if mat.indices.dtype == np.int32 and mat.indptr.dtype == np.int32:
        return mat

    return sp.csr_matrix(
        (
            mat.data,
            mat.indices.astype(np.int32),
            mat.indptr.astype(np.int32),
        ),
        shape=mat.shape,
    )


def pattern_mask(mat):
    """Boolean CSR matrix marking every explicitly stored coordinate.

    Zeros or negative values stored explicitly are still coordinates (this
    matches how training interactions are excluded from ranking); only the
    sparsity pattern matters.
    """

    csr = mat.tocsr(copy=True)
    csr.data = np.ones_like(csr.data, dtype=np.bool_)
    if not csr.has_sorted_indices:
        csr.sort_indices()

    return csr


def positive_mask(interactions):
    """Build a boolean CSR matrix of entries explicitly marked positive.

    In the pairwise losses an interaction only acts as a positive when its
    stored label is greater than zero.  Explicit negative labels therefore do
    not consume a negative candidate slot.  Duplicate coordinates collapse.
    """

    coo = interactions.tocoo()
    keep = coo.data > 0

    if not np.any(keep):
        return sp.csr_matrix(interactions.shape, dtype=np.bool_)

    mat = sp.csr_matrix(
        (
            np.ones(np.count_nonzero(keep), dtype=np.bool_),
            (coo.row[keep], coo.col[keep]),
        ),
        shape=interactions.shape,
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    if not mat.has_sorted_indices:
        mat.sort_indices()

    return mat


def negative_candidates(candidate_items, positives):
    """Return ``candidate_items`` with the positive entries removed.

    The result is the per-user uniform sampling universe for BPR and WARP:
    candidate items that are not marked as a positive by this training
    interaction matrix.  It is sparse and has at most as many entries as the
    candidate matrix.
    """

    neg = candidate_items - candidate_items.multiply(positives)
    neg = sp.csr_matrix(neg, dtype=np.bool_)
    neg.eliminate_zeros()
    neg.sum_duplicates()
    if not neg.has_sorted_indices:
        neg.sort_indices()

    return neg


def _raise_for_missing(kind, missing_nnz):
    raise ValueError(
        "The {} interactions contain {} positive entr{} that are not "
        "eligible candidate items.  Every {} positive must be a candidate "
        "for its user.".format(
            kind,
            missing_nnz,
            "y" if missing_nnz == 1 else "ies",
            kind,
        )
    )


def check_positives_are_candidates(interactions, candidate_items):
    """Raise a ``ValueError`` if a training positive is not a candidate.

    The check happens before any model parameter is changed.
    """

    positives = positive_mask(interactions)
    missing = positives - positives.multiply(candidate_items)

    if missing.nnz:
        _raise_for_missing("training", missing.nnz)


def check_test_positives_are_candidates(test_interactions, candidate_items):
    """Raise a ``ValueError`` if a test interaction is not a candidate.

    ``predict_rank`` assigns a rank to every stored test coordinate (the
    evaluation functions count stored coordinates as positives), so the check
    uses the stored sparsity pattern, matching the ranking loop exactly.
    """

    test_pattern = pattern_mask(test_interactions)
    missing = test_pattern - test_pattern.multiply(candidate_items)

    if missing.nnz:
        _raise_for_missing("test", missing.nnz)


def eligible_counts(candidate_items, train_interactions=None):
    """Number of candidate items per user, excluding training interactions.

    This is the ranking universe size (``C_eff``) used by ``predict_rank``
    and by the evaluation metrics.  ``train_interactions`` are excluded by
    sparsity pattern, matching ``predict_rank``'s exclusion rule.
    """

    counts = np.asarray(candidate_items.getnnz(axis=1), dtype=np.int64)

    if train_interactions is not None:
        train = pattern_mask(train_interactions)
        excluded = candidate_items.multiply(train).getnnz(axis=1)
        counts = counts - np.asarray(excluded, dtype=np.int64)

    return np.maximum(counts, 0)
