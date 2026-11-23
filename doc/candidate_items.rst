.. _candidate-items:

Candidate-restricted training and evaluation
============================================

LightFM 1.17 supports an optional per-user *candidate eligibility* matrix.
A candidate matrix declares which items each user is allowed to see.  When it
is supplied, training, ranking and evaluation all use the same universe, so a
model is never trained against an item that the user cannot be recommended and
never evaluated on one either.

API
---

The following entry points accept ``candidate_items=None``:

* :meth:`lightfm.LightFM.fit`
* :meth:`lightfm.LightFM.fit_partial`
* :meth:`lightfm.LightFM.predict_rank`
* :func:`lightfm.evaluation.precision_at_k`
* :func:`lightfm.evaluation.recall_at_k`
* :func:`lightfm.evaluation.auc_score`
* :func:`lightfm.evaluation.reciprocal_rank`

``candidate_items`` must be a SciPy **sparse boolean** matrix with the same
shape as the interaction matrix.  A stored ``True`` means "eligible"; an
explicitly stored ``False``, and every coordinate that is not stored, means
"ineligible".  Duplicate coordinates are collapsed: the item is eligible if
any stored value is ``True``, and it is counted once.  A wrong shape or a
non-boolean dtype raises ``ValueError``.  Passing ``None`` (the default) keeps
the historical behaviour exactly.

Training
--------

Candidate-restricted training is implemented for the ``bpr`` and ``warp``
losses; ``logistic`` and ``warp-kos`` raise ``ValueError`` when a candidate
matrix is supplied.

For ``bpr`` and ``warp`` the sampled negative is drawn uniformly from the
user's eligible items **that are not marked positive by the current training
interactions**.  Positive items and ineligible items therefore never consume
one of the ``max_sampled`` attempts.  WARP's rank estimate uses the size of
that same universe, ``log(rank_universe / sampled)``, so the sampling
distribution and the rank estimate are consistent by construction.

Every training positive must itself be a candidate.  If it is not, the call
raises ``ValueError`` before any parameter is modified (``fit`` validates
before discarding the old model, too).  A user whose eligible items are all
positives has no negative to compare against: the comparison is skipped, so
training terminates without an update for that user rather than sampling
forever.

Each ``fit_partial`` call may pass a different table.  Previously learned
parameters are retained and the new eligibility takes effect immediately.

Ranking
-------

``predict_rank`` compares test positives only against eligible items, walking
the candidate rows directly.  Training interactions are still excluded from
the ranking (by sparsity pattern, as before), and tie-breaking is unchanged:
an eligible item whose score is greater than or equal to the target's
increments the target's rank.  A test positive outside the candidate set
raises ``ValueError``; the existing train/test intersection check still runs
first.  For a test item that is itself a candidate but also a training
interaction, the historical rule applies: the item is not compared with
itself, so a top-ranked test item has rank 0.

Evaluation
----------

All four metrics forward ``candidate_items`` to ``predict_rank`` and adjust
their denominators:

* ``precision_at_k`` divides by ``min(k, C_eff)``, where ``C_eff`` is the
  number of candidate items after excluding training interactions.
* ``recall_at_k`` keeps the number of test positives as the denominator.
* ``auc_score`` compares positives only against the remaining candidate
  negatives.  If a user with test positives has no negative candidate left,
  the value is ``NaN``.
* ``reciprocal_rank`` uses the candidate-restricted ranks.

``k`` must be a positive integer.  ``preserve_rows=False`` keeps only users
with test positives; with ``preserve_rows=True`` users without test positives
retain each metric's original convention (precision 0, recall ``NaN`` from
``0 / 0``, AUC 0.5, reciprocal rank 0).

Cost and trade-offs
-------------------

* **Preprocessing.** ``normalize_candidate_items`` is ``O(nnz(candidates))``
  (canonicalise, drop explicit ``False``).  Each training call additionally
  builds two sparse helpers in ``O(nnz(candidates) + nnz(interactions))``:
  the positive mask and the negative sampling universe.  Neither ever
  materialises a dense user x item matrix.
* **Memory.** All candidate structures are sparse and proportional to the
  number of eligible entries.  The Cython side only holds the CSR
  ``indices``/``indptr`` (no float payload) for the candidate matrix.  Peak
  memory is therefore driven by the interaction and feature matrices and the
  number of eligible entries, not by the catalogue size.
* **Ranking.** ``predict_rank`` scores only the user's candidate row; its
  cost is ``O(C_eff(user) * (features + components))`` instead of
  ``O(n_items * ...)``.  Ranking a user never scores the full catalogue.
* **Sampling.** BPR performs one uniform draw from the precomputed negative
  universe (``O(1)``, then one representation lookup).  WARP performs up to
  ``max_sampled`` draws; each draw is still ``O(1)`` but requires a scoring
  pass.  As the positive fraction of a user's candidates grows the negative
  universe shrinks: WARP reaches ``max_sampled`` more often and eventually
  skips the user entirely when the universe is empty, while BPR skips
  immediately.  This is the intended behaviour: there is nothing meaningful to
  rank against when no negative remains.
* The per-call preprocessing is repeated on every ``fit_partial`` call because
  the eligibility table and the training positives can both change; this
  trades a small, linear cost for correctness.  If callers repeatedly train on
  the same table, the preprocessing could be cached externally.

Re-running the checks
---------------------

The unit tests are in ``tests/test_candidates.py``::

    python3 -m pytest tests/test_candidates.py -v

The resource-oriented checks (large catalogue, small candidate set; the
denominator regression; the sampling distribution; all-positive candidate
sets) are a standalone script::

    python3 tests/manual_candidate_checks.py
