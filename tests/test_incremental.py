"""
Regression tests for joint Dataset/LightFM incremental extension.

Covers:
- identity-feature and side-info-only datasets,
- exact preservation of indices, parameters, optimiser state and scores,
- AdaGrad and AdaDelta initial state for new parameters,
- continued training and prediction on the extended feature space,
- atomic rollback on iterator and allocation failures,
- no-op calls that do not move the random state,
- dimension mismatch and bad-argument validation.
"""

import numpy as np

import pytest

import scipy.sparse as sp

from lightfm.data import Dataset
from lightfm.lightfm import LightFM


PARAM_NAMES = (
    "item_embeddings",
    "item_embedding_gradients",
    "item_embedding_momentum",
    "item_biases",
    "item_bias_gradients",
    "item_bias_momentum",
    "user_embeddings",
    "user_embedding_gradients",
    "user_embedding_momentum",
    "user_biases",
    "user_bias_gradients",
    "user_bias_momentum",
)


def snapshot_model(model):
    """Independent value copies, for asserting content preservation."""

    return {name: getattr(model, name).copy() for name in PARAM_NAMES}


def snapshot_model_refs(model):
    """Live array references, for asserting that nothing was reassigned."""

    return {name: getattr(model, name) for name in PARAM_NAMES}


def snapshot_mappings(dataset):
    return [dict(mapping) for mapping in dataset.mapping()]


def assert_refs_unchanged(model, refs):
    for name in PARAM_NAMES:
        assert getattr(model, name) is refs[name]


def assert_old_state_untouched(model, snapshot, old_user_dim, old_item_dim):
    """Every pre-extension element must survive bit for bit."""

    for name in PARAM_NAMES:
        current = getattr(model, name)
        previous = snapshot[name]
        if name.startswith("user"):
            old = old_user_dim
        else:
            old = old_item_dim
        if current.ndim == 1:
            assert np.array_equal(current[:old], previous)
        else:
            assert np.array_equal(current[:old, :], previous)


def assert_mappings_untouched(dataset, snapshot):
    for current, previous in zip(dataset.mapping(), snapshot):
        assert current == previous


def rng_state(model):
    # MT19937 state is (name, key array, position, has_gauss, cached).
    # Drawing advances the position (and eventually the key), so all five
    # components must be captured and copied.
    name, key, pos, has_gauss, cached = model.random_state.get_state()
    return (name, key.copy(), pos, has_gauss, cached)


def assert_rng_unchanged(model, state):
    current = rng_state(model)
    assert len(current) == len(state)
    for a, b in zip(current, state):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b


def assert_rng_advanced(model, state):
    current = rng_state(model)
    changed = any(
        not np.array_equal(a, b) if isinstance(a, np.ndarray) else a != b
        for a, b in zip(current, state)
    )
    assert changed, "the random state did not advance"


# ---------------------------------------------------------------------------
# Identity-feature dataset
# ---------------------------------------------------------------------------


def _identity_workflow(learning_schedule):
    users = ["u0", "u1", "u2"]
    items = ["i0", "i1", "i2", "i3"]

    dataset = Dataset()
    dataset.fit(users, items)

    interactions, weights = dataset.build_interactions(
        [("u0", "i0"), ("u0", "i1"), ("u1", "i1"), ("u2", "i2")]
    )

    model = LightFM(
        no_components=5,
        learning_schedule=learning_schedule,
        loss="warp",
        random_state=np.random.RandomState(42),
    )
    model.fit_partial(interactions, sample_weight=weights, epochs=3)

    old_user_features = dataset.build_user_features([])
    old_item_features = dataset.build_item_features([])

    old_user_dim, old_item_dim = dataset.model_dimensions()

    pair_users = np.array([0, 0, 1, 2, 1], dtype=np.int32)
    pair_items = np.array([0, 2, 1, 3, 0], dtype=np.int32)
    old_scores = model.predict(
        pair_users,
        pair_items,
        user_features=old_user_features,
        item_features=old_item_features,
    )
    old_user_repr = model.get_user_representations(old_user_features)
    old_item_repr = model.get_item_representations(old_item_features)

    state_before = snapshot_model(model)
    mappings_before = snapshot_mappings(dataset)
    rng_before = rng_state(model)

    # Add two users, two items and one standalone side feature on each side.
    returned = dataset.fit_partial(
        users=["u3", "u4"],
        items=["i4", "i5"],
        user_features=["u_brand_a"],
        item_features=["i_genre_x"],
        model=model,
    )
    assert returned is dataset

    # Mappings grew, old indices never moved.
    assert dataset.interactions_shape() == (5, 6)
    assert dataset.model_dimensions() == (
        old_user_dim + 3,
        old_item_dim + 3,
    )
    for mapping, previous in zip(dataset.mapping()[:2], mappings_before[:2]):
        for key, value in previous.items():
            assert mapping[key] == value
    assert dataset.mapping()[0]["u3"] == 3
    assert dataset.mapping()[0]["u4"] == 4
    assert dataset.mapping()[2]["i4"] == 4
    assert dataset.mapping()[2]["i5"] == 5

    # Model capacity follows the mappings.
    assert model.user_embeddings.shape == (old_user_dim + 3, 5)
    assert model.item_embeddings.shape == (old_item_dim + 3, 5)

    assert_old_state_untouched(model, state_before, old_user_dim, old_item_dim)

    # New parameters carry the right optimiser-specific initial state.
    if learning_schedule == "adagrad":
        assert np.all(model.user_embedding_gradients[old_user_dim:] == 1)
        assert np.all(model.item_embedding_gradients[old_item_dim:] == 1)
        assert np.all(model.user_bias_gradients[old_user_dim:] == 1)
        assert np.all(model.item_bias_gradients[old_item_dim:] == 1)
        assert np.all(model.user_embedding_momentum[old_user_dim:] == 0)
        assert np.all(model.item_embedding_momentum[old_item_dim:] == 0)
    else:
        # AdaDelta starts every accumulator and momentum at zero; only the
        # embedding rows are random, so check the state arrays and biases.
        zero_state_names = [
            name for name in PARAM_NAMES if "embeddings" not in name
        ]
        for name in zero_state_names:
            array = getattr(model, name)
            if name.startswith("user"):
                assert np.all(array[old_user_dim:] == 0), name
            else:
                assert np.all(array[old_item_dim:] == 0), name

    assert np.all(model.user_biases[old_user_dim:] == 0)
    assert np.all(model.item_biases[old_item_dim:] == 0)

    # Scores for old feature rows are bitwise identical, no retraining done.
    new_user_features = dataset.build_user_features([])
    new_item_features = dataset.build_item_features([])
    assert new_user_features.shape == (5, old_user_dim + 3)
    assert new_item_features.shape == (6, old_item_dim + 3)

    same_scores = model.predict(
        pair_users,
        pair_items,
        user_features=new_user_features,
        item_features=new_item_features,
    )
    assert np.array_equal(same_scores, old_scores)

    # Representation interfaces read the same parameter arrays.
    user_biases, user_embeddings = model.get_user_representations()
    assert user_biases is model.user_biases
    assert user_embeddings is model.user_embeddings
    item_biases, item_embeddings = model.get_item_representations()
    assert item_biases is model.item_biases
    assert item_embeddings is model.item_embeddings

    new_user_repr = model.get_user_representations(new_user_features)
    new_item_repr = model.get_item_representations(new_item_features)
    assert np.array_equal(new_user_repr[0][:3], old_user_repr[0])
    assert np.array_equal(new_user_repr[1][:3], old_user_repr[1])
    assert np.array_equal(new_item_repr[0][:4], old_item_repr[0])
    assert np.array_equal(new_item_repr[1][:4], old_item_repr[1])

    # A successful extension consumed random draws (embedding initialisation).
    assert_rng_advanced(model, rng_before)

    # Continue training on interactions spanning old and new entities.
    more_interactions, more_weights = dataset.build_interactions(
        [("u3", "i4"), ("u4", "i5"), ("u0", "i5"), ("u3", "i0")]
    )
    model.fit_partial(
        more_interactions,
        sample_weight=more_weights,
        user_features=dataset.build_user_features([]),
        item_features=dataset.build_item_features([]),
        epochs=2,
    )

    # Training updated parameters; prediction still works on the full space.
    trained_scores = model.predict(
        pair_users,
        pair_items,
        user_features=dataset.build_user_features([]),
        item_features=dataset.build_item_features([]),
    )
    assert trained_scores.shape == old_scores.shape
    assert np.isfinite(trained_scores).all()

    return model, dataset


def test_identity_features_adagrad():
    _identity_workflow("adagrad")


def test_identity_features_adadelta():
    _identity_workflow("adadelta")


def test_multiple_extension_batches_keep_scores_stable():
    dataset = Dataset()
    dataset.fit(["u0", "u1"], ["i0", "i1"])

    interactions, _ = dataset.build_interactions([("u0", "i0"), ("u1", "i1")])
    model = LightFM(no_components=4, random_state=7)
    model.fit_partial(interactions, epochs=2)

    user_features = dataset.build_user_features([])
    item_features = dataset.build_item_features([])
    scores = model.predict(
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        user_features=user_features,
        item_features=item_features,
    )

    for batch in range(5):
        dataset.fit_partial(
            users=[f"u{2 + batch}"],
            items=[f"i{2 + batch}"],
            model=model,
        )
        rebuilt_user = dataset.build_user_features([])
        rebuilt_item = dataset.build_item_features([])
        current = model.predict(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            user_features=rebuilt_user,
            item_features=rebuilt_item,
        )
        assert np.array_equal(current, scores)

    assert dataset.interactions_shape() == (7, 7)


# ---------------------------------------------------------------------------
# Side-info-only dataset (identity features switched off)
# ---------------------------------------------------------------------------


def test_side_info_only_dataset():
    dataset = Dataset(
        user_identity_features=False, item_identity_features=False
    )
    dataset.fit(
        users=["u0", "u1", "u2"],
        items=["i0", "i1", "i2"],
        user_features=["ua", "ub"],
        item_features=["ia", "ib"],
    )

    user_features = dataset.build_user_features(
        [
            ("u0", ["ua"]),
            ("u1", ["ub"]),
            ("u2", ["ua", "ub"]),
        ],
        normalize=False,
    )
    item_features = dataset.build_item_features(
        [
            ("i0", ["ia"]),
            ("i1", ["ib"]),
            ("i2", ["ia", "ib"]),
        ],
        normalize=False,
    )

    interactions, weights = dataset.build_interactions(
        [("u0", "i0"), ("u1", "i2"), ("u2", "i1")]
    )

    model = LightFM(no_components=6, random_state=123)
    model.fit_partial(
        interactions,
        sample_weight=weights,
        user_features=user_features,
        item_features=item_features,
        epochs=2,
    )

    old_user_dim, old_item_dim = dataset.model_dimensions()
    state_before = snapshot_model(model)
    mappings_before = snapshot_mappings(dataset)
    rng_before = rng_state(model)

    pair_users = np.array([0, 1, 2], dtype=np.int32)
    pair_items = np.array([0, 1, 2], dtype=np.int32)
    old_scores = model.predict(
        pair_users,
        pair_items,
        user_features=user_features,
        item_features=item_features,
    )

    # New entities rely exclusively on new side features.
    dataset.fit_partial(
        users=["u3"],
        items=["i3"],
        user_features=["uc"],
        item_features=["ic"],
        model=model,
    )

    assert dataset.model_dimensions() == (
        old_user_dim + 1,
        old_item_dim + 1,
    )
    assert dataset.user_features_shape() == (4, old_user_dim + 1)
    assert dataset.item_features_shape() == (4, old_item_dim + 1)
    assert_old_state_untouched(model, state_before, old_user_dim, old_item_dim)
    for current, previous in zip(dataset.mapping(), mappings_before):
        for key, value in previous.items():
            assert current[key] == value

    new_user_features = dataset.build_user_features(
        [
            ("u0", ["ua"]),
            ("u1", ["ub"]),
            ("u2", ["ua", "ub"]),
            ("u3", ["uc"]),
        ],
        normalize=False,
    )
    new_item_features = dataset.build_item_features(
        [
            ("i0", ["ia"]),
            ("i1", ["ib"]),
            ("i2", ["ia", "ib"]),
            ("i3", ["ic"]),
        ],
        normalize=False,
    )

    same_scores = model.predict(
        pair_users,
        pair_items,
        user_features=new_user_features,
        item_features=new_item_features,
    )
    assert np.array_equal(same_scores, old_scores)
    assert_rng_advanced(model, rng_before)

    # Train on the new entity and confirm learning touches the new feature.
    new_interactions, new_weights = dataset.build_interactions(
        [("u3", "i3"), ("u0", "i3")]
    )
    new_embedding_before = model.user_embeddings[old_user_dim].copy()
    model.fit_partial(
        new_interactions,
        sample_weight=new_weights,
        user_features=new_user_features,
        item_features=new_item_features,
        epochs=3,
    )
    assert not np.array_equal(
        model.user_embeddings[old_user_dim], new_embedding_before
    )


# ---------------------------------------------------------------------------
# New-parameter initialisation matches a fresh model
# ---------------------------------------------------------------------------


def test_new_embeddings_use_the_same_initialisation_as_fit():
    dataset = Dataset()
    dataset.fit(["u0"], ["i0"])
    interactions, _ = dataset.build_interactions([("u0", "i0")])

    seed = 2026
    model = LightFM(no_components=8, random_state=np.random.RandomState(seed))
    model.fit_partial(interactions)

    dataset.fit_partial(users=["u1", "u2"], items=["i1", "i2"], model=model)

    # Replay the RNG draws a fresh fit of four features would make.
    reference = np.random.RandomState(seed)
    # The first fit drew one item and one user embedding.
    reference.rand(1, 8)
    reference.rand(1, 8)
    expected_new_items = ((reference.rand(2, 8) - 0.5) / 8).astype(np.float32)
    expected_new_users = ((reference.rand(2, 8) - 0.5) / 8).astype(np.float32)

    assert np.allclose(model.item_embeddings[1:], expected_new_items)
    assert np.allclose(model.user_embeddings[1:], expected_new_users)


# ---------------------------------------------------------------------------
# No-op behaviour
# ---------------------------------------------------------------------------


def test_known_entries_are_a_noop():
    dataset = Dataset()
    dataset.fit(["u0", "u1"], ["i0", "i1"], ["uf"], ["if"])

    interactions, _ = dataset.build_interactions([("u0", "i0")])
    model = LightFM(random_state=99)
    model.fit_partial(interactions)

    mappings_before = snapshot_mappings(dataset)
    refs_before = snapshot_model_refs(model)
    rng_before = rng_state(model)

    dataset.fit_partial(
        users=["u0", "u1"],
        items=["i0", "i1"],
        user_features=["uf"],
        item_features=["if"],
        model=model,
    )

    assert_mappings_untouched(dataset, mappings_before)
    assert_refs_unchanged(model, refs_before)
    assert_rng_unchanged(model, rng_before)

    # Omitting the model argument remains a no-op as well.
    dataset.fit_partial(users=["u0"], items=["i0"])
    assert_mappings_untouched(dataset, mappings_before)


def test_extend_dimensions_noop_does_not_move_rng():
    dataset = Dataset()
    dataset.fit(["u0"], ["i0"])
    interactions, _ = dataset.build_interactions([("u0", "i0")])
    model = LightFM(random_state=5)
    model.fit_partial(interactions)

    refs_before = snapshot_model_refs(model)
    rng_before = rng_state(model)
    model.extend_dimensions(
        no_item_features=model.item_embeddings.shape[0],
        no_user_features=model.user_embeddings.shape[0],
    )
    assert_refs_unchanged(model, refs_before)
    assert_rng_unchanged(model, rng_before)


# ---------------------------------------------------------------------------
# Atomicity and validation
# ---------------------------------------------------------------------------


def test_iterator_failure_leaves_everything_untouched():
    dataset = Dataset()
    dataset.fit(["u0"], ["i0"])
    interactions, _ = dataset.build_interactions([("u0", "i0")])

    model = LightFM(random_state=11)
    model.fit_partial(interactions)

    mappings_before = snapshot_mappings(dataset)
    refs_before = snapshot_model_refs(model)
    rng_before = rng_state(model)

    def broken_users():
        yield "u1"
        raise RuntimeError("iterator exploded halfway through")

    with pytest.raises(RuntimeError, match="exploded"):
        dataset.fit_partial(users=broken_users(), items=["i1"], model=model)

    assert_mappings_untouched(dataset, mappings_before)
    assert_refs_unchanged(model, refs_before)
    assert_rng_unchanged(model, rng_before)

    # Same guarantee without a model.
    with pytest.raises(RuntimeError, match="exploded"):
        dataset.fit_partial(users=broken_users(), items=["i1"])
    assert_mappings_untouched(dataset, mappings_before)


def test_allocation_failure_rolls_back_mappings_model_and_rng(monkeypatch):
    dataset = Dataset()
    dataset.fit(["u0", "u1"], ["i0", "i1"])
    interactions, _ = dataset.build_interactions(
        [("u0", "i0"), ("u1", "i1")]
    )
    model = LightFM(random_state=13)
    model.fit_partial(interactions)

    mappings_before = snapshot_mappings(dataset)
    refs_before = snapshot_model_refs(model)
    rng_before = rng_state(model)

    import lightfm.lightfm as lightfm_module

    def boom(*args, **kwargs):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(lightfm_module.np, "concatenate", boom)

    with pytest.raises(MemoryError):
        dataset.fit_partial(users=["u2"], items=["i2"], model=model)

    monkeypatch.undo()

    assert_mappings_untouched(dataset, mappings_before)
    assert_refs_unchanged(model, refs_before)
    assert_rng_unchanged(model, rng_before)

    # The model is still fully usable after the failed extension.
    assert np.isfinite(
        model.predict(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32)
        )
    ).all()


def test_dimension_mismatch_is_rejected_before_any_change():
    dataset = Dataset()
    dataset.fit(["u0", "u1"], ["i0", "i1"])

    # A model fitted on a smaller, unrelated feature space.
    small_model = LightFM(random_state=3)
    small_model.fit_partial(sp.coo_matrix((1, 1), dtype=np.float32))

    mappings_before = snapshot_mappings(dataset)
    refs_before = snapshot_model_refs(small_model)
    rng_before = rng_state(small_model)

    with pytest.raises(ValueError, match="smaller than"):
        dataset.fit_partial(users=["u2"], items=["i2"], model=small_model)

    assert_mappings_untouched(dataset, mappings_before)
    assert_refs_unchanged(small_model, refs_before)
    assert_rng_unchanged(small_model, rng_before)


def test_unfitted_model_is_rejected():
    dataset = Dataset()
    dataset.fit(["u0"], ["i0"])
    model = LightFM()

    with pytest.raises(ValueError, match="not been fitted"):
        dataset.fit_partial(users=["u1"], items=["i1"], model=model)

    with pytest.raises(ValueError, match="fit the model"):
        model.extend_dimensions(no_item_features=3, no_user_features=3)


def test_wrong_model_type_and_shrink_are_rejected():
    dataset = Dataset()
    dataset.fit(["u0"], ["i0"])
    with pytest.raises(TypeError, match="LightFM"):
        dataset.fit_partial(users=["u1"], model="not a model")

    interactions, _ = dataset.build_interactions([("u0", "i0")])
    model = LightFM()
    model.fit_partial(interactions)

    with pytest.raises(ValueError, match="Cannot shrink"):
        model.extend_dimensions(
            no_item_features=1,
            no_user_features=model.user_embeddings.shape[0] - 1,
        )


def test_fit_partial_without_model_keeps_legacy_behaviour():
    dataset = Dataset()
    dataset.fit_partial(["u0"], ["i0"])
    dataset.fit_partial(["u1", "u2"], ["i1"], user_features=["uf"])

    assert dataset.interactions_shape() == (3, 2)
    user_id_map, user_feature_map, item_id_map, item_feature_map = (
        dataset.mapping()
    )
    assert user_id_map == {"u0": 0, "u1": 1, "u2": 2}
    assert item_id_map == {"i0": 0, "i1": 1}
    # Identity features plus the standalone feature.
    assert user_feature_map == {"u0": 0, "u1": 1, "u2": 2, "uf": 3}
    assert item_feature_map == {"i0": 0, "i1": 1}


def test_extended_model_survives_pickle_roundtrip():
    import pickle

    dataset = Dataset()
    dataset.fit(["u0", "u1"], ["i0", "i1"])
    interactions, weights = dataset.build_interactions([("u0", "i0")])
    model = LightFM(no_components=4, random_state=7)
    model.fit_partial(interactions, sample_weight=weights, epochs=2)

    dataset.fit_partial(users=["u2"], items=["i2"], model=model)
    more_interactions, more_weights = dataset.build_interactions(
        [("u2", "i2"), ("u0", "i2")]
    )
    model.fit_partial(
        more_interactions,
        sample_weight=more_weights,
        user_features=dataset.build_user_features([]),
        item_features=dataset.build_item_features([]),
    )

    user_features = dataset.build_user_features([])
    item_features = dataset.build_item_features([])
    scores_before = model.predict(
        np.array([0, 2], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        user_features=user_features,
        item_features=item_features,
    )

    restored = pickle.loads(pickle.dumps(model))

    for name in PARAM_NAMES:
        assert np.array_equal(
            getattr(restored, name), getattr(model, name)
        )

    scores_after = restored.predict(
        np.array([0, 2], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        user_features=user_features,
        item_features=item_features,
    )
    assert np.array_equal(scores_after, scores_before)


def test_extension_works_with_bpr_and_all_losses():
    losses = ("bpr", "logistic", "warp", "warp-kos")
    for loss in losses:
        dataset = Dataset()
        dataset.fit(["u0", "u1", "u2", "u4"], ["i0", "i1", "i2", "i4"])
        interactions, weights = dataset.build_interactions(
            [
                ("u0", "i0"),
                ("u0", "i1"),
                ("u1", "i1"),
                ("u1", "i2"),
                ("u2", "i2"),
                ("u2", "i0"),
                ("u4", "i4"),
            ]
        )
        # warp-kos needs a gentler schedule on small synthetic data.
        model = LightFM(
            loss=loss,
            no_components=4,
            random_state=4,
            learning_rate=0.005 if loss == "warp-kos" else 0.05,
        )
        # warp-kos does not implement sample weights.
        model.fit_partial(
            interactions,
            sample_weight=None if loss == "warp-kos" else weights,
            epochs=1,
        )

        old_scores = model.predict(
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
        )

        dataset.fit_partial(users=["u3"], items=["i3"], model=model)

        # Extension alone must not move old scores.
        assert np.array_equal(
            model.predict(
                np.array([0, 1, 2], dtype=np.int32),
                np.array([0, 1, 2], dtype=np.int32),
            ),
            old_scores,
        )

        # Continued training exercises the new parameters under this loss.
        new_interactions, new_weights = dataset.build_interactions(
            [("u3", "i3")]
        )
        model.fit_partial(
            new_interactions,
            sample_weight=None if loss == "warp-kos" else new_weights,
            user_features=dataset.build_user_features([]),
            item_features=dataset.build_item_features([]),
            epochs=2,
        )
        assert np.isfinite(
            model.predict(
                np.array([3], dtype=np.int32),
                np.array([3], dtype=np.int32),
                user_features=dataset.build_user_features([]),
                item_features=dataset.build_item_features([]),
            )
        ).all()
