"""
Incremental training with Dataset.fit_partial(..., model=model).

This script demonstrates the joint data-mapping / model-capacity extension:

1. Fit a dataset and a LightFM model on the first batch of users and items.
2. Append new users, items and side features; the model grows in the same
   call without losing learned embeddings, biases or optimiser state.
3. Continue training on interactions that mix old and new entities.
4. Verify that predictions for the original entities are unchanged by the
   extension itself, and that a batch of already-known entries is a no-op.

Run with: python3 -I examples/incremental_training.py
"""

import numpy as np

from lightfm.data import Dataset
from lightfm.lightfm import LightFM


def main():
    # --- Batch 1: initial catalogue -----------------------------------------
    dataset = Dataset()
    dataset.fit(users=["alice", "bob"], items=["hat", "scarf"])

    interactions, weights = dataset.build_interactions(
        [("alice", "hat"), ("alice", "scarf"), ("bob", "hat")]
    )

    model = LightFM(loss="warp", no_components=10, random_state=42)
    model.fit_partial(interactions, sample_weight=weights, epochs=5)

    user_features = dataset.build_user_features([])
    item_features = dataset.build_item_features([])
    original_scores = model.predict(
        np.array([0, 0, 1], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        user_features=user_features,
        item_features=item_features,
    )

    # --- Batch 2: new users, items and side features arrive -----------------
    dataset.fit_partial(
        users=["carol"],
        items=["gloves"],
        user_features=["vip"],
        item_features=["winter"],
        model=model,
    )

    print("Extended dimensions:", dataset.model_dimensions())
    print("Model embedding shapes:", model.user_embeddings.shape,
          model.item_embeddings.shape)

    # Old feature rows still produce exactly the same scores.
    unchanged_scores = model.predict(
        np.array([0, 0, 1], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        user_features=dataset.build_user_features([]),
        item_features=dataset.build_item_features([]),
    )
    assert np.array_equal(original_scores, unchanged_scores)

    # --- Batch 3: keep training on the grown feature space ------------------
    new_interactions, new_weights = dataset.build_interactions(
        [("carol", "gloves"), ("carol", "hat"), ("alice", "gloves")]
    )
    model.fit_partial(
        new_interactions,
        sample_weight=new_weights,
        user_features=dataset.build_user_features([]),
        item_features=dataset.build_item_features([]),
        epochs=5,
    )

    scores = model.predict(
        np.array([2, 2, 2], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        user_features=dataset.build_user_features([]),
        item_features=dataset.build_item_features([]),
    )
    print("Carol's scores after continued training:", scores)

    # Re-feeding only known entries changes nothing, including the RNG.
    state_before = model.random_state.get_state()
    dataset.fit_partial(
        users=["alice", "bob", "carol"],
        items=["hat", "scarf", "gloves"],
        user_features=["vip"],
        item_features=["winter"],
        model=model,
    )
    assert all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for a, b in zip(state_before, model.random_state.get_state())
    )
    print("Known-only batch was a no-op. Done.")


if __name__ == "__main__":
    main()
