import numpy as np
import pytest

from hovsg.utils.uncertainty import compute_semantic_uncertainty


def test_strong_class_preference_has_low_uncertainty():
    text_feats = np.eye(3)

    similarity, uncertainty, label_cos_sim = compute_semantic_uncertainty(
        np.array([2.0, 0.0, 0.0]),
        text_feats,
        label_idx=0,
    )

    np.testing.assert_allclose(similarity, [1.0, 0.0, 0.0])
    assert uncertainty < 1e-10
    assert label_cos_sim == pytest.approx(1.0)


def test_uniform_compatibility_has_maximum_uncertainty():
    text_feats = np.eye(3)

    similarity, uncertainty = compute_semantic_uncertainty(
        np.ones(3),
        text_feats,
    )

    assert np.ptp(similarity) == pytest.approx(0.0)
    assert uncertainty == pytest.approx(1.0)


def test_zero_embedding_uses_conservative_defaults():
    similarity, uncertainty, label_cos_sim = compute_semantic_uncertainty(
        np.zeros(3),
        np.eye(3),
        label_idx=1,
    )

    np.testing.assert_array_equal(similarity, np.zeros(3))
    assert uncertainty == 1.0
    assert label_cos_sim == 0.0


def test_text_features_are_normalized_defensively():
    similarity, _, label_cos_sim = compute_semantic_uncertainty(
        np.array([10.0, 0.0]),
        np.array([[5.0, 0.0], [0.0, 2.0]]),
        label_idx=0,
    )

    np.testing.assert_allclose(similarity, [1.0, 0.0])
    assert label_cos_sim == pytest.approx(1.0)
