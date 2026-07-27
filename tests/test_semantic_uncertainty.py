import numpy as np
import pytest

from hovsg.utils.uncertainty import (
    build_synonym_eligibility_mask,
    compute_semantic_margin_uncertainty,
)


def test_synonym_mask_normalizes_text_and_excludes_self_and_synonyms():
    text_feats = np.array([[10.0, 0.0], [9.9, 0.1], [0.0, 4.0]])

    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(np.diag(mask), [False, False, False])
    assert not mask[0, 1]
    assert mask[0, 2]


def test_strong_preference_uses_best_distinct_competitor():
    text_feats = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]])
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.array([2.0, 0.0]), text_feats, 0, mask
    )

    assert result["label_cos_sim"] == pytest.approx(1.0)
    assert result["runner_up_idx"] == 2
    assert result["runner_up_cos_sim"] == pytest.approx(0.0)
    assert result["semantic_margin"] == pytest.approx(1.0)
    assert result["c_sem"] == pytest.approx(1.0)
    assert result["u_sem"] < 1e-10


def test_equidistant_distinct_classes_have_half_uncertainty():
    text_feats = np.eye(2)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.ones(2), text_feats, 0, mask
    )

    assert result["semantic_margin"] == pytest.approx(0.0)
    assert result["c_sem"] == pytest.approx(0.5)
    assert result["u_sem"] == pytest.approx(0.5)


def test_zero_embedding_uses_conservative_override():
    text_feats = np.eye(3)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.zeros(3), text_feats, 1, mask
    )

    assert result == {
        "label_cos_sim": 0.0,
        "runner_up_idx": None,
        "runner_up_cos_sim": 0.0,
        "semantic_margin": 0.0,
        "c_sem": 0.0,
        "u_sem": 1.0,
    }


def test_empty_eligible_set_warns_and_uses_explicit_fallback(caplog):
    text_feats = np.array([[1.0, 0.0], [1.0, 0.0]])
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.array([1.0, 0.0]), text_feats, 0, mask
    )

    assert "empty-eligible-set fallback" in caplog.text
    assert result["label_cos_sim"] == pytest.approx(1.0)
    assert result["runner_up_idx"] is None
    assert result["runner_up_cos_sim"] is None
    assert result["semantic_margin"] is None
    assert result["c_sem"] == 1.0
    assert result["u_sem"] == 0.0


def test_precomputed_similarity_vector_is_reused():
    text_feats = np.eye(2)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.array([1.0, 0.0]),
        text_feats,
        0,
        mask,
        logit_scale=1.0,
        similarity=np.array([0.4, 0.3]),
    )

    assert result["label_cos_sim"] == pytest.approx(0.4)
    assert result["runner_up_cos_sim"] == pytest.approx(0.3)
    assert result["semantic_margin"] == pytest.approx(0.1)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_synonym_threshold_must_be_between_zero_and_one(tau):
    with pytest.raises(ValueError, match="strictly between"):
        build_synonym_eligibility_mask(np.eye(2), tau)
