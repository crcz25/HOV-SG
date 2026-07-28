import numpy as np
import pytest

from scipy.special import expit

from hovsg.utils.uncertainty import (
    MEMBERSHIP_FIELDS,
    SEMANTIC_FIELDS,
    build_synonym_eligibility_mask,
    compute_semantic_margin_uncertainty,
    compute_vocabulary_membership,
    normalize_rows,
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
    assert result["p_sem"] == pytest.approx(1.0)
    assert result["u_sem"] < 1e-10


def test_equidistant_distinct_classes_have_half_uncertainty():
    text_feats = np.eye(2)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.ones(2), text_feats, 0, mask
    )

    assert result["semantic_margin"] == pytest.approx(0.0)
    assert result["p_sem"] == pytest.approx(0.5)
    assert result["u_sem"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "embedding", [np.zeros(3), np.array([np.nan, 0.0, 0.0]), np.array([np.inf, 0.0, 0.0])]
)
def test_degenerate_embedding_is_undefined_not_a_fabricated_probability(embedding):
    """cos(v, t) is undefined here, so sigma(alpha m) must not be reported.

    The previous implementation returned margin 0.0 with P_sem 0.0, which are
    mutually inconsistent: sigma(alpha * 0) is 0.5, not 0.
    """
    text_feats = np.eye(3)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(embedding, text_feats, 1, mask)

    assert result == dict.fromkeys(SEMANTIC_FIELDS)
    assert result["p_sem"] is None and result["u_sem"] is None


def test_empty_eligible_set_is_undefined_and_warns(caplog):
    """Every competitor excluded as a synonym -> max over an empty set."""
    text_feats = np.array([[1.0, 0.0], [1.0, 0.0]])
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.array([1.0, 0.0]), text_feats, 0, mask
    )

    assert "excluded as a synonym" in caplog.text
    # The observed cosine to the assigned label is still reported; only the
    # margin and the probability derived from it are undefined.
    assert result["label_cos_sim"] == pytest.approx(1.0)
    assert result["runner_up_idx"] is None
    assert result["runner_up_cos_sim"] is None
    assert result["semantic_margin"] is None
    assert result["p_sem"] is None
    assert result["u_sem"] is None


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


def test_vocabulary_membership_uses_stable_log_partitions():
    result = compute_vocabulary_membership(
        np.array([1.0, 0.0]),
        np.eye(2),
        np.array([[0.0, 1.0]]),
        logit_scale=1000.0,
    )

    assert result["vocab_log_partition"] == pytest.approx(1000.0)
    assert result["negative_log_partition"] == pytest.approx(0.0)
    assert result["p_mem"] == pytest.approx(1.0)
    assert result["u_mem"] == pytest.approx(0.0)


def test_vocabulary_membership_degenerate_embedding_is_undefined():
    result = compute_vocabulary_membership(
        np.zeros(2), np.eye(2), np.eye(2), logit_scale=100.0
    )

    # P_mem = 0.0 would assert certain out-of-vocabulary status for an object
    # that merely has no usable embedding.
    assert result == dict.fromkeys(MEMBERSHIP_FIELDS)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_synonym_threshold_must_be_between_zero_and_one(tau):
    with pytest.raises(ValueError, match="strictly between"):
        build_synonym_eligibility_mask(np.eye(2), tau)


# ---------------------------------------------------------------------------
# Exact numerical agreement with the published equations
# ---------------------------------------------------------------------------


def reference_semantic(embedding, text_feats, label_idx, tau, alpha):
    """Independent re-derivation of eq. (semantic) from its definition."""
    v = np.asarray(embedding, float)
    v = v / np.linalg.norm(v)
    t = normalize_rows(text_feats)
    cos_v = t @ v
    cos_tt = t @ t[label_idx]
    competitors = [
        c for c in range(len(t)) if c != label_idx and cos_tt[c] < tau
    ]
    margin = cos_v[label_idx] - max(cos_v[c] for c in competitors)
    return margin, float(expit(alpha * margin))


def test_semantic_matches_equation_on_a_worked_three_class_example():
    # Deliberately unnormalized inputs: the implementation must normalize.
    text_feats = np.array([[2.0, 0.0], [1.9, 0.62], [0.0, 3.0]])
    embedding = np.array([0.6, 0.2])
    tau, alpha = 0.95, 100.0

    mask = build_synonym_eligibility_mask(text_feats, tau)
    result = compute_semantic_margin_uncertainty(
        embedding, text_feats, 0, mask, logit_scale=alpha
    )
    expected_margin, expected_p = reference_semantic(
        embedding, text_feats, 0, tau, alpha
    )

    assert result["semantic_margin"] == pytest.approx(expected_margin, abs=1e-12)
    assert result["p_sem"] == pytest.approx(expected_p, abs=1e-12)
    assert result["u_sem"] == pytest.approx(1.0 - expected_p, abs=1e-12)


def test_tau_selects_which_class_competes():
    """Raising tau admits the near-synonym and shrinks the margin."""
    # Class 1 is a near-synonym of class 0 (cos ~ 0.995); class 2 is distinct.
    text_feats = np.array([[1.0, 0.0, 0.0], [0.995, 0.0999, 0.0], [0.0, 0.0, 1.0]])
    embedding = np.array([0.9, 0.3, 0.2])

    strict = compute_semantic_margin_uncertainty(
        embedding, text_feats, 0, build_synonym_eligibility_mask(text_feats, 0.9)
    )
    permissive = compute_semantic_margin_uncertainty(
        embedding, text_feats, 0, build_synonym_eligibility_mask(text_feats, 0.999)
    )

    assert strict["runner_up_idx"] == 2
    assert permissive["runner_up_idx"] == 1
    assert permissive["semantic_margin"] < strict["semantic_margin"]


def test_clip_logit_scale_is_applied_to_the_margin_not_the_cosine():
    text_feats = np.eye(2)
    mask = build_synonym_eligibility_mask(text_feats, 0.9)
    embedding = np.array([0.8, 0.6])  # margin = 0.8 - 0.6 = 0.2

    for alpha in (1.0, 10.0, 100.0):
        result = compute_semantic_margin_uncertainty(
            embedding, text_feats, 0, mask, logit_scale=alpha
        )
        assert result["semantic_margin"] == pytest.approx(0.2)
        assert result["p_sem"] == pytest.approx(float(expit(alpha * 0.2)))

    # At CLIP's converged scale a 0.2 margin is effectively certain.
    saturated = compute_semantic_margin_uncertainty(
        embedding, text_feats, 0, mask, logit_scale=100.0
    )
    assert saturated["p_sem"] > 1.0 - 1e-8


def test_assume_normalized_matches_defensive_normalization():
    text_feats = np.array([[2.0, 0.0], [0.0, 5.0], [1.0, 1.0]])
    normalized = normalize_rows(text_feats)
    mask = build_synonym_eligibility_mask(text_feats, 0.9)
    embedding = np.array([0.7, 0.4])

    defensive = compute_semantic_margin_uncertainty(embedding, text_feats, 0, mask)
    fast = compute_semantic_margin_uncertainty(
        embedding, normalized, 0, mask, assume_normalized=True
    )

    assert fast["semantic_margin"] == pytest.approx(defensive["semantic_margin"])
    assert fast["p_sem"] == pytest.approx(defensive["p_sem"])


def reference_membership(embedding, text_feats, negative_feats, alpha):
    """Direct evaluation of eq. (membership) as written, without log-sum-exp."""
    v = np.asarray(embedding, float)
    v = v / np.linalg.norm(v)
    z_c = np.exp(alpha * (normalize_rows(text_feats) @ v)).sum()
    z_n = np.exp(alpha * (normalize_rows(negative_feats) @ v)).sum()
    return z_c / (z_c + z_n)


def test_membership_matches_the_ratio_of_partition_sums():
    text_feats = np.array([[1.0, 0.0, 0.0], [0.9, 0.4, 0.0]])
    negative_feats = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.7, 0.7]])
    embedding = np.array([0.8, 0.5, 0.1])
    alpha = 5.0  # small enough that the direct form does not overflow

    result = compute_vocabulary_membership(
        embedding, text_feats, negative_feats, logit_scale=alpha
    )

    assert result["p_mem"] == pytest.approx(
        reference_membership(embedding, text_feats, negative_feats, alpha), abs=1e-12
    )
    assert result["u_mem"] == pytest.approx(1.0 - result["p_mem"])


def test_membership_uses_the_full_vocabulary_mass_not_only_the_best_class():
    """Adding more in-vocabulary classes must raise P_mem."""
    negative_feats = np.array([[0.0, 1.0]])
    embedding = np.array([1.0, 0.0])
    alpha = 5.0

    one_class = compute_vocabulary_membership(
        embedding, np.array([[1.0, 0.0]]), negative_feats, logit_scale=alpha
    )
    # A second, equally similar class doubles Z_C while leaving Z_N unchanged.
    two_classes = compute_vocabulary_membership(
        embedding, np.array([[1.0, 0.0], [1.0, 0.0]]), negative_feats, logit_scale=alpha
    )

    assert two_classes["p_mem"] > one_class["p_mem"]
    z_c = 2 * np.exp(alpha * 1.0)
    z_n = np.exp(alpha * 0.0)
    assert two_classes["p_mem"] == pytest.approx(z_c / (z_c + z_n))


def test_membership_is_numerically_stable_at_the_clip_logit_scale():
    """alpha=100 overflows exp() in float64; log-sum-exp must survive it."""
    text_feats = np.array([[1.0, 0.0], [0.99, 0.14]])
    negative_feats = np.array([[0.0, 1.0], [-1.0, 0.0]])

    result = compute_vocabulary_membership(
        np.array([1.0, 0.0]), text_feats, negative_feats, logit_scale=100.0
    )

    assert np.isfinite(result["vocab_log_partition"])
    assert np.isfinite(result["negative_log_partition"])
    assert 0.0 <= result["p_mem"] <= 1.0
    assert result["p_mem"] > 1.0 - 1e-12  # strongly in-vocabulary


def test_out_of_vocabulary_object_shifts_mass_to_the_negative_bank():
    text_feats = np.array([[1.0, 0.0, 0.0], [0.9, 0.44, 0.0]])
    negative_feats = np.array([[0.0, 0.0, 1.0], [0.0, 0.3, 0.95]])
    alpha = 20.0

    in_vocab = compute_vocabulary_membership(
        np.array([1.0, 0.0, 0.0]), text_feats, negative_feats, logit_scale=alpha
    )
    out_of_vocab = compute_vocabulary_membership(
        np.array([0.0, 0.0, 1.0]), text_feats, negative_feats, logit_scale=alpha
    )

    assert in_vocab["p_mem"] > 0.99
    assert out_of_vocab["p_mem"] < 0.01
    assert out_of_vocab["u_mem"] > 0.99


def test_membership_and_semantic_confidence_are_distinct_quantities():
    """A near-uniform in-vocabulary profile: high P_mem, low P_sem."""
    text_feats = np.array([[1.0, 0.0], [0.0, 1.0]])
    negative_feats = np.array([[-1.0, 0.0], [0.0, -1.0]])
    embedding = np.array([1.0, 1.0])  # equidistant from both classes

    membership = compute_vocabulary_membership(
        embedding, text_feats, negative_feats, logit_scale=20.0
    )
    semantic = compute_semantic_margin_uncertainty(
        embedding,
        text_feats,
        0,
        build_synonym_eligibility_mask(text_feats, 0.9),
        logit_scale=20.0,
    )

    assert membership["p_mem"] > 0.99  # clearly inside the vocabulary
    assert semantic["p_sem"] == pytest.approx(0.5)  # but the label is a coin flip


@pytest.mark.parametrize(
    ("text_feats", "negative_feats"),
    [
        (np.empty((0, 2)), np.eye(2)),
        (np.eye(2), np.empty((0, 2))),
    ],
)
def test_membership_rejects_empty_vocabulary_or_negative_bank(text_feats, negative_feats):
    with pytest.raises(ValueError, match="non-empty"):
        compute_vocabulary_membership(np.array([1.0, 0.0]), text_feats, negative_feats)


def test_no_entropy_is_used_anywhere_in_the_uncertainty_implementation():
    """Regression guard: the signals are margin/partition based, never entropy."""
    import pathlib

    import hovsg.utils.uncertainty as uncertainty_module

    sources = [
        pathlib.Path(uncertainty_module.__file__),
        pathlib.Path(uncertainty_module.__file__).parent / "cross_view_consistency.py",
        pathlib.Path(uncertainty_module.__file__).parent / "detection_uncertainty.py",
        pathlib.Path(uncertainty_module.__file__).parent / "negative_labels.py",
        pathlib.Path(uncertainty_module.__file__).parents[1] / "graph" / "object.py",
        pathlib.Path(uncertainty_module.__file__).parents[1] / "graph" / "graph.py",
    ]
    for source in sources:
        text = source.read_text(encoding="utf-8").lower()
        assert "entropy" not in text, f"entropy-based computation found in {source}"
