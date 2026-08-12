"""Numerical agreement with uncertainty_signals.tex, signal by signal.

Every expected value in this module is derived by hand from the equations in
the paper and written as an explicit arithmetic expression, so a failure means
the implementation left the paper rather than that two implementations of the
same idea disagree.

    Detection Confidence     P_det  = s_i in [0, 1]
    Semantic Uncertainty     P_sem  = sigma(alpha * m_i),  U_sem = 1 - P_sem
                             m_i    = cos(v, t_l) - max_{c: cos(t_c,t_l)<tau} cos(v, t_c)
    Label Coherence          P_coh  = sigma(alpha * m'_i)
                             m'_i   = cos(v, mu_l) - max_{c: cos(t_c,t_l)<tau} cos(v, mu_c)
                             P_sem_bar = min(P_sem, P_coh)
    Vocabulary Membership    P_mem  = L_C / (L_C + L_N)
                             L_C    = (1/|C|) sum_C exp(alpha cos), L_N likewise over N
    Cross-view Consistency   P_view = (1 + c_bar) / 2
                             ||m||^2 = 1/n + (1 - 1/n) c_bar
"""

import numpy as np
import pytest

from hovsg.utils.cross_view_consistency import (
    accumulate_unit_embeddings,
    compute_cross_view_consistency,
)
from hovsg.utils.detection_uncertainty import (
    accumulate_confidence,
    confidence_from_sum,
    finalize_confidence_array,
    mask_predicted_iou,
    uncertainty_from_confidence,
)
from hovsg.utils.uncertainty import (
    build_synonym_eligibility_mask,
    combine_semantic_confidence,
    compute_label_coherence_uncertainty,
    compute_semantic_margin_uncertainty,
    compute_vocabulary_membership,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Detection Confidence: s_i in [0, 1], an estimate of the probability that the
# detection corresponds to a real object. U_det is its complement.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [0.05, 0.5, 0.97])
def test_detection_confidence_is_the_detector_score_and_its_complement(score):
    assert mask_predicted_iou({"predicted_iou": score}) == pytest.approx(score)
    assert uncertainty_from_confidence(score) == pytest.approx(1.0 - score)


def test_detection_confidence_pools_over_the_points_of_the_object():
    """Two points, one seen by a high-score mask and one by a low-score mask."""
    sum_conf = np.zeros((2, 1))
    counter = np.zeros((2, 1))
    accumulate_confidence(sum_conf, counter, np.array([0]), 0.9)
    accumulate_confidence(sum_conf, counter, np.array([1]), 0.5)

    per_point = finalize_confidence_array(sum_conf, counter).reshape(-1)
    np.testing.assert_allclose(per_point, [0.9, 0.5])
    assert confidence_from_sum(per_point.sum(), per_point.size) == pytest.approx(0.7)


def test_detection_confidence_is_undefined_without_evidence():
    assert confidence_from_sum(0.0, 0) is None


# ---------------------------------------------------------------------------
# Semantic Uncertainty, eq. (semantic)
# ---------------------------------------------------------------------------


def test_semantic_worked_example():
    """v = (0.8, 0.6) against orthogonal class embeddings, alpha = 10.

    cos(v, t_0) = 0.8 and cos(v, t_1) = 0.6, and the two classes are not
    synonyms, so m = 0.2 and P_sem = sigma(2).
    """
    text_feats = np.eye(2)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)

    result = compute_semantic_margin_uncertainty(
        np.array([0.8, 0.6]), text_feats, label_idx=0, eligibility_mask=mask,
        logit_scale=10.0,
    )

    assert result["label_cos_sim"] == pytest.approx(0.8)
    assert result["runner_up_cos_sim"] == pytest.approx(0.6)
    assert result["semantic_margin"] == pytest.approx(0.2)
    assert result["p_sem"] == pytest.approx(sigmoid(2.0))
    assert result["u_sem"] == pytest.approx(1.0 - sigmoid(2.0))


def test_semantic_strong_agreement_and_strong_disagreement():
    text_feats = np.eye(2)
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)
    alpha = 100.0

    # The embedding sits on class 0: m = 1 - 0 = 1, P_sem = sigma(100) ~ 1.
    decisive = compute_semantic_margin_uncertainty(
        np.array([1.0, 0.0]), text_feats, 0, mask, logit_scale=alpha
    )
    assert decisive["semantic_margin"] == pytest.approx(1.0)
    assert decisive["p_sem"] == pytest.approx(1.0)

    # Exactly between the two classes: m = 0, P_sem = sigma(0) = 0.5 for any
    # alpha, so the label is a coin flip between them.
    ambiguous = compute_semantic_margin_uncertainty(
        np.array([1.0, 1.0]), text_feats, 0, mask, logit_scale=alpha
    )
    assert ambiguous["semantic_margin"] == pytest.approx(0.0)
    assert ambiguous["p_sem"] == pytest.approx(0.5)
    assert ambiguous["u_sem"] == pytest.approx(0.5)


def test_semantic_margin_skips_the_synonym_and_uses_the_next_distinct_class():
    """tau removes the near-synonym of the label from the maximum."""
    # t_1 is a near-synonym of t_0 (cos = 0.99...), t_2 is distinct.
    text_feats = np.array(
        [[1.0, 0.0, 0.0], [0.995, 0.0999, 0.0], [0.0, 0.0, 1.0]]
    )
    mask = build_synonym_eligibility_mask(text_feats, tau=0.9)
    embedding = np.array([0.8, 0.0, 0.6])

    result = compute_semantic_margin_uncertainty(
        embedding, text_feats, 0, mask, logit_scale=1.0
    )

    # cos(v, t_0) = 0.8, cos(v, t_2) = 0.6; the synonym's 0.796 is excluded.
    assert result["runner_up_idx"] == 2
    assert result["semantic_margin"] == pytest.approx(0.2)
    assert result["p_sem"] == pytest.approx(sigmoid(0.2))


# ---------------------------------------------------------------------------
# Label Coherence, eq. (3), and its combination with eq. (semantic)
# ---------------------------------------------------------------------------


def coherence_inputs(class_embeddings):
    """Build the prototype state the graph maintains, from unit embeddings."""
    class_embedding_sum = {
        label: np.sum(np.asarray(members, dtype=np.float64), axis=0)
        for label, members in class_embeddings.items()
    }
    class_count = {label: len(members) for label, members in class_embeddings.items()}
    class_prototype_full = {}
    for label, embedding_sum in class_embedding_sum.items():
        prototype = embedding_sum / class_count[label]
        class_prototype_full[label] = prototype / np.linalg.norm(prototype)
    return class_embedding_sum, class_count, class_prototype_full


def test_coherence_identical_embeddings_give_margin_one():
    """Two identical class members and an orthogonal competing class."""
    alpha = 2.0
    members = {
        0: [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        1: [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    }
    sums, counts, prototypes = coherence_inputs(members)

    result = compute_label_coherence_uncertainty(
        members[0][0],
        0,
        sums,
        counts,
        prototypes,
        build_synonym_eligibility_mask(np.eye(3), 0.9),
        logit_scale=alpha,
    )

    # Leave-one-out prototype of class 0 is the other identical member.
    assert result["coherence_prototype_cos_sim"] == pytest.approx(1.0)
    assert result["coherence_runner_up_cos_sim"] == pytest.approx(0.0)
    assert result["label_coherence_margin"] == pytest.approx(1.0)
    assert result["p_coh"] == pytest.approx(sigmoid(alpha))


def test_coherence_orthogonal_class_members_give_margin_zero():
    """A member that shares nothing with the rest of its class."""
    members = {
        0: [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
        1: [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    }
    sums, counts, prototypes = coherence_inputs(members)

    result = compute_label_coherence_uncertainty(
        members[0][0],
        0,
        sums,
        counts,
        prototypes,
        build_synonym_eligibility_mask(np.eye(3), 0.9),
        logit_scale=50.0,
    )

    # cos to the leave-one-out prototype (0,0,1) is 0, as is cos to class 1.
    assert result["coherence_prototype_cos_sim"] == pytest.approx(0.0)
    assert result["label_coherence_margin"] == pytest.approx(0.0)
    assert result["p_coh"] == pytest.approx(0.5)


def test_coherence_is_undefined_for_a_class_with_no_other_instance():
    members = {0: [np.array([1.0, 0.0, 0.0])], 1: [np.array([0.0, 1.0, 0.0])]}
    sums, counts, prototypes = coherence_inputs(members)

    result = compute_label_coherence_uncertainty(
        members[0][0],
        0,
        sums,
        counts,
        prototypes,
        build_synonym_eligibility_mask(np.eye(3), 0.9),
    )

    assert result["p_coh"] is None
    assert result["label_coherence_margin"] is None


def test_combined_estimator_is_the_minimum_of_the_two():
    """eq. (coherence): a low value from either estimator lowers confidence."""
    assert combine_semantic_confidence(sigmoid(2.0), sigmoid(0.5)) == pytest.approx(
        sigmoid(0.5)
    )
    # Undefined coherence reduces the combination to P_sem.
    assert combine_semantic_confidence(sigmoid(2.0), None) == pytest.approx(
        sigmoid(2.0)
    )


# ---------------------------------------------------------------------------
# Vocabulary Membership, eq. (membership)
# ---------------------------------------------------------------------------


def test_membership_in_vocabulary_worked_example():
    """v on the single class, orthogonal to the single negative label.

    L_C = e^1, L_N = e^0, so P_mem = e / (e + 1).
    """
    result = compute_vocabulary_membership(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
        logit_scale=1.0,
    )

    assert result["p_mem"] == pytest.approx(np.e / (np.e + 1.0))
    assert result["u_mem"] == pytest.approx(1.0 / (np.e + 1.0))


def test_membership_out_of_vocabulary_worked_example():
    """The same geometry with the object on the negative label instead."""
    result = compute_vocabulary_membership(
        np.array([0.0, 1.0]),
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
        logit_scale=1.0,
    )

    assert result["p_mem"] == pytest.approx(1.0 / (1.0 + np.e))
    assert result["u_mem"] == pytest.approx(np.e / (np.e + 1.0))


def test_membership_near_uniform_profile_splits_the_posterior():
    """Equally similar to C and to N: the posterior is one half."""
    result = compute_vocabulary_membership(
        np.array([1.0, 0.0]),
        np.array([[0.0, 1.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [0.0, -1.0]]),
        logit_scale=100.0,
    )

    assert result["p_mem"] == pytest.approx(0.5)


def test_membership_stored_log_likelihoods_reproduce_the_probability():
    text_feats = np.array([[1.0, 0.0], [0.6, 0.8]])
    negative_feats = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    alpha = 3.0

    result = compute_vocabulary_membership(
        np.array([1.0, 0.0]), text_feats, negative_feats, logit_scale=alpha
    )

    likelihood_c = (np.exp(alpha * 1.0) + np.exp(alpha * 0.6)) / 2.0
    likelihood_n = (np.exp(0.0) + np.exp(-alpha) + np.exp(0.0)) / 3.0
    assert result["vocab_log_likelihood"] == pytest.approx(np.log(likelihood_c))
    assert result["negative_log_likelihood"] == pytest.approx(np.log(likelihood_n))
    assert result["p_mem"] == pytest.approx(
        likelihood_c / (likelihood_c + likelihood_n)
    )


# ---------------------------------------------------------------------------
# Cross-view Consistency, eq. (crossview)
# ---------------------------------------------------------------------------


def test_cross_view_worked_examples():
    # n = 1: c_bar undefined, P = 1 by definition.
    assert compute_cross_view_consistency(np.array([1.0, 0.0]), 1)[0] == pytest.approx(1.0)
    # Two identical views: ||m|| = 1, c_bar = 1, P = 1.
    assert compute_cross_view_consistency(np.array([2.0, 0.0]), 2)[0] == pytest.approx(1.0)
    # Two orthogonal views: ||m||^2 = 1/2, c_bar = 0, P = 1/2.
    assert compute_cross_view_consistency(np.array([1.0, 1.0]), 2)[0] == pytest.approx(0.5)
    # Two opposite views: ||m|| = 0, c_bar = -1, P = 0.
    assert compute_cross_view_consistency(np.array([0.0, 0.0]), 2)[0] == pytest.approx(0.0)


def test_cross_view_over_sequential_updates_of_one_object():
    """Six sequential views of one point, three of them disagreeing."""
    sums, counts = np.zeros((1, 3)), np.zeros(1, dtype=np.int64)
    views = [[1.0, 0.0, 0.0]] * 3 + [[0.0, 1.0, 0.0]] * 3
    expected_after = {
        1: 1.0,          # single view
        2: 1.0,          # two identical views, c_bar = 1
        3: 1.0,          # three identical views
        4: (1.0 + 3.0 / 6.0) / 2.0,   # pairs: 3 agreeing of 6 -> c_bar = 1/2
        5: (1.0 + 4.0 / 10.0) / 2.0,  # 3 + 1 agreeing of 10 -> c_bar = 2/5
        6: (1.0 + 6.0 / 15.0) / 2.0,  # 3 + 3 agreeing of 15 -> c_bar = 2/5
    }

    for step, view in enumerate(views, start=1):
        accumulate_unit_embeddings(
            sums, counts, np.array([0]), np.asarray([view], dtype=np.float64)
        )
        p_view, u_view = compute_cross_view_consistency(sums[0], int(counts[0]))
        assert int(counts[0]) == step
        assert p_view == pytest.approx(expected_after[step])
        assert u_view == pytest.approx(1.0 - p_view)
