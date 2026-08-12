from types import SimpleNamespace

import numpy as np
import pytest

from hovsg.graph.graph import Graph
from hovsg.graph.object import Object
from hovsg.utils.uncertainty import build_synonym_eligibility_mask


def make_graph(objects):
    text_feats = np.eye(2)
    return SimpleNamespace(
        cfg=SimpleNamespace(
            pipeline={
                "semantic_uncertainty_logit_scale": 1.0,
                "semantic_uncertainty_synonym_threshold": 0.75,
            }
        ),
        objects=objects,
        label_text_feats=text_feats,
        label_classes=["chair", "table"],
        label_synonym_mask=build_synonym_eligibility_mask(text_feats, 0.75),
        semantic_uncertainty_logit_scale=1.0,
        negative_text_feats=np.array([[0.0, 1.0]]),
    )


def test_recompute_uses_existing_assignment_without_changing_name():
    obj = Object("0_0_0", "0_0", name="original name")
    obj.embedding = np.array([1.0, 0.0])
    obj.label_idx = 0
    graph = make_graph([obj])

    Graph.recompute_semantic_uncertainty(graph)

    assert obj.name == "original name"
    assert obj.label_idx == 0
    assert obj.label_cos_sim == pytest.approx(1.0)
    assert obj.runner_up_idx == 1
    assert obj.semantic_margin == pytest.approx(1.0)
    # |C| = 2 with cosines 1 and 0, |N| = 1 with cosine 0, so the size-normalized
    # likelihoods are (e + 1) / 2 and 1.
    assert obj.vocab_log_likelihood == pytest.approx(
        np.log((np.exp(1.0) + 1.0) / 2.0)
    )
    assert obj.negative_log_likelihood == pytest.approx(0.0)
    likelihood_c = (np.exp(1.0) + 1.0) / 2.0
    assert obj.p_mem == pytest.approx(likelihood_c / (likelihood_c + 1.0))


def test_recompute_does_not_guess_missing_legacy_label_index():
    obj = Object("0_0_0", "0_0", name="chair")
    obj.embedding = np.array([1.0, 0.0])
    for field in (
        "label_cos_sim",
        "runner_up_idx",
        "runner_up_cos_sim",
        "semantic_margin",
        "p_sem",
        "u_sem",
    ):
        setattr(obj, field, 0.5)
    graph = make_graph([obj])

    Graph.recompute_semantic_uncertainty(graph)

    assert obj.label_idx is None
    assert obj.name == "chair"
    assert obj.label_cos_sim is None
    assert obj.runner_up_idx is None
    assert obj.runner_up_cos_sim is None
    assert obj.semantic_margin is None
    assert obj.p_sem is None
    assert obj.u_sem is None
    # |C| = 2 with cosines 1 and 0, |N| = 1 with cosine 0, so the size-normalized
    # likelihoods are (e + 1) / 2 and 1.
    assert obj.vocab_log_likelihood == pytest.approx(
        np.log((np.exp(1.0) + 1.0) / 2.0)
    )
    assert obj.negative_log_likelihood == pytest.approx(0.0)
    likelihood_c = (np.exp(1.0) + 1.0) / 2.0
    assert obj.p_mem == pytest.approx(likelihood_c / (likelihood_c + 1.0))


def test_identify_object_labels_a_valid_embedding_with_zero_scores():
    """All-zero similarities do not imply that the visual embedding is invalid."""
    graph = SimpleNamespace(label_text_feats_normalized=None)
    name, label_idx, similarity = Graph.identify_object(
        graph,
        np.array([0.0, 0.0, 1.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ["chair", "table"],
    )

    np.testing.assert_allclose(similarity, [0.0, 0.0])
    assert (name, label_idx) == ("chair", 0)
