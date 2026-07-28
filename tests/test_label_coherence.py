"""Label Coherence: leave-one-out visual prototypes.

    m'_i = cos(v_i, mu_l_i) - max over c with cos(t_c, t_l_i) < tau of cos(v_i, mu_c)
    P_coh = sigma(alpha * m'_i),  U_coh = 1 - P_coh

with mu_c the normalized mean of the visual embeddings of the objects labeled
c, and v_i excluded from the mean of its own class. The signal is undefined
when l_i has no other instance.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from hovsg.graph.graph import Graph
from hovsg.graph.object import Object
from hovsg.utils.uncertainty import (
    COHERENCE_FIELDS,
    build_synonym_eligibility_mask,
    compute_label_coherence_uncertainty,
    normalize_rows,
)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def make_object(object_id, embedding, label_idx):
    obj = Object(object_id, "0_0", name=f"class{label_idx}")
    obj.embedding = unit(embedding)
    obj.label_idx = label_idx
    return obj


def make_graph(objects, text_feats, classes, tau=0.9, alpha=100.0):
    return SimpleNamespace(
        cfg=SimpleNamespace(pipeline={}),
        objects=objects,
        label_text_feats=text_feats,
        label_text_feats_normalized=normalize_rows(text_feats),
        label_classes=classes,
        label_coherence_synonym_threshold=tau,
        label_coherence_logit_scale=alpha,
        class_embedding_sum={},
        class_count={},
        class_prototype_full={},
    )


# ---------------------------------------------------------------------------
# Leave-one-out prototype construction
# ---------------------------------------------------------------------------


def test_object_is_excluded_from_its_own_class_prototype():
    """Including v_i would inflate the similarity toward its own prototype."""
    text_feats = np.eye(3)
    # Three "chair" objects, one of which points away from the other two.
    embeddings = [unit([1.0, 0.1, 0.0]), unit([1.0, -0.1, 0.0]), unit([0.2, 1.0, 0.0])]
    objects = [make_object(f"0_0_{i}", e, 0) for i, e in enumerate(embeddings)]
    objects.append(make_object("0_0_3", [0.0, 0.0, 1.0], 2))
    graph = make_graph(objects, text_feats, ["a", "b", "c"], alpha=1.0)

    Graph.recompute_label_coherence(graph)

    outlier = objects[2]
    # Prototype of class 0 excluding the outlier itself.
    expected_prototype = unit(embeddings[0] + embeddings[1])
    assert outlier.coherence_prototype_cos_sim == pytest.approx(
        float(np.dot(outlier.embedding, expected_prototype))
    )
    # The self-inclusive prototype would score strictly higher.
    inclusive = unit(sum(embeddings))
    assert float(np.dot(outlier.embedding, inclusive)) > (
        outlier.coherence_prototype_cos_sim
    )


def test_leave_one_out_margin_matches_the_equation():
    text_feats = np.eye(3)
    alpha = 20.0
    class_a = [unit([1.0, 0.2, 0.0]), unit([1.0, -0.2, 0.0])]
    class_b = [unit([0.0, 1.0, 0.3]), unit([0.0, 1.0, -0.3])]
    objects = [make_object(f"a{i}", e, 0) for i, e in enumerate(class_a)]
    objects += [make_object(f"b{i}", e, 1) for i, e in enumerate(class_b)]
    graph = make_graph(objects, text_feats, ["a", "b", "c"], alpha=alpha)

    Graph.recompute_label_coherence(graph)

    scored = objects[0]
    own_prototype = unit(class_a[1])  # leave-one-out over a two-object class
    other_prototype = unit(class_b[0] + class_b[1])
    expected_margin = float(
        np.dot(scored.embedding, own_prototype)
        - np.dot(scored.embedding, other_prototype)
    )

    assert scored.label_coherence_margin == pytest.approx(expected_margin, abs=1e-12)
    assert scored.p_coh == pytest.approx(float(expit(alpha * expected_margin)))
    assert scored.u_coh == pytest.approx(1.0 - scored.p_coh)


def test_prototypes_are_visual_not_text_embeddings():
    """Text features only gate synonyms; they never form the prototype."""
    # Text features deliberately orthogonal to the visual embeddings.
    text_feats = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    objects = [
        make_object("a0", [1.0, 0.0, 0.0], 0),
        make_object("a1", [1.0, 0.0, 0.0], 0),
        make_object("b0", [0.0, 1.0, 0.0], 1),
        make_object("b1", [0.0, 1.0, 0.0], 1),
    ]
    graph = make_graph(objects, text_feats, ["a", "b", "c"], alpha=1.0)

    Graph.recompute_label_coherence(graph)

    # Identical visual embeddings within class 0 -> cosine 1 to its prototype,
    # and 0 to class 1's, regardless of what the text embeddings look like.
    assert objects[0].coherence_prototype_cos_sim == pytest.approx(1.0)
    assert objects[0].coherence_runner_up_cos_sim == pytest.approx(0.0)
    assert objects[0].label_coherence_margin == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Undefined cases
# ---------------------------------------------------------------------------


def test_single_instance_class_is_undefined():
    text_feats = np.eye(3)
    objects = [
        make_object("only", [1.0, 0.0, 0.0], 0),  # sole member of class 0
        make_object("b0", [0.0, 1.0, 0.0], 1),
        make_object("b1", [0.0, 0.9, 0.1], 1),
    ]
    graph = make_graph(objects, text_feats, ["a", "b", "c"])

    Graph.recompute_label_coherence(graph)

    for field in COHERENCE_FIELDS:
        assert getattr(objects[0], field) is None
    # Its class still contributes a prototype for the others to compete with.
    assert 0 in graph.class_prototype_full
    assert objects[1].p_coh is not None


def test_missing_competing_prototype_is_undefined():
    """Only one class instantiated -> no distinct prototype to compete."""
    text_feats = np.eye(2)
    objects = [
        make_object("a0", [1.0, 0.0], 0),
        make_object("a1", [0.9, 0.1], 0),
    ]
    graph = make_graph(objects, text_feats, ["a", "b"])

    Graph.recompute_label_coherence(graph)

    for obj in objects:
        assert obj.p_coh is None
        assert obj.u_coh is None


def test_synonym_classes_are_excluded_from_competing_prototypes():
    # Classes 0 and 1 have near-identical text embeddings; class 2 is distinct.
    text_feats = np.array(
        [[1.0, 0.0, 0.0], [0.999, 0.0447, 0.0], [0.0, 0.0, 1.0]]
    )
    objects = [
        make_object("a0", [1.0, 0.0, 0.0], 0),
        make_object("a1", [0.98, 0.2, 0.0], 0),
        make_object("syn0", [0.95, 0.31, 0.0], 1),  # visually closest competitor
        make_object("syn1", [0.96, 0.28, 0.0], 1),
        make_object("far0", [0.0, 0.0, 1.0], 2),
        make_object("far1", [0.0, 0.1, 0.99], 2),
    ]
    graph = make_graph(objects, text_feats, ["a", "a2", "c"], tau=0.9)

    Graph.recompute_label_coherence(graph)

    # The synonym class would win on visual similarity, but tau removes it.
    assert objects[0].coherence_runner_up_class == "c"


def test_undefined_is_none_never_zero_or_one():
    """A degenerate embedding must not be scored as certain or impossible."""
    result = compute_label_coherence_uncertainty(
        np.zeros(3),
        0,
        {0: np.ones(3)},
        {0: 5},
        {0: unit([1.0, 0.0, 0.0]), 1: unit([0.0, 1.0, 0.0])},
        build_synonym_eligibility_mask(np.eye(3), 0.9),
    )

    assert result == dict.fromkeys(COHERENCE_FIELDS)


def test_unlabeled_objects_do_not_contribute_to_prototypes():
    text_feats = np.eye(2)
    labeled = [make_object("a0", [1.0, 0.0], 0), make_object("a1", [1.0, 0.0], 0)]
    unlabeled = Object("x", "0_0")
    unlabeled.embedding = unit([0.0, 1.0])
    unlabeled.label_idx = None
    degenerate = Object("z", "0_0")
    degenerate.embedding = np.zeros(2)
    degenerate.label_idx = 0

    graph = make_graph(labeled + [unlabeled, degenerate], text_feats, ["a", "b"])
    Graph.recompute_label_coherence(graph)

    assert graph.class_count == {0: 2}
    assert degenerate.p_coh is None
    assert unlabeled.p_coh is None


# ---------------------------------------------------------------------------
# Cache freshness across relabeling, merging and deletion
# ---------------------------------------------------------------------------


def build_two_class_graph():
    text_feats = np.eye(3)
    objects = [
        make_object("a0", [1.0, 0.1, 0.0], 0),
        make_object("a1", [1.0, -0.1, 0.0], 0),
        make_object("b0", [0.0, 1.0, 0.1], 1),
        make_object("b1", [0.0, 1.0, -0.1], 1),
    ]
    return make_graph(objects, text_feats, ["a", "b", "c"], alpha=10.0), objects


def test_prototypes_refresh_after_relabeling():
    graph, objects = build_two_class_graph()
    Graph.recompute_label_coherence(graph)
    assert graph.class_count == {0: 2, 1: 2}

    # Relabel one object into the other class and recompute.
    objects[0].label_idx = 1
    Graph.recompute_label_coherence(graph)

    assert graph.class_count == {0: 1, 1: 3}
    # The now-single-instance class 0 is undefined for its remaining member.
    assert objects[1].p_coh is None
    assert objects[0].p_coh is not None


def test_prototypes_refresh_after_deletion():
    graph, objects = build_two_class_graph()
    Graph.recompute_label_coherence(graph)
    before = objects[2].coherence_runner_up_cos_sim

    graph.objects = [obj for obj in graph.objects if obj.object_id != "a1"]
    Graph.recompute_label_coherence(graph)

    # Class 0's prototype is now a1-free, so the competing similarity changed.
    assert objects[2].coherence_runner_up_cos_sim != pytest.approx(before)
    assert graph.class_count == {0: 1, 1: 2}


def test_merge_clears_stale_coherence_until_recomputed():
    import open3d as o3d

    graph, objects = build_two_class_graph()
    Graph.recompute_label_coherence(graph)
    assert objects[0].p_coh is not None

    for obj in objects:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).normal(size=(3, 3)))
        obj.pcd = pcd

    merged = objects[0] + objects[1]

    # The merged embedding is new, so every signal read off it is stale and
    # must read as undefined rather than as a number for the merged object.
    for field in COHERENCE_FIELDS:
        assert getattr(merged, field) is None
    assert merged.p_sem is None and merged.p_mem is None
