"""End-to-end coverage of the five object-level uncertainty signals.

Exercises the path a real object takes: per-view accumulation -> mask-level
fusion -> object node -> merge -> recomputation -> serialization, checking that
every signal survives each step with the value the equations prescribe.
"""

import itertools
import json
from types import MethodType, SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree
from scipy.special import expit

from hovsg.graph.graph import Graph
from hovsg.graph.object import Object
from hovsg.graph.room import Room
from hovsg.utils.cross_view_consistency import (
    accumulate_unit_embeddings,
    compute_cross_view_consistency,
)
from hovsg.utils.detection_uncertainty import (
    accumulate_confidence,
    confidence_from_sum,
    finalize_confidence_array,
    object_confidence_sum_from_points,
)
from hovsg.utils.uncertainty import (
    COHERENCE_FIELDS,
    MEMBERSHIP_FIELDS,
    SEMANTIC_FIELDS,
    normalize_rows,
)


ALL_SIGNAL_FIELDS = (
    SEMANTIC_FIELDS
    + MEMBERSHIP_FIELDS
    + COHERENCE_FIELDS
    + (
        "p_sem_bar",  # eq. (coherence)
        "u_sem_bar",
        "p_det",
        "u_det",
        "p_view",
        "u_view",
        "p_obj",
        "u_obj",
    )
)


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


# ---------------------------------------------------------------------------
# Per-view accumulation through to the object node
# ---------------------------------------------------------------------------


def test_multi_view_accumulation_feeds_the_object_node():
    """Three views of two points, then the object-level roll-up."""
    n_points, dim = 2, 3
    cross_view_sum = np.zeros((n_points, dim), dtype=np.float64)
    cross_view_count = np.zeros(n_points, dtype=np.int64)
    sum_conf = np.zeros((n_points, 1), dtype=np.float64)
    counter_conf = np.zeros((n_points, 1), dtype=np.float64)

    # View 0 and 1 agree; view 2 disagrees on point 0 only.
    views = [
        (np.array([0, 1]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.9),
        (np.array([0, 1]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.7),
        (np.array([0, 1]), np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), 0.8),
    ]
    for indices, embeddings, confidence in views:
        accumulate_unit_embeddings(cross_view_sum, cross_view_count, indices, embeddings)
        accumulate_confidence(sum_conf, counter_conf, indices, confidence)

    np.testing.assert_array_equal(cross_view_count, [3, 3])
    np.testing.assert_allclose(cross_view_sum[1], [0.0, 3.0, 0.0])

    # Object roll-up over both points: six unit observations, of which four are
    # +y, two are +x and one is +z. c_bar is the mean over the 15 pairs.
    object_sum = cross_view_sum.sum(axis=0)
    object_count = int(cross_view_count.sum())
    p_view, u_view = compute_cross_view_consistency(object_sum, object_count)

    observations = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ] + [np.array([0.0, 1.0, 0.0])] * 3
    pairs = list(itertools.combinations(observations, 2))
    expected_mean_cosine = float(np.mean([np.dot(a, b) for a, b in pairs]))
    assert p_view == pytest.approx((1.0 + expected_mean_cosine) / 2.0)
    assert u_view == pytest.approx(1.0 - p_view)

    # Detection: the mean predicted_iou is the same at every point here.
    full_conf = finalize_confidence_array(sum_conf, counter_conf)
    np.testing.assert_allclose(full_conf.reshape(-1), [0.8, 0.8])
    tree = cKDTree(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    conf_sum, count = object_confidence_sum_from_points(
        full_conf, tree, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert confidence_from_sum(conf_sum, count) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# segment_objects: object creation with all five signals
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_graph(monkeypatch, tmp_path):
    import hovsg.graph.graph as graph_module

    class Pipeline(dict):
        def __getattr__(self, key):
            return self[key]

    # Class 0 and 1 are distinct; class 2 is a near-synonym of class 0.
    text_feats = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.99, 0.141, 0.0]])
    negative_feats = np.array([[0.0, 0.0, 1.0]])

    cfg = SimpleNamespace(
        main=SimpleNamespace(save_path=str(tmp_path)),
        pipeline=Pipeline(
            save_intermediate_results=False,
            obj_labels="synthetic",
            semantic_uncertainty_logit_scale=10.0,
            semantic_uncertainty_synonym_threshold=0.75,
        ),
    )

    monkeypatch.setattr(graph_module, "pcd_denoise_dbscan", lambda pcd, **_: pcd)
    monkeypatch.setattr(
        graph_module, "get_label_feats", lambda *_: (text_feats, ["chair", "table", "seat"])
    )
    monkeypatch.setattr(graph_module, "find_intersection_share", lambda *_: 1.0)

    room = Room("0_0", "0", name="room")
    room.vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    floor = SimpleNamespace(
        floor_zero_level=0.0, floor_height=2.0, pcd=make_pcd([[0.0, 0.0, 0.0]]), rooms=[room]
    )

    graph = SimpleNamespace(
        cfg=cfg,
        floors=[floor],
        rooms=[room],
        objects=[],
        mask_pcds=[
            make_pcd([[0.1, 0.5, 0.1], [0.2, 0.5, 0.1]]),
            make_pcd([[1.0, 0.5, 1.0], [1.1, 0.5, 1.0]]),
            make_pcd([[1.5, 0.5, 1.5], [1.6, 0.5, 1.5]]),
        ],
        # The first two are nearest to class 0 (not to the class-2 synonym, whose
        # text embedding leans toward +y), the third to class 1.
        mask_feats=[unit([1.0, -0.05, 0.0]), unit([1.0, -0.1, 0.0]), unit([0.0, 1.0, 0.0])],
        mask_confs=[(0.5, 2), (3.6, 4), (1.0, 2)],
        mask_cross_view_sums=[
            np.array([2.0, 0.0, 0.0]),  # 2 agreeing views
            np.array([1.0, 1.0, 0.0]),  # 2 orthogonal views
            np.array([1.0, 0.0, 0.0]),  # 1 view
        ],
        mask_cross_view_counts=[2, 2, 1],
        negative_text_feats=negative_feats,
        _negative_text_feats_normalized=None,
        label_text_feats_normalized=None,
        clip_model=None,
        clip_feat_dim=3,
        graph_tmp_folder=str(tmp_path),
        class_embedding_sum={},
        class_count={},
        class_prototype_full={},
        identify_object=None,
    )
    graph.identify_object = MethodType(Graph.identify_object, graph)
    return graph, text_feats, negative_feats


def test_segment_objects_populates_every_signal(synthetic_graph):
    graph, text_feats, negative_feats = synthetic_graph

    Graph.segment_objects(graph)

    assert len(graph.objects) == 3
    for obj in graph.objects:
        # Detection
        assert obj.p_det is not None
        assert obj.u_det == pytest.approx(1.0 - obj.p_det)
        # Semantic
        assert obj.p_sem is not None
        assert obj.u_sem == pytest.approx(1.0 - obj.p_sem)
        # Membership
        assert obj.p_mem is not None
        assert obj.u_mem == pytest.approx(1.0 - obj.p_mem)
        # Cross-view
        assert obj.p_view is not None
        assert obj.u_view == pytest.approx(1.0 - obj.p_view)

    assert [obj.p_det for obj in graph.objects] == pytest.approx([0.25, 0.9, 0.5])
    # Two agreeing views (c_bar = 1), two orthogonal views (c_bar = 0), and a
    # single view, where c_bar is undefined and P^view is 1 by definition.
    assert [obj.p_view for obj in graph.objects] == pytest.approx([1.0, 0.5, 1.0])


def test_synonym_class_is_never_the_semantic_runner_up(synthetic_graph):
    """Class 2 is a near-synonym of class 0 and must be excluded by tau."""
    graph, _, _ = synthetic_graph

    Graph.segment_objects(graph)

    chair_like = [obj for obj in graph.objects if obj.label_idx == 0]
    assert chair_like
    for obj in chair_like:
        assert obj.runner_up_idx == 1  # the distinct class, not the synonym


def test_coherence_is_computed_only_after_the_full_object_set_exists(synthetic_graph):
    graph, _, _ = synthetic_graph
    Graph.segment_objects(graph)

    # segment_objects does not assign coherence: prototypes need every object.
    assert all(obj.p_coh is None for obj in graph.objects)

    Graph.recompute_semantic_uncertainty(graph)

    for obj in graph.objects:
        assert obj.p_obj is not None
        assert obj.u_obj == pytest.approx(1.0 - obj.p_obj)

    # Objects 0 and 1 share class 0, so each is scored against the other.
    class_zero = [obj for obj in graph.objects if obj.label_idx == 0]
    assert len(class_zero) == 2
    for obj in class_zero:
        assert obj.p_coh is not None
        assert obj.u_coh == pytest.approx(1.0 - obj.p_coh)
    # The sole member of its class has no leave-one-out prototype.
    singletons = [obj for obj in graph.objects if obj.label_idx == 1]
    assert all(obj.p_coh is None for obj in singletons)


def test_recompute_is_idempotent(synthetic_graph):
    graph, _, _ = synthetic_graph
    Graph.segment_objects(graph)
    Graph.recompute_semantic_uncertainty(graph)

    first = [
        {field: getattr(obj, field) for field in ALL_SIGNAL_FIELDS}
        for obj in graph.objects
    ]
    Graph.recompute_semantic_uncertainty(graph)
    second = [
        {field: getattr(obj, field) for field in ALL_SIGNAL_FIELDS}
        for obj in graph.objects
    ]

    assert first == second


def test_relabeling_refreshes_semantic_and_coherence(synthetic_graph):
    graph, text_feats, _ = synthetic_graph
    Graph.segment_objects(graph)
    Graph.recompute_semantic_uncertainty(graph)

    target = graph.objects[0]
    before_margin = target.semantic_margin

    # Relabel to the other distinct class and recompute.
    target.label_idx = 1
    Graph.recompute_semantic_uncertainty(graph)

    assert target.semantic_margin != pytest.approx(before_margin)
    normalized = normalize_rows(text_feats)
    expected_label_cos = float(np.dot(target.embedding, normalized[1]))
    assert target.label_cos_sim == pytest.approx(expected_label_cos)
    assert target.p_sem == pytest.approx(
        float(expit(10.0 * target.semantic_margin))
    )


def test_degenerate_embedding_yields_no_label_and_no_signals(synthetic_graph):
    graph, _, _ = synthetic_graph
    graph.mask_feats[2] = np.zeros(3)

    Graph.segment_objects(graph)
    Graph.recompute_semantic_uncertainty(graph)

    degenerate = graph.objects[2]
    assert degenerate.label_idx is None
    assert degenerate.name is None
    for field in SEMANTIC_FIELDS + MEMBERSHIP_FIELDS + COHERENCE_FIELDS:
        assert getattr(degenerate, field) is None
    # It must not have polluted the visual prototypes either.
    assert all(count >= 1 for count in graph.class_count.values())
    assert 2 not in graph.class_count


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_full_signal_round_trip_through_json(synthetic_graph, tmp_path):
    graph, _, _ = synthetic_graph
    Graph.segment_objects(graph)
    Graph.recompute_semantic_uncertainty(graph)

    source = graph.objects[0]
    source.save(tmp_path)
    saved = json.loads((tmp_path / f"{source.object_id}.json").read_text("utf-8"))

    # Raw evidence must persist, not only the derived probabilities: the
    # signals have to be recomputable after a merge on a loaded graph.
    for field in (
        "detection_conf_sum",
        "detection_point_count",
        "cross_view_resultant_sum",
        "cross_view_count",
    ):
        assert field in saved

    restored = Object(source.object_id, source.room_id)
    restored.load(str(tmp_path))

    for field in ALL_SIGNAL_FIELDS:
        expected = getattr(source, field)
        actual = getattr(restored, field)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected
    np.testing.assert_allclose(
        restored.cross_view_resultant_sum, source.cross_view_resultant_sum
    )
    assert restored.detection_point_count == source.detection_point_count


def test_confidence_and_uncertainty_are_complements_everywhere(synthetic_graph):
    graph, _, _ = synthetic_graph
    Graph.segment_objects(graph)
    Graph.recompute_semantic_uncertainty(graph)

    for obj in graph.objects:
        for signal in ("sem", "mem", "coh", "det", "view", "obj"):
            confidence = getattr(obj, f"p_{signal}")
            uncertainty = getattr(obj, f"u_{signal}")
            if confidence is None:
                assert uncertainty is None, f"u_{signal} defined while p_{signal} is not"
                continue
            assert 0.0 <= confidence <= 1.0
            assert uncertainty == pytest.approx(1.0 - confidence)


def test_signals_are_not_fused_into_a_single_score(synthetic_graph):
    """The five source signals stay separate beside the explicit fused score."""
    graph, _, _ = synthetic_graph
    Graph.segment_objects(graph)
    Graph.recompute_semantic_uncertainty(graph)

    obj = graph.objects[0]
    for attribute in vars(obj):
        assert attribute not in {
            "combined_uncertainty",
            "total_uncertainty",
            "fused_confidence",
            "overall_confidence",
        }
