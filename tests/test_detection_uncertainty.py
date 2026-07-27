import json
from types import MethodType, SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree

from hovsg.graph.object import Object
from hovsg.utils.detection_uncertainty import (
    accumulate_confidence,
    finalize_confidence_array,
    mask_predicted_iou,
    object_confidence_from_points,
    uncertainty_from_confidence,
)


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def test_complement_invariant_for_confidence_values():
    rng = np.random.default_rng(7)

    for c_det in rng.random(25):
        u_det = uncertainty_from_confidence(c_det)

        assert u_det == pytest.approx(1.0 - c_det)
        assert c_det + u_det == pytest.approx(1.0)


@pytest.mark.parametrize(("value", "expected"), [(1.0001, 1.0), (-1e-9, 0.0)])
def test_confidence_values_are_clamped(value, expected):
    with pytest.warns(RuntimeWarning):
        c_det = mask_predicted_iou({"predicted_iou": value})

    assert c_det == expected
    assert uncertainty_from_confidence(c_det) == pytest.approx(1.0 - expected)


def test_nan_predicted_iou_raises():
    with pytest.raises(ValueError, match="predicted_iou"):
        mask_predicted_iou({"predicted_iou": np.nan})


def test_accumulate_and_finalize_confidence_array():
    sum_conf = np.zeros((4, 1), dtype=np.float64)
    counter = np.zeros((4, 1), dtype=np.float64)

    accumulate_confidence(sum_conf, counter, np.array([0, 1, 1]), 0.5)
    accumulate_confidence(sum_conf, counter, np.array([1, 2]), 1.0)
    result = finalize_confidence_array(sum_conf, counter)

    np.testing.assert_allclose(result.reshape(-1), [0.5, 2.0 / 3.0, 1.0, 0.0])
    assert np.isfinite(result).all()


def test_object_confidence_from_points_averages_nearest_scene_points():
    scene_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    tree = cKDTree(scene_points)
    full_conf_array = np.array([[0.2], [0.6], [1.0]])
    object_points = np.array([[0.05, 0.0, 0.0], [1.95, 0.0, 0.0]])

    result = object_confidence_from_points(full_conf_array, tree, object_points)

    assert result == pytest.approx(0.6)
    assert object_confidence_from_points(full_conf_array, tree, np.empty((0, 3))) == 0.0


def test_object_detection_metadata_round_trip(tmp_path):
    source = Object("0_0_0", "0_0", name="chair")
    source.pcd = make_pcd([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    source.vertices = np.zeros((2, 3))
    source.embedding = np.array([1.0, 0.0])
    source.c_det = 0.625
    source.u_det = 0.375
    source.save(tmp_path)

    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))

    assert restored.c_det == 0.625
    assert restored.u_det == 0.375


def test_old_object_metadata_loads_without_detection_fields(tmp_path):
    object_id = "0_0_0"
    o3d.io.write_point_cloud(str(tmp_path / f"{object_id}.ply"), make_pcd([[0, 0, 0]]))
    metadata = {
        "object_id": object_id,
        "vertices": [],
        "room_id": "0_0",
        "name": "chair",
        "embedding": [1.0, 0.0],
    }
    (tmp_path / f"{object_id}.json").write_text(json.dumps(metadata), encoding="utf-8")

    restored = Object(object_id, "0_0")
    restored.load(str(tmp_path))

    assert restored.c_det is None
    assert restored.u_det is None


def test_object_merge_averages_confidence_and_recomputes_uncertainty():
    left = Object("0_0_0", "0_0", name="chair")
    left.pcd = make_pcd([[0.0, 0.0, 0.0]])
    left.embedding = np.array([1.0, 0.0])
    left.c_det = 0.2
    left.u_det = 0.8

    right = Object("0_0_1", "0_0", name="chair")
    right.pcd = make_pcd([[1.0, 0.0, 0.0]])
    right.embedding = np.array([0.0, 1.0])
    right.c_det = 0.8
    right.u_det = 0.1

    merged = left + right

    assert merged.c_det == pytest.approx(0.5)
    assert merged.u_det == pytest.approx(1.0 - merged.c_det)


def test_object_merge_confidence_fallbacks_and_empty_short_circuits():
    empty = Object("0_0_0", "0_0")
    empty.pcd = o3d.geometry.PointCloud()

    right = Object("0_0_1", "0_0")
    right.pcd = make_pcd([[0.0, 0.0, 0.0]])
    right.embedding = np.array([1.0, 0.0])
    right.c_det = 0.7
    right.u_det = 0.3

    assert empty + right is right

    left = Object("0_0_2", "0_0")
    left.pcd = make_pcd([[1.0, 0.0, 0.0]])
    left.embedding = np.array([0.0, 1.0])
    left.c_det = None
    left.u_det = None

    merged = left + right

    assert merged.c_det == pytest.approx(0.7)
    assert merged.u_det == pytest.approx(0.3)


def test_segment_objects_assigns_detection_confidence(monkeypatch, tmp_path):
    import hovsg.graph.graph as graph_module
    from hovsg.graph.graph import Graph
    from hovsg.graph.room import Room

    class Pipeline(dict):
        def __getattr__(self, key):
            return self[key]

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
        graph_module,
        "get_label_feats",
        lambda *_: (np.eye(2, dtype=np.float64), ["chair", "table"]),
    )
    monkeypatch.setattr(graph_module, "find_intersection_share", lambda *_: 1.0)

    room = Room("0_0", "0", name="room")
    room.vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    floor = SimpleNamespace(
        floor_zero_level=0.0,
        floor_height=2.0,
        pcd=make_pcd([[0.0, 0.0, 0.0]]),
        rooms=[room],
    )
    graph = SimpleNamespace(
        cfg=cfg,
        floors=[floor],
        rooms=[room],
        objects=[],
        mask_pcds=[
            make_pcd([[0.1, 0.5, 0.1], [0.2, 0.5, 0.1]]),
            make_pcd([[1.0, 0.5, 1.0], [1.1, 0.5, 1.0]]),
        ],
        mask_feats=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        mask_confs=[0.25, 0.9],
        clip_model=None,
        clip_feat_dim=2,
        graph_tmp_folder=str(tmp_path),
        identify_object=None,
    )
    graph.identify_object = MethodType(Graph.identify_object, graph)

    Graph.segment_objects(graph)

    assert [object.c_det for object in graph.objects] == [0.25, 0.9]
    for object in graph.objects:
        assert object.u_det == pytest.approx(1.0 - object.c_det)
        assert object.label_cos_sim == pytest.approx(1.0)
        assert object.runner_up_cos_sim == pytest.approx(0.0)
        assert object.semantic_margin == pytest.approx(1.0)
        assert object.c_sem > 0.999
        assert object.u_sem < 0.001

    assert graph.semantic_uncertainty_synonym_threshold == pytest.approx(0.75)
    np.testing.assert_array_equal(
        graph.label_synonym_mask, np.array([[False, True], [True, False]])
    )
