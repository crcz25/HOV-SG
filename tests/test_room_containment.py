import numpy as np
import pytest
from types import SimpleNamespace

from hovsg.graph.graph import Graph
from hovsg.graph.room import Room
from hovsg.utils.uncertainty import compute_room_containment_probs


def test_empty_room_equals_prior():
    result = compute_room_containment_probs(np.empty((0, 3)), prior=0.05)

    np.testing.assert_allclose(result, np.full(3, 0.05))


def test_one_object_equals_its_compatibility_without_prior():
    result = compute_room_containment_probs(np.array([[0.8]]), prior=0.0)

    assert result[0] == pytest.approx(0.8)


def test_two_objects_follow_noisy_or():
    result = compute_room_containment_probs(
        np.array([[0.5], [0.4]]),
        prior=0.0,
    )

    assert result[0] == pytest.approx(0.7)


def test_detection_reliability_scales_object_contribution():
    result = compute_room_containment_probs(
        np.array([[0.8]]),
        detection_reliabilities=[0.25],
        prior=0.0,
    )

    assert result[0] == pytest.approx(0.2)


def test_more_support_is_monotonic():
    one_object = compute_room_containment_probs(np.array([[0.5, 0.1]]))
    two_objects = compute_room_containment_probs(
        np.array([[0.5, 0.1], [0.4, 0.2]])
    )

    assert np.all(two_objects >= one_object)


def test_lower_reliability_cannot_increase_belief():
    high_reliability = compute_room_containment_probs(
        np.array([[0.8, 0.2]]), detection_reliabilities=[1.0]
    )
    low_reliability = compute_room_containment_probs(
        np.array([[0.8, 0.2]]), detection_reliabilities=[0.25]
    )

    assert np.all(low_reliability <= high_reliability)


def test_empty_room_has_empty_three_signal_beliefs():
    room = Room("0_0", "0")

    room.compute_class_containment_beliefs()

    assert room.object_beliefs_semantic == {}
    assert room.object_beliefs_detection == {}
    assert room.object_beliefs_combined == {}
    assert room.class_containment_beliefs_semantic == {}
    assert room.class_containment_beliefs_detection == {}
    assert room.class_containment_beliefs_combined == {}


def test_one_object_reduces_to_each_signal_q():
    room = Room("0_0", "0")
    room.object_beliefs_semantic = {
        "obj": {"class_idx": 2, "class_name": "chair", "q": 0.8}
    }
    room.object_beliefs_detection = {
        "obj": {"class_idx": 2, "class_name": "chair", "q": 0.6}
    }
    room.object_beliefs_combined = {
        "obj": {"class_idx": 2, "class_name": "chair", "q": 0.48}
    }

    room.compute_class_containment_beliefs()

    assert room.class_containment_beliefs_semantic[2] == pytest.approx(0.8)
    assert room.class_containment_beliefs_detection[2] == pytest.approx(0.6)
    assert room.class_containment_beliefs_combined[2] == pytest.approx(0.48)


def test_two_objects_same_class_fuse_independently_per_signal():
    room = Room("0_0", "0")
    room.object_beliefs_semantic = {
        "a": {"class_idx": 1, "class_name": "table", "q": 0.8},
        "b": {"class_idx": 1, "class_name": "table", "q": 0.5},
    }
    room.object_beliefs_detection = {
        "a": {"class_idx": 1, "class_name": "table", "q": 1.0},
        "b": {"class_idx": 1, "class_name": "table", "q": 0.4},
    }
    room.object_beliefs_combined = {
        "a": {"class_idx": 1, "class_name": "table", "q": 0.8},
        "b": {"class_idx": 1, "class_name": "table", "q": 0.2},
    }

    room.compute_class_containment_beliefs()

    assert room.class_containment_beliefs_semantic[1] == pytest.approx(0.9)
    assert room.class_containment_beliefs_detection[1] == pytest.approx(1.0)
    assert room.class_containment_beliefs_combined[1] == pytest.approx(0.84)


def test_different_assigned_classes_do_not_cross_contribute():
    room = Room("0_0", "0")
    for signal in ("semantic", "detection", "combined"):
        setattr(
            room,
            f"object_beliefs_{signal}",
            {
                "a": {"class_idx": 0, "class_name": "chair", "q": 0.5},
                "b": {"class_idx": 1, "class_name": "table", "q": 0.9},
            },
        )

    room.compute_class_containment_beliefs()

    assert room.class_containment_beliefs_semantic[0] == pytest.approx(0.5)
    assert room.class_containment_beliefs_detection[0] == pytest.approx(0.5)
    assert room.class_containment_beliefs_combined[0] == pytest.approx(0.5)


def test_room_propagation_shares_assignment_but_keeps_signal_values_separate():
    room = Room("0_0", "0")
    first = SimpleNamespace(
        object_id="0_0_0", label_idx=0, c_sem=0.8, c_det=0.6
    )
    second = SimpleNamespace(
        object_id="0_0_1", label_idx=1, c_sem=0.7, c_det=0.25
    )
    room.objects = [first, second]
    graph = SimpleNamespace(
        rooms=[room],
        label_text_feats=np.eye(2, dtype=np.float64),
        label_classes=["chair", "table"],
    )

    Graph.propagate_semantic_uncertainty_to_rooms(graph)

    for obj in room.objects:
        obj_id = str(obj.object_id)
        semantic = room.object_beliefs_semantic[obj_id]
        detection = room.object_beliefs_detection[obj_id]
        combined = room.object_beliefs_combined[obj_id]
        assert semantic["class_idx"] == detection["class_idx"] == combined["class_idx"]
        assert semantic["class_name"] == detection["class_name"] == combined["class_name"]
        assert detection["q"] == pytest.approx(obj.c_det)
        assert combined["q"] == pytest.approx(semantic["q"] * detection["q"])

    assert set(room.class_containment_beliefs_semantic) == {0, 1}
    assert set(room.class_containment_beliefs_detection) == {0, 1}
    assert set(room.class_containment_beliefs_combined) == {0, 1}
    assert room.class_containment_probs is None
    assert room.class_containment_topk is None


def test_signal_independence_when_inputs_change():
    base = Room("0_0", "0")
    changed_detection = Room("0_0", "0")
    changed_semantic = Room("0_0", "0")

    base.object_beliefs_semantic = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.7}
    }
    base.object_beliefs_detection = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.5}
    }
    base.object_beliefs_combined = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.35}
    }

    changed_detection.object_beliefs_semantic = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.7}
    }
    changed_detection.object_beliefs_detection = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.2}
    }
    changed_detection.object_beliefs_combined = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.14}
    }

    changed_semantic.object_beliefs_semantic = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.9}
    }
    changed_semantic.object_beliefs_detection = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.5}
    }
    changed_semantic.object_beliefs_combined = {
        "obj": {"class_idx": 0, "class_name": "chair", "q": 0.45}
    }

    for room in (base, changed_detection, changed_semantic):
        room.compute_class_containment_beliefs()

    assert changed_detection.class_containment_beliefs_semantic[0] == pytest.approx(
        base.class_containment_beliefs_semantic[0]
    )
    assert changed_semantic.class_containment_beliefs_detection[0] == pytest.approx(
        base.class_containment_beliefs_detection[0]
    )
    assert changed_detection.class_containment_beliefs_combined[0] != pytest.approx(
        base.class_containment_beliefs_combined[0]
    )
    assert changed_semantic.class_containment_beliefs_combined[0] != pytest.approx(
        base.class_containment_beliefs_combined[0]
    )
