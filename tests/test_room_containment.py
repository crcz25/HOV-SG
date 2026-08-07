from types import SimpleNamespace

import numpy as np
import pytest

from hovsg.graph.graph import Graph
from hovsg.graph.room import Room


def make_object(object_id, class_id, class_label, probability, minimum, maximum):
    vertices = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        object_id=object_id,
        label_idx=class_id,
        name=class_label,
        p_obj=probability,
        vertices=vertices,
        pcd=None,
    )


def beliefs_by_class_id(room):
    return {entry["class_id"]: entry for entry in room.class_containment_belief}


def test_noisy_or_includes_contributing_object_class_metadata():
    room = Room("0_0", "0")
    room.objects = [
        make_object("a", 2, "chair", 0.5, (0, 0, 0), (1, 1, 1)),
        make_object("b", 2, "chair", 0.4, (2, 0, 0), (3, 1, 1)),
    ]

    room.compute_fused_class_containment_beliefs(merge_threshold=0.5)

    assert room.class_containment_belief == [
        {"class_id": 2, "class_label": "chair", "belief": pytest.approx(0.7)}
    ]


def test_single_object_preserves_its_class_metadata():
    room = Room("0_0", "0")
    room.objects = [
        make_object("a", 2, "chair", 0.48, (0, 0, 0), (1, 1, 1))
    ]

    room.compute_fused_class_containment_beliefs()

    assert room.class_containment_belief == [
        {"class_id": 2, "class_label": "chair", "belief": pytest.approx(0.48)}
    ]


def test_connected_duplicate_components_use_their_maximum_probability():
    room = Room("0_0", "0")
    room.objects = [
        make_object("a", 1, "table", 0.3, (0, 0, 0), (2, 1, 1)),
        make_object("b", 1, "table", 0.8, (1, 0, 0), (3, 1, 1)),
        make_object("c", 1, "table", 0.4, (2, 0, 0), (4, 1, 1)),
        make_object("d", 1, "table", 0.2, (10, 0, 0), (11, 1, 1)),
    ]

    room.compute_fused_class_containment_beliefs(merge_threshold=0.25)

    belief = beliefs_by_class_id(room)[1]
    assert belief["class_label"] == "table"
    assert belief["belief"] == pytest.approx(0.84)


def test_merging_is_idempotent_and_endpoint_thresholds_are_well_defined():
    objects = [
        make_object("a", 0, "chair", 0.3, (0, 0, 0), (1, 1, 1)),
        make_object("b", 0, "chair", 0.8, (0, 0, 0), (1, 1, 1)),
        make_object("c", 0, "chair", 0.4, (3, 0, 0), (4, 1, 1)),
    ]

    no_merge = Room.merge_object_groups(objects, overlap_threshold=0.0)
    strict_merge = Room.merge_object_groups(objects, overlap_threshold=1.0)
    assert [len(group) for group in no_merge] == [1, 1, 1]
    assert sorted(len(group) for group in strict_merge) == [1, 2]

    room = Room("0_0", "0")
    room.objects = objects
    room.compute_fused_class_containment_beliefs(merge_threshold=1.0)
    first = list(room.class_containment_belief)
    room.compute_fused_class_containment_beliefs(merge_threshold=1.0)
    assert room.class_containment_belief == first


def test_undefined_object_probability_is_excluded():
    room = Room("0_0", "0")
    room.objects = [
        make_object("defined", 0, "chair", 0.6, (0, 0, 0), (1, 1, 1)),
        make_object("undefined", 0, "chair", None, (0, 0, 0), (1, 1, 1)),
        make_object("all-undefined", 3, "lamp", None, (2, 0, 0), (3, 1, 1)),
    ]

    room.compute_fused_class_containment_beliefs()

    assert room.class_containment_belief == [
        {"class_id": 0, "class_label": "chair", "belief": pytest.approx(0.6)}
    ]
    assert 3 not in beliefs_by_class_id(room)


def test_graph_propagation_uses_assigned_label_and_fused_probability():
    room = Room("0_0", "0")
    room.objects = [
        make_object("chair", 0, "chair", 0.7, (0, 0, 0), (1, 1, 1)),
        make_object("table", 1, "table", 0.2, (2, 0, 0), (3, 1, 1)),
    ]
    graph = SimpleNamespace(
        rooms=[room],
        room_belief_merge_threshold=0.5,
        cfg=SimpleNamespace(pipeline=SimpleNamespace(room_belief_merge_threshold=0.5)),
    )

    Graph.propagate_semantic_uncertainty_to_rooms(graph)

    assert room.class_containment_belief == [
        {"class_id": 0, "class_label": "chair", "belief": pytest.approx(0.7)},
        {"class_id": 1, "class_label": "table", "belief": pytest.approx(0.2)},
    ]
