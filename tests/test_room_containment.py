"""Numerical agreement with propagation.tex, object-to-room propagation.

Every expected value below is written as the explicit arithmetic of the
paper's equation rather than as a number copied from the implementation, so a
failure means the code left the paper.

    Object probability   q_i    = s_i * P_view_i * P_mem_i * P_sem_bar_i
    Room-level belief    b(r,c) = 1 - prod_{o_i in O(r,c)} (1 - q_i)
                         O(r,c) = {o_i in O(r) : l_i = c}
    Bracket              max_i q_i <= b(r,c)
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hovsg.graph.graph import Graph
from hovsg.graph.room import Room
from hovsg.utils.uncertainty import compute_class_containment_belief


def make_object(object_id, class_id, class_label, probability):
    """A minimal object node carrying what eq. (noisyor) reads: l_i and q_i."""
    return SimpleNamespace(
        object_id=object_id,
        label_idx=class_id,
        name=class_label,
        p_obj=probability,
    )


def make_room(room_id, objects):
    room = Room(room_id, room_id.split("_")[0])
    room.objects = list(objects)
    return room


def beliefs_by_class_id(room):
    return {entry["class_id"]: entry for entry in room.class_containment_belief}


# ---------------------------------------------------------------------------
# eq. (noisyor) with a single object: the product has one factor, so the room
# belief is the object probability itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probability", [0.0, 0.48, 1e-9, 1.0 - 1e-9, 1.0])
def test_single_object_belief_is_its_own_probability(probability):
    room = make_room("0_0", [make_object("a", 2, "chair", probability)])

    room.compute_class_containment_beliefs()

    assert room.class_containment_belief == [
        {
            "class_id": 2,
            "class_label": "chair",
            "belief": pytest.approx(1.0 - (1.0 - probability)),
        }
    ]


# ---------------------------------------------------------------------------
# eq. (noisyor) with several objects of one class: every object contributes a
# factor, and none of them is dropped, merged or replaced by a maximum.
# ---------------------------------------------------------------------------


def test_two_objects_of_one_class_multiply_their_complements():
    room = make_room(
        "0_0",
        [
            make_object("a", 2, "chair", 0.5),
            make_object("b", 2, "chair", 0.4),
        ],
    )

    room.compute_class_containment_beliefs()

    expected = 1.0 - (1.0 - 0.5) * (1.0 - 0.4)  # 0.7
    assert room.class_containment_belief == [
        {"class_id": 2, "class_label": "chair", "belief": pytest.approx(expected)}
    ]


def test_three_objects_of_one_class_multiply_all_three_complements():
    probabilities = [0.5, 0.4, 0.2]
    room = make_room(
        "0_0",
        [
            make_object(str(index), 1, "table", probability)
            for index, probability in enumerate(probabilities)
        ],
    )

    room.compute_class_containment_beliefs()

    expected = 1.0 - (1.0 - 0.5) * (1.0 - 0.4) * (1.0 - 0.2)  # 0.76
    assert beliefs_by_class_id(room)[1]["belief"] == pytest.approx(expected)


def test_spatially_coincident_objects_are_not_collapsed():
    """Duplicates are still two terms: the paper's product has no dedup step."""
    shared_geometry = np.zeros((8, 3))
    objects = []
    for object_id, probability in (("a", 0.3), ("b", 0.8)):
        objectt = make_object(object_id, 1, "table", probability)
        objectt.vertices = shared_geometry
        objectt.pcd = SimpleNamespace(points=shared_geometry)
        objects.append(objectt)
    room = make_room("0_0", objects)

    room.compute_class_containment_beliefs()

    expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.8)  # 0.86, not the maximum 0.8
    assert beliefs_by_class_id(room)[1]["belief"] == pytest.approx(expected)


def test_each_further_object_raises_the_belief_above_the_bracket_bound():
    """eq. (bracket): max_i q_i <= b(r, c), and b grows with every object."""
    probabilities = [0.2, 0.6, 0.35, 0.05]
    beliefs = []
    for count in range(1, len(probabilities) + 1):
        room = make_room(
            "0_0",
            [
                make_object(str(index), 4, "lamp", probability)
                for index, probability in enumerate(probabilities[:count])
            ],
        )
        room.compute_class_containment_beliefs()
        belief = beliefs_by_class_id(room)[4]["belief"]
        # The bound holds exactly in real arithmetic. Evaluating the paper's
        # 1 - prod(1 - q_i) in float64 can land one ulp below it, which the
        # tolerance absorbs without hiding a formula that fails the bound.
        assert belief >= max(probabilities[:count]) - 1e-12
        assert belief <= 1.0
        beliefs.append(belief)

    assert beliefs == sorted(beliefs)


# ---------------------------------------------------------------------------
# Boundary values. q_i = 0 contributes the factor 1 and leaves the belief
# unchanged; q_i = 1 saturates the product regardless of the other objects.
# ---------------------------------------------------------------------------


def test_zero_probability_object_contributes_no_evidence():
    room = make_room(
        "0_0",
        [
            make_object("a", 2, "chair", 0.6),
            make_object("b", 2, "chair", 0.0),
        ],
    )

    room.compute_class_containment_beliefs()

    assert beliefs_by_class_id(room)[2]["belief"] == pytest.approx(0.6)


def test_certain_object_saturates_the_belief():
    room = make_room(
        "0_0",
        [
            make_object("a", 2, "chair", 0.6),
            make_object("b", 2, "chair", 1.0),
        ],
    )

    room.compute_class_containment_beliefs()

    assert beliefs_by_class_id(room)[2]["belief"] == 1.0


def test_near_boundary_probabilities_stay_inside_the_unit_interval():
    room = make_room(
        "0_0",
        [
            make_object("a", 2, "chair", 1e-12),
            make_object("b", 2, "chair", 1.0 - 1e-12),
        ],
    )

    room.compute_class_containment_beliefs()

    expected = 1.0 - (1.0 - 1e-12) * 1e-12
    belief = beliefs_by_class_id(room)[2]["belief"]
    assert belief == pytest.approx(expected)
    assert 0.0 <= belief <= 1.0


# ---------------------------------------------------------------------------
# Classes are propagated independently: one entry per class, no mixing.
# ---------------------------------------------------------------------------


def test_classes_are_propagated_independently():
    room = make_room(
        "0_0",
        [
            make_object("chair_a", 10, "chair", 0.5),
            make_object("chair_b", 10, "chair", 0.4),
            make_object("table", 12, "desk", 0.9),
            make_object("lamp_a", 7, "lamp", 0.25),
            make_object("lamp_b", 7, "lamp", 0.75),
        ],
    )

    room.compute_class_containment_beliefs()

    assert room.class_containment_belief == [
        {
            "class_id": 7,
            "class_label": "lamp",
            "belief": pytest.approx(1.0 - (1.0 - 0.25) * (1.0 - 0.75)),
        },
        {
            "class_id": 10,
            "class_label": "chair",
            "belief": pytest.approx(1.0 - (1.0 - 0.5) * (1.0 - 0.4)),
        },
        {
            "class_id": 12,
            "class_label": "desk",
            "belief": pytest.approx(0.9),
        },
    ]
    class_ids = [entry["class_id"] for entry in room.class_containment_belief]
    assert len(class_ids) == len(set(class_ids))


def test_objects_of_other_classes_do_not_enter_the_product():
    """Restricting the product to O(r, c) is what separates the classes."""
    room = make_room(
        "0_0",
        [
            make_object("chair", 10, "chair", 0.5),
            make_object("desk", 12, "desk", 0.99),
        ],
    )

    room.compute_class_containment_beliefs()

    assert beliefs_by_class_id(room)[10]["belief"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Undefined q_i. The paper's product has no factor for a missing probability.
# ---------------------------------------------------------------------------


def test_object_with_undefined_probability_is_left_out_of_the_product():
    room = make_room(
        "0_0",
        [
            make_object("defined", 0, "chair", 0.6),
            make_object("undefined", 0, "chair", None),
            make_object("unlabeled", None, "chair", 0.9),
            make_object("all-undefined", 3, "lamp", None),
        ],
    )

    room.compute_class_containment_beliefs()

    assert room.class_containment_belief == [
        {"class_id": 0, "class_label": "chair", "belief": pytest.approx(0.6)}
    ]
    assert 3 not in beliefs_by_class_id(room)


def test_empty_object_set_has_no_belief():
    assert compute_class_containment_belief([]) is None
    assert make_room("0_0", []).compute_class_containment_beliefs() == []


# ---------------------------------------------------------------------------
# Repeated evaluation and object ordering.
# ---------------------------------------------------------------------------


def test_repeated_propagation_does_not_accumulate():
    room = make_room(
        "0_0",
        [
            make_object("a", 2, "chair", 0.5),
            make_object("b", 2, "chair", 0.4),
        ],
    )

    first = list(room.compute_class_containment_beliefs())
    second = list(room.compute_class_containment_beliefs())

    assert second == first
    assert first == [
        {
            "class_id": 2,
            "class_label": "chair",
            "belief": pytest.approx(1.0 - (1.0 - 0.5) * (1.0 - 0.4)),
        }
    ]


def test_object_order_does_not_change_the_room_node():
    probabilities = [0.1, 0.5, 0.9]
    forward = make_room(
        "0_0",
        [
            make_object(str(index), 2, "chair", probability)
            for index, probability in enumerate(probabilities)
        ],
    )
    backward = make_room("0_0", list(reversed(forward.objects)))

    forward.compute_class_containment_beliefs()
    backward.compute_class_containment_beliefs()

    expected = 1.0 - (1.0 - 0.1) * (1.0 - 0.5) * (1.0 - 0.9)
    assert forward.class_containment_belief[0]["belief"] == pytest.approx(expected)
    assert backward.class_containment_belief[0]["belief"] == pytest.approx(expected)


def test_class_entries_are_ordered_by_class_id_whatever_the_object_order():
    room = make_room(
        "0_0",
        [
            make_object("c", 12, "desk", 0.3),
            make_object("a", 7, "lamp", 0.3),
            make_object("b", 10, "chair", 0.3),
        ],
    )

    room.compute_class_containment_beliefs()

    assert [entry["class_id"] for entry in room.class_containment_belief] == [7, 10, 12]


# ---------------------------------------------------------------------------
# Graph-level propagation: one pass over rooms, each reading only its own
# objects.
# ---------------------------------------------------------------------------


def test_graph_propagation_uses_the_assigned_label_and_fused_probability():
    room = make_room(
        "0_0",
        [
            make_object("chair", 0, "chair", 0.7),
            make_object("table", 1, "table", 0.2),
        ],
    )
    graph = SimpleNamespace(rooms=[room])

    Graph.propagate_object_probabilities_to_rooms(graph)

    assert room.class_containment_belief == [
        {"class_id": 0, "class_label": "chair", "belief": pytest.approx(0.7)},
        {"class_id": 1, "class_label": "table", "belief": pytest.approx(0.2)},
    ]


def test_rooms_do_not_share_objects():
    kitchen = make_room(
        "0_0",
        [
            make_object("kitchen_chair_a", 10, "chair", 0.5),
            make_object("kitchen_chair_b", 10, "chair", 0.4),
        ],
    )
    bedroom = make_room("0_1", [make_object("bedroom_bed", 20, "bed", 0.8)])
    graph = SimpleNamespace(rooms=[kitchen, bedroom])

    Graph.propagate_object_probabilities_to_rooms(graph)

    assert kitchen.class_containment_belief == [
        {
            "class_id": 10,
            "class_label": "chair",
            "belief": pytest.approx(1.0 - (1.0 - 0.5) * (1.0 - 0.4)),
        }
    ]
    assert bedroom.class_containment_belief == [
        {"class_id": 20, "class_label": "bed", "belief": pytest.approx(0.8)}
    ]
    # The chairs of one room leave no trace in the other.
    assert 10 not in beliefs_by_class_id(bedroom)
    assert 20 not in beliefs_by_class_id(kitchen)


# ---------------------------------------------------------------------------
# Serialization: the propagated value reaches the scene graph unchanged.
# ---------------------------------------------------------------------------


def test_serialized_belief_is_the_propagated_value(tmp_path, monkeypatch):
    monkeypatch.setattr("hovsg.graph.room.o3d.io.write_point_cloud", lambda *_: True)
    room = make_room(
        "0_0",
        [
            make_object("a", 2, "chair", 0.5),
            make_object("b", 2, "chair", 0.4),
            make_object("c", 5, "sink", 0.123456789),
        ],
    )
    room.pcd = object()
    room.centroid = [0.0, 0.0, 0.0]
    room.vertices = np.zeros((4, 2))

    room.compute_class_containment_beliefs()
    room.save(tmp_path)

    serialized = json.loads((tmp_path / "0_0.json").read_text(encoding="utf-8"))
    assert serialized["class_containment_belief"] == [
        {
            "class_id": 2,
            "class_label": "chair",
            "belief": 1.0 - (1.0 - 0.5) * (1.0 - 0.4),
        },
        # Written as the paper's expression, not as 0.123456789: the room node
        # holds exactly what float64 evaluation of eq. (noisyor) produces.
        {"class_id": 5, "class_label": "sink", "belief": 1.0 - (1.0 - 0.123456789)},
    ]
