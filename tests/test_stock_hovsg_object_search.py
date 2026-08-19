"""The stock HOV-SG object search, measured by the belief-guided metrics.

``application/visualize_query_graph.py`` turns HOV-SG's own object ranking
into an ordered room list and hands it to the route, oracle and metric
evaluator of ``uncertsg_eval``.  These tests pin the two halves of that: that
the retrieval stays HOV-SG's (full ranked list, its own background filtering,
its own ordering), and that the measurement is the one the belief-guided
search is measured by, down to equal figures for an equal room order.

The scene below is a star with the start pose at its centre:

    room 0_2 "far"                       10 m from the start
        |
    room 0_0 "start" --- 2 m --- room 0_1 "near"
        |
        +-------------- 5 m --- room 0_3 "mid"

HOV-SG places mugs in "near" and "far".  The scene really holds one in "far"
and one in "mid", and it placed nothing in "mid", so the oracle stops in a
room no system could have retrieved.
"""

import pytest

from application.visualize_query_graph import (HOVSG_ORDER, NEGATIVE_LABELS,
                                               hovsg_room_order, search_query)
from metrics.evaluate_baselines import format_query_debug
from metrics.navigation import build_navigation_graph
from metrics.object_search import (RouteEvaluator, Vocabulary,
                                   assign_gt_classes_to_rooms, search_scene)

#: A stand-in for the HM3DSem vocabulary; an index here is the ``label_idx``
#: the graph stores and the row of the embedding matrix below.
CLASSES = ("mug", "kettle")
EMBEDDINGS = [[1.0, 0.0], [0.0, 1.0]]

START_NODE = "hovsg:nav:0:start"
NEAR = "hovsg:room:0_1"
FAR = "hovsg:room:0_2"
MID = "hovsg:room:0_3"

#: Raw HOV-SG room identifier -> the node id the adapter normalizes it into.
ROOM_NODE_IDS = {
    "0_0": "hovsg:room:0_0",
    "0_1": NEAR,
    "0_2": FAR,
    "0_3": MID,
}


def _footprint(x0, x1, z0, z1):
    """A room floor footprint sampled on the pipeline's 0.5 m grid."""
    steps = lambda low, high: [low + 0.5 * i for i in range(int((high - low) * 2) + 1)]
    return [[x, z] for x in steps(x0, x1) for z in steps(z0, z1)]


def _room(room_id, footprint, beliefs):
    return {
        "id": room_id,
        "raw_id": room_id.rsplit(":", 1)[-1],
        "attributes": {
            "floor_id": "0",
            "vertices": footprint,
            "class_containment_belief": beliefs,
        },
    }


def _belief(label, belief):
    return {"class_id": CLASSES.index(label), "class_label": label, "belief": belief}


def _object(object_id, label, similarity):
    return {
        "id": object_id,
        "attributes": {
            "name": label,
            "label_idx": CLASSES.index(label),
            "label_cos_sim": similarity,
        },
    }


def _nav(node_id, x, z):
    return {"id": node_id, "attributes": {"pos": [x, 0.0, z], "floor_id": "0"}}


def _scene_graph():
    return {
        "Nodes": {
            "rooms": [
                _room("hovsg:room:0_0", _footprint(-1, 1, -1, 1), []),
                _room(NEAR, _footprint(1.5, 3, -1, 1), [_belief("mug", 0.4)]),
                _room(FAR, _footprint(-1, 1, 9, 11), [_belief("mug", 0.9)]),
                _room(MID, _footprint(4, 6, -1, 1), []),
            ],
            "objects": [
                _object("object:near", "mug", 0.20),
                _object("object:far", "mug", 0.30),
            ],
            "nav": [
                _nav(START_NODE, 0.0, 0.0),
                _nav("hovsg:nav:0:near", 2.0, 0.0),
                _nav("hovsg:nav:0:far", 0.0, 10.0),
                _nav("hovsg:nav:0:mid", 5.0, 0.0),
            ],
        },
        "Edges": [
            ("object:near", NEAR, "OBJECT_BELONGS_TO_ROOM"),
            ("object:far", FAR, "OBJECT_BELONGS_TO_ROOM"),
            (START_NODE, "hovsg:room:0_0", "NAV_BELONGS_TO_ROOM"),
            ("hovsg:nav:0:near", NEAR, "NAV_BELONGS_TO_ROOM"),
            ("hovsg:nav:0:far", FAR, "NAV_BELONGS_TO_ROOM"),
            ("hovsg:nav:0:mid", MID, "NAV_BELONGS_TO_ROOM"),
            (START_NODE, "hovsg:nav:0:near", "NAVIGABLE_PATH"),
            (START_NODE, "hovsg:nav:0:far", "NAVIGABLE_PATH"),
            (START_NODE, "hovsg:nav:0:mid", "NAVIGABLE_PATH"),
        ],
    }


def _gt():
    """One mug in "far" and one in "mid", the room the graph placed nothing in."""
    return {
        "all_objects": [
            {"id": 0, "category": "mug", "centroid": [0.0, 1.0, 10.0]},
            {"id": 1, "category": "mug", "centroid": [5.0, 1.0, 0.0]},
        ]
    }


def _routes(scene_graph=None, gt_data=None):
    scene_graph = scene_graph or _scene_graph()
    ground_truth = assign_gt_classes_to_rooms(
        gt_data or _gt(), scene_graph, floor_id="0"
    )
    navigation = build_navigation_graph(scene_graph, floor_id="0")
    return RouteEvaluator.for_scene(navigation, START_NODE, ground_truth)


class _StubRoom:
    def __init__(self, room_id):
        self.room_id = room_id


class _StubGraph:
    """Stands in for ``hovsg.graph.Graph``, recording how it was queried.

    ``ranking`` names, per query class, the parent rooms of the objects
    HOV-SG returns, in the order it returns them.  One entry is one retrieved
    object, so a room named twice is a room two retrieved objects lie in.
    """

    def __init__(self, ranking, room_ids=("0_0", "0_1", "0_2", "0_3"), objects=6):
        self.rooms = [_StubRoom(room_id) for room_id in room_ids]
        self.objects = list(range(objects))
        self._ranking = ranking
        self.calls = []

    def query_object(self, query, room_ids, top_k, negative_prompt):
        self.calls.append(
            {
                "query": query,
                "room_ids": list(room_ids),
                "top_k": top_k,
                "negative_prompt": list(negative_prompt),
            }
        )
        rooms = self._ranking.get(query, [])
        indices = [
            next(i for i, room in enumerate(self.rooms) if room.room_id == raw)
            for raw in rooms
        ]
        return list(range(len(indices))), indices


def _search(ranking, query_class="mug", routes=None):
    graph = _StubGraph({query_class: ranking})
    record = search_query(
        graph, query_class, routes or _routes(), ROOM_NODE_IDS, "scene"
    )
    return graph, record


def _belief_guided(query_class="mug"):
    """The belief-guided search over the same scene, from the same start pose."""
    scene_graph = _scene_graph()
    scene = search_scene(
        "scene",
        scene_graph,
        _gt(),
        build_navigation_graph(scene_graph, floor_id="0"),
        START_NODE,
        Vocabulary(CLASSES, EMBEDDINGS),
        floor_id="0",
    )
    return next(q for q in scene["queries"] if q["class_label"] == query_class)


class TestRetrievalIsStock:
    def test_the_full_ranked_object_list_is_requested(self):
        graph, _ = _search(["0_1", "0_2"])

        call = graph.calls[0]
        # top_k is the size of the graph, so nothing is truncated away.
        assert call["top_k"] == len(graph.objects)
        assert call["query"] == "mug"

    def test_hovsgs_own_background_filtering_is_used_unchanged(self):
        graph, _ = _search(["0_1"])

        assert graph.calls[0]["negative_prompt"] == NEGATIVE_LABELS == ["background"]

    def test_every_room_of_the_graph_is_searched(self):
        graph, _ = _search(["0_1"])

        assert graph.calls[0]["room_ids"] == list(range(len(graph.rooms)))

    def test_the_query_class_reaches_hovsg_verbatim(self):
        """No label resolution stands between the query and the retrieval."""
        graph, record = _search(["0_2"], query_class="mug")

        assert graph.calls[0]["query"] == "mug"
        assert record["resolved_class"] == "mug"
        assert record["resolution"] is None


class TestRoomOrder:
    def test_a_room_keeps_the_position_of_its_first_retrieved_object(self):
        graph = _StubGraph({"mug": ["0_2", "0_1", "0_2", "0_1", "0_3"]})

        rooms, objects = hovsg_room_order(graph, "mug", ROOM_NODE_IDS)

        assert rooms == [FAR, NEAR, MID]
        # Every retrieved object was read, only the repeated rooms collapsed.
        assert objects == 5

    def test_the_retrieval_order_is_preserved(self):
        graph = _StubGraph({"mug": ["0_1", "0_2"]})

        rooms, _ = hovsg_room_order(graph, "mug", ROOM_NODE_IDS)

        assert rooms == [NEAR, FAR]

    def test_a_room_missing_from_the_serialized_graph_is_an_error(self):
        graph = _StubGraph({"mug": ["0_9"]}, room_ids=("0_9",))

        with pytest.raises(KeyError):
            hovsg_room_order(graph, "mug", ROOM_NODE_IDS)


class TestRouteAndMetrics:
    def test_a_search_ends_in_the_first_gt_positive_room_entered(self):
        _, record = _search(["0_1", "0_2"])

        walked = record["orders"][HOVSG_ORDER]
        # "near" holds no mug, so the robot goes on to "far", which does.
        assert walked["route"] == [NEAR, FAR]
        assert record["correct_candidate_rooms"] == [FAR]
        assert record["searched"] is True
        assert record["failure"] is None

    def test_the_metrics_are_the_route_against_the_oracle(self):
        _, record = _search(["0_1", "0_2"])

        walked = record["orders"][HOVSG_ORDER]
        # 2 m out to "near", then 12 m back through the start pose to "far".
        assert walked["distance_m"] == pytest.approx(14.0)
        assert walked["excess_m"] == pytest.approx(14.0 - 5.0)
        assert walked["path_efficiency"] == pytest.approx(5.0 / 14.0)
        assert walked["rooms_entered"] == 2

    def test_the_oracle_is_read_off_the_ground_truth_not_the_candidates(self):
        _, record = _search(["0_1", "0_2"])

        # The scene holds a mug 5 m away in "mid", which HOV-SG never
        # retrieved and its graph placed nothing in.  The oracle goes there
        # anyway, so the excess above charges the mapping for it.
        assert record["oracle_room_id"] == MID
        assert record["oracle_distance_m"] == pytest.approx(5.0)
        assert record["oracle_room_is_candidate"] is False
        assert MID not in [candidate["room_id"] for candidate in record["candidates"]]

class TestUnreachableRoomsStopTheSearch:
    """An unreachable room is walked at, not removed from the order."""

    @staticmethod
    def _cut_off_far():
        """The same scene with "far" severed from the navigation graph."""
        scene_graph = _scene_graph()
        scene_graph["Edges"] = [
            edge
            for edge in scene_graph["Edges"]
            if edge[:2] != (START_NODE, "hovsg:nav:0:far")
        ]
        return _routes(scene_graph)

    def test_it_stays_a_candidate(self):
        _, record = _search(["0_2", "0_1"], routes=self._cut_off_far())

        # Both rooms are still offered; the unreachable one is flagged, not
        # removed, and carries no distance from the start pose.
        assert [c["room_id"] for c in record["candidates"]] == [FAR, NEAR]
        assert record["unreachable_candidate_rooms"] == [FAR]
        far = record["candidates"][0]
        assert far["start_distance_m"] is None and far["entry_node_id"] is None

    def test_reaching_it_fails_the_query(self):
        _, record = _search(["0_2", "0_1"], routes=self._cut_off_far())

        assert record["failure"] == "failed_with_unreachable"
        assert record["failed_orders"] == {HOVSG_ORDER: "failed_with_unreachable"}
        assert record["blocked_rooms"] == {HOVSG_ORDER: FAR}
        assert record["searched"] is False
        assert record["orders"] == {}
        assert record["oracle_distance_m"] is None

    def test_the_rooms_behind_it_are_never_tried(self):
        # "mid" holds a mug and is reachable, but it is ranked behind the
        # unreachable "far", so the robot never gets to it.
        _, record = _search(["0_2", "0_3"], routes=self._cut_off_far())

        assert record["failure"] == "failed_with_unreachable"
        assert record["blocked_rooms"] == {HOVSG_ORDER: FAR}

    def test_a_correct_room_ranked_ahead_of_it_still_succeeds(self):
        # "mid" holds a mug and comes first, so the search ends there and the
        # unreachable room behind it never costs anything.
        _, record = _search(["0_3", "0_2"], routes=self._cut_off_far())

        assert record["failure"] is None
        assert record["searched"] is True
        walked = record["orders"][HOVSG_ORDER]
        assert walked["route"] == [MID]
        assert walked["distance_m"] == pytest.approx(5.0)
        # "far" is unreachable now, so the oracle is the mug in "mid".
        assert walked["path_efficiency"] == pytest.approx(1.0)

    def test_walking_every_reachable_room_without_finding_it_is_a_different_failure(self):
        _, record = _search(["0_1"])

        assert record["failure"] == "no_GT_positive_candidate"
        assert record["blocked_rooms"] == {}


class TestFailuresCarryNoDistance:
    @pytest.mark.parametrize(
        "ranking, query_class, failure",
        [
            (["0_1", "0_2"], "kettle", "absent_class"),
            ([], "mug", "no_candidate_rooms"),
            (["0_1"], "mug", "no_GT_positive_candidate"),
            (["0_0", "0_1"], "mug", "no_GT_positive_candidate"),
        ],
    )
    def test_the_failure_is_recorded_instead_of_a_distance(
        self, ranking, query_class, failure
    ):
        _, record = _search(ranking, query_class=query_class)

        assert record["failure"] == failure
        assert record["searched"] is False
        assert record["orders"] == {}
        assert record["oracle_distance_m"] is None

    def test_an_absent_class_is_never_queried_against_the_graph(self):
        graph, record = _search(["0_1"], query_class="kettle")

        assert record["failure"] == "absent_class"
        assert graph.calls == []


class TestTheTwoSearchesShareTheEvaluator:
    """Equal room orders must produce equal figures across the two scripts."""

    @pytest.mark.parametrize(
        "ranking, order",
        [(["0_1", "0_2"], "proximity"), (["0_2", "0_1"], "belief")],
    )
    def test_an_equal_room_order_is_measured_identically(self, ranking, order):
        _, record = _search(ranking)
        belief_guided = _belief_guided()

        stock = record["orders"][HOVSG_ORDER]
        reference = belief_guided["orders"][order]
        assert stock["order"] == reference["order"]
        assert stock == reference

    def test_both_read_the_same_ground_truth_and_oracle(self):
        _, record = _search(["0_1", "0_2"])
        belief_guided = _belief_guided()

        for field in (
            "holding_rooms",
            "reachable_holding_rooms",
            "oracle_room_id",
            "oracle_distance_m",
            "starts_in_a_holding_room",
        ):
            assert record[field] == belief_guided[field], field

    def test_the_shared_debug_trace_renders_a_stock_record(self):
        _, record = _search(["0_1", "0_2"])

        trace = format_query_debug(record)

        assert "[query] mug" in trace
        # The ranking position stands where the belief-guided trace prints the
        # belief, since a stock candidate carries no belief to print.
        assert "rank=1" in trace
        assert "belief=" not in trace
        assert "<- oracle" in trace
