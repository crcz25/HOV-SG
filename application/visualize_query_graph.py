"""Stock HOV-SG object search, measured by the belief-guided search metrics.

This script runs HOV-SG's own object query over every class of the HM3DSem
vocabulary and hands the rooms it retrieves to the *same* route, oracle and
metric evaluator the belief-guided search in ``uncertsg_eval`` is measured by,
so the two are directly comparable.  Nothing here scores, re-ranks or
re-labels what HOV-SG returns:

* **Retrieval is stock.**  ``Graph.query_object`` embeds the query class with
  CLIP, compares it against the stored object embeddings, applies HOV-SG's own
  ``background`` negative filtering and returns its ranking.  No belief and no
  label resolution enters this side; the query class is used verbatim, both
  against the graph and against the ground truth.
* **Only the truncation is lifted.**  HOV-SG's ``top_k`` is set to the number
  of objects in the graph, so the full ranked object list is returned instead
  of a fixed-size head.
* **Rooms follow the objects.**  Each retrieved object's parent room, in
  retrieval order, with repeats of a room already seen dropped.  That ordered
  room list is the *only* thing this system contributes to the evaluation.

Everything downstream of that list -- the ground-truth projection ``y(r, c)``
onto the predicted rooms, the room entry nodes ``v_r``, the navigation graph,
the start pose, the walk, the success condition, the oracle ``d*(c)`` and the
four reported metrics -- is :class:`~metrics.object_search.RouteEvaluator`
from ``uncertsg_eval``, called here exactly as the belief-guided search calls
it.  The scene inputs come from the belief-guided run's own configuration
file, so neither the graph, the ground truth, the evaluated floor nor the
start pose can drift between them.

No room is removed from the order before the walk.  The robot goes at every
room HOV-SG ranked, in that order, so a room with no navigation entry node
reachable from the start pose is a room it drives at and cannot enter: the
query ends there and HOV-SG is charged for having ranked it ahead of a room
holding the class.

The reported figures are the shared ones: ``distance_m`` is the distance
walked, ``excess_m`` is ``distance_m - d*(c)``, ``path_efficiency`` is
``d*(c) / distance_m`` and ``rooms_entered`` counts the rooms visited.  A
query with no route records why instead: ``absent_class`` when the scene holds
no instance of the class, ``no_candidate_rooms`` when HOV-SG returns no room,
``failed_with_unreachable`` when an unreachable room stopped the robot before
it found the class, and ``no_GT_positive_candidate`` when it entered every
ranked room and none of them held it.

Run it with the original interactive query-and-visualize loop by setting
``main.mode=interactive``.
"""

from __future__ import annotations

import contextlib
import io
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import open3d as o3d
from omegaconf import DictConfig

from hovsg.graph.graph import Graph

# The evaluation lives in its own top-level package layout, imported the same
# way its own entry point imports it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVAL_ROOT = _REPO_ROOT / "uncertsg_eval"
if str(_EVAL_ROOT) not in sys.path:
    sys.path.append(str(_EVAL_ROOT))

from main import load_params  # noqa: E402  (needs the path entry above)
from metrics.evaluate_baselines import (  # noqa: E402
    SearchConfig, format_query_debug, format_table, latex_rows, load_scene,
    scene_parameters, write_result)
from metrics.object_search import (RouteEvaluator, Vocabulary,  # noqa: E402
                                   aggregate, assign_gt_classes_to_rooms,
                                   blank_query_record)

# pylint: disable=all

#: The name this system's single ranking is reported under, in the column the
#: belief-guided run fills with one of its four orders.
HOVSG_ORDER = "hovsg"

#: HOV-SG's own negative prompt, as ``Graph.query_hierarchy`` sets it: an
#: object whose best-matching category is this one rather than the query is
#: dropped from the ranking.
NEGATIVE_LABELS = ["background"]

#: Decimals in the progress lines printed while a scene runs.
_PRINTED_DIGITS = 2


def _resolve(value: Any, root: Path) -> Path:
    """Resolve a configured path, relative ones against ``root``."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def hovsg_object_ranking(graph: Graph, query_class: str) -> Tuple[List[int], List[int]]:
    """Return HOV-SG's full ranking of its objects for one query class.

    This is ``Graph.query_object`` as HOV-SG ships it, with two arguments
    chosen and nothing else touched:

    * ``top_k`` is the number of objects in the graph, which returns the whole
      ranked list rather than a fixed-size head.
    * ``room_ids`` names every room, which is the global object search this
      evaluation queries.  HOV-SG builds its candidate object list from those
      rooms, so this covers exactly the objects of the graph.

    The negative filtering, the CLIP embedding of the query and the ordering
    of the survivors by their score are all HOV-SG's own.

    Returns:
        The retrieved object indices into ``graph.objects`` and, aligned with
        them, the index into ``graph.rooms`` of each object's parent room.
    """
    return graph.query_object(
        query_class,
        room_ids=list(range(len(graph.rooms))),
        top_k=len(graph.objects),
        negative_prompt=NEGATIVE_LABELS,
    )


def hovsg_room_order(
    graph: Graph, query_class: str, room_node_ids: Dict[str, str]
) -> Tuple[List[str], int]:
    """Turn HOV-SG's ranked objects into the ordered room list it implies.

    Each retrieved object contributes its parent room, in retrieval order; a
    room already contributed is not contributed again, so the first object
    that puts a room on the list fixes that room's position.  Rooms are named
    by the normalized node id the evaluation uses, which is what makes this
    list comparable with the belief-guided one.

    Returns:
        The ordered, de-duplicated room ids and how many objects HOV-SG ranked.
    """
    object_ids, room_ids = hovsg_object_ranking(graph, query_class)

    ordered: List[str] = []
    seen = set()
    for room_index in room_ids:
        room_id = graph.rooms[room_index].room_id
        node_id = room_node_ids.get(room_id)
        if node_id is None:
            raise KeyError(
                f"HOV-SG room '{room_id}' has no node in the serialized scene graph"
            )
        if node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)
    return ordered, len(object_ids)


def search_query(
    graph: Graph,
    query_class: str,
    routes: RouteEvaluator,
    room_node_ids: Dict[str, str],
    scene_id: str,
) -> Dict[str, Any]:
    """Run one query of the vocabulary and measure the route it produces.

    The record has the shape the belief-guided search emits, so the two runs
    aggregate and print through the same code.
    """
    record = blank_query_record(scene_id, query_class)

    # y(r, c) over the predicted rooms and the oracle d*(c), both read from
    # the ground truth alone and both independent of what HOV-SG retrieves.  A
    # class the scene does not hold has no route to measure.
    target = routes.target(query_class)
    if target is None:
        record["failure"] = "absent_class"
        return record
    record["present_in_gt"] = True
    record["holding_rooms"] = list(target.holding_rooms)
    record["reachable_holding_rooms"] = [
        dict(room) for room in target.reachable_holding_rooms
    ]
    # HOV-SG queries the class itself: there is no resolution step to record.
    record["resolved_class"] = query_class

    ranked_rooms, ranked_objects = hovsg_room_order(graph, query_class, room_node_ids)
    record["retrieved_objects"] = ranked_objects

    # The shared evaluator decides everything from here.  Every retrieved room
    # stays in the order, unreachable ones included: the robot walks the
    # ranking and a room it cannot enter ends the query where it stands.
    outcome = target.evaluate(ranked_rooms)
    record["unreachable_candidate_rooms"] = list(outcome.unreachable_rooms)
    record["candidates"] = [
        {
            "room_id": room_id,
            "rank": rank,
            "entry_node_id": target.entry_node(room_id),
            "start_distance_m": target.start_distance(room_id),
            "holds_class": target.holds(room_id),
        }
        for rank, room_id in enumerate(ranked_rooms, start=1)
    ]
    record["correct_candidate_rooms"] = list(outcome.correct_rooms)
    if outcome.failure is not None:
        record["failure"] = outcome.failure
        record["failed_orders"] = {HOVSG_ORDER: outcome.failure}
        if outcome.blocked_room_id is not None:
            record["blocked_rooms"] = {HOVSG_ORDER: outcome.blocked_room_id}
        return record

    record["oracle_distance_m"] = target.oracle_distance_m
    record["oracle_room_id"] = target.oracle_room_id
    record["oracle_room_is_candidate"] = (
        target.start_distance(target.oracle_room_id) is not None
        and target.oracle_room_id in set(ranked_rooms)
    )
    record["starts_in_a_holding_room"] = target.starts_in_a_holding_room
    record["orders"][HOVSG_ORDER] = outcome.metrics
    record["searched"] = True
    return record


def search_scene(
    params: DictConfig,
    scene_params: Dict[str, Any],
    vocabulary: Vocabulary,
    *,
    config: SearchConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run every class of the vocabulary against one scene's HOV-SG graph."""
    loaded = load_scene(scene_params, floor_id=config.floor_id)
    ground_truth = assign_gt_classes_to_rooms(
        loaded.gt_data, loaded.scene_graph, floor_id=config.floor_id
    )
    routes = RouteEvaluator.for_scene(
        loaded.navigation, loaded.start_node, ground_truth
    )
    # HOV-SG names its rooms by the raw identifier it serialized them under;
    # the evaluation names them by the node id the adapter normalized that
    # into.  The serialized graph carries both, so the mapping is read from it
    # rather than reconstructed.
    room_node_ids = {
        node["raw_id"]: node["id"] for node in loaded.scene_graph["Nodes"]["rooms"]
    }

    if verbose:
        print(loaded.describe_start())
        print(f"  loading the HOV-SG graph from {scene_params['scene_graph']}")
    graph = Graph(params)
    graph.load_graph(str(scene_params["scene_graph"]))

    queries = []
    for query_class in vocabulary.classes:
        # HOV-SG's query path narrates every candidate it scores, which is
        # thousands of lines per scene at this query set's size.  Silencing it
        # changes nothing it computes.
        with contextlib.redirect_stdout(io.StringIO()):
            query = search_query(
                graph, query_class, routes, room_node_ids, loaded.scene_id
            )
        queries.append(query)
        if config.per_query_debug:
            trace = format_query_debug(query)
            if trace:
                print(trace)

    scene = {
        "scene_id": loaded.scene_id,
        "floor_id": config.floor_id,
        "start_node_id": loaded.start_node,
        "rooms": sorted(loaded.navigation.room_nodes),
        "vocabulary_classes": len(vocabulary),
        "predicted_labels": sorted({str(obj.name) for obj in graph.objects}),
        "ground_truth_placement": ground_truth.diagnostics,
        "queries": queries,
        "start_pose": loaded.start,
        "navigation": loaded.navigation_summary(),
        "objects": len(graph.objects),
    }
    if verbose:
        placement = scene["ground_truth_placement"]
        print(
            f"  ground truth: {placement['placed']} of {placement['objects']} objects lie "
            f"in a predicted room "
            f"({placement['footprint_cell_m']:.{_PRINTED_DIGITS}f} m footprint cell), "
            f"{len(placement['classes_outside_every_room'])} of {placement['classes']} "
            f"classes lie outside every one"
        )
        present = [query for query in queries if query["present_in_gt"]]
        searched = sum(1 for query in present if query["searched"])
        print(
            f"  {len(queries)} vocabulary classes queried against "
            f"{len(graph.objects)} HOV-SG objects: {len(present)} present in the "
            f"ground truth, of which {searched} searched and "
            f"{len(present) - searched} failed"
        )
    return scene


def run_search(params: DictConfig, *, verbose: bool = True) -> Dict[str, Any]:
    """Evaluate the stock HOV-SG search over every scene of the eval config."""
    eval_config = _resolve(params.main.eval_config, _REPO_ROOT)
    if verbose:
        print(f"Reading the belief-guided run's configuration: {eval_config}")
    eval_params = load_params(eval_config)
    config = SearchConfig.from_params(eval_params)

    vocabulary = Vocabulary(
        eval_params["hm3dsem_class_map"], eval_params["hm3dsem_embeddings"]
    )
    if verbose:
        print(f"Query set: {len(vocabulary)} HM3DSem classes")

    scenes = []
    for scene_params in scene_parameters(eval_params):
        if verbose:
            print(f"Scene {scene_params['scene_id']}: loading")
        scenes.append(
            search_scene(
                params, scene_params, vocabulary, config=config, verbose=verbose
            )
        )

    result = {
        "system": "hovsg (stock object search)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": {
            "floor_id": config.floor_id,
            "orders": [HOVSG_ORDER],
            "bootstrap_samples": config.bootstrap_samples,
            "confidence": config.confidence,
            "bootstrap_seed": config.bootstrap_seed,
            "vocabulary_classes": len(vocabulary),
            "eval_config": str(eval_config),
            "negative_prompt": list(NEGATIVE_LABELS),
        },
        "scenes": scenes,
        "aggregate": aggregate(
            scenes,
            orders=(HOVSG_ORDER,),
            bootstrap_samples=config.bootstrap_samples,
            confidence=config.confidence,
            seed=config.bootstrap_seed,
        ),
    }
    result["results_table_rows"] = latex_rows(result)

    result_file = params.main.get("result_file")
    if result_file:
        # Resolved next to the belief-guided run's own result file, which is
        # written relative to the uncertsg_eval directory.
        path = write_result(result, _resolve(result_file, _EVAL_ROOT))
        if verbose:
            print(f"Saved object-search results: {path}")
    return result


def run_interactive(params: DictConfig) -> None:
    """The original loop: query the graph and visualize what comes back."""
    hovsg = Graph(params)
    hovsg.load_graph(params.main.graph_path)
    # generate room names
    hovsg.generate_room_names(
            generate_method="view_embedding",
            default_room_types=[
                "office",
                "kitchen",
                "bathroom",
                "seminar room",
                "meeting room",
                "dinning room",
                "corridor",
            ])

    # loop forever and ask for query, until user click 'q'
    while True:
        query = input("Enter query: ")
        if query == "q":
            break
        floor, room, obj = hovsg.query_hierarchy(query, top_k=5)
        # visualize the query
        print(floor.floor_id, [(r.room_id, r.name) for r in room], [o.object_id for o in obj])
        # use open3d to visualize room.pcd and color the points where obj.pcd is
        for i in range(len(obj)):
            obj_pcd = obj[i].pcd.paint_uniform_color([0, 1, 0])
            room_pcd = room[i].pcd
            obj_pcd = deepcopy(obj[i].pcd)
            room_pcd = deepcopy(room[i].pcd)
            print(obj_pcd.get_center())
            o3d.visualization.draw_geometries([room_pcd, obj_pcd])


@hydra.main(version_base=None, config_path="../config", config_name="visualize_query_graph")
def main(params: DictConfig):
    mode = str(params.main.get("mode", "search"))
    if mode == "interactive":
        run_interactive(params)
        return None

    if mode != "search":
        raise ValueError(
            f"Unknown main.mode '{mode}': use 'search' to evaluate the stock "
            f"HOV-SG object search or 'interactive' for the query-and-visualize loop"
        )

    result = run_search(params)
    print()
    print(format_table(result))
    print()
    print("results.tex rows:")
    for row in result["results_table_rows"]:
        print(row)
    return result


if __name__ == "__main__":
    main()
