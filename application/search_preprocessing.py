"""Raw object-search data from the *stock* HOV-SG search, and nothing else.

This script runs HOV-SG's own object query over the complete HM3DSem
vocabulary and writes what it computed, unmodified, to three CSV files.  It is
a preprocessing stage: it produces the raw material an evaluation reads, and
it deliberately computes no evaluation of its own.  There is no re-ranking, no
normalization, no calibration, no threshold, no room score, no final rank and
no retrieval metric anywhere in this file.

What is reproduced, and from where
----------------------------------

* **Retrieval is ``Graph.query_object``.**  The candidate object list, the
  CLIP text embedding of the query, the query-versus-``background``
  comparison, the surviving-object rule and the score the survivors are
  ordered by are the lines of ``hovsg/graph/graph.py``, followed step for
  step.  ``query_object`` returns only a truncated list of node indices, so
  the scores behind it are recomputed here through the same calls in the same
  order -- and then checked, class by class, against what stock
  ``query_object`` actually returns (see ``--validation``).
* **Only the truncation is lifted.**  ``top_k`` is never applied: every
  predicted object gets a row for every vocabulary class, whether or not it
  survives the query-versus-background filter.  The score on each row is the
  one stock orders its surviving candidates by; no rank and no metric is
  written.
* **The vocabulary is the graph's own.**  ``get_label_feats`` with
  ``HM3DSEM_LABELS`` is the bank ``Graph.segment_objects`` labeled the objects
  with, so a class id in these files is exactly an object node's stored
  ``label_idx``.
* **Ground truth is the HM3DSem evaluator's.**
  ``HM3DSemanticEvaluator.load_gt_graph_from_json`` loads the walk's
  ``scene_info.json`` and per-object point clouds, which live in the same
  world frame as the predicted graph.
* **Room containment is ``Graph.segment_objects``'.**  A ground-truth object
  is placed in the predicted room whose 2-D footprint covers its centroid,
  scored by ``find_intersection_share`` at the same radius, inside the same
  floor height band, with the same ``np.argmax`` tie-break.  A GT object no
  predicted room covers is left unassigned and counted, never nudged into the
  nearest room.

No predicted object is ever matched to a ground-truth instance: the two sides
meet only through the predicted room a ground-truth centroid falls in.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from hovsg.eval.hm3dsem_evaluator import HM3DSemanticEvaluator
from hovsg.graph.graph import Graph
from hovsg.utils.clip_utils import get_text_feats_multiple_templates
from hovsg.utils.graph_utils import find_intersection_share
from hovsg.utils.label_feats import get_label_feats

# pylint: disable=all

#: The name every row of every output file is attributed to.
METHOD = "hovsg_stock"

#: HOV-SG's own negative prompt, as ``Graph.query_hierarchy`` sets it before
#: calling ``Graph.query_object``: an object whose best-matching category is
#: this one rather than the query does not survive the query.
NEGATIVE_LABELS = ["background"]

#: The radius ``Graph.segment_objects`` associates an object's 2-D points with
#: a room footprint at, and the height margin it admits an object to a floor
#: with.  Both are read from that method; neither is a threshold introduced
#: here.
ROOM_ASSIGNMENT_RADIUS = 0.2
FLOOR_HEIGHT_MARGIN = 0.2

_REPO_ROOT = Path(__file__).resolve().parents[1]

OBJECT_FIELDS = [
    "scene_id",
    "method",
    "object_id",
    "predicted_class_id",
    "class_label",
    "predicted_room",
    "object_x",
    "object_y",
    "object_z",
]

SCORE_FIELDS = [
    "scene_id",
    "method",
    "class_id",
    "class_label",
    "object_id",
    "predicted_room",
    "score",
]

ROOM_GT_FIELDS = [
    "scene_id",
    "method",
    "room_id",
    "class_id",
    "class_label",
    "room_x",
    "room_y",
    "room_z",
    "gt_contains_class",
    "gt_class_present_in_scene",
    "num_gt_instances_in_room",
]


def _resolve(value, root: Path = _REPO_ROOT) -> Path:
    """Resolve a configured path, relative ones against the repository root."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@contextlib.contextmanager
def _in_repo_root():
    """Run a block from the repository root.

    ``get_label_feats`` reads the HM3DSem vocabulary from the relative path
    ``hovsg/labels``, so it only resolves from there.  Hydra 1.3 leaves the
    working directory alone, which makes that depend on where the script was
    invoked from; this removes the dependency.
    """
    previous = Path.cwd()
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# Stock HOV-SG object search
# ---------------------------------------------------------------------------


def stock_object_candidates(graph: Graph) -> Tuple[List, List[int]]:
    """Build the candidate list ``Graph.query_object`` searches.

    These are its own lines for the global search, with ``room_ids`` naming
    every room: the objects of the graph, walked room by room, each paired
    with the index of the room it hangs under.  Reproducing them here is what
    makes the score rows below line up with the node indices ``query_object``
    returns.

    Returns:
        The candidate objects and, aligned with them, their room indices into
        ``graph.rooms``.
    """
    room_ids = list(range(len(graph.rooms)))
    objects_list = []
    room_ids_list: List[int] = []
    for i in room_ids:
        objects_list.extend(graph.rooms[i].objects)
        room_ids_list.extend([i] * len(graph.rooms[i].objects))
    return objects_list, room_ids_list


@dataclass(frozen=True)
class StockQuery:
    """One vocabulary class scored against every candidate object.

    Attributes:
        query_id: The row of ``sim_mat`` holding the query, as
            ``Graph.query_object`` picks it.
        categories: The category list the query was embedded with.
        scores: ``sim_mat[query_id]``, the query-to-object similarity.  This
            is the value ``query_object`` orders its surviving candidates by.
        max_scores: ``np.max(sim_mat, axis=0)`` -- identical to ``scores`` on
            every surviving object, which is asserted rather than assumed.
        survives: Whether each object's best-matching category is the query
            rather than a negative label.
    """

    query_id: int
    categories: List[str]
    scores: np.ndarray
    max_scores: np.ndarray
    survives: np.ndarray


def stock_query(
    graph: Graph, query: str, object_embs: np.ndarray
) -> StockQuery:
    """Score one query against the candidate objects, as stock HOV-SG does.

    Every step below is ``Graph.query_object``'s ``clip`` branch, in its
    order: the query/negative category list, the CLIP text features from
    ``get_text_feats_multiple_templates``, the plain dot product against the
    stored object embeddings, the per-object best category and the per-object
    maximum score.  Nothing is normalized, rescaled or thresholded on top.
    """
    negative_prompt = NEGATIVE_LABELS
    if query in negative_prompt:
        query_id = negative_prompt.index(query)
    else:
        query_id = None

    if query_id is None:
        categories = [query, *negative_prompt]
        query_id = 0
    else:
        categories = list(negative_prompt)

    query_text_feats = get_text_feats_multiple_templates(
        categories, graph.clip_model, graph.clip_feat_dim
    )
    sim_mat = np.dot(query_text_feats, object_embs.T)

    # category id for each object, and its score under that category
    cls_ids = np.argmax(sim_mat, axis=0)
    max_scores = np.max(sim_mat, axis=0)
    survives = cls_ids == query_id
    return StockQuery(
        query_id=query_id,
        categories=categories,
        scores=sim_mat[query_id],
        max_scores=max_scores,
        survives=survives,
    )


def stock_retrieved_order(query: StockQuery, top_k: int) -> np.ndarray:
    """Reproduce the candidate order ``Graph.query_object`` returns.

    Used only to check this run against stock ``query_object``; the raw
    tables store no rank.  The fallback branch is stock's own: when no object
    survives the negative filter, ``query_object`` keeps the plain
    score-ordered list it built before filtering.
    """
    top_index = np.argsort(query.scores)[::-1][:top_k]
    if len(NEGATIVE_LABELS) > 0:
        obj_ids = np.where(query.survives)[0]
        if len(obj_ids) > 0:
            obj_scores = query.max_scores[obj_ids]
            resort_ids = np.argsort(-obj_scores)
            top_index = obj_ids[resort_ids]
            top_index = top_index[:top_k]
    return top_index


# ---------------------------------------------------------------------------
# Ground-truth objects in predicted rooms
# ---------------------------------------------------------------------------


@dataclass
class RoomPlacement:
    """Where the ground-truth objects landed among the predicted rooms."""

    #: (room_id, class_id) -> number of GT instances of that class in the room
    instances: Dict[Tuple[str, int], int]
    #: class ids present anywhere in the ground truth of this scene
    classes_in_scene: set
    assigned: int
    unassigned: int
    #: GT categories of the objects no predicted room contains
    unassigned_categories: Counter
    #: GT categories that are not part of the HM3DSem vocabulary at all
    categories_outside_vocabulary: Counter
    #: GT objects whose loaded point cloud is empty, so they have no centroid
    without_geometry: int


def room_containment_shares(rooms: Sequence, point: np.ndarray) -> np.ndarray:
    """Score a world point against every predicted room's extent.

    This is ``Graph.segment_objects``' room association applied to a single
    point: the room's floor height band admits it, and
    ``find_intersection_share`` measures how much of the room's 2-D footprint
    lies within ``ROOM_ASSIGNMENT_RADIUS`` of it.  A share of zero means the
    room does not contain the point.
    """
    point = np.asarray(point, dtype=float)
    point_2d = point[[0, 2]].reshape(1, 2)
    shares = np.zeros(len(rooms), dtype=float)
    for index, room in enumerate(rooms):
        lower = room.room_zero_level - FLOOR_HEIGHT_MARGIN
        upper = room.room_zero_level + room.room_height + FLOOR_HEIGHT_MARGIN
        if not lower <= point[1] <= upper:
            continue
        shares[index] = find_intersection_share(
            np.asarray(room.vertices, dtype=float), point_2d, ROOM_ASSIGNMENT_RADIUS
        )
    return shares


def assign_gt_objects_to_rooms(
    gt_objects: Sequence, rooms: Sequence, class_ids: Dict[str, int]
) -> RoomPlacement:
    """Place every ground-truth object in the predicted room containing it.

    The object's centroid is the mean of the ground-truth points the walk
    observed, which is what ``create_hm3dsem_walks_gt.object_centroid``
    records for an observed object and the same statistic
    ``Room.update_centroid`` uses.  An object no room contains is counted, not
    reassigned: no predicted object and no ground-truth instance are matched
    to each other anywhere in this function.
    """
    instances: Dict[Tuple[str, int], int] = defaultdict(int)
    classes_in_scene = set()
    assigned = 0
    unassigned = 0
    unassigned_categories: Counter = Counter()
    outside_vocabulary: Counter = Counter()
    without_geometry = 0

    for gt_object in gt_objects:
        category = str(gt_object.category)
        class_id = class_ids.get(category)
        if class_id is None:
            # A ground-truth category the HM3DSem vocabulary does not contain
            # cannot be a query class, so it can support no y(r, c).
            outside_vocabulary[category] += 1
        else:
            classes_in_scene.add(class_id)

        points = np.asarray(gt_object.points, dtype=float)
        if not points.size:
            # No observed geometry means no centroid to place, so the object
            # is reported rather than placed by some other quantity.
            without_geometry += 1
            unassigned += 1
            unassigned_categories[category] += 1
            continue

        centroid = np.mean(points, axis=0)
        shares = room_containment_shares(rooms, centroid)
        if not shares.size or np.max(shares) == 0:
            unassigned += 1
            unassigned_categories[category] += 1
            continue

        assigned += 1
        room = rooms[int(np.argmax(shares))]
        if class_id is not None:
            instances[(room.room_id, class_id)] += 1

    return RoomPlacement(
        instances=dict(instances),
        classes_in_scene=classes_in_scene,
        assigned=assigned,
        unassigned=unassigned,
        unassigned_categories=unassigned_categories,
        categories_outside_vocabulary=outside_vocabulary,
        without_geometry=without_geometry,
    )


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def object_centroid(objectt) -> np.ndarray:
    """The predicted object's centroid, in the graph's world frame."""
    return np.asarray(objectt.pcd.get_center(), dtype=float)


def write_objects_csv(path: Path, scene_id: str, objects_list: Sequence) -> int:
    """Write one row per predicted HOV-SG object."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(OBJECT_FIELDS)
        for objectt in objects_list:
            centroid = object_centroid(objectt)
            label_idx = getattr(objectt, "label_idx", None)
            writer.writerow(
                [
                    scene_id,
                    METHOD,
                    objectt.object_id,
                    "" if label_idx is None else int(label_idx),
                    objectt.name,
                    objectt.room_id,
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                ]
            )
    return len(objects_list)


def write_room_ground_truth_csv(
    path: Path,
    scene_id: str,
    rooms: Sequence,
    classes: Sequence[str],
    placement: RoomPlacement,
) -> int:
    """Write the full predicted-room x HM3DSem-vocabulary table."""
    rows = 0
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(ROOM_GT_FIELDS)
        for room in rooms:
            centroid = room.centroid
            if centroid is None:
                centroid = room.update_centroid()
            centroid = np.asarray(centroid, dtype=float)
            for class_id, class_label in enumerate(classes):
                count = placement.instances.get((room.room_id, class_id), 0)
                writer.writerow(
                    [
                        scene_id,
                        METHOD,
                        room.room_id,
                        class_id,
                        class_label,
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                        int(count > 0),
                        int(class_id in placement.classes_in_scene),
                        int(count),
                    ]
                )
                rows += 1
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def comparison_class_ids(setting, class_count: int) -> List[int]:
    """Which class ids to re-run stock ``query_object`` for."""
    text = str(setting).strip().lower()
    if text in {"none", "0", "false"}:
        return []
    if text in {"all", "true"}:
        return list(range(class_count))
    sample = int(text)
    if sample <= 0:
        return []
    sample = min(sample, class_count)
    return sorted(set(np.linspace(0, class_count - 1, sample).astype(int).tolist()))


@dataclass(frozen=True)
class Comparison:
    """The outcome of checking one class against stock ``query_object``.

    Attributes:
        agrees: Whether stock retrieved the same objects, in the same order,
            with the same score.
        sorting_deviation: ``|max_scores - scores|`` over the surviving
            objects: how far the score stock sorts its candidates by is from
            the score written to the raw table.  Zero by construction of
            ``np.argmax``/``np.max``, and measured rather than assumed.
        score_deviation: ``|recomputed - scores|`` over every object, where
            the recomputed row comes from running the stock CLIP embedding
            and dot product a second, independent time.
        message: What differed, when something did.
    """

    agrees: bool
    sorting_deviation: float
    score_deviation: float
    message: str = ""


def compare_with_stock_query_object(
    graph: Graph,
    query_class: str,
    query: StockQuery,
    objects_list: Sequence,
    room_ids_list: Sequence[int],
    object_index: Dict[str, int],
    object_embs: np.ndarray,
) -> Comparison:
    """Run stock ``Graph.query_object`` and compare it 1-to-1 with this run.

    Three things are compared, which together are what the raw tables are
    made of: which objects the stock search retrieves, in which order, and
    with which score.  ``top_k`` is the object count, so stock returns its
    whole list and no difference can hide behind truncation.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        stock_ids, stock_rooms = graph.query_object(
            query_class,
            room_ids=list(range(len(graph.rooms))),
            top_k=len(graph.objects),
            negative_prompt=NEGATIVE_LABELS,
        )

    top_index = stock_retrieved_order(query, top_k=len(graph.objects))
    expected_ids = [object_index[objects_list[i].object_id] for i in top_index]
    expected_rooms = [room_ids_list[i] for i in top_index]

    # (1) the objects stock retrieves, in stock's order.
    if list(stock_ids) != expected_ids:
        detail = (
            " in a different order"
            if len(stock_ids) == len(expected_ids)
            else ""
        )
        return Comparison(
            False,
            float("nan"),
            float("nan"),
            f"'{query_class}': stock query_object retrieved {len(stock_ids)} "
            f"objects, this run reproduces {len(expected_ids)}{detail}",
        )
    if list(stock_rooms) != expected_rooms:
        return Comparison(
            False, float("nan"), float("nan"), f"'{query_class}': parent rooms differ"
        )

    # (2) the score stock sorts its survivors by is the score stored for them.
    survivors = np.where(query.survives)[0]
    sorting_deviation = 0.0
    if survivors.size:
        sorting_deviation = float(
            np.max(np.abs(query.max_scores[survivors] - query.scores[survivors]))
        )

    # (3) the score itself, re-derived through the same stock calls a second
    # time, exactly as query_object derives it internally on every call.
    recomputed = np.dot(
        get_text_feats_multiple_templates(
            query.categories, graph.clip_model, graph.clip_feat_dim
        ),
        object_embs.T,
    )[query.query_id]
    score_deviation = float(np.max(np.abs(recomputed - query.scores)))

    if sorting_deviation != 0.0 or score_deviation != 0.0:
        return Comparison(
            False,
            sorting_deviation,
            score_deviation,
            f"'{query_class}': scores differ (sorting {sorting_deviation:.3e}, "
            f"recomputed {score_deviation:.3e})",
        )
    return Comparison(True, sorting_deviation, score_deviation)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(params: DictConfig) -> Dict[str, object]:
    """Generate the raw stock-HOV-SG object-search tables for one scene."""
    scene_id = str(params.main.scene_id)
    graph_path = _resolve(params.main.graph_path)
    scene_dir = _resolve(params.main.dataset_path) / str(params.main.split) / scene_id
    scene_info = scene_dir / "scene_info.json"
    output_dir = _resolve(params.main.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene {scene_id}")
    print(f"  predicted graph: {graph_path}")
    print(f"  HM3DSem ground truth: {scene_info}")
    print(f"  output directory: {output_dir}")

    # --- 1. Load ----------------------------------------------------------
    graph = Graph(params)
    graph.load_graph(str(graph_path))

    evaluator = HM3DSemanticEvaluator(params)
    evaluator.load_gt_graph_from_json(str(scene_info))

    with _in_repo_root():
        _, classes = get_label_feats(
            graph.clip_model,
            graph.clip_feat_dim,
            params.eval.obj_labels,
            None,
        )
    classes = [str(label) for label in classes]
    class_ids = {label: index for index, label in enumerate(classes)}
    print(f"  HM3DSem vocabulary: {len(classes)} classes ({params.eval.obj_labels})")

    # --- 2. Ground-truth objects in predicted rooms ------------------------
    placement = assign_gt_objects_to_rooms(evaluator.gt_objects, graph.rooms, class_ids)
    print(
        f"  ground truth: {placement.assigned} of {len(evaluator.gt_objects)} objects "
        f"lie in a predicted room, {placement.unassigned} in none"
    )

    # --- 3. Predicted objects ---------------------------------------------
    objects_list, room_ids_list = stock_object_candidates(graph)
    object_embs = np.array([obj.embedding for obj in objects_list])
    object_index = {obj.object_id: i for i, obj in enumerate(graph.objects)}
    print(f"  predicted graph: {len(graph.rooms)} rooms, {len(objects_list)} objects")

    # The class ids of the score table index the same vocabulary the graph
    # was labeled with, so an object's stored label_idx must name its stored
    # class label in that vocabulary.  Checked, not assumed.
    inconsistent_labels = [
        obj.object_id
        for obj in objects_list
        if getattr(obj, "label_idx", None) is None
        or classes[int(obj.label_idx)] != obj.name
    ]

    objects_csv = output_dir / "object_search_objects.csv"
    object_rows = write_objects_csv(objects_csv, scene_id, objects_list)
    print(f"  wrote {object_rows} rows: {objects_csv}")

    room_gt_csv = output_dir / "room_search_ground_truth.csv"
    room_rows = write_room_ground_truth_csv(
        room_gt_csv, scene_id, graph.rooms, classes, placement
    )
    print(f"  wrote {room_rows} rows: {room_gt_csv}")

    # --- 4. Query the full vocabulary --------------------------------------
    compared = comparison_class_ids(
        params.validation.stock_comparison_classes, len(classes)
    )
    compared_set = set(compared)
    scores_csv = output_dir / "object_search_scores.csv"

    score_rows = 0
    survivors_total = 0
    classes_with_survivors = 0
    empty_survivor_classes: List[str] = []
    mismatches: List[str] = []
    max_sorting_deviation = 0.0
    max_score_deviation = 0.0
    object_ids = [obj.object_id for obj in objects_list]
    room_labels = [graph.rooms[i].room_id for i in room_ids_list]

    with scores_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SCORE_FIELDS)
        for class_id, class_label in enumerate(classes):
            query = stock_query(graph, class_label, object_embs)
            survivors = int(np.count_nonzero(query.survives))
            survivors_total += survivors
            if survivors:
                classes_with_survivors += 1
            elif len(empty_survivor_classes) < 10:
                empty_survivor_classes.append(class_label)

            writer.writerows(
                [
                    scene_id,
                    METHOD,
                    class_id,
                    class_label,
                    object_id,
                    room_label,
                    float(score),
                ]
                for object_id, room_label, score in zip(
                    object_ids, room_labels, query.scores
                )
            )
            score_rows += len(object_ids)

            if class_id in compared_set:
                comparison = compare_with_stock_query_object(
                    graph,
                    class_label,
                    query,
                    objects_list,
                    room_ids_list,
                    object_index,
                    object_embs,
                )
                if not comparison.agrees:
                    mismatches.append(comparison.message)
                else:
                    max_sorting_deviation = max(
                        max_sorting_deviation, comparison.sorting_deviation
                    )
                    max_score_deviation = max(
                        max_score_deviation, comparison.score_deviation
                    )

            if (class_id + 1) % 200 == 0 or class_id + 1 == len(classes):
                print(f"    queried {class_id + 1}/{len(classes)} classes")

    print(f"  wrote {score_rows} rows: {scores_csv}")

    summary = {
        "scene_id": scene_id,
        "objects": len(objects_list),
        "rooms": len(graph.rooms),
        "classes": len(classes),
        "object_rows": object_rows,
        "score_rows": score_rows,
        "room_rows": room_rows,
        "survivors_total": survivors_total,
        "classes_with_survivors": classes_with_survivors,
        "empty_survivor_classes": empty_survivor_classes,
        "compared_classes": len(compared),
        "mismatches": mismatches,
        "max_sorting_deviation": max_sorting_deviation,
        "max_score_deviation": max_score_deviation,
        "inconsistent_labels": inconsistent_labels,
        "gt_objects": len(evaluator.gt_objects),
        "placement": placement,
        "files": {
            "objects": objects_csv,
            "scores": scores_csv,
            "rooms": room_gt_csv,
        },
    }
    report(summary, objects_list, graph)
    return summary


def report(summary: Dict[str, object], objects_list: Sequence, graph: Graph) -> None:
    """Print the checks of section 8, each with the number behind it."""
    placement: RoomPlacement = summary["placement"]
    objects = summary["objects"]
    classes = summary["classes"]
    rooms = summary["rooms"]

    duplicate_ids = [
        object_id
        for object_id, count in Counter(obj.object_id for obj in objects_list).items()
        if count > 1
    ]

    print()
    print("--- validation ---------------------------------------------------")
    checks = [
        (
            summary["object_rows"] == objects and not duplicate_ids,
            f"every predicted object appears once: {summary['object_rows']} rows "
            f"for {objects} objects, {len(duplicate_ids)} duplicate ids",
        ),
        (
            len(objects_list) == len(graph.objects),
            f"the candidate list covers the graph: {len(objects_list)} candidates "
            f"for {len(graph.objects)} object nodes",
        ),
        (
            summary["score_rows"] == objects * classes,
            f"object_search_scores.csv holds N x C rows: {summary['score_rows']} "
            f"= {objects} x {classes}",
        ),
        (
            not summary["inconsistent_labels"],
            f"predicted class ids index the queried vocabulary: "
            f"{len(summary['inconsistent_labels'])} of {objects} objects disagree "
            f"with their stored class label",
        ),
        (
            not summary["mismatches"],
            f"stock query_object reproduced 1-to-1 on {summary['compared_classes']} "
            f"of {classes} classes: {len(summary['mismatches'])} mismatches, "
            f"max score deviation {summary['max_score_deviation']:.3e}, "
            f"max sorting-score deviation {summary['max_sorting_deviation']:.3e}",
        ),
        (
            summary["score_rows"] == objects * classes,
            f"no top_k truncation: every class keeps all {objects} objects",
        ),
        (
            summary["survivors_total"] < objects * classes,
            f"failed-filter objects remain in the raw table: "
            f"{objects * classes - summary['survivors_total']} of "
            f"{objects * classes} rows are objects the background filter "
            f"dropped, and every one of them still has a row",
        ),
        (
            summary["room_rows"] == rooms * classes,
            f"room ground truth is the full Cartesian product: "
            f"{summary['room_rows']} = {rooms} x {classes}",
        ),
        (
            True,
            f"unassigned ground-truth objects: {placement.unassigned} of "
            f"{summary['gt_objects']} lie in no predicted room "
            f"({placement.assigned} assigned)",
        ),
        (
            not placement.categories_outside_vocabulary,
            f"ground-truth categories inside the vocabulary: "
            f"{len(placement.categories_outside_vocabulary)} outside it",
        ),
    ]
    failed = 0
    for passed, message in checks:
        print(f"  [{'ok' if passed else 'FAILED'}] {message}")
        failed += 0 if passed else 1

    print(
        f"  [ok] no predicted-object-to-GT matching, no room score, no rank and "
        f"no retrieval metric is computed or written"
    )

    if placement.unassigned_categories:
        common = ", ".join(
            f"{category} x{count}"
            for category, count in placement.unassigned_categories.most_common(10)
        )
        print(f"  unassigned GT objects by category: {common}")
    if placement.categories_outside_vocabulary:
        print(
            "  GT categories outside the HM3DSem vocabulary: "
            + ", ".join(sorted(placement.categories_outside_vocabulary))
        )
    empty_classes = classes - summary["classes_with_survivors"]
    if empty_classes:
        print(
            f"  no object survived the background filter for {empty_classes} of "
            f"{classes} classes (first {len(summary['empty_survivor_classes'])}: "
            + ", ".join(summary["empty_survivor_classes"])
            + ")"
        )
    if placement.without_geometry:
        print(
            f"  ground-truth objects with an empty point cloud, counted as "
            f"unassigned: {placement.without_geometry}"
        )
    for message in summary["mismatches"][:10]:
        print(f"  MISMATCH {message}")

    print("------------------------------------------------------------------")
    if failed:
        raise SystemExit(f"{failed} validation check(s) failed")


@hydra.main(
    version_base=None, config_path="../config", config_name="search_preprocessing"
)
def main(params: DictConfig):
    return run(params)


if __name__ == "__main__":
    main()
