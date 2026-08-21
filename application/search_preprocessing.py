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
* **The rows are stock's answer.**  Every row of
  ``object_search_scores.csv`` is an object ``query_object`` actually returned
  for that class, written in the order it returned them, carrying the score it
  ranked them by.  The rows come from the return value of the stock call
  itself, not from a re-derivation of it.  The query-versus-background filter
  is therefore already applied: an object stock discarded for a class has no
  row for that class, and no reconstruction step is needed to find that out.
* **``top_k`` is stock's own.**  The shipped application calls
  ``query_hierarchy(query, top_k=5)``, so 5 is what a query against this
  system actually returns and 5 is what the table records.  Nothing about the
  call is changed, truncation included.
* **Stock's empty-survivor fallback is stock's answer too.**  When no object
  survives the filter for a class, ``query_object`` leaves the unfiltered,
  score-ordered list in place and returns its ``top_k`` head.  That is what
  the table records for those classes, because that is what the search
  returns.
* **The vocabulary is the graph's own.**  ``get_label_feats`` with
  ``HM3DSEM_LABELS`` is the bank ``Graph.segment_objects`` labeled the objects
  with, so a class id in these files is exactly an object node's stored
  ``label_idx``.
* **Ground truth is every annotated object of the scene.**  The walk's
  ``scene_info.json`` carries two collections: ``objects``, the
  trajectory-visible ones the evaluator reads, and ``all_objects``, every
  annotated object of the complete HM3D-Sem scene including the ones the walk
  never saw.  This run reads ``all_objects`` (``eval.gt_object_collection``),
  positioned by each entry's serialized ``centroid``, and cross-checks it
  against ``HM3DSemanticEvaluator.load_gt_graph_from_json``: every object that
  loader covers must be present with the same position, to zero deviation.
  ``gt_objects.csv`` flags which were observed, since an object the walk never
  saw cannot be in the predicted graph at all.
* **y(r, c) is over predicted rooms, and its evidence is on disk.**  The
  room table answers, for each predicted room and class, whether the room
  contains a ground-truth object of that class.  ``gt_objects.csv`` records
  every ground-truth object individually -- its HM3DSem region, the predicted
  room its centroid fell in, and whether it fell in one at all -- so the
  objects behind any ``gt_class_present_in_room = 1`` are recoverable, and the ones
  that landed nowhere are visible rather than merely counted.
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
import json
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

#: The number of objects a query returns.  This is the value HOV-SG's own
#: application passes -- ``application/visualize_query_graph_back.py`` calls
#: ``query_hierarchy(query, top_k=5)`` -- and therefore what the stock search
#: answers with.  ``Graph.query_object``'s own signature default is 1.
#: Overridable through ``main.top_k``.
STOCK_TOP_K = 5

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
    "predicted_room_id",
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
    "predicted_room_id",
    "score",
]

GT_OBJECT_FIELDS = [
    "scene_id",
    "method",
    "gt_object_id",
    "gt_category",
    "class_id",
    "gt_room_id",
    "predicted_room_id",
    "gt_x",
    "gt_y",
    "gt_z",
    "observed_in_walk",
    "assigned",
]

ROOM_GT_FIELDS = [
    "scene_id",
    "method",
    "predicted_room_id",
    "class_id",
    "class_label",
    "room_x",
    "room_y",
    "room_z",
    "gt_class_present_in_room",
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

    #: (predicted_room_id, class_id) -> how many GT objects of that class
    #: have their centroid inside that predicted room.  This is the count
    #: behind y(r, c).
    instances: Dict[Tuple[str, int], int]
    #: One record per GT object, in ground-truth order: where it is annotated,
    #: where it landed, and whether it landed anywhere at all.
    records: List[Dict[str, object]]
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


def load_gt_objects(scene_info: Path, collection: str) -> List[Dict[str, object]]:
    """Read one of ``scene_info.json``'s two ground-truth object collections.

    ``objects`` is the trajectory-visible set, each entry backed by an
    ``objects/<id>.ply``.  ``all_objects`` is the superset holding every
    annotated object of the complete HM3D-Sem scene, observed or not; its
    unobserved entries have no point cloud and carry the source Habitat boxes
    instead.  ``HM3DSemanticEvaluator`` reads only the former and requires the
    point clouds, so the latter is read here directly.

    Each entry's ``centroid`` is used as its world position.  That field is
    written by ``create_hm3dsem_walks_gt.object_centroid``: the mean of the
    mapped points for an observed object, falling back to the OBB and then the
    AABB centre for one the walk never saw.  For observed objects it is
    therefore the same statistic the evaluator's point clouds give, which
    ``verify_gt_against_evaluator`` checks rather than assumes.
    """
    with scene_info.open() as file:
        payload = json.load(file)
    if collection not in payload:
        raise KeyError(
            f"{scene_info} has no '{collection}' collection; it holds "
            f"{sorted(k for k in payload if isinstance(payload[k], list))}"
        )
    return [
        {
            "id": entry["id"],
            "category": str(entry["category"]),
            "region_id": entry["region_id"],
            "centroid": entry.get("centroid"),
            "observed_in_walk": bool(entry.get("observed_in_walk", True)),
        }
        for entry in payload[collection]
    ]


def verify_gt_against_evaluator(
    gt_objects: Sequence[Dict[str, object]], evaluator_objects: Sequence
) -> Tuple[int, float]:
    """Check the JSON-read ground truth against the repository's own loader.

    Every object ``HM3DSemanticEvaluator`` loaded must appear here with the
    same world position, where the evaluator's position is the mean of the
    point cloud it read from disk.  This is what makes reading the JSON
    directly equivalent to the loader for the objects the loader covers, and
    it is measured, not asserted in a comment.

    Returns:
        How many evaluator objects were matched, and the largest coordinate
        deviation found.
    """
    by_id = {str(entry["id"]): entry for entry in gt_objects}
    matched = 0
    deviation = 0.0
    for objectt in evaluator_objects:
        entry = by_id.get(str(objectt.id))
        if entry is None or entry["centroid"] is None:
            continue
        points = np.asarray(objectt.points, dtype=float)
        if not points.size:
            continue
        matched += 1
        deviation = max(
            deviation,
            float(
                np.max(
                    np.abs(
                        np.mean(points, axis=0)
                        - np.asarray(entry["centroid"], dtype=float)
                    )
                )
            ),
        )
    return matched, deviation


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

    The object's position is its serialized ``centroid``, written by
    ``create_hm3dsem_walks_gt.object_centroid``: the mean of the mapped points
    for an object the walk observed -- the same statistic
    ``Room.update_centroid`` uses -- and the Habitat OBB or AABB centre for
    one it never saw.  An object no room contains is counted, not
    reassigned: no predicted object and no ground-truth instance are matched
    to each other anywhere in this function.
    """
    instances: Dict[Tuple[str, int], int] = defaultdict(int)
    records: List[Dict[str, object]] = []
    classes_in_scene = set()
    assigned = 0
    unassigned = 0
    unassigned_categories: Counter = Counter()
    outside_vocabulary: Counter = Counter()
    without_geometry = 0

    for gt_object in gt_objects:
        category = str(gt_object["category"])
        class_id = class_ids.get(category)
        if class_id is None:
            # A ground-truth category the HM3DSem vocabulary does not contain
            # cannot be a query class, so it can support no y(r, c).
            outside_vocabulary[category] += 1
        else:
            classes_in_scene.add(class_id)

        record = {
            "gt_object_id": gt_object["id"],
            "gt_category": category,
            "class_id": "" if class_id is None else class_id,
            "gt_room_id": gt_object["region_id"],
            "predicted_room_id": "",
            "gt_x": "",
            "gt_y": "",
            "gt_z": "",
            "observed_in_walk": int(bool(gt_object["observed_in_walk"])),
            "assigned": 0,
        }
        records.append(record)

        if gt_object["centroid"] is None:
            # No geometry at all: neither mapped points nor a usable Habitat
            # box, so there is no position to place.  Reported, not guessed.
            without_geometry += 1
            unassigned += 1
            unassigned_categories[category] += 1
            continue

        centroid = np.asarray(gt_object["centroid"], dtype=float)
        record["gt_x"], record["gt_y"], record["gt_z"] = (
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
        )
        shares = room_containment_shares(rooms, centroid)
        if not shares.size or np.max(shares) == 0:
            unassigned += 1
            unassigned_categories[category] += 1
            continue

        assigned += 1
        room = rooms[int(np.argmax(shares))]
        record["predicted_room_id"] = room.room_id
        record["assigned"] = 1
        if class_id is not None:
            instances[(room.room_id, class_id)] += 1

    return RoomPlacement(
        instances=dict(instances),
        records=records,
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


def write_gt_objects_csv(
    path: Path, scene_id: str, placement: RoomPlacement
) -> int:
    """Write every HM3DSem ground-truth object and where it landed.

    One row per ground-truth object, in ground-truth order, carrying both of
    its room identities: ``gt_room_id`` is the HM3DSem region it is annotated
    in, ``predicted_room_id`` is the predicted room its centroid falls in.
    The objects no predicted room contains are here too, with an empty
    ``predicted_room_id`` and ``assigned = 0``, so the ones that produced no
    y(r, c) are on disk rather than only in a printed count.

    Grouping the assigned rows by ``(predicted_room_id, class_id)`` gives
    exactly the objects behind each ``gt_class_present_in_room = 1``.
    """
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(GT_OBJECT_FIELDS)
        writer.writerows(
            [scene_id, METHOD] + [record[field] for field in GT_OBJECT_FIELDS[2:]]
            for record in placement.records
        )
    return len(placement.records)


def write_room_ground_truth_csv(
    path: Path,
    scene_id: str,
    rooms: Sequence,
    classes: Sequence[str],
    placement: RoomPlacement,
) -> int:
    """Write the full predicted-room x HM3DSem-vocabulary table.

    ``gt_class_present_in_room`` is y(r, c): whether predicted room ``r``
    holds at least one ground-truth object of class ``c``, where "holds" means
    the object's centroid lies in the room.  Its neighbour
    ``gt_class_present_in_scene`` asks the same of the whole scene, so the two
    differ only in scope.  Which objects those are is recorded
    per object in ``gt_objects.csv``; this table carries the label and the
    count.
    """
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


def verify_reproduction(
    graph: Graph,
    query_class: str,
    query: StockQuery,
    objects_list: Sequence,
    room_ids_list: Sequence[int],
    object_index: Dict[str, int],
    object_embs: np.ndarray,
    stock_ids: Sequence[int],
    stock_rooms: Sequence[int],
    top_k: int,
) -> Comparison:
    """Check what stock returned against an independent reproduction of it.

    ``stock_ids``/``stock_rooms`` are the live return value of
    ``Graph.query_object`` -- the same value the rows were written from.  This
    re-derives that list from the stock procedure followed step by step, and
    re-derives the scores through a second, independent CLIP embedding, so a
    silent divergence between what the table says and what the search does
    would have to survive both.
    """
    top_index = stock_retrieved_order(query, top_k=top_k)
    expected_ids = [object_index[objects_list[i].object_id] for i in top_index]
    expected_rooms = [room_ids_list[i] for i in top_index]

    if list(stock_ids) != expected_ids:
        detail = " in a different order" if len(stock_ids) == len(expected_ids) else ""
        return Comparison(
            False,
            float("nan"),
            float("nan"),
            f"'{query_class}': stock query_object returned {len(stock_ids)} "
            f"objects, this run reproduces {len(expected_ids)}{detail}",
        )
    if list(stock_rooms) != expected_rooms:
        return Comparison(
            False, float("nan"), float("nan"), f"'{query_class}': parent rooms differ"
        )

    # The score stock sorts its survivors by is the score written to the table.
    survivors = np.where(query.survives)[0]
    sorting_deviation = 0.0
    if survivors.size:
        sorting_deviation = float(
            np.max(np.abs(query.max_scores[survivors] - query.scores[survivors]))
        )

    # The score itself, re-derived through the same stock calls a second time.
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

    # The ground truth this run evaluates against.  ``all_objects`` is every
    # annotated object of the scene; ``objects`` is only what the walk saw.
    collection = str(params.eval.get("gt_object_collection", "all_objects"))
    gt_objects = load_gt_objects(scene_info, collection)
    observed = sum(1 for entry in gt_objects if entry["observed_in_walk"])
    print(
        f"  HM3DSem ground truth: {len(gt_objects)} objects from "
        f"'{collection}', {observed} observed in the walk"
    )

    # The repository's own loader, kept as the cross-check that reading the
    # JSON directly is equivalent for the objects it covers.  It reads
    # 'objects' only, and needs a point cloud per object.
    evaluator = HM3DSemanticEvaluator(params)
    evaluator.load_gt_graph_from_json(str(scene_info))
    matched_gt, gt_centroid_deviation = verify_gt_against_evaluator(
        gt_objects, evaluator.gt_objects
    )

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
    placement = assign_gt_objects_to_rooms(gt_objects, graph.rooms, class_ids)
    print(
        f"  ground truth: {placement.assigned} of {len(gt_objects)} objects "
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

    gt_objects_csv = output_dir / "gt_objects.csv"
    gt_object_rows = write_gt_objects_csv(gt_objects_csv, scene_id, placement)
    print(f"  wrote {gt_object_rows} rows: {gt_objects_csv}")

    room_gt_csv = output_dir / "room_search_ground_truth.csv"
    room_rows = write_room_ground_truth_csv(
        room_gt_csv, scene_id, graph.rooms, classes, placement
    )
    print(f"  wrote {room_rows} rows: {room_gt_csv}")

    # --- 4. Query the full vocabulary --------------------------------------
    # Every class goes through the stock call, and the rows are its return
    # value.  ``validation.stock_comparison_classes`` selects how many of
    # those returns are additionally checked against an independent
    # reproduction of the procedure behind them.
    compared = comparison_class_ids(
        params.validation.stock_comparison_classes, len(classes)
    )
    compared_set = set(compared)
    scores_csv = output_dir / "object_search_scores.csv"
    top_k = int(params.main.get("top_k", STOCK_TOP_K))
    print(f"  top_k: {top_k} (the value HOV-SG's own application queries with)")
    candidate_position = {obj.object_id: i for i, obj in enumerate(objects_list)}

    score_rows = 0
    rows_per_class: List[int] = []
    survivors_total = 0
    classes_with_survivors = 0
    fallback_classes: List[str] = []
    shape_mismatches: List[str] = []
    mismatches: List[str] = []
    max_sorting_deviation = 0.0
    max_score_deviation = 0.0

    with scores_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SCORE_FIELDS)
        for class_id, class_label in enumerate(classes):
            query = stock_query(graph, class_label, object_embs)
            survivors = int(np.count_nonzero(query.survives))
            survivors_total += survivors
            if survivors:
                classes_with_survivors += 1
            else:
                # Stock keeps its unfiltered, score-ordered list when nothing
                # survives, and returns every object.  Recorded, not special-cased.
                fallback_classes.append(class_label)

            # The final result of the search for this class, taken from the
            # stock function itself, truncated exactly as stock truncates it.
            with contextlib.redirect_stdout(io.StringIO()):
                stock_ids, stock_rooms = graph.query_object(
                    class_label,
                    room_ids=list(range(len(graph.rooms))),
                    top_k=top_k,
                    negative_prompt=NEGATIVE_LABELS,
                )

            writer.writerows(
                [
                    scene_id,
                    METHOD,
                    class_id,
                    class_label,
                    graph.objects[node_index].object_id,
                    graph.rooms[room_index].room_id,
                    float(
                        query.scores[
                            candidate_position[graph.objects[node_index].object_id]
                        ]
                    ),
                ]
                for node_index, room_index in zip(stock_ids, stock_rooms)
            )
            score_rows += len(stock_ids)
            rows_per_class.append(len(stock_ids))

            # The row count must be stock's answer: its top_k head of the
            # survivors, or of every object when the fallback applies.
            expected = min(survivors if survivors else len(objects_list), top_k)
            if len(stock_ids) != expected:
                shape_mismatches.append(
                    f"'{class_label}': {len(stock_ids)} rows, expected {expected}"
                )

            if class_id in compared_set:
                comparison = verify_reproduction(
                    graph,
                    class_label,
                    query,
                    objects_list,
                    room_ids_list,
                    object_index,
                    object_embs,
                    stock_ids,
                    stock_rooms,
                    top_k,
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
        "fallback_classes": fallback_classes,
        "rows_per_class": rows_per_class,
        "top_k": top_k,
        "shape_mismatches": shape_mismatches,
        "compared_classes": len(compared),
        "mismatches": mismatches,
        "max_sorting_deviation": max_sorting_deviation,
        "max_score_deviation": max_score_deviation,
        "inconsistent_labels": inconsistent_labels,
        "gt_objects": len(gt_objects),
        "gt_collection": collection,
        "gt_observed": observed,
        "evaluator_objects": len(evaluator.gt_objects),
        "matched_gt": matched_gt,
        "gt_centroid_deviation": gt_centroid_deviation,
        "gt_object_rows": gt_object_rows,
        "positive_cells": sum(1 for count in placement.instances.values() if count),
        "placed_instances": sum(placement.instances.values()),
        "placement": placement,
        "files": {
            "objects": objects_csv,
            "scores": scores_csv,
            "gt_objects": gt_objects_csv,
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
    rows_per_class = summary["rows_per_class"]
    fallback = len(summary["fallback_classes"])
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
            not summary["inconsistent_labels"],
            f"predicted class ids index the queried vocabulary: "
            f"{len(summary['inconsistent_labels'])} of {objects} objects disagree "
            f"with their stored class label",
        ),
        (
            len(rows_per_class) == classes,
            f"object_search_scores.csv is the search's answer: "
            f"{summary['score_rows']} rows over {len(rows_per_class)} of {classes} "
            f"queried classes "
            f"({min(rows_per_class)}-{max(rows_per_class)} objects returned per class)",
        ),
        (
            not summary["shape_mismatches"],
            f"the filter and the truncation are applied as stock applies "
            f"them: {len(summary['shape_mismatches'])} classes with an "
            f"unexpected row count, "
            f"{summary['survivors_total']} objects passed the background filter "
            f"across the vocabulary and {summary['score_rows']} rows survived "
            f"stock's top-{summary['top_k']} cut",
        ),
        (
            not summary["shape_mismatches"],
            f"top_k is stock's own: every class carries at most "
            f"{summary['top_k']} objects, the value HOV-SG's application "
            f"queries with, {fallback} of them supplied by the empty-survivor "
            f"fallback",
        ),
        (
            not summary["mismatches"],
            f"stock query_object reproduced 1-to-1 on {summary['compared_classes']} "
            f"of {classes} classes: {len(summary['mismatches'])} mismatches, "
            f"max score deviation {summary['max_score_deviation']:.3e}, "
            f"max sorting-score deviation {summary['max_sorting_deviation']:.3e}",
        ),
        (
            summary["room_rows"] == rooms * classes,
            f"room ground truth is the full Cartesian product: "
            f"{summary['room_rows']} = {rooms} predicted rooms x {classes} "
            f"classes, {summary['positive_cells']} of them y(r,c)=1",
        ),
        (
            summary["matched_gt"] == summary["evaluator_objects"]
            and summary["gt_centroid_deviation"] == 0.0,
            f"the ground truth agrees with the repository's own loader: all "
            f"{summary['evaluator_objects']} objects HM3DSemanticEvaluator "
            f"loads are in '{summary['gt_collection']}' "
            f"({summary['matched_gt']} matched), max centroid deviation "
            f"{summary['gt_centroid_deviation']:.3e}",
        ),
        (
            summary["gt_object_rows"] == summary["gt_objects"],
            f"every ground-truth object is in gt_objects.csv: "
            f"{summary['gt_object_rows']} rows for {summary['gt_objects']} GT "
            f"objects of '{summary['gt_collection']}' "
            f"({summary['gt_observed']} observed in the walk, "
            f"{summary['gt_objects'] - summary['gt_observed']} never seen), "
            f"{placement.unassigned} of them in no predicted room",
        ),
        (
            summary["placed_instances"] == placement.assigned,
            f"the room counts and the inventory agree: "
            f"num_gt_instances_in_room sums to {summary['placed_instances']}, "
            f"and {placement.assigned} objects are marked assigned",
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
    if summary["fallback_classes"]:
        shown = summary["fallback_classes"][:10]
        print(
            f"  no object survived the background filter for "
            f"{len(summary['fallback_classes'])} of {classes} classes, so stock "
            f"returned every object for them (first {len(shown)}: "
            + ", ".join(shown)
            + ")"
        )
    print(
        f"  objects returned across the vocabulary: {summary['score_rows']} rows "
        f"from {classes} queries, at most {summary['top_k']} each; "
        f"{len(summary['fallback_classes'])} of those queries answered from "
        f"stock's empty-survivor fallback"
    )
    if placement.without_geometry:
        print(
            f"  ground-truth objects with an empty point cloud, counted as "
            f"unassigned: {placement.without_geometry}"
        )
    for message in summary["shape_mismatches"][:10]:
        print(f"  ROW COUNT {message}")
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
