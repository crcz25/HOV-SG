"""Raw object-search data for two search methods, and nothing else.

This script runs one object search over the complete HM3DSem vocabulary under
two ranking methods and writes what each computed, unmodified, to CSV.  It is a
preprocessing stage: it produces the raw material an evaluation reads, and it
deliberately computes no evaluation of its own.  There is no normalization, no
calibration, no threshold, no room score, and no retrieval metric anywhere in
this file.  ``rank`` is recorded because it is the search's own output, not
because a metric is computed from it here.

The two methods
---------------

Both methods search one candidate list -- every object of the graph, walked
room by room -- and both cut their ranking at the same ``top_k``.  They differ
in the value they rank by and, as a consequence of it, in whether they need a
negative label at all.

* **``hovsg``** orders by ``sim_mat[query_id]``, the query-to-object CLIP
  similarity, after discarding every object whose best-matching category is
  ``background`` rather than the query.  This is stock ``Graph.query_object``,
  negative label included, and it is checked class by class against what the
  live stock call actually returns.
* **``belief``** orders by the query-conditioned object probability

      q_i(q) = P_det,i * P_view,i * P_mem,i * P_sem,i(q)

  and uses **no negative label**.  The first three factors are read off the
  object node: they are properties of the object, computed once when the graph
  was built.  The fourth is computed here, at query time, and is the only part
  that depends on what was asked:

      P_sem,i(q) = sigma(alpha * (cos(v_i, t_q)
                                  - max_{c: cos(t_c, t_q) < tau} cos(v_i, t_c)))

  -- the gap between the query and the best *semantically distinct* class the
  object could otherwise be.  This is eq. (semantic) with the object's own
  label replaced by the query, so an object that matches some other class
  better than the query gets a negative margin and is driven toward zero by the
  sigmoid.  That is what makes a hard ``background`` filter unnecessary: the
  vocabulary itself supplies the "none of the above" reference, continuously,
  instead of one generic word supplying it as a yes/no gate.

  Label Coherence does not enter.  It compares against in-map visual
  prototypes, which exist only for classes some object was labeled with, so it
  has nothing to say about an arbitrary query; the semantic factor is
  ``P_sem(q)`` alone.

  Because ``P_sem(q)`` is a margin against the whole vocabulary and not a
  lookup of the object's stored label, the belief search never reads
  ``label_idx``.  It asks how well the object matches *this query* relative to
  everything else it could be.

What is reproduced, and from where
----------------------------------

* **Retrieval is ``Graph.query_object``.**  The candidate object list, the
  CLIP text embedding of the query, the query-versus-``background``
  comparison, the surviving-object rule and the truncation are the lines of
  ``hovsg/graph/graph.py``, followed step for step in
  :func:`stock_object_candidates`, :func:`stock_query` and
  :func:`rank_candidates`.  Sharing them is what lets the belief method reuse
  the stock search rather than imitate it.
* **The ``hovsg`` rows are stock's answer, and it is proven.**  Every class is
  additionally put through the live ``graph.query_object`` call, and the rows
  this file writes for ``hovsg`` must equal that return value -- same objects,
  same order, same parent rooms (see ``--validation``).  When a previous
  ``object_search_hovsg.csv`` is present in the output directory, the new rows
  are also checked against it, so a change in stock behaviour cannot pass
  unnoticed.
* **``top_k`` is stock's own.**  The shipped application calls
  ``query_hierarchy(query, top_k=5)``, so 5 is what a query against this
  system actually returns and 5 is what the tables record, for both methods.
* **Stock's empty-survivor fallback is stock's answer too.**  When no object
  survives the ``background`` filter for a class, stock leaves the unfiltered
  list in place and returns its ``top_k`` head.  The ``hovsg`` method does the
  same.  The ``belief`` method has no filter and therefore no fallback: it
  always ranks the whole candidate list.
* **The semantic factor is checked against the graph's own.**  Evaluated at the
  object's *own* label, ``P_sem(q)`` must reproduce the ``p_sem`` the graph
  stored, to floating-point tolerance.  That single equality is what proves the
  text bank, the cosine convention, the synonym threshold tau and the logit
  scale alpha used here are the ones the graph was built with -- none of which
  is persisted with the graph, and all of which would otherwise be assumed.
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
* **y(r, c) is over predicted rooms, and its evidence is on disk.**
  ``rooms.csv`` answers, for each predicted room and class, whether the room
  contains a ground-truth object of that class.  ``gt_objects.csv`` records
  every ground-truth object individually -- its HM3DSem region, the predicted
  room its centroid fell in, and whether it fell in one at all -- so the
  objects behind any ``gt_class_present_in_room = 1`` are recoverable, and the
  ones that landed nowhere are visible rather than merely counted.
* **Room containment is ``Graph.segment_objects``'.**  A ground-truth object
  is placed in the predicted room whose 2-D footprint covers its centroid,
  scored by ``find_intersection_share`` at the same radius, inside the same
  floor height band, with the same ``np.argmax`` tie-break.  A GT object no
  predicted room covers is left unassigned and counted, never nudged into the
  nearest room.
* **``rooms.csv``'s ``belief`` is the room node's own.**  It is
  ``Room.class_containment_belief``, eq. (noisyor) evaluated over the objects
  the graph assigned to that room and labeled with that class, computed when
  the graph was built and persisted with the node.  It is empty for a class no
  object in the room was labeled with, because the room then carries no
  evidence about that class -- which is not the belief 0.

Output files
------------

``objects.csv``               one row per predicted object node, with its stored
                              ``p_obj`` and the factors behind it
``rooms.csv``                 predicted room x vocabulary, with y(r, c) and the belief
``object_search_hovsg.csv``   the ``hovsg`` search result, ranked by ``score``
``object_search_belief.csv``  the ``belief`` search result, ranked by ``score``
``gt_objects.csv``            one row per ground-truth object, and where it landed

Both search files name their ranking value ``score``, so the two are read the
same way and differ only in what that value is: a CLIP similarity for
``hovsg``, a probability for ``belief``.  The stored ``p_obj`` of
``objects.csv`` is a third thing again -- confidence in the object's *own*
label, query-independent -- and is what ``rooms.csv``'s belief comes from.

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
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.special import expit

from hovsg.eval.hm3dsem_evaluator import HM3DSemanticEvaluator
from hovsg.graph.graph import Graph
from hovsg.utils.clip_utils import get_text_feats_multiple_templates
from hovsg.utils.graph_utils import find_intersection_share
from hovsg.utils.label_feats import get_label_feats
from hovsg.utils.uncertainty import (
    build_synonym_eligibility_mask,
    compute_cosine_similarities,
    normalize_rows,
)

# pylint: disable=all

#: The ranking method every row of a search file is attributed to.  ``hovsg``
#: orders candidates by the stock CLIP query score, ``belief`` by the stored
#: fused object probability ``p_obj``.  Everything else about the two searches
#: is the same code on the same inputs.
HOVSG_METHOD = "hovsg"
BELIEF_METHOD = "belief"

#: HOV-SG's own negative prompt, as ``Graph.query_hierarchy`` sets it before
#: calling ``Graph.query_object``: an object whose best-matching category is
#: this one rather than the query does not survive the query.  It belongs to
#: the ``hovsg`` method alone.  The ``belief`` method uses no negative label:
#: its semantic factor measures the query against the whole vocabulary, which
#: supplies the same "none of the above" reference continuously rather than as
#: a gate.
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

#: One row per predicted object node, carrying the per-object probabilities the
#: graph computed and stored.  ``p_obj`` is eq. (obj-prob) evaluated against the
#: object's *own* label -- ``p_det * p_view * p_mem * p_sem_bar`` -- so it says
#: how confident the map is that the object is what it was labeled, and it is
#: the quantity ``rooms.csv``'s ``belief`` is propagated from.  It is
#: deliberately *not* what the belief search ranks by; that is a
#: query-conditioned score and it lives in ``object_search_belief.csv``.
#: ``p_sem_bar`` is ``min(p_sem, p_coh)`` by eq. (coherence), with ``p_sem`` and
#: ``p_coh`` beside it so the combination is visible rather than only its result.
OBJECT_FIELDS = [
    "scene_id",
    "object_id",
    "predicted_class_id",
    "class_label",
    "predicted_room_id",
    "object_x",
    "object_y",
    "object_z",
    "p_obj",
    "p_det",
    "p_view",
    "p_mem",
    "p_sem_bar",
    "p_sem",
    "p_coh",
]

#: The stored per-object probabilities written to ``objects.csv``, in order.
OBJECT_PROBABILITY_FIELDS = [
    "p_obj",
    "p_det",
    "p_view",
    "p_mem",
    "p_sem_bar",
    "p_sem",
    "p_coh",
]

#: The columns both search files share: which query, which object it returned,
#: and at which rank.  Each method appends its own score columns.
SEARCH_KEY_FIELDS = [
    "scene_id",
    "method",
    "class_id",
    "class_label",
    "object_id",
    "predicted_room_id",
    "rank",
]

HOVSG_SEARCH_FIELDS = SEARCH_KEY_FIELDS + ["score"]

#: ``score`` is the value the belief search ranks by, named as in
#: ``object_search_hovsg.csv`` so the two files answer the same question in the
#: same column: what this method ordered the candidates by.  Here it is the
#: query-conditioned probability, and its four factors sit beside it, so
#: score = p_det * p_view * p_mem * p_sem_q is checkable from the row alone.
#: ``p_sem_q`` is the only one that depends on the query; the margin behind it
#: is verified in :func:`verify_belief_ranking` rather than written out.
BELIEF_FACTOR_FIELDS = ["p_det", "p_view", "p_mem", "p_sem_q"]
BELIEF_SEARCH_FIELDS = SEARCH_KEY_FIELDS + ["score"] + BELIEF_FACTOR_FIELDS

GT_OBJECT_FIELDS = [
    "scene_id",
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

ROOM_FIELDS = [
    "scene_id",
    "predicted_room_id",
    "class_id",
    "class_label",
    "gt_class_present_in_room",
    "gt_class_present_in_scene",
    "num_gt_instances_in_room",
    "belief",
]


def _resolve(value, root: Path = _REPO_ROOT) -> Path:
    """Resolve a configured path, relative ones against the repository root."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _csv_float(value) -> str:
    """Render an optional probability, leaving an undefined one empty.

    An undefined factor is written as an empty cell rather than as a number:
    the uncertainty utilities return ``None`` precisely where the evidence does
    not support a probability, and substituting one here would invent it.
    """
    return "" if value is None else float(value)


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
# The shared object search
# ---------------------------------------------------------------------------


def stock_object_candidates(graph: Graph) -> Tuple[List, List[int]]:
    """Build the candidate list ``Graph.query_object`` searches.

    These are its own lines for the global search, with ``room_ids`` naming
    every room: the objects of the graph, walked room by room, each paired
    with the index of the room it hangs under.  Both methods search this one
    list, so a difference between their results can only come from the value
    they rank it by.

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
            is the value the ``hovsg`` method ranks by.
        max_scores: ``np.max(sim_mat, axis=0)`` -- identical to ``scores`` on
            every surviving object, which is asserted rather than assumed.
        survives: Whether each object's best-matching category is the query
            rather than a negative label.  This is the shared filter: both
            methods rank exactly these objects.
    """

    query_id: int
    categories: List[str]
    scores: np.ndarray
    max_scores: np.ndarray
    survives: np.ndarray


def stock_query(graph: Graph, query: str, object_embs: np.ndarray) -> StockQuery:
    """Score one query against the candidate objects, as stock HOV-SG does.

    Every step below is ``Graph.query_object``'s ``clip`` branch, in its
    order: the query/negative category list, the CLIP text features from
    ``get_text_feats_multiple_templates``, the plain dot product against the
    stored object embeddings, the per-object best category and the per-object
    maximum score.  Nothing is normalized, rescaled or thresholded on top.

    The result is shared by both methods: ``survives`` is the filter they
    agree on and ``scores`` is what only ``hovsg`` ranks by.
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


@dataclass(frozen=True)
class BeliefSignals:
    """The query-independent half of the belief score, prepared once per scene.

    Attributes:
        stored: ``P_det * P_view * P_mem`` per candidate -- the three factors
            the graph computed and persisted, multiplied together once.  Not
            one of them depends on the query, so this whole array is reused by
            every class.  ``nan`` marks a candidate one of whose factors is
            undefined.
        vocabulary_cos: ``cos(v_i, t_c)`` for every candidate against every
            vocabulary class, from the same cached CLIP text bank the graph was
            labeled with.  The margin of eq. (semantic) is two lookups into
            this matrix, so it too is computed once rather than per query.
        synonym_mask: ``build_synonym_eligibility_mask`` at the build-time tau.
            Row ``c`` selects the classes that are semantically distinct from
            class ``c`` and may therefore compete with it.
        logit_scale: alpha of eq. (semantic).
        synonym_threshold: tau of eq. (semantic).
    """

    stored: np.ndarray
    vocabulary_cos: np.ndarray
    synonym_mask: np.ndarray
    logit_scale: float
    synonym_threshold: float


def prepare_belief_signals(
    objects_list: Sequence,
    text_feats: np.ndarray,
    logit_scale: float,
    synonym_threshold: float,
) -> BeliefSignals:
    """Precompute everything in the belief score that the query cannot change.

    Three of the four factors are stored on the object node, and the fourth is
    a margin between two cosines drawn from one object-by-vocabulary matrix.
    That matrix does not depend on the query either -- only *which two of its
    columns* the margin uses does.  So the entire per-query cost is one masked
    maximum over a matrix that already exists, which is what makes the belief
    search no more expensive than stock's.

    The rows are filled one object at a time through the uncertainty module's
    own ``compute_cosine_similarities``, which is the call
    ``Graph._assign_object_semantic_signals`` used when it wrote the stored
    ``p_sem``.  Building the matrix with a single matrix-matrix product instead
    is mathematically the same and measurably not: BLAS accumulates a gemm in a
    different order than the gemv behind the per-object call, the cosines then
    differ in their last bits, and alpha = 100 amplifies that into ~1e-6 on
    ``p_sem``.  Going through the same function keeps this file's semantic
    factor bit-comparable with the graph's, which is what
    :func:`verify_semantic_factor_matches_graph` then measures.
    """
    stored = np.asarray(
        [
            [
                np.nan if getattr(obj, field, None) is None else float(getattr(obj, field))
                for field in ("p_det", "p_view", "p_mem")
            ]
            for obj in objects_list
        ],
        dtype=np.float64,
    ).prod(axis=1)

    normalized_text_feats = normalize_rows(np.asarray(text_feats, dtype=np.float64))
    vocabulary_cos = np.empty(
        (len(objects_list), normalized_text_feats.shape[0]), dtype=np.float64
    )
    for index, objectt in enumerate(objects_list):
        vocabulary_cos[index] = compute_cosine_similarities(
            objectt.embedding, normalized_text_feats, assume_normalized=True
        )
    return BeliefSignals(
        stored=stored,
        vocabulary_cos=vocabulary_cos,
        synonym_mask=build_synonym_eligibility_mask(
            normalized_text_feats, synonym_threshold
        ),
        logit_scale=float(logit_scale),
        synonym_threshold=float(synonym_threshold),
    )


@dataclass(frozen=True)
class BeliefQuery:
    """One class scored against every candidate by the query-conditioned belief.

    Attributes:
        query_cos: ``cos(v_i, t_q)`` per candidate.
        competitor_cos: the largest ``cos(v_i, t_c)`` over the classes that are
            semantically distinct from the query -- the runner-up the margin is
            measured against.
        competitor_class: which class that was, per candidate.
        margin: ``query_cos - competitor_cos``, eq. (semantic)'s m_i with the
            object's own label replaced by the query.
        p_sem: ``sigma(alpha * margin)``.
        values: the ranking value ``P_det * P_view * P_mem * P_sem(q)``, with
            ``-inf`` where it is undefined so such a candidate ranks last.
        defined: whether the ranking value is a real probability per candidate.
    """

    query_cos: np.ndarray
    competitor_cos: np.ndarray
    competitor_class: np.ndarray
    margin: np.ndarray
    p_sem: np.ndarray
    values: np.ndarray
    defined: np.ndarray


def belief_query(signals: BeliefSignals, class_id: int) -> BeliefQuery:
    """Evaluate the query-conditioned belief of one class against every object.

    This is eq. (semantic) read with the query in the place of the object's own
    label::

        m_i(q)     = cos(v_i, t_q) - max_{c: cos(t_c, t_q) < tau} cos(v_i, t_c)
        P_sem,i(q) = sigma(alpha * m_i(q))
        q_i(q)     = P_det,i * P_view,i * P_mem,i * P_sem,i(q)

    The maximum is taken over the classes the synonym mask keeps for the query,
    so a near-synonym of the query cannot be its own competitor and drive the
    margin to zero.  Nothing here reads the object's stored label: the same
    vocabulary is the reference for every object, and the query is what moves.

    When the query has no semantically distinct competitor at all the margin is
    undefined, and the ranking value is left undefined with it rather than
    invented -- the same choice ``compute_semantic_margin_uncertainty`` makes.
    """
    eligible = np.flatnonzero(signals.synonym_mask[class_id])
    query_cos = signals.vocabulary_cos[:, class_id]
    candidates = len(query_cos)

    if eligible.size == 0:
        undefined = np.full(candidates, np.nan, dtype=np.float64)
        return BeliefQuery(
            query_cos=query_cos,
            competitor_cos=undefined,
            competitor_class=np.full(candidates, -1, dtype=np.int64),
            margin=undefined,
            p_sem=undefined,
            values=np.full(candidates, -np.inf, dtype=np.float64),
            defined=np.zeros(candidates, dtype=bool),
        )

    competitor_block = signals.vocabulary_cos[:, eligible]
    competitor_position = np.argmax(competitor_block, axis=1)
    competitor_cos = competitor_block[np.arange(candidates), competitor_position]
    margin = query_cos - competitor_cos
    p_sem = expit(signals.logit_scale * margin)
    values = signals.stored * p_sem
    defined = np.isfinite(values)
    return BeliefQuery(
        query_cos=query_cos,
        competitor_cos=competitor_cos,
        competitor_class=eligible[competitor_position],
        margin=margin,
        p_sem=p_sem,
        values=np.where(defined, values, -np.inf),
        defined=defined,
    )


def rank_candidates(
    values: np.ndarray, survives: Optional[np.ndarray], top_k: int
) -> np.ndarray:
    """Order and truncate the candidates, optionally behind a filter.

    With ``survives`` given this is ``Graph.query_object``'s selection,
    expression for expression: the survivors of the query-versus-``background``
    filter are ordered by ``values`` descending and cut to ``top_k``, and when
    nothing survives, stock's fallback ranks the unfiltered list instead.
    Stock's two sort expressions are kept as stock writes them
    (``np.argsort(-x)`` on the survivor branch, ``np.argsort(x)[::-1]`` on the
    fallback branch) so the ``hovsg`` ranking is bit-identical to the live
    call, tie order included.

    With ``survives`` as ``None`` there is no filter and no fallback: the whole
    candidate list is ordered and cut.  That is the ``belief`` method, whose
    ranking value already answers the question the negative label was asked --
    an object that matches some other class better than the query carries a
    negative margin and sinks on its own.
    """
    top_index = np.argsort(values)[::-1][:top_k]
    if survives is not None and len(NEGATIVE_LABELS) > 0:
        obj_ids = np.where(survives)[0]
        if len(obj_ids) > 0:
            obj_values = values[obj_ids]
            resort_ids = np.argsort(-obj_values)
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
    """Write one row per predicted HOV-SG object, with its stored probabilities.

    The object inventory is a property of the graph, not of a search, so it
    carries no ``method`` column: both searches rank these same objects, and
    both read these same per-object signals.  The probabilities are copied off
    the object node exactly as ``Object.load`` read them back, so a factor the
    graph left undefined stays an empty cell here rather than becoming a
    number.
    """
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(OBJECT_FIELDS)
        for objectt in objects_list:
            centroid = object_centroid(objectt)
            label_idx = getattr(objectt, "label_idx", None)
            writer.writerow(
                [
                    scene_id,
                    objectt.object_id,
                    "" if label_idx is None else int(label_idx),
                    objectt.name,
                    objectt.room_id,
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                ]
                + [
                    _csv_float(getattr(objectt, field, None))
                    for field in OBJECT_PROBABILITY_FIELDS
                ]
            )
    return len(objects_list)


def write_gt_objects_csv(path: Path, scene_id: str, placement: RoomPlacement) -> int:
    """Write every HM3DSem ground-truth object and where it landed.

    One row per ground-truth object, in ground-truth order, carrying both of
    its room identities: ``gt_room_id`` is the HM3DSem region it is annotated
    in, ``predicted_room_id`` is the predicted room its centroid falls in.
    The objects no predicted room contains are here too, with an empty
    ``predicted_room_id`` and ``assigned = 0``, so the ones that produced no
    y(r, c) are on disk rather than only in a printed count.

    Grouping the assigned rows by ``(predicted_room_id, class_id)`` gives
    exactly the objects behind each ``gt_class_present_in_room = 1`` of
    ``rooms.csv``.  The ground truth is method-independent, so there is no
    ``method`` column.
    """
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(GT_OBJECT_FIELDS)
        writer.writerows(
            [scene_id] + [record[field] for field in GT_OBJECT_FIELDS[1:]]
            for record in placement.records
        )
    return len(placement.records)


def room_class_beliefs(rooms: Sequence) -> Dict[Tuple[str, int], float]:
    """Collect the rooms' stored eq. (noisyor) beliefs, keyed by (room, class).

    ``Room.class_containment_belief`` is computed when the graph is built and
    persisted with the room node, one entry per class instantiated in that
    room.  A class no object in the room was labeled with has no entry, and
    gets no cell value below.
    """
    beliefs: Dict[Tuple[str, int], float] = {}
    for room in rooms:
        for entry in getattr(room, "class_containment_belief", None) or []:
            class_id = entry.get("class_id")
            if class_id is None:
                continue
            beliefs[(room.room_id, int(class_id))] = entry.get("belief")
    return beliefs


def write_rooms_csv(
    path: Path,
    scene_id: str,
    rooms: Sequence,
    classes: Sequence[str],
    placement: RoomPlacement,
    beliefs: Dict[Tuple[str, int], float],
) -> int:
    """Write the full predicted-room x HM3DSem-vocabulary table.

    ``gt_class_present_in_room`` is y(r, c): whether predicted room ``r``
    holds at least one ground-truth object of class ``c``, where "holds" means
    the object's centroid lies in the room.  Its neighbour
    ``gt_class_present_in_scene`` asks the same of the whole scene, so the two
    differ only in scope.  Which objects those are is recorded per object in
    ``gt_objects.csv``; this table carries the label and the count.

    ``belief`` is the prediction side of the same cell: the room node's stored
    eq. (noisyor) belief that it contains an instance of ``c``, propagated from
    the ``p_obj`` of the objects the graph assigned to ``r`` and labeled ``c``.
    It is empty where no object in the room carries that label, because an
    empty O(r, c) is an absence of evidence rather than a belief of zero.
    """
    rows = 0
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(ROOM_FIELDS)
        for room in rooms:
            for class_id, class_label in enumerate(classes):
                count = placement.instances.get((room.room_id, class_id), 0)
                writer.writerow(
                    [
                        scene_id,
                        room.room_id,
                        class_id,
                        class_label,
                        int(count > 0),
                        int(class_id in placement.classes_in_scene),
                        int(count),
                        _csv_float(beliefs.get((room.room_id, class_id))),
                    ]
                )
                rows += 1
    return rows


def hovsg_search_row(
    scene_id: str,
    class_id: int,
    class_label: str,
    objectt,
    room_id: str,
    rank: int,
    score: float,
) -> List:
    """One ``object_search_hovsg.csv`` row: the score the search ranked by."""
    return [
        scene_id,
        HOVSG_METHOD,
        class_id,
        class_label,
        objectt.object_id,
        room_id,
        rank,
        float(score),
    ]


def belief_search_row(
    scene_id: str,
    class_id: int,
    class_label: str,
    objectt,
    room_id: str,
    rank: int,
    belief: BeliefQuery,
    index: int,
) -> List:
    """One ``object_search_belief.csv`` row: the ranking score and its factors.

    ``score`` is the query-conditioned probability the row was ranked by.  The
    three stored factors are read off the object node and ``p_sem_q`` off this
    class's :class:`BeliefQuery`, so the score is rebuildable from the row:
    ``score`` must equal ``p_det * p_view * p_mem * p_sem_q``.  That, and the
    margin behind ``p_sem_q``, are checked in :func:`verify_belief_ranking`.
    """
    defined = bool(belief.defined[index])
    return [
        scene_id,
        BELIEF_METHOD,
        class_id,
        class_label,
        objectt.object_id,
        room_id,
        rank,
        _csv_float(float(belief.values[index]) if defined else None),
        _csv_float(getattr(objectt, "p_det", None)),
        _csv_float(getattr(objectt, "p_view", None)),
        _csv_float(getattr(objectt, "p_mem", None)),
        _csv_float(
            float(belief.p_sem[index]) if np.isfinite(belief.p_sem[index]) else None
        ),
    ]


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
            the score written to the ``hovsg`` table.  Zero by construction of
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
    hovsg_order: Sequence[int],
    stock_ids: Sequence[int],
    stock_rooms: Sequence[int],
) -> Comparison:
    """Check the shared pipeline's ``hovsg`` ranking against the live stock call.

    ``stock_ids``/``stock_rooms`` are the live return value of
    ``Graph.query_object``.  ``hovsg_order`` is what :func:`rank_candidates`
    produced from ``query.scores`` -- the rows this file actually writes.  The
    two must name the same objects in the same order under the same parent
    rooms, which is what makes the refactored pipeline stock's own search
    rather than a lookalike.  The scores are re-derived through a second,
    independent CLIP embedding on top, so a silent divergence would have to
    survive both checks.
    """
    expected_ids = [object_index[objects_list[i].object_id] for i in hovsg_order]
    expected_rooms = [room_ids_list[i] for i in hovsg_order]

    if list(stock_ids) != expected_ids:
        detail = " in a different order" if len(stock_ids) == len(expected_ids) else ""
        return Comparison(
            False,
            float("nan"),
            float("nan"),
            f"'{query_class}': stock query_object returned {len(stock_ids)} "
            f"objects, the shared pipeline ranks {len(expected_ids)}{detail}",
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


def verify_candidate_pool(
    query_class: str,
    query: StockQuery,
    hovsg_order: Sequence[int],
    belief_order: Sequence[int],
    candidates: int,
    top_k: int,
) -> List[str]:
    """Check each method searched the pool its own definition gives it.

    The two no longer share a filter, so what must hold of the *results* is
    narrower than before and worth stating exactly:

    * both rank objects drawn from the one candidate list, and nothing else;
    * ``hovsg`` returns only objects that beat ``background``, unless nothing
      did and stock's fallback opened the list up again;
    * ``belief`` is never filtered, so it returns a full ``top_k`` whenever the
      graph has that many objects -- a short belief answer would mean a filter
      leaked into a method that is defined without one.
    """
    problems: List[str] = []
    for name, order in ((HOVSG_METHOD, hovsg_order), (BELIEF_METHOD, belief_order)):
        if any(index < 0 or index >= candidates for index in order):
            problems.append(
                f"'{query_class}': {name} returned an object outside the "
                f"candidate list"
            )

    survivors = int(np.count_nonzero(query.survives))
    if survivors and not set(hovsg_order).issubset(
        set(np.where(query.survives)[0].tolist())
    ):
        problems.append(
            f"'{query_class}': hovsg returned an object that did not survive "
            f"the background filter"
        )
    if len(belief_order) != min(candidates, top_k):
        problems.append(
            f"'{query_class}': belief returned {len(belief_order)} rows, but it "
            f"has no filter and should always return {min(candidates, top_k)}"
        )
    return problems


def verify_belief_ranking(
    query_class: str,
    class_id: int,
    belief: BeliefQuery,
    belief_order: Sequence[int],
    signals: BeliefSignals,
    objects_list: Sequence,
    top_k: int,
) -> List[str]:
    """Check the belief rows are ranked strictly by the query-conditioned score.

    Four independent things are asserted, none of them true by construction of
    the ranker:

    * the returned sequence is non-increasing, so the file is in ranked order;
    * the returned values are the ``top_k`` largest of the *whole* candidate
      list, recomputed here by sorting it directly -- which is also what proves
      no filter was applied;
    * every ranking value really is ``P_det * P_view * P_mem * P_sem(q)``;
    * ``P_sem(q)`` really is ``sigma(alpha * (query_cos - competitor_cos))``,
      and its competitor really is semantically distinct from the query under
      the same tau the mask was built with.
    """
    problems: List[str] = []
    ranked = belief.values[list(belief_order)]
    if ranked.size and np.any(np.diff(ranked) > 0):
        problems.append(f"'{query_class}': belief rows are not ordered by score")

    expected = np.sort(belief.values)[::-1][: min(len(objects_list), top_k)]
    if ranked.size != expected.size or not np.array_equal(ranked, expected):
        problems.append(
            f"'{query_class}': belief rows are not the top-{top_k} of the whole "
            f"candidate list"
        )

    for index in belief_order:
        if not belief.defined[index]:
            continue
        objectt = objects_list[index]
        factors = [
            getattr(objectt, field, None) for field in ("p_det", "p_view", "p_mem")
        ]
        if any(value is None for value in factors):
            problems.append(
                f"'{query_class}': object {objectt.object_id} has a defined "
                f"score but a missing stored factor"
            )
            continue
        product = float(np.prod(factors)) * float(belief.p_sem[index])
        if not np.isclose(
            float(belief.values[index]), product, rtol=0, atol=1e-12
        ):
            problems.append(
                f"'{query_class}': object {objectt.object_id} score "
                f"{float(belief.values[index]):.6e} is not the product of its "
                f"factors {product:.6e}"
            )
        recomputed = float(
            expit(
                signals.logit_scale
                * (float(belief.query_cos[index]) - float(belief.competitor_cos[index]))
            )
        )
        if not np.isclose(float(belief.p_sem[index]), recomputed, rtol=0, atol=1e-12):
            problems.append(
                f"'{query_class}': object {objectt.object_id} p_sem_q is not "
                f"sigma(alpha * margin)"
            )
        competitor = int(belief.competitor_class[index])
        if competitor >= 0 and not signals.synonym_mask[class_id, competitor]:
            problems.append(
                f"'{query_class}': object {objectt.object_id} was measured "
                f"against competitor class {competitor}, which tau excludes as "
                f"a synonym of the query"
            )
    return problems


#: How far the recomputed ``P_sem`` may sit from the graph's stored ``p_sem``.
#:
#: The two are the same formula on the same inputs, but not the same
#: arithmetic: the cached CLIP text bank is float32, so every cosine carries a
#: relative error near float32 epsilon (~1.2e-7), and eq. (semantic) multiplies
#: the margin by alpha = 100 before the sigmoid, which carries that error into
#: the fifth decimal of a probability.  Measured deviation on HM3DSem is ~5e-6.
#: The bound below is loose enough to admit that and far too tight to admit a
#: wrong tau, a wrong alpha or a misaligned vocabulary, each of which moves
#: ``p_sem`` by whole percentage points -- and the competitor identity is
#: checked exactly, without a tolerance, for precisely that reason.
SEMANTIC_FACTOR_TOLERANCE = 1e-4


def verify_semantic_factor_matches_graph(
    signals: BeliefSignals, objects_list: Sequence
) -> Tuple[int, float, int, List[str]]:
    """Check ``P_sem(q)`` reproduces the graph's ``p_sem`` at the object's label.

    The query-conditioned semantic factor is eq. (semantic) with the query in
    the place of the object's own label, so setting the query *to* that label
    must give back the ``p_sem`` the graph computed and stored.  That agreement
    is the only evidence that this file's CLIP text bank, cosine convention,
    synonym threshold tau and logit scale alpha are the ones the graph was
    built with: none of those four is persisted with the graph, and a silent
    mismatch in any of them would produce a plausible but wrong belief ranking.

    Two things are compared, and the discrete one carries the weight:

    * **which class the margin was measured against**, exactly.  It is an index,
      so it is either the graph's choice or it is not, and it can only be the
      graph's choice if tau, the mask and the vocabulary ordering all agree.
    * **the resulting probability**, within
      :data:`SEMANTIC_FACTOR_TOLERANCE`, which is set by the float32 precision
      of the text bank rather than by what would be convenient.

    Returns:
        How many objects were checked, the largest probability deviation, how
        many chose a different competitor, and the first few disagreements.
    """
    checked = 0
    deviation = 0.0
    competitor_mismatches = 0
    problems: List[str] = []
    by_class: Dict[int, BeliefQuery] = {}
    for index, objectt in enumerate(objects_list):
        label_idx = getattr(objectt, "label_idx", None)
        stored = getattr(objectt, "p_sem", None)
        if label_idx is None or stored is None:
            continue
        label_idx = int(label_idx)
        if label_idx not in by_class:
            by_class[label_idx] = belief_query(signals, label_idx)
        recomputed = by_class[label_idx].p_sem[index]
        if not np.isfinite(recomputed):
            continue
        checked += 1

        stored_competitor = getattr(objectt, "runner_up_idx", None)
        competitor = int(by_class[label_idx].competitor_class[index])
        if stored_competitor is not None and int(stored_competitor) != competitor:
            competitor_mismatches += 1
            if len(problems) < 5:
                problems.append(
                    f"object {objectt.object_id}: the graph measured its margin "
                    f"against class {int(stored_competitor)}, this run against "
                    f"class {competitor}"
                )

        difference = abs(float(recomputed) - float(stored))
        deviation = max(deviation, difference)
        if difference > SEMANTIC_FACTOR_TOLERANCE and len(problems) < 5:
            problems.append(
                f"object {objectt.object_id}: stored p_sem {float(stored):.9f}, "
                f"recomputed at its own label {float(recomputed):.9f}"
            )
    return checked, deviation, competitor_mismatches, problems


def load_previous_hovsg_rows(output_dir: Path) -> Tuple[Optional[Path], List[Tuple]]:
    """Read a previous run's HOV-SG search result, if one is on disk.

    Only the columns that carry the search's answer are read -- which class,
    which object, which parent room, and the score -- since the method label
    and the rank column are presentation, not result.
    """
    path = output_dir / "object_search_hovsg.csv"
    if path.exists():
        rows: List[Tuple] = []
        with path.open(newline="") as file:
            for record in csv.DictReader(file):
                rows.append(
                    (
                        int(record["class_id"]),
                        record["object_id"],
                        record["predicted_room_id"],
                        float(record["score"]),
                    )
                )
        return path, rows
    return None, []


def compare_previous_hovsg(
    previous: Sequence[Tuple], current: Sequence[Tuple]
) -> Tuple[bool, float, str]:
    """Check that HOV-SG still answers what it answered before.

    The comparison is positional, so it covers the returned order as well as
    the membership: row *n* of the previous file must name the same object in
    the same room for the same class as row *n* of this run.  The score is
    compared numerically and the largest deviation is reported, since a change
    there would mean the search itself moved even if the ranking did not.
    """
    if not previous:
        return True, 0.0, "no previous result on disk to compare against"
    if len(previous) != len(current):
        return (
            False,
            float("nan"),
            f"{len(current)} rows now, {len(previous)} before",
        )
    deviation = 0.0
    for index, (before, now) in enumerate(zip(previous, current)):
        if before[:3] != now[:3]:
            return (
                False,
                float("nan"),
                f"row {index} was {before[:3]}, is now {now[:3]}",
            )
        deviation = max(deviation, abs(before[3] - now[3]))
    return deviation == 0.0, deviation, ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(params: DictConfig) -> Dict[str, object]:
    """Generate the raw two-method object-search tables for one scene."""
    scene_id = str(params.main.scene_id)
    graph_path = _resolve(params.main.graph_path)
    scene_dir = _resolve(params.main.dataset_path) / str(params.main.split) / scene_id
    scene_info = scene_dir / "scene_info_remapped.json"
    output_dir = _resolve(params.main.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene {scene_id}")
    print(f"  predicted graph: {graph_path}")
    print(f"  HM3DSem ground truth: {scene_info}")
    print(f"  output directory: {output_dir}")

    # Read the previous HOV-SG answer before anything overwrites it.
    previous_path, previous_rows = load_previous_hovsg_rows(output_dir)
    if previous_path is not None:
        print(f"  previous HOV-SG result to reproduce: {previous_path}")

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
        label_text_feats, classes = get_label_feats(
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

    # The query-independent half of the belief score, built once: the three
    # stored factors multiplied together, and the object-by-vocabulary cosine
    # matrix the query-conditioned semantic margin reads two columns of.
    logit_scale = float(params.belief.semantic_uncertainty_logit_scale)
    synonym_threshold = float(params.belief.semantic_uncertainty_synonym_threshold)
    signals = prepare_belief_signals(
        objects_list, label_text_feats, logit_scale, synonym_threshold
    )
    undefined_stored = int(np.count_nonzero(~np.isfinite(signals.stored)))
    print(
        f"  belief signals: alpha={logit_scale}, tau={synonym_threshold}, "
        f"P_det*P_view*P_mem defined for "
        f"{len(objects_list) - undefined_stored} of {len(objects_list)} objects"
    )

    # Setting the query to the object's own label must give back the p_sem the
    # graph stored.  This is what proves alpha, tau, the text bank and the
    # cosine convention here are the graph's own.
    (
        semantic_checked,
        semantic_deviation,
        semantic_competitor_mismatches,
        semantic_problems,
    ) = verify_semantic_factor_matches_graph(signals, objects_list)

    # The class ids of the search tables index the same vocabulary the graph
    # was labeled with, so an object's stored label_idx must name its stored
    # class label in that vocabulary.  Checked, not assumed.
    inconsistent_labels = [
        obj.object_id
        for obj in objects_list
        if getattr(obj, "label_idx", None) is None
        or classes[int(obj.label_idx)] != obj.name
    ]

    objects_csv = output_dir / "objects.csv"
    object_rows = write_objects_csv(objects_csv, scene_id, objects_list)
    print(f"  wrote {object_rows} rows: {objects_csv}")

    gt_objects_csv = output_dir / "gt_objects.csv"
    gt_object_rows = write_gt_objects_csv(gt_objects_csv, scene_id, placement)
    print(f"  wrote {gt_object_rows} rows: {gt_objects_csv}")

    beliefs = room_class_beliefs(graph.rooms)
    rooms_csv = output_dir / "rooms.csv"
    room_rows = write_rooms_csv(
        rooms_csv, scene_id, graph.rooms, classes, placement, beliefs
    )
    print(
        f"  wrote {room_rows} rows: {rooms_csv} "
        f"({len(beliefs)} cells carry a room belief)"
    )

    # --- 4. Query the full vocabulary under both methods --------------------
    # One search per class feeds both rankings: the CLIP query embedding, the
    # background filter and the truncation are computed once and shared, and
    # only the value passed to rank_candidates differs.
    compared = comparison_class_ids(
        params.validation.stock_comparison_classes, len(classes)
    )
    compared_set = set(compared)
    hovsg_csv = output_dir / "object_search_hovsg.csv"
    belief_csv = output_dir / "object_search_belief.csv"
    top_k = int(params.main.get("top_k", STOCK_TOP_K))
    print(f"  top_k: {top_k} (the value HOV-SG's own application queries with)")

    hovsg_rows = 0
    belief_rows = 0
    rows_per_class: List[int] = []
    survivors_total = 0
    classes_with_survivors = 0
    fallback_classes: List[str] = []
    undefined_belief_classes: List[str] = []
    shape_mismatches: List[str] = []
    mismatches: List[str] = []
    pipeline_problems: List[str] = []
    belief_problems: List[str] = []
    current_hovsg_rows: List[Tuple] = []
    belief_answer_signature: set = set()
    max_sorting_deviation = 0.0
    max_score_deviation = 0.0

    with hovsg_csv.open("w", newline="") as hovsg_file, belief_csv.open(
        "w", newline=""
    ) as belief_file:
        hovsg_writer = csv.writer(hovsg_file)
        hovsg_writer.writerow(HOVSG_SEARCH_FIELDS)
        belief_writer = csv.writer(belief_file)
        belief_writer.writerow(BELIEF_SEARCH_FIELDS)

        for class_id, class_label in enumerate(classes):
            query = stock_query(graph, class_label, object_embs)
            survivors = int(np.count_nonzero(query.survives))
            survivors_total += survivors
            if survivors:
                classes_with_survivors += 1
            else:
                # Stock keeps its unfiltered list when nothing survives and
                # returns its top_k head.  Both methods do the same, each with
                # its own ordering value.  Recorded, not special-cased.
                fallback_classes.append(class_label)

            # Same candidate list, same truncation.  hovsg ranks the survivors
            # of its background filter by CLIP score; belief has no filter and
            # ranks everything by the query-conditioned probability.
            belief = belief_query(signals, class_id)
            if not belief.defined.any():
                undefined_belief_classes.append(class_label)
            hovsg_order = rank_candidates(query.scores, query.survives, top_k)
            belief_order = rank_candidates(belief.values, None, top_k)

            for rank, index in enumerate(hovsg_order, start=1):
                objectt = objects_list[index]
                room_id = graph.rooms[room_ids_list[index]].room_id
                score = float(query.scores[index])
                hovsg_writer.writerow(
                    hovsg_search_row(
                        scene_id, class_id, class_label, objectt, room_id, rank, score
                    )
                )
                current_hovsg_rows.append(
                    (class_id, objectt.object_id, room_id, score)
                )
            for rank, index in enumerate(belief_order, start=1):
                objectt = objects_list[index]
                room_id = graph.rooms[room_ids_list[index]].room_id
                belief_writer.writerow(
                    belief_search_row(
                        scene_id,
                        class_id,
                        class_label,
                        objectt,
                        room_id,
                        rank,
                        belief,
                        index,
                    )
                )

            hovsg_rows += len(hovsg_order)
            belief_rows += len(belief_order)
            rows_per_class.append(len(hovsg_order))
            belief_answer_signature.add(
                tuple(objects_list[i].object_id for i in belief_order)
            )

            # hovsg's row count must be stock's answer: its top_k head of the
            # survivors, or of every object when the fallback applies.  belief
            # is unfiltered, so it always returns a full head.
            expected = min(survivors if survivors else len(objects_list), top_k)
            if len(hovsg_order) != expected:
                shape_mismatches.append(
                    f"'{class_label}': {len(hovsg_order)} hovsg rows, expected "
                    f"{expected}"
                )

            pipeline_problems.extend(
                verify_candidate_pool(
                    class_label,
                    query,
                    hovsg_order,
                    belief_order,
                    len(objects_list),
                    top_k,
                )
            )
            belief_problems.extend(
                verify_belief_ranking(
                    class_label,
                    class_id,
                    belief,
                    belief_order,
                    signals,
                    objects_list,
                    top_k,
                )
            )

            if class_id in compared_set:
                # The final result of the stock search for this class, taken
                # from the stock function itself, is what the hovsg ranking
                # above must equal.
                with contextlib.redirect_stdout(io.StringIO()):
                    stock_ids, stock_rooms = graph.query_object(
                        class_label,
                        room_ids=list(range(len(graph.rooms))),
                        top_k=top_k,
                        negative_prompt=NEGATIVE_LABELS,
                    )
                comparison = verify_reproduction(
                    graph,
                    class_label,
                    query,
                    objects_list,
                    room_ids_list,
                    object_index,
                    object_embs,
                    hovsg_order,
                    stock_ids,
                    stock_rooms,
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

    print(f"  wrote {hovsg_rows} rows: {hovsg_csv}")
    print(f"  wrote {belief_rows} rows: {belief_csv}")

    reproduces_previous, previous_deviation, previous_message = compare_previous_hovsg(
        previous_rows, current_hovsg_rows
    )

    summary = {
        "scene_id": scene_id,
        "objects": len(objects_list),
        "rooms": len(graph.rooms),
        "classes": len(classes),
        "object_rows": object_rows,
        "hovsg_rows": hovsg_rows,
        "belief_rows": belief_rows,
        "room_rows": room_rows,
        "room_belief_cells": len(beliefs),
        "survivors_total": survivors_total,
        "classes_with_survivors": classes_with_survivors,
        "fallback_classes": fallback_classes,
        "belief_answer_signatures": len(belief_answer_signature),
        "undefined_belief_classes": undefined_belief_classes,
        "rows_per_class": rows_per_class,
        "top_k": top_k,
        "undefined_stored": undefined_stored,
        "logit_scale": logit_scale,
        "synonym_threshold": synonym_threshold,
        "semantic_checked": semantic_checked,
        "semantic_deviation": semantic_deviation,
        "semantic_competitor_mismatches": semantic_competitor_mismatches,
        "semantic_problems": semantic_problems,
        "shape_mismatches": shape_mismatches,
        "compared_classes": len(compared),
        "mismatches": mismatches,
        "pipeline_problems": pipeline_problems,
        "belief_problems": belief_problems,
        "max_sorting_deviation": max_sorting_deviation,
        "max_score_deviation": max_score_deviation,
        "inconsistent_labels": inconsistent_labels,
        "previous_path": previous_path,
        "previous_rows": len(previous_rows),
        "reproduces_previous": reproduces_previous,
        "previous_deviation": previous_deviation,
        "previous_message": previous_message,
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
            "hovsg": hovsg_csv,
            "belief": belief_csv,
            "gt_objects": gt_objects_csv,
            "rooms": rooms_csv,
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
            f"both search files are the search's answer: "
            f"{summary['hovsg_rows']} hovsg and {summary['belief_rows']} belief "
            f"rows over {len(rows_per_class)} of {classes} queried classes "
            f"({min(rows_per_class)}-{max(rows_per_class)} objects per class)",
        ),
        (
            summary["belief_rows"] == classes * min(objects, summary["top_k"]),
            f"belief is unfiltered and answers every query in full: "
            f"{summary['belief_rows']} rows = {classes} classes x "
            f"{min(objects, summary['top_k'])}, against {summary['hovsg_rows']} "
            f"hovsg rows behind the background filter",
        ),
        (
            not summary["shape_mismatches"],
            f"hovsg's filter and truncation are applied as stock applies "
            f"them: {len(summary['shape_mismatches'])} classes with an "
            f"unexpected row count, "
            f"{summary['survivors_total']} objects passed the background "
            f"filter across the vocabulary and {summary['hovsg_rows']} rows "
            f"survived stock's top-{summary['top_k']} cut",
        ),
        (
            not summary["pipeline_problems"],
            f"each method searched the pool its own definition gives it: "
            f"{len(summary['pipeline_problems'])} classes where a method "
            f"returned something its own rule excludes",
        ),
        (
            not summary["belief_problems"],
            f"belief ranks strictly by score = p_det * p_view * p_mem * "
            f"p_sem_q, with p_sem_q = sigma(alpha * margin): "
            f"{len(summary['belief_problems'])} ordering, factorization or "
            f"synonym violations across {classes} classes",
        ),
        (
            summary["semantic_competitor_mismatches"] == 0
            and summary["semantic_deviation"] <= SEMANTIC_FACTOR_TOLERANCE,
            f"the query-conditioned semantic factor is the graph's own: "
            f"evaluated at each object's own label it picks the same competitor "
            f"class on {summary['semantic_checked'] - summary['semantic_competitor_mismatches']} "
            f"of {summary['semantic_checked']} objects and reproduces the stored "
            f"p_sem to {summary['semantic_deviation']:.3e} "
            f"(float32 text bank x alpha={summary['logit_scale']} bounds this at "
            f"{SEMANTIC_FACTOR_TOLERANCE:.0e}; tau={summary['synonym_threshold']})",
        ),
        (
            not summary["mismatches"],
            f"stock query_object reproduced 1-to-1 on {summary['compared_classes']} "
            f"of {classes} classes: {len(summary['mismatches'])} mismatches, "
            f"max score deviation {summary['max_score_deviation']:.3e}, "
            f"max sorting-score deviation {summary['max_sorting_deviation']:.3e}",
        ),
        (
            summary["reproduces_previous"],
            f"HOV-SG reproduces the previous result: "
            + (
                summary["previous_message"]
                if summary["previous_message"]
                else f"{summary['previous_rows']} rows matched row for row against "
                f"{summary['previous_path'].name if summary['previous_path'] else 'nothing'}"
                f", max score deviation {summary['previous_deviation']:.3e}"
            ),
        ),
        (
            summary["room_rows"] == rooms * classes,
            f"rooms.csv is the full Cartesian product: "
            f"{summary['room_rows']} = {rooms} predicted rooms x {classes} "
            f"classes, {summary['positive_cells']} of them y(r,c)=1 and "
            f"{summary['room_belief_cells']} carrying a room belief",
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
        f"  [ok] no predicted-object-to-GT matching, no room score and no "
        f"retrieval metric is computed or written"
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
    if summary["undefined_stored"]:
        print(
            f"  objects with an undefined stored factor, ranked last by belief: "
            f"{summary['undefined_stored']} of {objects}"
        )
    if summary["undefined_belief_classes"]:
        shown = summary["undefined_belief_classes"][:10]
        print(
            f"  classes with no semantically distinct competitor under "
            f"tau={summary['synonym_threshold']}, so P_sem(q) is undefined: "
            f"{len(summary['undefined_belief_classes'])} (first {len(shown)}: "
            + ", ".join(shown)
            + ")"
        )
    if summary["fallback_classes"]:
        shown = summary["fallback_classes"][:10]
        print(
            f"  no object survived the background filter for "
            f"{len(summary['fallback_classes'])} of {classes} classes, so hovsg "
            f"fell back to ranking every object for them (first {len(shown)}: "
            + ", ".join(shown)
            + ")"
        )
    print(
        f"  belief answers depend on the query: "
        f"{summary['belief_answer_signatures']} distinct object sets across "
        f"{classes} queries"
    )
    for message in summary["semantic_problems"][:5]:
        print(f"  SEMANTIC FACTOR {message}")
    print(
        f"  objects returned across the vocabulary: {summary['hovsg_rows']} hovsg "
        f"and {summary['belief_rows']} belief rows from {classes} queries, at "
        f"most {summary['top_k']} each; {len(summary['fallback_classes'])} hovsg "
        f"queries answered from stock's empty-survivor fallback"
    )
    if placement.without_geometry:
        print(
            f"  ground-truth objects with an empty point cloud, counted as "
            f"unassigned: {placement.without_geometry}"
        )
    for message in summary["shape_mismatches"][:10]:
        print(f"  ROW COUNT {message}")
    for message in summary["pipeline_problems"][:10]:
        print(f"  SHARED PIPELINE {message}")
    for message in summary["belief_problems"][:10]:
        print(f"  BELIEF RANKING {message}")
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
