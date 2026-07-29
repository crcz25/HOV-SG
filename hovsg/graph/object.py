"""
    This file contains the class definition for the Object in HOV-SG.
"""
import json
import os

import numpy as np
import open3d as o3d

from hovsg.utils.detection_uncertainty import (
    confidence_from_sum,
    uncertainty_from_confidence,
)
from hovsg.utils.cross_view_consistency import cross_view_values
from hovsg.utils.uncertainty import (
    COHERENCE_FIELDS,
    MEMBERSHIP_FIELDS,
    OBJECT_FIELDS,
    SEMANTIC_FIELDS,
)



class Object:
    """
    Class to represent an object in a room.
    :param object_id: Unique identifier for the object
    :param room_id: Identifier of the room this object belongs to
    :param name: Name of the object (e.g., "Chair", "Table")
    """
    def __init__(self, object_id, room_id, name=None):
        self.object_id = object_id  # Unique identifier for the object
        self.vertices = None  # Coordinates of the object in the point cloud 8 vertices
        self.embedding = None  # CLIP Embedding of the object
        self.pcd = None  # Point cloud of the object
        self.room_id = room_id  # Identifier of the room this object belongs to
        self.name = name  # Name of the object (e.g., "Chair", "Table")
        self.gt_name = None
        self.label_idx = None
        self.label_cos_sim = None
        self.runner_up_idx = None
        self.runner_up_cos_sim = None
        self.semantic_margin = None
        self.p_sem = None
        self.u_sem = None
        self.coherence_prototype_cos_sim = None
        self.coherence_runner_up_class = None
        self.coherence_runner_up_cos_sim = None
        self.label_coherence_margin = None
        self.p_coh = None
        self.u_coh = None
        self.vocab_log_partition = None
        self.negative_log_partition = None
        self.p_mem = None
        self.u_mem = None
        # Detection: the raw point-confidence accumulators are stored so that
        # merges recover the exact pooled mean regardless of merge order.
        self.detection_conf_sum = None
        self.detection_point_count = None
        self.p_det = None
        self.u_det = None
        # Cross-view: the *unnormalized* resultant sum of unit per-view
        # embeddings and its observation count. The normalized fused embedding
        # never replaces these; ||sum / count|| is the signal itself.
        self.cross_view_resultant_sum = None
        self.cross_view_count = None
        self.cross_view_point_count = None
        self.p_view = None
        self.u_view = None
        self.cross_view_sufficient = None
        self.cross_view_consistency_min_observations = 2
        self.p_obj = None
        self.u_obj = None

    def set_vertices(self, vertices):
        """
        Method to set the vertices of the object
        :param vertices: Coordinates of the object in the point cloud 8 vertices
        """
        self.vertices = vertices  # Method to set the vertices of the object

    def save(self, path):
        """
        Save the object in folder as ply for the point cloud
        and json for the metadata
        """
        # save the point cloud
        o3d.io.write_point_cloud(os.path.join(path, str(self.object_id) + ".ply"), self.pcd)
        # save the metadata
        metadata = {
            "object_id": self.object_id,
            "vertices": np.array(self.vertices).tolist(),
            "room_id": self.room_id,
            "name": self.name,
            "embedding": self.embedding.tolist() if self.embedding is not None else "",
            "label_idx": int(self.label_idx) if self.label_idx is not None else None,
            "label_cos_sim": (
                float(self.label_cos_sim) if self.label_cos_sim is not None else None
            ),
            "runner_up_idx": (
                int(self.runner_up_idx) if self.runner_up_idx is not None else None
            ),
            "runner_up_cos_sim": (
                float(self.runner_up_cos_sim)
                if self.runner_up_cos_sim is not None
                else None
            ),
            "semantic_margin": (
                float(self.semantic_margin)
                if self.semantic_margin is not None
                else None
            ),
            "p_sem": float(self.p_sem) if self.p_sem is not None else None,
            "u_sem": float(self.u_sem) if self.u_sem is not None else None,
            "coherence_prototype_cos_sim": (
                float(self.coherence_prototype_cos_sim)
                if self.coherence_prototype_cos_sim is not None
                else None
            ),
            "coherence_runner_up_class": (
                int(self.coherence_runner_up_class)
                if isinstance(self.coherence_runner_up_class, (int, np.integer))
                else self.coherence_runner_up_class
            ),
            "coherence_runner_up_cos_sim": (
                float(self.coherence_runner_up_cos_sim)
                if self.coherence_runner_up_cos_sim is not None
                else None
            ),
            "label_coherence_margin": (
                float(self.label_coherence_margin)
                if self.label_coherence_margin is not None
                else None
            ),
            "p_coh": float(self.p_coh) if self.p_coh is not None else None,
            "u_coh": float(self.u_coh) if self.u_coh is not None else None,
            "vocab_log_partition": (
                float(self.vocab_log_partition)
                if self.vocab_log_partition is not None
                else None
            ),
            "negative_log_partition": (
                float(self.negative_log_partition)
                if self.negative_log_partition is not None
                else None
            ),
            "p_mem": float(self.p_mem) if self.p_mem is not None else None,
            "u_mem": float(self.u_mem) if self.u_mem is not None else None,
            "detection_conf_sum": (
                float(self.detection_conf_sum)
                if self.detection_conf_sum is not None
                else None
            ),
            "detection_point_count": (
                int(self.detection_point_count)
                if self.detection_point_count is not None
                else None
            ),
            "p_det": float(self.p_det) if self.p_det is not None else None,
            "u_det": float(self.u_det) if self.u_det is not None else None,
            "cross_view_resultant_sum": (
                np.asarray(self.cross_view_resultant_sum).tolist()
                if self.cross_view_resultant_sum is not None
                else ""
            ),
            "cross_view_count": (
                int(self.cross_view_count) if self.cross_view_count is not None else None
            ),
            "cross_view_point_count": (
                int(self.cross_view_point_count)
                if self.cross_view_point_count is not None
                else None
            ),
            "p_view": float(self.p_view) if self.p_view is not None else None,
            "u_view": float(self.u_view) if self.u_view is not None else None,
            "cross_view_sufficient": (
                bool(self.cross_view_sufficient)
                if self.cross_view_sufficient is not None
                else None
            ),
            "p_obj": float(self.p_obj) if self.p_obj is not None else None,
            "u_obj": float(self.u_obj) if self.u_obj is not None else None,
        }
        with open(os.path.join(path, str(self.object_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the object from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, str(self.object_id) + ".ply"))
        # load the metadata
        with open(path + "/" + str(self.object_id) + ".json") as json_file:
            metadata = json.load(json_file)
            self.vertices = np.asarray(metadata["vertices"])
            self.room_id = metadata["room_id"]
            self.name = metadata["name"]
            self.embedding = np.asarray(metadata["embedding"]) if metadata["embedding"] != "" else None
            self.label_idx = metadata.get("label_idx")
            self.label_cos_sim = metadata.get("label_cos_sim")
            self.runner_up_idx = metadata.get("runner_up_idx")
            self.runner_up_cos_sim = metadata.get("runner_up_cos_sim")
            self.semantic_margin = metadata.get("semantic_margin")
            self.p_sem = metadata.get("p_sem")
            self.u_sem = metadata.get("u_sem")
            self.coherence_prototype_cos_sim = metadata.get(
                "coherence_prototype_cos_sim"
            )
            self.coherence_runner_up_class = metadata.get("coherence_runner_up_class")
            self.coherence_runner_up_cos_sim = metadata.get(
                "coherence_runner_up_cos_sim"
            )
            self.label_coherence_margin = metadata.get("label_coherence_margin")
            self.p_coh = metadata.get("p_coh")
            self.u_coh = metadata.get("u_coh")
            self.vocab_log_partition = metadata.get("vocab_log_partition")
            self.negative_log_partition = metadata.get("negative_log_partition")
            self.p_mem = metadata.get("p_mem")
            self.u_mem = metadata.get("u_mem")
            self.detection_conf_sum = metadata.get("detection_conf_sum")
            self.detection_point_count = metadata.get("detection_point_count")
            if self.detection_point_count is not None:
                self.detection_point_count = int(self.detection_point_count)
            self.p_det = metadata.get("p_det")
            self.u_det = metadata.get("u_det")
            cross_view_sum = metadata.get("cross_view_resultant_sum", "")
            self.cross_view_resultant_sum = (
                np.asarray(cross_view_sum, dtype=np.float64)
                if cross_view_sum is not None and cross_view_sum != ""
                else None
            )
            self.cross_view_count = metadata.get("cross_view_count")
            if self.cross_view_count is not None:
                self.cross_view_count = int(self.cross_view_count)
            self.cross_view_point_count = metadata.get("cross_view_point_count")
            if self.cross_view_point_count is not None:
                self.cross_view_point_count = int(self.cross_view_point_count)
            self.p_view = metadata.get("p_view")
            self.u_view = metadata.get("u_view")
            self.cross_view_sufficient = metadata.get("cross_view_sufficient")
            self.p_obj = metadata.get("p_obj")
            self.u_obj = metadata.get("u_obj")

    def __add__(self, other):
        """
            Method to add two objects together
            :param other: Object to add to self
        """
        def invalidate_coherence(target):
            for field in COHERENCE_FIELDS:
                setattr(target, field, None)

        def invalidate_object_probability(target):
            for field in OBJECT_FIELDS:
                setattr(target, field, None)

        if self.pcd.is_empty():
            invalidate_coherence(other)
            invalidate_object_probability(other)
            return other
        if other.pcd.is_empty():
            invalidate_coherence(self)
            invalidate_object_probability(self)
            return self
        self.pcd += other.pcd
        self.vertices = self.pcd.get_axis_aligned_bounding_box().get_box_points()
        self.embedding = np.mean([self.embedding, other.embedding], axis=0)
        embedding_norm = np.linalg.norm(self.embedding)
        if embedding_norm > 1e-8:
            self.embedding = self.embedding / embedding_norm

        # Detection: add the raw point-confidence accumulators, so P_det stays
        # the pooled mean over the union of both objects' points. Averaging the
        # two already-averaged P_det values instead would weight a chain of
        # merges by 1/2, 1/4, 1/8, ... and depend on merge order.
        detection_sums = (self.detection_conf_sum, getattr(other, "detection_conf_sum", None))
        detection_counts = (
            self.detection_point_count,
            getattr(other, "detection_point_count", None),
        )
        if all(value is not None for value in detection_sums + detection_counts):
            self.detection_conf_sum = float(detection_sums[0]) + float(detection_sums[1])
            self.detection_point_count = int(detection_counts[0]) + int(detection_counts[1])
            self.p_det = confidence_from_sum(
                self.detection_conf_sum, self.detection_point_count
            )
            self.u_det = (
                uncertainty_from_confidence(self.p_det) if self.p_det is not None else None
            )
        else:
            self.detection_conf_sum = None
            self.detection_point_count = None
            self.p_det = None
            self.u_det = None

        # Cross-view: sums and counts are additive, so the merged running mean
        # is exactly the mean over the union of both observation sets.
        cross_view_sums = (
            self.cross_view_resultant_sum,
            getattr(other, "cross_view_resultant_sum", None),
        )
        cross_view_counts = (
            self.cross_view_count,
            getattr(other, "cross_view_count", None),
        )
        if all(value is not None for value in cross_view_sums + cross_view_counts):
            self.cross_view_resultant_sum = np.asarray(cross_view_sums[0]) + np.asarray(
                cross_view_sums[1]
            )
            self.cross_view_count = int(cross_view_counts[0]) + int(cross_view_counts[1])
            point_counts = (
                self.cross_view_point_count,
                getattr(other, "cross_view_point_count", None),
            )
            self.cross_view_point_count = (
                int(point_counts[0]) + int(point_counts[1])
                if all(value is not None for value in point_counts)
                else None
            )
            min_observations = max(
                int(getattr(self, "cross_view_consistency_min_observations", 2)),
                int(getattr(other, "cross_view_consistency_min_observations", 2)),
            )
            self.cross_view_consistency_min_observations = min_observations
            self.p_view, self.u_view, self.cross_view_sufficient = cross_view_values(
                self.cross_view_resultant_sum,
                self.cross_view_count,
                min_observations=min_observations,
                point_count=self.cross_view_point_count,
            )
        else:
            self.cross_view_resultant_sum = None
            self.cross_view_count = None
            self.cross_view_point_count = None
            self.p_view = None
            self.u_view = None
            self.cross_view_sufficient = None
        # The merged embedding v_i is new, so every signal read off it is stale.
        # Clearing them makes a missed recomputation visible as an undefined
        # value rather than as a stale number attributed to the merged object;
        # Graph.recompute_semantic_uncertainty refills them.
        for field in SEMANTIC_FIELDS + MEMBERSHIP_FIELDS:
            setattr(self, field, None)
        invalidate_coherence(self)
        invalidate_object_probability(self)
        return self

    def __str__(self) -> str:
        return f"Name: {self.name}" + f"_{self.object_id}"
