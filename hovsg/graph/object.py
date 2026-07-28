"""
    This file contains the class definition for the Object in HOV-SG.
"""
import json
import os

import numpy as np
import open3d as o3d

from hovsg.utils.detection_uncertainty import uncertainty_from_confidence



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
        self.c_sem = None
        self.u_sem = None
        self.vocab_log_partition = None
        self.negative_log_partition = None
        self.c_mem = None
        self.u_mem = None
        self.c_det = None
        self.u_det = None

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
            "c_sem": float(self.c_sem) if self.c_sem is not None else None,
            "u_sem": float(self.u_sem) if self.u_sem is not None else None,
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
            "c_mem": float(self.c_mem) if self.c_mem is not None else None,
            "u_mem": float(self.u_mem) if self.u_mem is not None else None,
            "c_det": float(self.c_det) if self.c_det is not None else None,
            "u_det": float(self.u_det) if self.u_det is not None else None,
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
            self.c_sem = metadata.get("c_sem")
            self.u_sem = metadata.get("u_sem")
            self.vocab_log_partition = metadata.get("vocab_log_partition")
            self.negative_log_partition = metadata.get("negative_log_partition")
            self.c_mem = metadata.get("c_mem")
            self.u_mem = metadata.get("u_mem")
            self.c_det = metadata.get("c_det")
            self.u_det = metadata.get("u_det")

    def __add__(self, other):
        """
            Method to add two objects together
            :param other: Object to add to self
        """
        if self.pcd.is_empty():
            return other
        if other.pcd.is_empty():
            return self
        self.pcd += other.pcd
        self.vertices = self.pcd.get_axis_aligned_bounding_box().get_box_points()
        self.embedding = np.mean([self.embedding, other.embedding], axis=0)
        embedding_norm = np.linalg.norm(self.embedding)
        if embedding_norm > 1e-8:
            self.embedding = self.embedding / embedding_norm
        c_det_values = [
            float(c_det)
            for c_det in (self.c_det, getattr(other, "c_det", None))
            if c_det is not None
        ]
        if c_det_values:
            self.c_det = float(np.clip(np.mean(c_det_values), 0.0, 1.0))
            self.u_det = uncertainty_from_confidence(self.c_det)
        else:
            self.c_det = None
            self.u_det = None
        self.label_cos_sim = None
        self.runner_up_idx = None
        self.runner_up_cos_sim = None
        self.semantic_margin = None
        self.c_sem = None
        self.u_sem = None
        self.vocab_log_partition = None
        self.negative_log_partition = None
        self.c_mem = None
        self.u_mem = None
        return self

    def __str__(self) -> str:
        return f"Name: {self.name}" + f"_{self.object_id}"
