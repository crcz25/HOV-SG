"""
    This file contains the class definition for the Object in HOV-SG.
"""
import json
import os

import numpy as np
import open3d as o3d



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
        self.semantic_uncertainty = None

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
            "semantic_uncertainty": (
                float(self.semantic_uncertainty)
                if self.semantic_uncertainty is not None
                else None
            ),
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
            self.semantic_uncertainty = metadata.get("semantic_uncertainty")

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
        self.label_cos_sim = None
        self.semantic_uncertainty = None
        return self

    def __str__(self) -> str:
        return f"Name: {self.name}" + f"_{self.object_id}"
