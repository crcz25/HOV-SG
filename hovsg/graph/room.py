"""
Room class to represent a room in a HOV-SGraph.
"""

import json
import os
from collections import defaultdict
from typing import Any, List

import numpy as np
import open3d as o3d

from hovsg.utils.clip_utils import get_img_feats, get_text_feats_multiple_templates
from hovsg.utils.uncertainty import compute_class_containment_belief


class Room:
    """
    Class to represent a room in a building.
    :param room_id: Unique identifier for the room
    :param floor_id: Identifier of the floor this room belongs to
    :param name: Name of the room (e.g., "Living Room", "Bedroom")
    """
    def __init__(self, room_id, floor_id, name=None):
        self.room_id = room_id  # Unique identifier for the room
        self.name = name  # Name of the room (e.g., "Living Room", "Bedroom")
        self.category = None  # placeholder for a GT category
        self.floor_id = floor_id  # Identifier of the floor this room belongs to
        self.objects = []  # List of objects inside the room
        self.vertices = []  # indices of the room in the point cloud 8 vertices
        self.embeddings = []  # List of tensors of embeddings of the room
        self.pcd = None  # Point cloud of the room
        # Mean of the final world-frame room point cloud.  It is populated
        # when room segmentation assigns the point cloud and persisted with
        # the room node.
        self.centroid = None
        self.room_height = None  # Height of the room
        self.room_zero_level = None  # Zero level of the room
        self.represent_images = []  # 5 images that represent the appearance of the room
        self.object_counter = 0
        # eq. (noisyor) per class instantiated in this room, as
        # {"class_id", "class_label", "belief"} entries.
        self.class_containment_belief = []

    def add_object(self, objectt):
        """
        Method to add objects to the room
        :param objectt: Object object to be added to the room
        """
        self.objects.append(objectt)  # Method to add objects to the room

    def update_centroid(self):
        """Compute this room's geometric centroid in map/world coordinates.

        ``self.pcd`` is selected from ``Graph.full_pcd`` without a final
        coordinate transform.  Averaging its points therefore preserves the
        Habitat world frame shared with the HM3D-Sem ground-truth regions.
        """
        if self.pcd is None:
            raise ValueError(f"Room {self.room_id} has no point cloud")
        points = np.asarray(self.pcd.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"Room {self.room_id} has no valid 3-D points")
        if not np.isfinite(points).all():
            raise ValueError(f"Room {self.room_id} contains non-finite points")
        self.centroid = np.mean(points, axis=0).tolist()
        return self.centroid

    def set_txt_embeddings(self, text):
        self.embeddings.append(get_text_feats_multiple_templates(text))

    def infer_room_type_from_view_embedding(
        self,
        default_room_types: List[str],
        clip_model: Any,
        clip_feat_dim: int,
    ) -> str:
        """Use the embeddings stored inside the room to infer room type. We should already
           save k views CLIP embeddings for each room. We match the k embeddings with room
           types' textual CLIP embeddings to get a room label for each of the k views. Then
           we count which room type has the most votes and return that.

        Args:
            default_room_types (List[str]): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: a room type from the default_room_types list
        """
        if len(self.embeddings) == 0:
            print("empty embeddings")
            return "unknown room type"
        text_feats = get_text_feats_multiple_templates(default_room_types, clip_model, clip_feat_dim)
        embeddings = np.array(self.embeddings)
        sim_mat = np.dot(embeddings, text_feats.T)
        # sim_mat = compute_similarity(embeddings, text_feats)
        print(sim_mat)
        col_ids = np.argmax(sim_mat, axis=1)
        votes = [default_room_types[i] for i in col_ids]
        print(f"the votes are: {votes}")
        unique, counts = np.unique(col_ids, return_counts=True)
        unique_id = np.argmax(counts)
        type_id = unique[unique_id]
        self.name = default_room_types[type_id]
        print(f"The room view ids are {self.represent_images}")
        print(f"The room type is {default_room_types[type_id]}")
        return default_room_types[type_id]

    def infer_room_type_from_objects(
        self,
        infer_method: str = "label",
        default_room_types: List[str] = None,
        clip_model: Any = None,
        clip_feat_dim: int = None,
    ) -> str:
        """Use the objects contained in the room to infer a room type. We want to ask GPT what kind of room it is from
        the names for the objects contained in the room.

        Args:
            infer_method (str): "label" if we want to directly use the pre-computed object names in the children nodes.
                                "obj_embedding" if we want to use the embedding of the room node's children to infer the
                                room type. default_room_types can not be None if infer_method is "embedding".
            default_room_types (List[str] = None): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: room type name
        """
        from llm.llm_utils import infer_room_type_from_object_list_chat

        if infer_method == "label":
            objects_list = []
            for obj_i, obj in enumerate(self.objects):
                obj: Object
                if not any(
                    substring in obj.name.lower()
                    for substring in [
                        "wall",
                        "floor",
                        "ceiling",
                        "railing",
                        "roof",
                        "void",
                        "unlabeled",
                        "misc",
                    ]
                ):
                    objects_list.append(obj.name)
            room_type = infer_room_type_from_object_list_chat(objects_list, default_room_type=default_room_types)
            self.name = room_type
        if infer_method == "obj_embedding":
            assert default_room_types, "default_room_types can not be None if infer_method is 'embedding'"
            object_embs = []
            for obj_i, obj in enumerate(self.objects):
                obj: Object
                object_embs.append(obj.embedding)

            represent_feat = feats_denoise_dbscan(object_embs).reshape((1, -1))
            text_feats = get_text_feats_multiple_templates(default_room_types, clip_model, clip_feat_dim)
            sim_mat = compute_similarity(represent_feat, text_feats)
            col_id = np.argmax(sim_mat)
            self.name = default_room_types[col_id]
        print("room_id, name: ", self.room_id, self.name)

    def group_objects_by_class(self):
        """Return this room's objects grouped by their assigned label.

        The groups are the sets O(r, c) = {o_i in O(r) : l_i = c} of
        eq. (noisyor), keyed by the class id c.  Only the objects added to
        this room are considered, so no object contributes to a room it is not
        assigned to, and each object enters exactly one group: the one of the
        single class the pipeline labeled it.

        An object whose fused probability q_i is undefined is left out.  The
        paper's product has no factor for a missing q_i, and substituting one
        would report a belief the evidence does not support.
        """
        objects_by_class = defaultdict(list)
        for objectt in self.objects:
            label_idx = getattr(objectt, "label_idx", None)
            if label_idx is None or getattr(objectt, "p_obj", None) is None:
                continue
            objects_by_class[int(label_idx)].append(objectt)
        return objects_by_class

    def compute_class_containment_beliefs(self):
        """Propagate the objects' q_i to one room-level belief per class.

        Applies eq. (noisyor) independently to every class instantiated in
        this room and stores the result in ``class_containment_belief`` as one
        entry per class, ordered by class id so the room node is independent
        of the order the objects were added in.
        """
        self.class_containment_belief = [
            {
                "class_id": class_id,
                # Every object in O(r, c) was labeled c, so they all carry
                # the same class name.
                "class_label": objects[0].name,
                "belief": compute_class_containment_belief(
                    objectt.p_obj for objectt in objects
                ),
            }
            for class_id, objects in sorted(self.group_objects_by_class().items())
        ]
        return self.class_containment_belief

    def save(self, path):
        """
        Save the room in folder as ply for the point cloud
        and json for the metadata
        """
        # Segmenting a room computes this eagerly, but keep direct Room users
        # on the same geometry-derived serialization path.
        if self.centroid is None:
            self.update_centroid()
        # save the point cloud
        o3d.io.write_point_cloud(os.path.join(path, str(self.room_id) + ".ply"), self.pcd)
        # save the metadata
        metadata = {
            "room_id": self.room_id,
            "name": self.name,
            "floor_id": self.floor_id,
            "centroid": self.centroid,
            "objects": [obj.object_id for obj in self.objects],
            "vertices": self.vertices.tolist(),
            "room_height": self.room_height,
            "room_zero_level": self.room_zero_level,
            "embeddings": [i.tolist() for i in self.embeddings],
            "represent_images": self.represent_images,
            "class_containment_belief": self.class_containment_belief,
        }
        with open(os.path.join(path, str(self.room_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the room from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, str(self.room_id) + ".ply"))
        # load the metadata
        with open(path + "/" + str(self.room_id) + ".json") as json_file:
            metadata = json.load(json_file)
            self.name = metadata["name"]
            self.floor_id = metadata["floor_id"]
            self.centroid = metadata["centroid"]
            self.vertices = np.asarray(metadata["vertices"])
            self.room_height = metadata["room_height"]
            self.room_zero_level = metadata["room_zero_level"]
            self.embeddings = [np.asarray(i) for i in metadata["embeddings"]]
            self.represent_images = metadata["represent_images"]
            self.class_containment_belief = metadata.get(
                "class_containment_belief", []
            )

    def __str__(self):
        return f"Room ID: {self.room_id}, Name: {self.name}, Floor ID: {self.floor_id}, Objects: {len(self.objects)}"
