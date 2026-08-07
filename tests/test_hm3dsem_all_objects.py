"""Ground-truth export of the complete semantic scene alongside the walk."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hovsg.data.hm3dsem.create_hm3dsem_walks_gt import (
    PanopticObject,
    PanopticScene,
    is_finite_vector,
    object_centroid,
    serialize_object,
)


SEPARATIONS = [0.0, 3.0, 6.0]


def make_object(object_id: str, hex_color: str, category: str, region_id: str) -> PanopticObject:
    return PanopticObject([object_id, hex_color, f'"{category}"', region_id])


def habitat_object(object_id: int, center) -> SimpleNamespace:
    """A stand-in for one ``sim.semantic_scene.objects`` entry."""
    center = list(map(float, center))
    return SimpleNamespace(
        id=f"level_region_{object_id}",
        aabb=SimpleNamespace(center=center, size=[1.0, 1.0, 1.0]),
        obb=SimpleNamespace(
            center=center,
            sizes=[0.9, 1.0, 0.8],
            rotation=[0.0, 0.0, 0.0, 1.0],
            local_to_world=np.eye(4).tolist(),
            world_to_local=np.eye(4).tolist(),
            volume=0.72,
            half_extents=[0.45, 0.5, 0.4],
        ),
    )


def make_scene():
    """One observed object, plus three that only the semantic scene knows."""
    objects = [
        make_object("1", "2AAEF2", "chair", "3"),  # observed on floor 0
        make_object("2", "FEDF6F", "table", "3"),  # unobserved, region 3 is known
        make_object("3", "C074ED", "lamp", "9"),   # unobserved, unknown region, floor 1 height
        make_object("4", "200EF8", "roof", "9"),   # unobserved, unknown region, above every storey
    ]
    habitat_scene = SimpleNamespace(
        objects=[
            habitat_object(1, [1.0, 1.0, 1.0]),
            habitat_object(2, [2.0, 1.5, 2.0]),
            habitat_object(3, [3.0, 4.5, 3.0]),
            habitat_object(4, [4.0, 42.0, 4.0]),
        ]
    )
    scene = PanopticScene("00000-scene", habitat_scene, objects)

    observed = scene.get_object(1)
    observed.points = np.array([[1.0, 0.9, 1.0], [1.2, 1.1, 1.0], [1.0, 1.0, 1.2]])
    observed.colors = np.zeros_like(observed.points)
    scene.label_mapped_objects()
    scene.construct_regions()
    scene.assign_regions_to_floors(SEPARATIONS)
    scene.assign_remaining_object_floors(SEPARATIONS)
    return scene


def test_habitat_geometry_reaches_every_annotated_object():
    scene = make_scene()

    assert scene.habitat_object_count == 4
    assert [obj.habitat_matched for obj in scene.objects] == [True] * 4
    assert scene.get_object(4).obb_center == [4.0, 42.0, 4.0]


def test_all_objects_is_a_superset_of_the_visible_objects(tmp_path):
    scene = make_scene()
    scene.write_metadata(str(tmp_path))
    scene_info = json.loads((tmp_path / "scene_info.json").read_text())

    objects = scene_info["objects"]
    all_objects = scene_info["all_objects"]
    assert [obj["id"] for obj in objects] == [1]
    assert [obj["id"] for obj in all_objects] == [1, 2, 3, 4]
    # The visible collection keeps its meaning and stays field-for-field equal
    # to its counterpart in the complete-scene collection.
    by_id = {obj["id"]: obj for obj in all_objects}
    assert all(by_id[obj["id"]] == obj for obj in objects)
    assert [obj["observed_in_walk"] for obj in all_objects] == [True, False, False, False]


def test_regions_export_the_mean_of_their_reconstructed_world_points(tmp_path):
    scene = make_scene()
    scene.write_metadata(str(tmp_path))

    region = json.loads((tmp_path / "scene_info.json").read_text())["regions"][0]
    assert region["centroid"] == pytest.approx([1.0666666, 1.0, 1.0666666])
    assert len(region["centroid"]) == 3


def test_unobserved_objects_take_a_floor_from_their_region_then_their_height(tmp_path):
    scene = make_scene()
    scene.write_metadata(str(tmp_path))
    floors = {obj["id"]: obj["floor_id"] for obj in json.loads((tmp_path / "scene_info.json").read_text())["all_objects"]}

    assert floors[1] == 0  # observed, via its reconstructed region
    assert floors[2] == 0  # unobserved, but region 3 sits on floor 0
    assert floors[3] == 1  # unobserved and region-less, from its 4.5 m centroid
    assert floors[4] is None  # 42 m is on no storey, so nothing is guessed


def test_unmatched_annotations_are_excluded_with_a_warning(tmp_path, caplog):
    objects = [make_object("1", "2AAEF2", "chair", "3"), make_object("7", "FEDF6F", "ghost", "3")]
    scene = PanopticScene("00000-scene", SimpleNamespace(objects=[habitat_object(1, [1.0, 1.0, 1.0])]), objects)
    scene.get_object(1).points = np.array([[1.0, 1.0, 1.0]])
    scene.get_object(1).colors = np.zeros((1, 3))
    scene.label_mapped_objects()
    scene.construct_regions()
    scene.assign_regions_to_floors(SEPARATIONS)
    scene.assign_remaining_object_floors(SEPARATIONS)
    with caplog.at_level("WARNING"):
        scene.write_metadata(str(tmp_path))

    scene_info = json.loads((tmp_path / "scene_info.json").read_text())
    assert [obj["id"] for obj in scene_info["all_objects"]] == [1]
    assert "excluded from all_objects" in caplog.text


def test_centroid_prefers_observed_points_then_the_source_boxes():
    scene = make_scene()

    observed = scene.get_object(1)
    assert object_centroid(observed) == pytest.approx([1.0666666, 1.0, 1.0666666])
    # Without points the complete-scene OBB centre stands in for the centroid.
    assert object_centroid(scene.get_object(3)) == [3.0, 4.5, 3.0]

    unusable = scene.get_object(3)
    unusable.obb_center = [float("nan")] * 3
    assert object_centroid(unusable) == [3.0, 4.5, 3.0]  # falls back to the AABB centre
    unusable.aabb_center = None
    assert object_centroid(unusable) is None


def test_serialized_objects_expose_the_required_ground_truth_fields():
    scene = make_scene()
    item = serialize_object(scene.get_object(2))

    for field in (
        "id", "category", "region_id", "floor_id", "observed_in_walk", "centroid",
        "aabb_center", "aabb_dims", "obb_center", "obb_dims", "obb_rotation",
        "obb_local_to_world", "obb_world_to_local", "obb_volume", "obb_half_extents",
    ):
        assert field in item
    assert item["category"] == "table"
    assert item["observed_in_walk"] is False
    assert all(is_finite_vector(item[key]) for key in ("centroid", "aabb_center", "aabb_dims", "obb_center", "obb_dims"))


def test_is_finite_vector_rejects_unusable_geometry():
    assert is_finite_vector([0.0, 1.0, 2.0])
    assert not is_finite_vector(None)
    assert not is_finite_vector([0.0, 1.0])
    assert not is_finite_vector([0.0, float("inf"), 2.0])
    assert not is_finite_vector([[0.0, 1.0, 2.0]])
