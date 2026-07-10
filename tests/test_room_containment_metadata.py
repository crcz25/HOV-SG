import json

import numpy as np
import pytest

from hovsg.graph.room import Room


@pytest.fixture
def stub_open3d_io(monkeypatch):
    monkeypatch.setattr("hovsg.graph.room.o3d.io.write_point_cloud", lambda *_: True)
    monkeypatch.setattr(
        "hovsg.graph.room.o3d.io.read_point_cloud", lambda *_: object()
    )


def test_room_containment_metadata_round_trip(tmp_path, stub_open3d_io):
    source = Room("0_0", "0", name="room_0")
    source.pcd = object()
    source.vertices = np.zeros((4, 2))
    source.class_containment_probs = np.array([0.9, 0.2])
    source.class_containment_topk = [
        {"class_idx": 0, "class_name": "chair", "prob": 0.9}
    ]
    source.object_beliefs_semantic = {
        "0_0_0": {"class_idx": 0, "class_name": "chair", "q": 0.8}
    }
    source.object_beliefs_detection = {
        "0_0_0": {"class_idx": 0, "class_name": "chair", "q": 0.6}
    }
    source.object_beliefs_combined = {
        "0_0_0": {"class_idx": 0, "class_name": "chair", "q": 0.48}
    }
    source.class_containment_beliefs_semantic = {0: 0.8}
    source.class_containment_beliefs_detection = {0: 0.6}
    source.class_containment_beliefs_combined = {0: 0.48}
    source.save(tmp_path)

    saved_metadata = json.loads((tmp_path / "0_0.json").read_text(encoding="utf-8"))
    assert "class_containment_topk_semantic" not in saved_metadata
    assert "class_containment_topk_detection" not in saved_metadata
    assert "class_containment_topk_combined" not in saved_metadata

    restored = Room("0_0", "0")
    restored.load(str(tmp_path))

    np.testing.assert_allclose(restored.class_containment_probs, [0.9, 0.2])
    assert restored.class_containment_topk == source.class_containment_topk
    assert restored.object_beliefs_semantic == source.object_beliefs_semantic
    assert restored.object_beliefs_detection == source.object_beliefs_detection
    assert restored.object_beliefs_combined == source.object_beliefs_combined
    assert restored.class_containment_beliefs_semantic == {0: 0.8}
    assert restored.class_containment_beliefs_detection == {0: 0.6}
    assert restored.class_containment_beliefs_combined == {0: 0.48}


def test_old_room_metadata_loads_without_containment_fields(
    tmp_path, stub_open3d_io
):
    metadata = {
        "room_id": "0_0",
        "name": "room_0",
        "floor_id": "0",
        "objects": [],
        "vertices": [],
        "room_height": 2.5,
        "room_zero_level": 0.0,
        "embeddings": [],
        "represent_images": [],
    }
    (tmp_path / "0_0.json").write_text(json.dumps(metadata), encoding="utf-8")

    restored = Room("0_0", "0")
    restored.load(str(tmp_path))

    assert restored.class_containment_probs is None
    assert restored.class_containment_topk is None
    assert restored.object_beliefs_semantic == {}
    assert restored.object_beliefs_detection == {}
    assert restored.object_beliefs_combined == {}
    assert restored.class_containment_beliefs_semantic is None
    assert restored.class_containment_beliefs_detection is None
    assert restored.class_containment_beliefs_combined is None
