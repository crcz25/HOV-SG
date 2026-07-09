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
    source.save(tmp_path)

    restored = Room("0_0", "0")
    restored.load(str(tmp_path))

    np.testing.assert_allclose(restored.class_containment_probs, [0.9, 0.2])
    assert restored.class_containment_topk == source.class_containment_topk


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
