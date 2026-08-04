import json

import numpy as np

from hovsg.graph.room import Room


def test_room_containment_metadata_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("hovsg.graph.room.o3d.io.write_point_cloud", lambda *_: True)
    monkeypatch.setattr("hovsg.graph.room.o3d.io.read_point_cloud", lambda *_: object())

    source = Room("0_0", "0", name="room_0")
    source.pcd = object()
    source.vertices = np.zeros((4, 2))
    source.class_containment_belief = {0: 0.8, 2: 0.4}
    source.class_containment_belief_correlated_limit = {0: 0.8, 2: 0.3}
    source.save(tmp_path)

    saved_metadata = json.loads((tmp_path / "0_0.json").read_text(encoding="utf-8"))
    assert saved_metadata["class_containment_belief"] == {"0": 0.8, "2": 0.4}
    assert saved_metadata["class_containment_belief_correlated_limit"] == {
        "0": 0.8,
        "2": 0.3,
    }

    restored = Room("0_0", "0")
    restored.load(str(tmp_path))

    assert restored.class_containment_belief == source.class_containment_belief
    assert (
        restored.class_containment_belief_correlated_limit
        == source.class_containment_belief_correlated_limit
    )


def test_room_metadata_without_beliefs_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("hovsg.graph.room.o3d.io.read_point_cloud", lambda *_: object())
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

    assert restored.class_containment_belief == {}
    assert restored.class_containment_belief_correlated_limit == {}
