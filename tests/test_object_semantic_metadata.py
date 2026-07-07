import json

import numpy as np
import pytest

from hovsg.graph.object import Object


@pytest.fixture
def stub_open3d_io(monkeypatch):
    monkeypatch.setattr("hovsg.graph.object.o3d.io.write_point_cloud", lambda *_: True)
    monkeypatch.setattr(
        "hovsg.graph.object.o3d.io.read_point_cloud", lambda *_: object()
    )


@pytest.mark.parametrize(
    ("label_idx", "label_cos_sim", "semantic_uncertainty"),
    [(None, None, None), (np.int64(2), np.float32(0.75), np.float64(0.25))],
)
def test_semantic_metadata_round_trip(
    tmp_path,
    stub_open3d_io,
    label_idx,
    label_cos_sim,
    semantic_uncertainty,
):
    source = Object("0_0_0", "0_0", name="chair")
    source.pcd = object()
    source.vertices = np.zeros((2, 3))
    source.embedding = np.array([1.0, 0.0])
    source.label_idx = label_idx
    source.label_cos_sim = label_cos_sim
    source.semantic_uncertainty = semantic_uncertainty
    source.save(tmp_path)

    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))

    expected_idx = int(label_idx) if label_idx is not None else None
    assert restored.label_idx == expected_idx
    assert restored.label_cos_sim == (
        float(label_cos_sim) if label_cos_sim is not None else None
    )
    assert restored.semantic_uncertainty == (
        float(semantic_uncertainty) if semantic_uncertainty is not None else None
    )


def test_old_metadata_loads_without_semantic_fields(tmp_path, stub_open3d_io):
    metadata = {
        "object_id": "0_0_0",
        "vertices": [],
        "room_id": "0_0",
        "name": "chair",
        "embedding": [1.0, 0.0],
    }
    (tmp_path / "0_0_0.json").write_text(json.dumps(metadata), encoding="utf-8")

    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))

    assert restored.label_idx is None
    assert restored.label_cos_sim is None
    assert restored.semantic_uncertainty is None


def test_merge_normalizes_embedding_and_invalidates_semantic_cues():
    class FakeBoundingBox:
        def get_box_points(self):
            return np.zeros((8, 3))

    class FakePointCloud:
        def is_empty(self):
            return False

        def __iadd__(self, other):
            return self

        def get_axis_aligned_bounding_box(self):
            return FakeBoundingBox()

    left = Object("0_0_0", "0_0", name="chair")
    left.pcd = FakePointCloud()
    left.embedding = np.array([1.0, 0.0])
    left.label_idx = 3
    left.label_cos_sim = 0.8
    left.semantic_uncertainty = 0.2

    right = Object("0_0_1", "0_0", name="chair")
    right.pcd = FakePointCloud()
    right.embedding = np.array([0.0, 1.0])

    merged = left + right

    assert np.linalg.norm(merged.embedding) == pytest.approx(1.0)
    assert merged.label_idx == 3
    assert merged.label_cos_sim is None
    assert merged.semantic_uncertainty is None
