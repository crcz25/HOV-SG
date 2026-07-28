import json

import numpy as np
import pytest

from hovsg.graph.object import Object


SEMANTIC_FIELDS = (
    "label_idx",
    "label_cos_sim",
    "runner_up_idx",
    "runner_up_cos_sim",
    "semantic_margin",
    "c_sem",
    "u_sem",
)


@pytest.fixture
def stub_open3d_io(monkeypatch):
    monkeypatch.setattr("hovsg.graph.object.o3d.io.write_point_cloud", lambda *_: True)
    monkeypatch.setattr(
        "hovsg.graph.object.o3d.io.read_point_cloud", lambda *_: object()
    )


@pytest.mark.parametrize(
    "values",
    [
        (None, None, None, None, None, None, None),
        (
            np.int64(2),
            np.float32(0.75),
            np.int32(5),
            np.float64(0.5),
            np.float32(0.25),
            np.float64(0.9),
            np.float32(0.1),
        ),
    ],
)
def test_semantic_metadata_round_trip(tmp_path, stub_open3d_io, values):
    source = Object("0_0_0", "0_0", name="chair")
    source.pcd = object()
    source.vertices = np.zeros((2, 3))
    source.embedding = np.array([1.0, 0.0])
    for field, value in zip(SEMANTIC_FIELDS, values):
        setattr(source, field, value)
    source.save(tmp_path)

    saved = json.loads((tmp_path / "0_0_0.json").read_text(encoding="utf-8"))
    assert "semantic_uncertainty" not in saved
    assert all(field in saved for field in SEMANTIC_FIELDS)

    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))

    for field, value in zip(SEMANTIC_FIELDS, values):
        expected = int(value) if field.endswith("idx") and value is not None else value
        expected = float(expected) if expected is not None and not field.endswith("idx") else expected
        assert getattr(restored, field) == expected
def test_metadata_without_optional_uncertainty_fields_loads(tmp_path, stub_open3d_io):
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

    for field in SEMANTIC_FIELDS:
        assert getattr(restored, field) is None
    for field in ("vocab_log_partition", "negative_log_partition", "c_mem", "u_mem"):
        assert getattr(restored, field) is None
    assert not hasattr(restored, "semantic_uncertainty")


def test_membership_metadata_round_trip(tmp_path, stub_open3d_io):
    source = Object("0_0_0", "0_0", name="chair")
    source.pcd = object()
    source.vertices = np.zeros((2, 3))
    source.embedding = np.array([1.0, 0.0])
    source.vocab_log_partition = np.float64(12.5)
    source.negative_log_partition = np.float32(3.25)
    source.c_mem = np.float64(0.9)
    source.u_mem = np.float32(0.1)
    source.save(tmp_path)

    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))

    assert restored.vocab_log_partition == pytest.approx(12.5)
    assert restored.negative_log_partition == pytest.approx(3.25)
    assert restored.c_mem == pytest.approx(0.9)
    assert restored.u_mem == pytest.approx(0.1)


def test_merge_normalizes_embedding_and_invalidates_margin_fields():
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
    for field in SEMANTIC_FIELDS[1:]:
        setattr(left, field, 0.5)

    right = Object("0_0_1", "0_0", name="chair")
    right.pcd = FakePointCloud()
    right.embedding = np.array([0.0, 1.0])

    merged = left + right

    assert np.linalg.norm(merged.embedding) == pytest.approx(1.0)
    assert merged.label_idx == 3
    for field in SEMANTIC_FIELDS[1:]:
        assert getattr(merged, field) is None
    for field in ("vocab_log_partition", "negative_log_partition", "c_mem", "u_mem"):
        setattr(left, field, 0.5)
    merged = left + right
    for field in ("vocab_log_partition", "negative_log_partition", "c_mem", "u_mem"):
        assert getattr(merged, field) is None
