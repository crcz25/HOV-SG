import json

import numpy as np
import open3d as o3d
import pytest

from hovsg.graph.object import Object
from hovsg.utils.cross_view_consistency import (
    accumulate_unit_embeddings,
    cross_view_values,
)


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def test_accumulator_normalizes_and_skips_zero_vectors():
    sums = np.zeros((2, 2), dtype=np.float64)
    counts = np.zeros((2, 1), dtype=np.int64)

    accumulate_unit_embeddings(
        sums,
        counts,
        np.array([0, 0, 1]),
        np.array([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]]),
    )

    np.testing.assert_allclose(sums, [[0.6, 0.8], [0.0, -1.0]])
    np.testing.assert_array_equal(counts.reshape(-1), [1, 1])


def test_cross_view_resultant_length_and_low_support_flag():
    c_view, u_view, sufficient = cross_view_values(
        np.array([1.0, 1.0]), 2, min_observations=2
    )
    assert c_view == pytest.approx(np.sqrt(2.0) / 2.0)
    assert u_view == pytest.approx(1.0 - c_view)
    assert sufficient is True

    c_view, u_view, sufficient = cross_view_values(
        np.array([1.0, 0.0]), 1, min_observations=2
    )
    assert c_view == pytest.approx(1.0)
    assert u_view == pytest.approx(0.0)
    assert sufficient is False

    assert cross_view_values(np.zeros(2), 0) == (None, None, False)


def test_object_cross_view_metadata_round_trip_and_merge(tmp_path):
    left = Object("0_0_0", "0_0", name="chair")
    left.pcd = make_pcd([[0.0, 0.0, 0.0]])
    left.vertices = np.zeros((1, 3))
    left.embedding = np.array([1.0, 0.0])
    left.cross_view_resultant_sum = np.array([2.0, 0.0])
    left.cross_view_count = 2
    left.c_view, left.u_view, left.cross_view_sufficient = cross_view_values(
        left.cross_view_resultant_sum, left.cross_view_count
    )
    left.save(tmp_path)

    saved = json.loads((tmp_path / "0_0_0.json").read_text(encoding="utf-8"))
    assert saved["cross_view_resultant_sum"] == [2.0, 0.0]
    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))
    np.testing.assert_allclose(restored.cross_view_resultant_sum, [2.0, 0.0])
    assert restored.cross_view_count == 2

    right = Object("0_0_1", "0_0", name="chair")
    right.pcd = make_pcd([[1.0, 0.0, 0.0]])
    right.embedding = np.array([0.0, 1.0])
    right.cross_view_resultant_sum = np.array([0.0, 2.0])
    right.cross_view_count = 2

    merged = restored + right
    np.testing.assert_allclose(merged.cross_view_resultant_sum, [2.0, 2.0])
    assert merged.cross_view_count == 4
    assert merged.c_view == pytest.approx(np.sqrt(2.0) / 2.0)
    assert merged.u_view == pytest.approx(1.0 - merged.c_view)
