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
        np.array([0, 1]),
        np.array([[3.0, 4.0], [0.0, 0.0]]),
    )

    # Point 0 gets the unit vector; the zero feature of point 1 is not an
    # observation and is skipped entirely.
    np.testing.assert_allclose(sums, [[0.6, 0.8], [0.0, 0.0]])
    np.testing.assert_array_equal(counts.reshape(-1), [1, 0])


def test_accumulator_counts_one_observation_per_point_per_view():
    """Several pixels of one frame hitting the same point are one view.

    Counting each pixel separately would inflate the running mean with
    repeated copies of a single view, and the count would no longer match the
    number of embeddings the fusion averaged.
    """
    sums = np.zeros((1, 2), dtype=np.float64)
    counts = np.zeros(1, dtype=np.int64)

    accumulate_unit_embeddings(
        sums,
        counts,
        np.array([0, 0, 0]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
    )

    assert counts[0] == 1
    # HOV-SG's fusion (`sum_features[idx] += F_2D`) keeps the last write for a
    # repeated index, so the accumulator keeps the same one.
    np.testing.assert_allclose(sums, [[0.0, 1.0]])


def test_a_single_view_gives_perfect_consistency_regardless_of_pixel_count():
    sums = np.zeros((1, 2), dtype=np.float64)
    counts = np.zeros(1, dtype=np.int64)
    embedding = np.array([[0.6, 0.8]] * 5)

    accumulate_unit_embeddings(sums, counts, np.zeros(5, dtype=np.int64), embedding)
    p_view, u_view, _ = cross_view_values(sums[0], int(counts[0]))

    assert p_view == pytest.approx(1.0)
    assert u_view == pytest.approx(0.0)


def test_agreeing_views_give_one_and_opposing_views_give_zero():
    agree = np.zeros((1, 2)), np.zeros(1, dtype=np.int64)
    for _ in range(4):
        accumulate_unit_embeddings(
            agree[0], agree[1], np.array([0]), np.array([[1.0, 0.0]])
        )
    p_agree, _, _ = cross_view_values(agree[0][0], int(agree[1][0]))
    assert p_agree == pytest.approx(1.0)

    disagree = np.zeros((1, 2)), np.zeros(1, dtype=np.int64)
    for sign in (1.0, -1.0):
        accumulate_unit_embeddings(
            disagree[0], disagree[1], np.array([0]), np.array([[sign, 0.0]])
        )
    p_disagree, u_disagree, _ = cross_view_values(
        disagree[0][0], int(disagree[1][0])
    )
    assert p_disagree == pytest.approx(0.0)
    assert u_disagree == pytest.approx(1.0)


def test_orthogonal_views_give_the_norm_of_the_mean():
    """Two orthogonal unit views: ||m|| = ||(1,1)/2|| = sqrt(2)/2."""
    sums, counts = np.zeros((1, 2)), np.zeros(1, dtype=np.int64)
    for embedding in ([1.0, 0.0], [0.0, 1.0]):
        accumulate_unit_embeddings(sums, counts, np.array([0]), np.array([embedding]))

    p_view, _, _ = cross_view_values(sums[0], int(counts[0]))

    assert p_view == pytest.approx(np.sqrt(2.0) / 2.0)


def test_signal_uses_the_unnormalized_mean_not_the_fused_embedding():
    """Normalizing the accumulator first would always give exactly 1.0."""
    resultant_sum = np.array([1.0, 1.0])  # two orthogonal views
    p_view, _, _ = cross_view_values(resultant_sum, 2)

    fused = resultant_sum / np.linalg.norm(resultant_sum)
    assert np.linalg.norm(fused) == pytest.approx(1.0)
    assert p_view == pytest.approx(np.sqrt(2.0) / 2.0)
    assert p_view < 1.0


def test_low_support_flag_thresholds_views_per_point():
    # 10 point-view observations spread over 10 points is 1 view per point,
    # which is not two independent views of the object.
    _, _, sufficient = cross_view_values(
        np.ones(2), 10, min_observations=2, point_count=10
    )
    assert sufficient is False

    # The same total over 5 points is 2 views per point.
    _, _, sufficient = cross_view_values(
        np.ones(2), 10, min_observations=2, point_count=5
    )
    assert sufficient is True


def test_cross_view_edge_cases():
    p_view, u_view, sufficient = cross_view_values(
        np.array([1.0, 0.0]), 1, min_observations=2
    )
    assert p_view == pytest.approx(1.0)
    assert u_view == pytest.approx(0.0)
    assert sufficient is False

    assert cross_view_values(np.zeros(2), 0) == (None, None, False)
    assert cross_view_values(np.array([np.nan, 0.0]), 2) == (None, None, False)

    with pytest.raises(ValueError, match="must not be negative"):
        cross_view_values(np.zeros(2), -1)


def test_values_stay_within_the_unit_interval_for_random_view_sets():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n_views = int(rng.integers(1, 12))
        sums, counts = np.zeros((1, 4)), np.zeros(1, dtype=np.int64)
        for _ in range(n_views):
            accumulate_unit_embeddings(
                sums, counts, np.array([0]), rng.normal(size=(1, 4))
            )
        p_view, u_view, _ = cross_view_values(sums[0], int(counts[0]))

        assert -1e-12 <= p_view <= 1.0 + 1e-12
        assert p_view + u_view == pytest.approx(1.0)


def test_object_cross_view_metadata_round_trip_and_merge(tmp_path):
    left = Object("0_0_0", "0_0", name="chair")
    left.pcd = make_pcd([[0.0, 0.0, 0.0]])
    left.vertices = np.zeros((1, 3))
    left.embedding = np.array([1.0, 0.0])
    left.cross_view_resultant_sum = np.array([2.0, 0.0])
    left.cross_view_count = 2
    left.p_view, left.u_view, left.cross_view_sufficient = cross_view_values(
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
    assert merged.p_view == pytest.approx(np.sqrt(2.0) / 2.0)
    assert merged.u_view == pytest.approx(1.0 - merged.p_view)
