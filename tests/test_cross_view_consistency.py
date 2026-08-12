"""Cross-view Consistency, eq. (crossview).

    ||m_i||^2 = 1/n + (1 - 1/n) * c_bar_i    (resultant-length identity)
    P^view_i = (1 + c_bar_i) / 2,   U^view_i = 1 - P^view_i

with m_i the unweighted running mean of the n per-view unit embeddings and
c_bar_i their mean pairwise cosine similarity. A single view leaves c_bar_i
undefined and P^view_i is 1.
"""

import itertools
import json

import numpy as np
import open3d as o3d
import pytest

from hovsg.graph.object import Object
from hovsg.utils.cross_view_consistency import (
    accumulate_unit_embeddings,
    compute_cross_view_consistency,
    mean_pairwise_cosine_similarity,
)


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def direct_mean_pairwise_cosine(unit_views):
    """c_bar_i evaluated straight from its definition, without the identity."""
    pairs = list(itertools.combinations(range(len(unit_views)), 2))
    return float(
        np.mean([np.dot(unit_views[i], unit_views[j]) for i, j in pairs])
    )


def accumulate(views):
    """Accumulate a list of per-view embeddings for a single point."""
    dim = len(views[0])
    sums, counts = np.zeros((1, dim)), np.zeros(1, dtype=np.int64)
    for view in views:
        accumulate_unit_embeddings(
            sums, counts, np.array([0]), np.asarray([view], dtype=np.float64)
        )
    return sums[0], int(counts[0])


# ---------------------------------------------------------------------------
# Accumulation of the running mean
# ---------------------------------------------------------------------------


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
    repeated copies of a single view, and the count would no longer be the
    number n of embeddings the fusion averaged.
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


# ---------------------------------------------------------------------------
# Recovering c_bar from the resultant-length identity
# ---------------------------------------------------------------------------


def test_mean_pairwise_cosine_is_undefined_for_fewer_than_two_views():
    assert mean_pairwise_cosine_similarity(1.0, 1) is None
    assert mean_pairwise_cosine_similarity(0.0, 0) is None


def test_recovered_mean_cosine_matches_its_direct_definition():
    """The identity must reproduce the mean of the explicit pairwise cosines."""
    rng = np.random.default_rng(7)
    for n_views in range(2, 9):
        views = rng.normal(size=(n_views, 5))
        unit_views = views / np.linalg.norm(views, axis=1, keepdims=True)
        resultant_norm = float(np.linalg.norm(unit_views.mean(axis=0)))

        recovered = mean_pairwise_cosine_similarity(resultant_norm, n_views)

        assert recovered == pytest.approx(
            direct_mean_pairwise_cosine(unit_views), abs=1e-12
        )


# ---------------------------------------------------------------------------
# P^view on worked examples
# ---------------------------------------------------------------------------


def test_single_view_is_defined_as_perfect_consistency():
    resultant_sum, count = accumulate([[0.6, 0.8]])

    p_view, u_view = compute_cross_view_consistency(resultant_sum, count)

    assert count == 1
    assert p_view == pytest.approx(1.0)
    assert u_view == pytest.approx(0.0)


def test_identical_views_give_one_and_opposite_views_give_zero():
    # Four identical views: c_bar = 1, P = 1.
    agree_sum, agree_count = accumulate([[1.0, 0.0]] * 4)
    p_agree, u_agree = compute_cross_view_consistency(agree_sum, agree_count)
    assert p_agree == pytest.approx(1.0)
    assert u_agree == pytest.approx(0.0)

    # Two opposite views: ||m|| = 0, c_bar = (0 - 1/2) / (1/2) = -1, P = 0.
    disagree_sum, disagree_count = accumulate([[1.0, 0.0], [-1.0, 0.0]])
    p_disagree, u_disagree = compute_cross_view_consistency(
        disagree_sum, disagree_count
    )
    assert p_disagree == pytest.approx(0.0)
    assert u_disagree == pytest.approx(1.0)


def test_two_orthogonal_views_give_one_half():
    """c_bar = 0 for orthogonal views, so P = (1 + 0) / 2, not ||m|| = 0.707."""
    resultant_sum, count = accumulate([[1.0, 0.0], [0.0, 1.0]])

    p_view, _ = compute_cross_view_consistency(resultant_sum, count)

    assert float(np.linalg.norm(resultant_sum / count)) == pytest.approx(
        np.sqrt(2.0) / 2.0
    )
    assert p_view == pytest.approx(0.5)


def test_three_view_worked_example():
    """Two agreeing views and one orthogonal one.

    Pairwise cosines are 1, 0 and 0, so c_bar = 1/3 and P = 2/3. The identity
    gives the same: ||m||^2 = 5/9, c_bar = (5/9 - 1/3) / (1 - 1/3) = 1/3.
    """
    resultant_sum, count = accumulate([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    p_view, u_view = compute_cross_view_consistency(resultant_sum, count)

    np.testing.assert_allclose(resultant_sum, [2.0, 1.0])
    assert p_view == pytest.approx(2.0 / 3.0)
    assert u_view == pytest.approx(1.0 / 3.0)


def test_signal_does_not_depend_on_the_number_of_views():
    """Same agreement, different n: identical P^view, different ||m_i||."""
    # Two views at cos = 1/3.
    two_views = accumulate([[1.0, 0.0], [1.0 / 3.0, np.sqrt(8.0) / 3.0]])
    # Three views whose pairwise cosines are 1, 0, 0: mean 1/3 as well.
    three_views = accumulate([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    p_two, _ = compute_cross_view_consistency(*two_views)
    p_three, _ = compute_cross_view_consistency(*three_views)

    assert p_two == pytest.approx(2.0 / 3.0)
    assert p_three == pytest.approx(2.0 / 3.0)
    # The raw resultant length -- the quantity the paper rejects -- does differ.
    norm_two = np.linalg.norm(two_views[0] / two_views[1])
    norm_three = np.linalg.norm(three_views[0] / three_views[1])
    assert norm_two != pytest.approx(norm_three)


def test_later_views_update_the_running_state_incrementally():
    """A running mean, not a batch recomputation: adding a view updates it."""
    sums, counts = np.zeros((1, 2)), np.zeros(1, dtype=np.int64)

    accumulate_unit_embeddings(sums, counts, np.array([0]), np.array([[1.0, 0.0]]))
    after_one, _ = compute_cross_view_consistency(sums[0], int(counts[0]))
    assert after_one == pytest.approx(1.0)

    accumulate_unit_embeddings(sums, counts, np.array([0]), np.array([[0.0, 1.0]]))
    after_two, _ = compute_cross_view_consistency(sums[0], int(counts[0]))
    assert after_two == pytest.approx(0.5)  # c_bar = 0

    # A third, agreeing view pulls consistency back up to c_bar = 1/3.
    accumulate_unit_embeddings(sums, counts, np.array([0]), np.array([[1.0, 0.0]]))
    after_three, _ = compute_cross_view_consistency(sums[0], int(counts[0]))
    assert after_three == pytest.approx(2.0 / 3.0)


def test_cross_view_edge_cases():
    assert compute_cross_view_consistency(np.zeros(2), 0) == (None, None)
    assert compute_cross_view_consistency(np.array([np.nan, 0.0]), 2) == (None, None)

    with pytest.raises(ValueError, match="must not be negative"):
        compute_cross_view_consistency(np.zeros(2), -1)


def test_values_stay_within_the_unit_interval_for_random_view_sets():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n_views = int(rng.integers(1, 12))
        sums, counts = np.zeros((1, 4)), np.zeros(1, dtype=np.int64)
        for _ in range(n_views):
            accumulate_unit_embeddings(
                sums, counts, np.array([0]), rng.normal(size=(1, 4))
            )
        p_view, u_view = compute_cross_view_consistency(sums[0], int(counts[0]))

        assert 0.0 <= p_view <= 1.0
        assert p_view + u_view == pytest.approx(1.0)


def test_object_cross_view_metadata_round_trip(tmp_path):
    obj = Object("0_0_0", "0_0", name="chair")
    obj.pcd = make_pcd([[0.0, 0.0, 0.0]])
    obj.vertices = np.zeros((1, 3))
    obj.embedding = np.array([1.0, 0.0])
    obj.cross_view_resultant_sum = np.array([2.0, 0.0])
    obj.cross_view_count = 2
    obj.p_view, obj.u_view = compute_cross_view_consistency(
        obj.cross_view_resultant_sum, obj.cross_view_count
    )
    obj.save(tmp_path)

    saved = json.loads((tmp_path / "0_0_0.json").read_text(encoding="utf-8"))
    assert saved["cross_view_resultant_sum"] == [2.0, 0.0]
    assert saved["p_view"] == pytest.approx(1.0)

    # The raw evidence round-trips, so P^view stays recomputable after a load.
    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))
    np.testing.assert_allclose(restored.cross_view_resultant_sum, [2.0, 0.0])
    assert restored.cross_view_count == 2
    assert compute_cross_view_consistency(
        restored.cross_view_resultant_sum, restored.cross_view_count
    ) == (pytest.approx(obj.p_view), pytest.approx(obj.u_view))
