"""Coordinate-frame invariants shared by HM3D-Sem GT and HOV-SG."""

import numpy as np

from hovsg.data.hm3dsem.create_hm3dsem_walks_gt import read_camera_pose_hmp3d
from hovsg.dataloader.hm3dsem import HM3DSemDataset


def test_gt_and_graph_read_saved_habitat_poses_in_the_same_frame(tmp_path):
    """Both RGB-D reconstruction paths must apply the same camera-basis flip."""
    source_pose = np.array(
        [
            [0.0, 0.0, -1.0, 3.0],
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose_path = tmp_path / "pose.txt"
    pose_path.write_text(" ".join(map(str, source_pose.ravel())))

    expected = source_pose @ np.diag([1.0, -1.0, -1.0, 1.0])
    gt_pose = read_camera_pose_hmp3d(pose_path)
    graph_pose = HM3DSemDataset._load_pose(object(), pose_path)

    assert np.array_equal(gt_pose, expected)
    assert np.array_equal(graph_pose, expected)
