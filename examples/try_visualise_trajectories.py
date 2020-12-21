import numpy as np
import open3d as o3d

from typing import List, Tuple

from smg.open3d.visualisation_util import VisualisationUtil
from smg.utility.trajectory_util import TrajectoryUtil


def main():
    # TODO: I ultimately want to make this visualiser a lot more general and put it somewhere more central.

    # Load in the trajectories saved by the monocular pose globaliser example.
    relocaliser_trajectory: List[Tuple[float, np.ndarray]] = TrajectoryUtil.load_tum_trajectory(
        "trajectory-relocaliser.txt"
    )
    tracker_trajectory: List[Tuple[float, np.ndarray]] = TrajectoryUtil.load_tum_trajectory(
        "trajectory-tracker.txt"
    )
    unscaled_tracker_trajectory: List[Tuple[float, np.ndarray]] = TrajectoryUtil.load_tum_trajectory(
        "trajectory-tracker-unscaled.txt"
    )

    # Smooth the trajectories using Laplacian smoothing to make the visualisation look a bit nicer.
    relocaliser_trajectory = TrajectoryUtil.smooth_trajectory(relocaliser_trajectory)
    tracker_trajectory = TrajectoryUtil.smooth_trajectory(tracker_trajectory)
    unscaled_tracker_trajectory = TrajectoryUtil.smooth_trajectory(unscaled_tracker_trajectory)

    # Create the Open3D geometries for the visualisation.
    grid: o3d.geometry.LineSet = VisualisationUtil.make_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1])
    relocaliser_segments: o3d.geometry.LineSet = VisualisationUtil.make_trajectory_segments(
        relocaliser_trajectory, colour=(0.0, 1.0, 0.0)
    )
    tracker_segments: o3d.geometry.LineSet = VisualisationUtil.make_trajectory_segments(
        tracker_trajectory, colour=(0.0, 0.0, 1.0)
    )
    unscaled_tracker_segments: o3d.geometry.LineSet = VisualisationUtil.make_trajectory_segments(
        unscaled_tracker_trajectory, colour=(1.0, 0.0, 0.0)
    )

    # Visualise the geometries.
    VisualisationUtil.visualise_geometries([grid, relocaliser_segments, tracker_segments, unscaled_tracker_segments])


if __name__ == "__main__":
    main()
