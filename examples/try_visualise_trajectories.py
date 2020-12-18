import numpy as np

from open3d.cpu.pybind.geometry import Geometry, LineSet
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
    grid: LineSet = VisualisationUtil.make_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1])
    relocaliser_geoms: List[Geometry] = VisualisationUtil.make_geometries_for_trajectory(
        relocaliser_trajectory, (0.0, 1.0, 0.0)
    )
    tracker_geoms: List[Geometry] = VisualisationUtil.make_geometries_for_trajectory(
        tracker_trajectory, (0.0, 0.0, 1.0)
    )
    unscaled_tracker_geoms: List[Geometry] = VisualisationUtil.make_geometries_for_trajectory(
        unscaled_tracker_trajectory, (1.0, 0.0, 0.0)
    )

    # Visualise the geometries.
    VisualisationUtil.visualise_geometries(
        [grid] + relocaliser_geoms + tracker_geoms + unscaled_tracker_geoms
    )


if __name__ == "__main__":
    main()
