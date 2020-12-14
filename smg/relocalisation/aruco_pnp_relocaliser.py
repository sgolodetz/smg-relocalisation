import numpy as np

from typing import Dict, Tuple

from smg.relocalisation.backends.sparse_pnp_backend import SparsePnPBackend
from smg.relocalisation.frontends.aruco_correspondence_generator import ArUcoCorrespondenceGenerator


class ArUcoPnPRelocaliser:
    """Can be used to relocalise a camera by detecting ArUco-based correspondences and passing them to PnP."""

    # CONSTRUCTOR

    def __init__(self, fiducials: Dict[str, np.ndarray]):
        """
        Construct an ArUco-PnP relocaliser.

        :param fiducials:   The fiducials.
        """
        self.__correspondence_generator: ArUcoCorrespondenceGenerator = ArUcoCorrespondenceGenerator(fiducials)

    # PUBLIC METHODS

    def estimate_pose(self, colour_image: np.ndarray, intrinsics: Tuple[float, float, float, float], *,
                      draw_detections: bool = False, print_correspondences: bool = False) -> np.ndarray:
        """
        Try to estimate the pose of the camera from a colour image of the scene being viewed.

        :param colour_image:            The colour image of the scene being viewed.
        :param intrinsics:              The intrinsics for the colour camera.
        :param draw_detections:         Whether or not to draw the ArUco markers after detecting them (for debugging).
        :param print_correspondences:   Whether or not to print the generated correspondences (for debugging).
        :return:                        The estimated camera-to-world transform, if possible, or None otherwise.
        """
        # TODO
        fx, fy, cx, cy = intrinsics
        camera_matrix: np.ndarray = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # TODO
        correspondences = self.__correspondence_generator.generate_correspondences(
            colour_image, draw_detections=draw_detections, print_correspondences=print_correspondences
        )

        # TODO
        return SparsePnPBackend.estimate_pose(correspondences, camera_matrix)
