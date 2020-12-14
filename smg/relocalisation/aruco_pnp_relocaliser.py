import numpy as np

from smg.relocalisation.backends.sparse_pnp_backend import SparsePnPBackend
from smg.relocalisation.frontends.aruco_correspondence_generator import ArUcoCorrespondenceGenerator


class ArUcoPnPRelocaliser:
    """Can be used to relocalise a camera by detecting ArUco-based correspondences and passing them to PnP."""

    # CONSTRUCTORS

    def __init__(self, fiducials_filename: str):
        """
        Construct an ArUco-PnP relocaliser.

        :param fiducials_filename:  The name of the file containing the fiducials.
        """
        self.__correspondence_generator = ArUcoCorrespondenceGenerator(fiducials_filename)

    # PUBLIC METHODS

    def relocalise(self, colour_image: np.ndarray, camera_matrix: np.ndarray, *,
                   draw_detections: bool = False, print_correspondences: bool = False) -> np.ndarray:
        """
        Try to relocalise the camera from a colour image of the scene being viewed.

        :param colour_image:            The colour image of the scene being viewed.
        :param camera_matrix:           The intrinsics matrix for the colour camera.
        :param draw_detections:         Whether or not to draw the ArUco markers after detecting them (for debugging).
        :param print_correspondences:   Whether or not to print the generated correspondences (for debugging).
        :return:                        The estimated camera-to-world transform, if possible, or None otherwise.
        """
        correspondences = self.__correspondence_generator.generate_correspondences(
            colour_image, draw_detections=draw_detections, print_correspondences=print_correspondences
        )
        return SparsePnPBackend.estimate_pose(correspondences, camera_matrix)
