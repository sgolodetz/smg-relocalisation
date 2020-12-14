import cv2
import numpy as np

from typing import Optional


class SparsePnPBackend:
    """Can be used to estimate the camera-to-world transform from sparse point correspondences using PnP."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def estimate_pose(correspondences: [np.ndarray], camera_matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to estimate the camera-to-world transform from a set of >= 4 2D-to-3D point correspondences.

        :param correspondences: The point correspondences.
        :param camera_matrix:   The camera intrinsics matrix.
        :return:                The estimated transform, if >= 4 correspondences were provided, or None otherwise.
        """
        # If fewer than four point correspondences were provided, PnP cannot be used, so early out.
        if len(correspondences) < 4:
            return None

        # Construct the inputs to PnP from the point correspondences.
        object_points: np.ndarray = np.zeros((len(correspondences), 3))
        image_points: np.ndarray = np.zeros((len(correspondences), 2))
        for i in range(len(correspondences)):
            pos2d, pos3d = correspondences[i][:2], correspondences[i][2:]
            for j in range(3):
                object_points[i][j] = pos3d[j]
            for j in range(2):
                image_points[i][j] = pos2d[j]

        # print(object_points)
        # print(image_points)

        # Perform PnP to estimate the world-to-camera transform.
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distCoeffs=None)
        # ret, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, distCoeffs=None)

        # Convert the transform into a 4x4 matrix, then invert and return it.
        mat: np.ndarray = np.eye(4, 4)
        mat[0:3, 0:3], _ = cv2.Rodrigues(rvec)
        mat[0:3, 3:] = tvec
        return np.linalg.inv(mat)
