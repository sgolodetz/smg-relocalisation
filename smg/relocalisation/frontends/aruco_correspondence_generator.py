import cv2
import numpy as np

from typing import Dict, List


class ArUcoCorrespondenceGenerator:
    """Can be used to generate 2D-to-3D correspondences via detecting ArUco markers."""

    # CONSTRUCTOR

    def __init__(self, fiducials: Dict[str, np.ndarray]):
        """
        Construct an ArUco correspondence generator.

        :param fiducials:   The fiducials.
        """
        self.__fiducials: Dict[str, np.ndarray] = fiducials
        self.__marker_dict: cv2.aruco_Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # PUBLIC METHODS

    def generate_correspondences(self, img: np.ndarray, *, draw_detections: bool = False,
                                 print_correspondences: bool = False) -> List[np.ndarray]:
        """
        Generate 2D-to-3D correspondences via detecting ArUco markers in the specified image.

        :param img:                     The image in which to detect the markers.
        :param draw_detections:         Whether or not to draw the markers after detecting them (for debugging).
        :param print_correspondences:   Whether or not to print the generated correspondences (for debugging).
        :return:                        The generated correspondences.
        """
        correspondences: [np.ndarray] = []

        # Detect the markers in the image.
        parameters = cv2.aruco.DetectorParameters_create()
        corners, marker_ids, _ = cv2.aruco.detectMarkers(img, self.__marker_dict, parameters=parameters)

        # Draw the markers if requested.
        if draw_detections:
            img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, marker_ids) \
                if marker_ids is not None else img.copy()
            cv2.imshow("Markers", img_markers)
            cv2.waitKey(1)

        # If no markers were detected, early out.
        if marker_ids is None:
            return []

        # Construct the 2D-to-3D correspondences from the marker corners and their positions in 3D space (if known).

        # For each marker:
        for i in range(len(marker_ids)):
            marker_id = str(marker_ids[i][0])

            # For each corner:
            for j in range(4):
                # Determine the name that the corresponding fiducial would have.
                fiducial_id = marker_id + "_" + str(j)

                # Look to see if we know its position in 3D space.
                pos3d = self.__fiducials.get(fiducial_id, None)

                # If we don't, ignore this corner and carry on.
                if pos3d is None:
                    continue

                # If we do, add a 2D-to-3D correspondence for the corner.
                pos2d = corners[i][0][j]
                # correspondences.append((Vec2(pos2d[0], pos2d[1]), pos3d))
                correspondences.append(np.array([*pos2d, *pos3d]))

                # Print the correspondence if requested.
                if print_correspondences:
                    print(fiducial_id, pos2d, pos3d)

        return correspondences
