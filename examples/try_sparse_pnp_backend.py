import numpy as np

from typing import List

from smg.relocalisation.backends.sparse_pnp_backend import SparsePnPBackend
from smg.relocalisation.frontends.aruco_correspondence_generator import ArUcoCorrespondenceGenerator
from smg.rotory.drones.tello import Tello


def main():
    np.set_printoptions(suppress=True)

    with Tello(print_commands=False, print_responses=False, print_state_messages=False) as drone:
        while True:
            img: np.ndarray = drone.get_image()
            height: float = 1.5
            offset: float = 0.0705
            corr_gen: ArUcoCorrespondenceGenerator = ArUcoCorrespondenceGenerator({
                "0_0": np.array([-offset, -(height + offset), 0]),
                "0_1": np.array([offset, -(height + offset), 0]),
                "0_2": np.array([offset, -(height - offset), 0]),
                "0_3": np.array([-offset, -(height - offset), 0])
            })
            correspondences: List[np.ndarray] = corr_gen.generate_correspondences(
                img, draw_detections=True, print_correspondences=False
            )
            fx, fy, cx, cy = drone.get_intrinsics()
            camera_matrix: np.ndarray = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            print(SparsePnPBackend.estimate_pose(correspondences, camera_matrix))


if __name__ == "__main__":
    main()
