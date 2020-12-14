import numpy as np

from smg.relocalisation.aruco_pnp_relocaliser import ArUcoPnPRelocaliser
from smg.rotory.drones.tello import Tello


def main():
    np.set_printoptions(suppress=True)

    height: float = 1.5
    offset: float = 0.0705
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser({
        "0_0": np.array([-offset, -(height + offset), 0]),
        "0_1": np.array([offset, -(height + offset), 0]),
        "0_2": np.array([offset, -(height - offset), 0]),
        "0_3": np.array([-offset, -(height - offset), 0])
    })

    with Tello(print_commands=False, print_responses=False, print_state_messages=False) as drone:
        fx, fy, cx, cy = drone.get_intrinsics()
        camera_matrix: np.ndarray = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        while True:
            img: np.ndarray = drone.get_image()
            print(relocaliser.relocalise(img, camera_matrix, draw_detections=True, print_correspondences=False))


if __name__ == "__main__":
    main()
