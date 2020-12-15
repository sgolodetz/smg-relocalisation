import numpy as np

from smg.relocalisation import ArUcoPnPRelocaliser
from smg.rotory.drones import Tello


def main():
    np.set_printoptions(suppress=True)

    # Set up a relocaliser that uses an ArUco marker of a known size and at a known height to relocalise.
    height: float = 1.5  # 1.5m (the height of the centre of the printed marker)
    offset: float = 0.0705  # 7.05cm (half the width of the printed marker)
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser({
        "0_0": np.array([-offset, -(height + offset), 0]),
        "0_1": np.array([offset, -(height + offset), 0]),
        "0_2": np.array([offset, -(height - offset), 0]),
        "0_3": np.array([-offset, -(height - offset), 0])
    })

    # Construct a drone, and repeatedly print out the pose of its camera as estimated by the relocaliser.
    with Tello(print_commands=False, print_responses=False, print_state_messages=False) as drone:
        while True:
            print(relocaliser.estimate_pose(
                drone.get_image(), drone.get_intrinsics(), draw_detections=True, print_correspondences=False)
            )


if __name__ == "__main__":
    main()
