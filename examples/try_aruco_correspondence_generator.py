import numpy as np

from typing import List

from smg.relocalisation.frontends.aruco_correspondence_generator import ArUcoCorrespondenceGenerator
from smg.rotory.drones.tello import Tello


def main():
    np.set_printoptions(suppress=True)

    with Tello(print_commands=False, print_responses=False, print_state_messages=False) as drone:
        # img_filename: str = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "test.png")
        # img: np.ndarray = cv2.imread(img_filename)
        while True:
            img: np.ndarray = drone.get_image()
            height: float = 1.0
            offset: float = 0.1
            corr_gen: ArUcoCorrespondenceGenerator = ArUcoCorrespondenceGenerator({
                "0_0": np.array([-offset, -(height + offset), 0]),
                "0_1": np.array([offset, -(height + offset), 0]),
                "0_2": np.array([offset, -(height - offset), 0]),
                "0_3": np.array([-offset, -(height - offset), 0])
            })
            correspondences: List[np.ndarray] = corr_gen.generate_correspondences(
                img, draw_detections=True, print_correspondences=True
            )
            # cv2.waitKey()
            print(correspondences)


if __name__ == "__main__":
    main()
