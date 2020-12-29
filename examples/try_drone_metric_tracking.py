import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from threading import Event
from typing import Dict, Optional, Tuple

from smg.opengl import OpenGLUtil
from smg.pyorbslam2 import MonocularTracker
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.relocalisation.poseglobalisers import MonocularPoseGlobaliser
from smg.rotory import DroneFactory
from smg.rotory.drones import Drone
from smg.rotory.joysticks import FutabaT6K


class EDroneCalibrationState(int):
    pass


DCS_UNCALIBRATED: EDroneCalibrationState = 0
DCS_STARTING_TRAINING: EDroneCalibrationState = 1
DCS_LANDING_TO_TRAIN: EDroneCalibrationState = 2
DCS_TRAINING: EDroneCalibrationState = 3
DCS_CALIBRATED: EDroneCalibrationState = 4


class DroneFSM:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, drone: Drone, joystick: FutabaT6K):
        self.__calibration_state: EDroneCalibrationState = DCS_UNCALIBRATED
        self.__drone: Drone = drone
        self.__joystick: FutabaT6K = joystick
        self.__landing_event: Event = Event()
        self.__pose_globaliser: MonocularPoseGlobaliser = MonocularPoseGlobaliser()
        self.__takeoff_event: Event = Event()
        self.__throttle_down_event: Event = Event()
        self.__throttle_prev: Optional[float] = None
        self.__throttle_up_event: Event = Event()
        self.__should_terminate: bool = False

    # PUBLIC METHODS

    def alive(self) -> bool:
        return not self.__should_terminate

    def get_calibration_state(self) -> EDroneCalibrationState:
        return self.__calibration_state

    def iterate(self, tracker_c_t_i: Optional[np.ndarray], relocaliser_w_t_c: Optional[np.ndarray],
                takeoff_requested: bool, landing_requested: bool) -> None:
        # TODO: Comment here.
        if takeoff_requested:
            # self.__drone.takeoff()
            self.__takeoff_event.set()
        elif landing_requested:
            # self.__drone.land()
            self.__landing_event.set()

        # TODO: Comment here.
        throttle: float = self.__joystick.get_throttle()
        if self.__throttle_prev is not None:
            if throttle <= 0.25 < self.__throttle_prev:
                self.__throttle_down_event.set()
            if throttle >= 0.75 > self.__throttle_prev:
                self.__throttle_up_event.set()

        # Update the drone's movement based on the pitch, roll and yaw values output by the joystick.
        # self.__drone.move_forward(self.__joystick.get_pitch())
        # self.__drone.turn(self.__joystick.get_yaw())
        #
        # if self.__joystick.get_button(1) == 0:
        #     self.__drone.move_right(0)
        #     self.__drone.move_up(self.__joystick.get_roll())
        # else:
        #     self.__drone.move_right(self.__joystick.get_roll())
        #     self.__drone.move_up(0)

        # TODO: Comment here.
        tracker_i_t_c: Optional[np.ndarray] = np.linalg.inv(tracker_c_t_i) if tracker_c_t_i is not None else None

        # TODO: Comment here.
        if self.__calibration_state == DCS_UNCALIBRATED:
            self.__iterate_uncalibrated()
        elif self.__calibration_state == DCS_STARTING_TRAINING:
            self.__iterate_starting_training(tracker_i_t_c, relocaliser_w_t_c)
        elif self.__calibration_state == DCS_LANDING_TO_TRAIN:
            self.__iterate_landing_to_train()
        elif self.__calibration_state == DCS_TRAINING:
            self.__iterate_training(tracker_i_t_c, relocaliser_w_t_c)
        elif self.__calibration_state == DCS_CALIBRATED:
            self.__iterate_calibrated(tracker_i_t_c, relocaliser_w_t_c)

        # TODO: Comment here.
        self.__throttle_prev = throttle

        # TODO: Comment here.
        self.__landing_event.clear()
        self.__takeoff_event.clear()
        self.__throttle_down_event.clear()
        self.__throttle_up_event.clear()

    def terminate(self) -> None:
        self.__should_terminate = True

    # PRIVATE METHODS

    def __iterate_calibrated(self, tracker_i_t_c: Optional[np.ndarray], relocaliser_w_t_c: Optional[np.ndarray]) \
            -> None:
        """
        TODO

        .. note::
            TODO: Throttle starts down (no fixed height), can then be down or up (fixed height)

        :param tracker_i_t_c:       TODO
        :param relocaliser_w_t_c:   TODO
        """
        # TODO
        if self.__throttle_down_event.is_set():
            # TODO
            pass

        # TODO
        if tracker_i_t_c is not None:
            # TODO
            tracker_w_t_c:  np.ndarray = self.__pose_globaliser.apply(tracker_i_t_c)

            # TODO
            if self.__throttle_up_event.is_set():
                self.__pose_globaliser.set_fixed_height(tracker_w_t_c)

    def __iterate_landing_to_train(self) \
            -> None:
        """
        TODO

        .. note::
            TODO: Throttle is up; either landing or on ground; takeoff -> C1; throttle down -> C3
        """
        # If the user has told the drone to take off, return to the previous calibration step.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_STARTING_TRAINING

        # If the user has throttled down, move on to the next calibration step.
        if self.__throttle_down_event.is_set():
            self.__calibration_state = DCS_TRAINING

    def __iterate_training(self, tracker_i_t_c: Optional[np.ndarray], relocaliser_w_t_c: Optional[np.ndarray]) \
            -> None:
        """
        TODO

        .. note::
            TODO: Throttle is down; on ground; takeoff -> C4; throttle up -> C2

        :param tracker_i_t_c:       TODO
        :param relocaliser_w_t_c:   TODO
        """
        # Train the pose globaliser if possible.
        # FIXME: The relocaliser pose won't be available at this point - we should make it artificially from the
        #        last known relocaliser pose and the fact that we're now on the ground.
        if tracker_i_t_c is not None and relocaliser_w_t_c is not None:
            self.__pose_globaliser.train(tracker_i_t_c, relocaliser_w_t_c)

        # If the user has told the drone to take off, complete the calibration process.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_CALIBRATED

        # If the user has throttled up, return to the previous calibration step.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_LANDING_TO_TRAIN

    def __iterate_starting_training(self, tracker_i_t_c: Optional[np.ndarray],
                                    relocaliser_w_t_c: Optional[np.ndarray]) -> None:
        """
        TODO

        .. note::
            TODO: Throttle is up; flying; land -> C2; throttle down -> U

        :param tracker_i_t_c:       TODO
        :param relocaliser_w_t_c:   TODO
        """
        # If the drone's successfully relocalised using the marker:
        if relocaliser_w_t_c is not None:
            # Start to train the pose globaliser. Note that this can safely be called repeatedly: this has the effect
            # of using the poses from the most recent call to define the reference space for the globaliser.
            self.__pose_globaliser.start_training(tracker_i_t_c, relocaliser_w_t_c)

        # If the user has told the drone to land and the globaliser's now training, move on to the next calibration
        # step. Otherwise, stay on this step, and wait for the user to take off and try again.
        if self.__landing_event.is_set() and self.__pose_globaliser.get_state() == MonocularPoseGlobaliser.TRAINING:
            self.__calibration_state = DCS_LANDING_TO_TRAIN

        # If the user has throttled down, stop the calibration process.
        if self.__throttle_down_event.is_set():
            self.__calibration_state = DCS_UNCALIBRATED

    def __iterate_uncalibrated(self) -> None:
        """
        TODO

        .. note::
            The drone can be doing anything at this point - no calibration happens until the user throttles up.
        """
        # If the user throttles up, start the calibration process.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_STARTING_TRAINING


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Set up a relocaliser that uses an ArUco marker of a known size and at a known height to relocalise.
    height: float = 1.5  # 1.5m (the height of the centre of the printed marker)
    offset: float = 0.0705  # 7.05cm (half the width of the printed marker)
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser({
        "0_0": np.array([-offset, -(height + offset), 0]),
        "0_1": np.array([offset, -(height + offset), 0]),
        "0_2": np.array([offset, -(height - offset), 0]),
        "0_3": np.array([-offset, -(height - offset), 0])
    })

    # Initialise pygame and its joystick module.
    pygame.init()
    pygame.joystick.init()

    # Make sure pygame always gets the user inputs.
    pygame.event.set_grab(True)

    # Try to determine the joystick index of the Futaba T6K. If no joystick is plugged in, early out.
    joystick_count: int = pygame.joystick.get_count()
    joystick_idx: int = 0
    if joystick_count == 0:
        exit(0)
    elif joystick_count != 1:
        # TODO: Prompt the user for the joystick to use.
        pass

    # Construct and calibrate the Futaba T6K.
    joystick: FutabaT6K = FutabaT6K(joystick_idx)
    joystick.calibrate()

    # Use the Futaba T6K to control a drone.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=True, print_control_messages=True, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        with MonocularTracker(
                settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
                voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # Create the window.
            window_size: Tuple[int, int] = (640, 480)
            pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)

            # # Set the projection matrix.
            # glMatrixMode(GL_PROJECTION)
            # intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
            # OpenGLUtil.set_projection_matrix(intrinsics, *window_size)
            #
            # # Enable the z-buffer.
            # glEnable(GL_DEPTH_TEST)
            # glDepthFunc(GL_LESS)

            # Construct the state machine for the drone.
            state_machine: DroneFSM = DroneFSM(drone, joystick)

            # TODO: Comment here.
            while state_machine.alive():
                # TODO: Comment here.
                takeoff_requested: bool = False
                landing_requested: bool = False

                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 0:
                            takeoff_requested = True
                    elif event.type == pygame.JOYBUTTONUP:
                        if event.button == 0:
                            landing_requested = True
                    elif event.type == pygame.QUIT:
                        state_machine.terminate()

                if not state_machine.alive():
                    continue

                # Get an image from the drone.
                image: np.ndarray = drone.get_image()

                # Try to estimate a transformation from initial camera space to current camera space
                # using the tracker.
                tracker_c_t_i: Optional[np.ndarray] = tracker.estimate_pose(image) if tracker.is_ready() else None

                # Try to estimate a transformation from current camera space to world space using the relocaliser.
                relocaliser_w_t_c: Optional[np.ndarray] = relocaliser.estimate_pose(image, drone.get_intrinsics())

                # TODO: Comment here.
                state_machine.iterate(tracker_c_t_i, relocaliser_w_t_c, takeoff_requested, landing_requested)

                # TODO: Comment here.
                cv2.imshow("Image", image)
                cv2.waitKey(1)

                # TODO: Comment here.
                pygame.display.set_caption(
                    f"Calibration State: {int(state_machine.get_calibration_state())}; "
                    f"Battery Level: {drone.get_battery_level()}"
                )
                pygame.display.flip()

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
