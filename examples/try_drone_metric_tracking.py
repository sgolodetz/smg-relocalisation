import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from threading import Event
from typing import Dict, Optional

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

    def iterate(self, tracker_c_t_i: Optional[np.ndarray], relocaliser_w_t_c: Optional[np.ndarray]) -> None:
        # TODO: Comment here.
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                if event.button == 0:
                    self.__drone.takeoff()
                    self.__takeoff_event.set()
            elif event.type == pygame.JOYBUTTONUP:
                # If Button 0 on the Futaba T6K is set to its "released" state, land.
                if event.button == 0:
                    self.__drone.land()
                    self.__landing_event.set()

        # TODO: Comment here.
        throttle: float = self.__joystick.get_throttle()
        if self.__throttle_prev is not None:
            if throttle <= -0.5 < self.__throttle_prev:
                self.__throttle_down_event.set()
            if throttle >= 0.5 > self.__throttle_prev:
                self.__throttle_up_event.set()

        # Update the drone's movement based on the pitch, roll and yaw values output by the joystick.
        self.__drone.move_forward(self.__joystick.get_pitch())
        self.__drone.turn(self.__joystick.get_yaw())

        if self.__joystick.get_button(1) == 0:
            self.__drone.move_right(0)
            self.__drone.move_up(self.__joystick.get_roll())
        else:
            self.__drone.move_right(self.__joystick.get_roll())
            self.__drone.move_up(0)

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
        if tracker_i_t_c is not None and relocaliser_w_t_c is not None:
            self.__pose_globaliser.train(tracker_i_t_c, relocaliser_w_t_c)

        # If the user has told the drone to take off, complete the calibration process.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_CALIBRATED

        # If the user has throttled up, return to the previous calibration step.
        if self.__throttle_up_event:
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

    # Initialise pygame and its joystick module.
    pygame.init()
    pygame.joystick.init()

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
        "tello": dict(print_commands=True, print_responses=True, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        state_machine: DroneFSM = DroneFSM(drone, joystick)
        while state_machine.alive():
            state_machine.iterate(None, None)

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
