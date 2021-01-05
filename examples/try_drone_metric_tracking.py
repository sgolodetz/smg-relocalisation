import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from threading import Event
from typing import Dict, Optional, Tuple

from smg.opengl import OpenGLRenderer, OpenGLUtil
from smg.pyorbslam2 import MonocularTracker
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.relocalisation.poseglobalisers import MonocularPoseGlobaliser
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter, CameraRenderer
from smg.rotory import DroneFactory
from smg.rotory.drones import Drone
from smg.rotory.joysticks import FutabaT6K
from smg.utility import ImageUtil


class EDroneCalibrationState(int):
    pass


DCS_UNCALIBRATED: EDroneCalibrationState = 0
DCS_SETTING_REFERENCE: EDroneCalibrationState = 1
DCS_PREPARING_TO_TRAIN: EDroneCalibrationState = 2
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
        self.__pose_globaliser: MonocularPoseGlobaliser = MonocularPoseGlobaliser(debug=True)
        self.__relocaliser_w_t_c_for_training: Optional[np.ndarray] = None
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
        elif self.__calibration_state == DCS_SETTING_REFERENCE:
            self.__iterate_setting_reference(tracker_i_t_c, relocaliser_w_t_c)
        elif self.__calibration_state == DCS_PREPARING_TO_TRAIN:
            self.__iterate_preparing_to_train()
        elif self.__calibration_state == DCS_TRAINING:
            self.__iterate_training(tracker_i_t_c)
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

        :param tracker_i_t_c:   TODO
        """
        # TODO
        if self.__throttle_down_event.is_set():
            self.__pose_globaliser.clear_fixed_height()

        # TODO
        if tracker_i_t_c is not None:
            # TODO
            tracker_w_t_c:  np.ndarray = self.__pose_globaliser.apply(tracker_i_t_c)

            print("Tracker Pose:")
            print(tracker_w_t_c)

            if relocaliser_w_t_c is not None:
                print("Relocaliser Pose:")
                print(relocaliser_w_t_c)

            # TODO
            if self.__throttle_up_event.is_set():
                self.__pose_globaliser.set_fixed_height(tracker_w_t_c)

    def __iterate_preparing_to_train(self) \
            -> None:
        """
        TODO

        .. note::
            TODO: Throttle is up; either landing or on ground; takeoff -> C1; throttle down -> C3
        """
        # If the user has told the drone to take off, return to the previous calibration step.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_SETTING_REFERENCE

        # If the user has throttled down, move on to the next calibration step.
        if self.__throttle_down_event.is_set():
            self.__calibration_state = DCS_TRAINING

    def __iterate_setting_reference(self, tracker_i_t_c: Optional[np.ndarray],
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
            # Set the pose globaliser's reference space to get it ready for training. Note that this can safely be
            # called repeatedly (the poses from the most recent call will be used to define the reference space).
            self.__pose_globaliser.set_reference_space(tracker_i_t_c, relocaliser_w_t_c)

            # If the user has told the drone to land, move on to the next calibration step. Otherwise, stay on this
            # step, and wait for the user to take off and try again.
            if self.__landing_event.is_set():
                self.__calibration_state = DCS_PREPARING_TO_TRAIN

                # It's unlikely that we'll be able to see the ArUco marker to relocalise once we're on the ground,
                # so estimate the relocaliser pose we'll have at that point by using the pose currently output by
                # the relocaliser and the fact that we'll be on the ground (i.e. y = 0) then.
                self.__relocaliser_w_t_c_for_training = relocaliser_w_t_c.copy()
                self.__relocaliser_w_t_c_for_training[1, 3] = 0.0

        # If the user has throttled down, stop the calibration process.
        if self.__throttle_down_event.is_set():
            self.__calibration_state = DCS_UNCALIBRATED

    def __iterate_training(self, tracker_i_t_c: Optional[np.ndarray]) -> None:
        """
        TODO

        .. note::
            TODO: Throttle is down; on ground; takeoff -> C4; throttle up -> C2

        :param tracker_i_t_c:   TODO
        """
        # Train the pose globaliser if possible.
        if tracker_i_t_c is not None and self.__relocaliser_w_t_c_for_training is not None:
            self.__pose_globaliser.train(tracker_i_t_c, self.__relocaliser_w_t_c_for_training)

        # If the user has told the drone to take off, complete the calibration process.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_CALIBRATED

        # If the user has throttled up, return to the previous calibration step.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_PREPARING_TO_TRAIN

    def __iterate_uncalibrated(self) -> None:
        """
        TODO

        .. note::
            The drone can be doing anything at this point - no calibration happens until the user throttles up.
        """
        # If the user throttles up, start the calibration process.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_SETTING_REFERENCE


def render_window(*, drone_image: np.ndarray, renderer: OpenGLRenderer, window_size: Tuple[int, int]) -> None:
    # Clear the window.
    OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), window_size)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the drone image.
    OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), window_size)
    renderer.render_image(ImageUtil.flip_channels(drone_image))

    # TODO: Render the metric trajectory of the drone in 3D.
    OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), window_size)

    glDepthFunc(GL_LEQUAL)
    # glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    OpenGLUtil.set_projection_matrix((500.0, 500.0, 320.0, 240.0), 640, 480)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    cam: SimpleCamera = SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0])
    other_cam: SimpleCamera = SimpleCamera([1, 0, 5], [0, 0, 1], [0, -1, 0])
    glLoadMatrixf(CameraPoseConverter.pose_to_modelview(CameraPoseConverter.camera_to_pose(cam)).flatten(order='F'))
    CameraRenderer.render_camera(other_cam, body_colour=(1.0, 1.0, 0.0), body_scale=0.1)

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glDisable(GL_DEPTH_TEST)

    # Swap the buffers.
    pygame.display.flip()


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

    # Construct the drone.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        # Create the window.
        window_size: Tuple[int, int] = (1280, 480)
        pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)

        # Construct the renderer.
        with OpenGLRenderer() as renderer:
            # Construct the tracker.
            with MonocularTracker(
                settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
                voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
            ) as tracker:
                # Construct and calibrate the Futaba T6K.
                joystick: FutabaT6K = FutabaT6K(joystick_idx)
                joystick.calibrate()

                # Construct the state machine for the drone.
                state_machine: DroneFSM = DroneFSM(drone, joystick)

                # While the state machine is still running:
                while state_machine.alive():
                    # Process any pygame events.
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

                    # If the user closed the application and the state machine terminated, early out.
                    if not state_machine.alive():
                        continue

                    # Get an image from the drone.
                    image: np.ndarray = drone.get_image()

                    # Try to estimate a transformation from initial camera space to current camera space
                    # using the tracker.
                    tracker_c_t_i: Optional[np.ndarray] = tracker.estimate_pose(image) if tracker.is_ready() else None

                    # Try to estimate a transformation from current camera space to world space using the relocaliser.
                    relocaliser_w_t_c: Optional[np.ndarray] = relocaliser.estimate_pose(image, drone.get_intrinsics())

                    # Run an iteration of the state machine.
                    state_machine.iterate(tracker_c_t_i, relocaliser_w_t_c, takeoff_requested, landing_requested)

                    # Update the caption of the window to reflect the current state.
                    pygame.display.set_caption(
                        f"Calibration State: {int(state_machine.get_calibration_state())}; "
                        f"Battery Level: {drone.get_battery_level()}"
                    )

                    # Render the contents of the window.
                    render_window(
                        drone_image=image,
                        renderer=renderer,
                        window_size=window_size
                    )

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
