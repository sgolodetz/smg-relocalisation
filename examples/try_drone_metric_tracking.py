import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from threading import Event
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from smg.opengl import OpenGLImageRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.pyorbslam2 import MonocularTracker
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.relocalisation.poseglobalisers import MonocularPoseGlobaliser
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraRenderer
from smg.rotory import DroneFactory
from smg.rotory.drones import Drone
from smg.rotory.joysticks import FutabaT6K
from smg.utility import ImageUtil, TrajectoryUtil


class EDroneCalibrationState(int):
    """The different calibration states in which a drone can be."""
    pass


# Fly around as normal with non-metric tracking.
DCS_UNCALIBRATED: EDroneCalibrationState = 0
# Fly around in front of the marker to set the reference space.
DCS_SETTING_REFERENCE: EDroneCalibrationState = 1
# Land prior to training the globaliser to estimate the scale.
DCS_PREPARING_TO_TRAIN: EDroneCalibrationState = 2
# Whilst on the ground, train the globaliser to estimate the scale.
DCS_TRAINING: EDroneCalibrationState = 3
# Fly around as normal with metric tracking.
DCS_CALIBRATED: EDroneCalibrationState = 4


class DroneFSM:
    """A finite state machine for a drone."""

    # CONSTRUCTOR

    def __init__(self, drone: Drone, joystick: FutabaT6K):
        """
        Construct a finite state machine for a drone.

        :param drone:       The drone.
        :param joystick:    The joystick that will be used to control the drone's movement.
        """
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
        self.__tracker_w_t_c: Optional[np.ndarray] = None
        self.__should_terminate: bool = False

    # PUBLIC METHODS

    def alive(self) -> bool:
        """
        Get whether or not the state machine is still alive.

        :return:    True, if the state machine is still alive, or False otherwise.
        """
        return not self.__should_terminate

    def get_calibration_state(self) -> EDroneCalibrationState:
        """
        Get the calibration state of the drone.

        :return:    The calibration state of the drone.
        """
        return self.__calibration_state

    def get_tracker_w_t_c(self) -> Optional[np.ndarray]:
        """
        Try to get a metric transformation from current camera space to world space, as estimated by the tracker.

        .. note::
            This returns None iff either (i) the tracker failed, or (ii) the drone hasn't been calibrated yet.

        :return:    A metric transformation from current camera space to world space, as estimated by the tracker,
                    if available, or None otherwise.
        """
        return self.__tracker_w_t_c

    def iterate(self, tracker_c_t_i: Optional[np.ndarray], relocaliser_w_t_c: Optional[np.ndarray],
                takeoff_requested: bool, landing_requested: bool) -> None:
        """
        Run an iteration of the state machine.

        :param tracker_c_t_i:       A non-metric transformation from initial camera space to current camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
        :param takeoff_requested:   Whether or not the user has asked for the drone to take off.
        :param landing_requested:   Whether or not the user has asked for the drone to land.
        """
        # Process any take-off or landing requests, and set the corresponding events so that individual states
        # can respond to them later if desired.
        if takeoff_requested:
            # self.__drone.takeoff()
            self.__takeoff_event.set()
        elif landing_requested:
            # self.__drone.land()
            self.__landing_event.set()

        # Check for any throttle up/down events that have occurred so that individual states can respond to them later.
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

        # If the non-metric tracker pose is available, compute its inverse for later use.
        tracker_i_t_c: Optional[np.ndarray] = np.linalg.inv(tracker_c_t_i) if tracker_c_t_i is not None else None

        # Run an iteration of the current state.
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

        # Record the current setting of the throttle for later, so we can detect throttle up/down events that occur.
        self.__throttle_prev = throttle

        # Clear any events that have occurred during this iteration of the state machine.
        self.__landing_event.clear()
        self.__takeoff_event.clear()
        self.__throttle_down_event.clear()
        self.__throttle_up_event.clear()

    def terminate(self) -> None:
        """Tell the state machine to terminate."""
        self.__should_terminate = True

    # PRIVATE METHODS

    def __iterate_calibrated(self, tracker_i_t_c: Optional[np.ndarray],
                             relocaliser_w_t_c: Optional[np.ndarray]) -> None:
        """
        Run an iteration of the 'calibrated' state.

        .. note::
            The drone enters this state by taking off after training the globaliser. It then never leaves this state.
            On entering this state, the throttle will be down (as it was during the training of the globaliser).
            Moving the throttle up/down will then set/clear a fixed height.

        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
        """
        # If the user throttles down, clear the fixed height.
        if self.__throttle_down_event.is_set():
            self.__pose_globaliser.clear_fixed_height()

        # If the non-metric tracker pose is available:
        if tracker_i_t_c is not None:
            # Use the globaliser to obtain the metric tracker pose.
            self.__tracker_w_t_c = self.__pose_globaliser.apply(tracker_i_t_c)

            # If the user throttles up, set the current height as the fixed height. Note that it is theoretically
            # possible for the user to throttle up during a period of tracking failure. In that case, the throttle
            # will be up but no fixed height will have been set. However, if that happens, the user can simply
            # throttle down again with no ill effects (clearing a fixed height that hasn't been set is a no-op).
            if self.__throttle_up_event.is_set():
                self.__pose_globaliser.set_fixed_height(self.__tracker_w_t_c)
        else:
            # If the non-metric tracker pose isn't available, the metric tracker pose clearly can't be estimated.
            self.__tracker_w_t_c = None

        # Print the tracker pose.
        print("Tracker Pose:")
        print(self.__tracker_w_t_c)

        # If the relocaliser pose is available, also print that.
        if relocaliser_w_t_c is not None:
            print("Relocaliser Pose:")
            print(relocaliser_w_t_c)

    def __iterate_preparing_to_train(self) -> None:
        """
        Run an iteration of the 'preparing to train' state.

        .. note::
            The drone enters this state either by landing after setting the globaliser's reference space,
            or by throttling up after training the globaliser. It leaves this state either by throttling
            down to enter the training state, or by taking off to enter the setting reference state. On
            entering this state, the throttle will be up.
        .. note::
            In practice, this state exists to allow the drone to land prior to starting to train the
            globaliser. The training process should only be started once the drone is on the ground.
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
        Run an iteration of the 'setting reference' state.

        .. note::
            TODO: Throttle is up; flying; land -> C2; throttle down -> U

        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
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
        Run an iteration of the 'training' state.

        .. note::
            TODO: Throttle is down; on ground; takeoff -> C4; throttle up -> C2

        :param tracker_i_t_c:   A non-metric transformation from current camera space to initial camera space,
                                as estimated by the tracker.
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
        Run an iteration of the 'uncalibrated' state.

        .. note::
            TODO: The drone can be doing anything at this point - no calibration happens until the user throttles up.
        """
        # If the user throttles up, start the calibration process.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_SETTING_REFERENCE


def render_window(*, drone_image: np.ndarray, image_renderer: OpenGLImageRenderer,
                  relocaliser_trajectory: List[Tuple[float, np.ndarray]],
                  tracker_trajectory: List[Tuple[float, np.ndarray]],
                  viewing_pose: np.ndarray, window_size: Tuple[int, int]) -> None:
    """
    TODO

    :param drone_image:             TODO
    :param image_renderer:          TODO
    :param relocaliser_trajectory:  TODO
    :param tracker_trajectory:      TODO
    :param viewing_pose:            TODO
    :param window_size:             TODO
    """
    # Clear the window.
    OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), window_size)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the drone image.
    OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), window_size)
    image_renderer.render_image(ImageUtil.flip_channels(drone_image))

    # Render the drone's trajectories in 3D.
    OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), window_size)

    glDepthFunc(GL_LEQUAL)
    glEnable(GL_DEPTH_TEST)

    with OpenGLMatrixContext(
        GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix((500.0, 500.0, 320.0, 240.0), 640, 480)
    ):
        with OpenGLMatrixContext(
            GL_MODELVIEW, lambda: glLoadMatrixf(CameraPoseConverter.pose_to_modelview(viewing_pose).flatten(order='F'))
        ):
            glPushAttrib(GL_ENABLE_BIT)
            glColor3f(0.0, 0.0, 0.0)
            glLineStipple(1, 0x8888)
            glEnable(GL_LINE_STIPPLE)
            OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1])
            glPopAttrib()

            origin: SimpleCamera = SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0])
            CameraRenderer.render_camera(origin, body_colour=(1.0, 1.0, 0.0), body_scale=0.1)

            OpenGLUtil.render_trajectory(relocaliser_trajectory, colour=(0.0, 1.0, 0.0))
            OpenGLUtil.render_trajectory(tracker_trajectory, colour=(0.0, 0.0, 1.0))

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

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
        )

        # Construct the image renderer.
        with OpenGLImageRenderer() as image_renderer:
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

                # Initialise the timestamp and the drone's trajectories (used for visualisation).
                timestamp: float = 0.0
                relocaliser_trajectory: List[Tuple[float, np.ndarray]] = []
                tracker_trajectory: List[Tuple[float, np.ndarray]] = []

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
                        break

                    # Allow the user to control the camera.
                    camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                    # Get an image from the drone.
                    image: np.ndarray = drone.get_image()

                    # Try to estimate a transformation from initial camera space to current camera space
                    # using the tracker.
                    tracker_c_t_i: Optional[np.ndarray] = tracker.estimate_pose(image) if tracker.is_ready() else None

                    # Try to estimate a transformation from current camera space to world space using the relocaliser.
                    relocaliser_w_t_c: Optional[np.ndarray] = relocaliser.estimate_pose(image, drone.get_intrinsics())

                    # Run an iteration of the state machine.
                    state_machine.iterate(tracker_c_t_i, relocaliser_w_t_c, takeoff_requested, landing_requested)

                    # Update the drone's trajectories.
                    tracker_w_t_c: Optional[np.ndarray] = state_machine.get_tracker_w_t_c()
                    if tracker_w_t_c is not None:
                        tracker_trajectory.append((timestamp, tracker_w_t_c))
                        if relocaliser_w_t_c is not None:
                            relocaliser_trajectory.append((timestamp, relocaliser_w_t_c))

                    # Update the caption of the window to reflect the current state.
                    pygame.display.set_caption(
                        f"Calibration State: {int(state_machine.get_calibration_state())}; "
                        f"Battery Level: {drone.get_battery_level()}"
                    )

                    # Render the contents of the window.
                    render_window(
                        drone_image=image,
                        image_renderer=image_renderer,
                        relocaliser_trajectory=TrajectoryUtil.smooth_trajectory(relocaliser_trajectory),
                        tracker_trajectory=tracker_trajectory,
                        viewing_pose=camera_controller.get_pose(),
                        window_size=window_size
                    )

                    # Update the timestamp.
                    timestamp += 1.0

                # If the tracker's not ready yet, forcibly terminate the whole process (this isn't graceful, but
                # if we don't do it then we may have to wait a very long time for it to finish initialising).
                if not tracker.is_ready():
                    # noinspection PyProtectedMember
                    os._exit(0)

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
