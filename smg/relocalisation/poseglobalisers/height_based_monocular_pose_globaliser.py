import matplotlib.pyplot as plt
import numpy as np
import os
import vg

from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation
from typing import List, Optional

from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter


class HeightBasedMonocularPoseGlobaliser:
    """
    Used to correct the scale of monocular poses by using the known (e.g. sensed) metric height of the camera.

    .. note::
        This is particularly useful for globalising the camera poses of drones with an accurate height sensor,
        such as the DJI Tello.
    """

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False):
        """
        Construct a height-based monocular pose globaliser.

        :param debug:   Whether or not to output debug messages.
        """
        self.__debug: bool = debug
        self.__height_movement_sum: float = 0.0
        self.__last_height: Optional[float] = None
        self.__last_tracker_pos: Optional[np.ndarray] = None
        self.__metric_w_t_i: Optional[np.ndarray] = None
        self.__reference_height: Optional[float] = None
        self.__scale: float = 1.0
        self.__tracker_i_t_r: Optional[np.ndarray] = None
        self.__tracker_movement_sum: float = 0.0
        self.__up: Optional[np.ndarray] = None
        self.__up_count: int = 0
        self.__up_sum: np.ndarray = np.zeros(3)

        self.__ax: Optional[np.ndarray] = None
        self.__fig: Optional[Figure] = None

        self.__debug_height_movements: List[float] = []
        self.__debug_heights: List[float] = []
        self.__debug_scales: List[float] = []
        self.__debug_tracker_movements: List[float] = []

        if debug:
            self.__fig, self.__ax = plt.subplots(4, 1)
            self.__fig.canvas.set_window_title("Scale Estimation")

    # PUBLIC METHODS

    def apply(self, tracker_i_t_c: np.ndarray) -> np.ndarray:
        """
        Convert a non-metric transformation from current camera space to initial camera space into a metric
        transformation from current camera space to world space.

        .. note::
            "World" space is a coordinate system whose origin is on the ground below the reference origin,
            and whose x and z axes are level with the ground (the y axis points downwards).
        .. note::
            The transformations output by the monocular tracker are assumed to be non-metric, and their scale
            may drift over time.

        :param tracker_i_t_c:   A non-metric transformation from current camera space to initial camera space,
                                as estimated by the tracker.
        :return:                A metric transformation from current camera space to world space.
        """
        # Make a metric transformation from current camera space to initial camera space.
        metric_i_t_c: np.ndarray = tracker_i_t_c.copy()
        metric_i_t_c[0:3, 3] *= self.__scale

        # Make and return a metric transformation from current camera space to world space.
        # wTc = wTi . iTc
        return self.__metric_w_t_i @ metric_i_t_c

    # noinspection PyMethodMayBeStatic
    def finish_training(self) -> None:
        """Inform the globaliser that the training process has finished."""
        if self.__debug:
            self.__draw_scale_estimation_figure(start=0, end=len(self.__debug_scales))
            # folder: str = "D:/cyberphysicalsystems/droneflightsequences/output-scaletests"
            folder: str = "C:/smglib"
            os.makedirs(folder, exist_ok=True)
            self.__fig.savefig(os.path.join(folder, "scale.png"))
            plt.close("all")

    def train(self, tracker_i_t_c: np.ndarray, height: float) -> None:
        """
        Train the globaliser using a sample consisting of a non-metric pose estimated by the tracker
        and the known (e.g. sensed) metric height of the camera at the same point in time.

        :param tracker_i_t_c:   A non-metric transformation from current camera space to initial camera space,
                                as estimated by the tracker.
        :param height:          The known (e.g. sensed) metric height of the camera.
        """
        # Get the non-metric camera position estimated by the tracker.
        tracker_pos: np.ndarray = tracker_i_t_c[0:3, 3]

        # If this is the first sample we've received:
        if self.__last_height is None:
            # Use the current non-metric tracker pose and metric height to define the reference space.
            self.__reference_height = height
            self.__tracker_i_t_r = tracker_i_t_c.copy()

            # Set the height and tracker position of the most recent sample.
            self.__last_height = height
            self.__last_tracker_pos = tracker_pos

        # Otherwise:
        else:
            # Compute the absolute change in metric height since the last sample.
            height_movement: float = abs(height - self.__last_height)

            # If the height has changed by at least 2cm:
            if height_movement >= 0.02:
                # Compute the non-metric camera movement estimated by the tracker since the last sample.
                tracker_offset: np.ndarray = tracker_pos - self.__last_tracker_pos
                tracker_movement: float = np.linalg.norm(tracker_offset)

                # Update the two records of how far we've moved overall during training. Note that the height one
                # will be metric, whilst the tracker one will be non-metric.
                self.__height_movement_sum += height_movement
                self.__tracker_movement_sum += tracker_movement

                # As long as we've moved:
                if self.__tracker_movement_sum > 0.0:
                    # Estimate the scale.
                    self.__scale = self.__height_movement_sum / self.__tracker_movement_sum

                    # Compute an estimate of the up vector from the camera movement since the last sample,
                    # as estimated by the tracker. Note that we assume the camera only moves either up or
                    # down during training. The initial up estimate should point roughly in the direction
                    # of either -y (up) or y (down). We multiply it by -1 if needed to make it point in
                    # the direction of -y.
                    up_estimate: np.ndarray = tracker_offset.copy()
                    if up_estimate[1] > 0:
                        up_estimate *= -1

                    # Use the estimate to update the overall up vector (this is just an average of all our estimates).
                    self.__up_sum += up_estimate
                    self.__up_count += 1
                    self.__up = vg.normalize(self.__up_sum / self.__up_count)

                    # Update the metric transformation from initial camera space to world space.
                    self.__update_w_t_i()

                    # If we're debugging:
                    if self.__debug:
                        # Print out some relevant values.
                        print(
                            height_movement, tracker_movement, tracker_offset,
                            self.__height_movement_sum, self.__tracker_movement_sum,
                            self.__scale, self.__up
                        )

                        # Visualise some relevant sequences.
                        self.__debug_height_movements.append(height_movement)
                        self.__debug_heights.append(height)
                        self.__debug_tracker_movements.append(tracker_movement)
                        self.__debug_scales.append(self.__scale)

                        tick_step: int = 10
                        start: int = max(len(self.__debug_scales) - 4 * tick_step - 1, 0)
                        end: int = len(self.__debug_scales)
                        self.__draw_scale_estimation_figure(
                            start=start, end=end, xticks=np.arange(start, end, tick_step)
                        )
                        plt.waitforbuttonpress(0.001)

                # Update the height and tracker position of the most recent sample.
                self.__last_height = height
                self.__last_tracker_pos = tracker_pos

    # PRIVATE METHODS

    def __draw_scale_estimation_figure(self, *, start: int, end: int, xticks: Optional[List[int]] = None) -> None:
        """
        TODO

        :param start:   TODO
        :param end:     TODO
        :param xticks:  TODO
        """
        for i in range(4):
            self.__ax[i].clear()
            if xticks is not None:
                self.__ax[i].xaxis.set_ticks(xticks)

        plt.xlabel("Iteration")
        xs: List[int] = np.arange(start, end)
        self.__ax[0].set_ylabel("Scale")
        self.__ax[0].plot(xs, self.__debug_scales[start:end])
        self.__ax[1].set_ylabel("Height")
        self.__ax[1].plot(xs, self.__debug_heights[start:end])
        self.__ax[2].set_ylabel("Height\nMovement")
        self.__ax[2].plot(xs, self.__debug_height_movements[start:end])
        self.__ax[3].set_ylabel("Tracker\nMovement")
        self.__ax[3].plot(xs, self.__debug_tracker_movements[start:end])

        plt.subplots_adjust(
            wspace=1.5,
            hspace=1.5
        )

        plt.draw()

    def __update_w_t_i(self) -> None:
        """Update the metric transformation from initial camera space to world space."""
        # Make a metric transformation from reference space to initial camera space.
        metric_i_t_r: np.ndarray = self.__tracker_i_t_r.copy()
        metric_i_t_r[0:3, 3] *= self.__scale

        # Make a metric transformation from reference space to world space.
        metric_w_t_r: np.ndarray = np.eye(4)
        angle: float = vg.angle(self.__up, np.array([0, -1, 0]), units="rad")
        reference_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(metric_i_t_r)
        r: Rotation = Rotation.from_rotvec(reference_cam.u() * angle)
        metric_w_t_r[0:3, 0:3] = r.as_matrix()
        metric_w_t_r[1, 3] = -self.__reference_height

        # Update the metric transformation from initial camera space to world space.
        # wTi = wTr . (iTr)^-1
        self.__metric_w_t_i = metric_w_t_r @ np.linalg.inv(metric_i_t_r)
