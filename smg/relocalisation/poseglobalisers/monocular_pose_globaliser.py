import numpy as np
import vg

from typing import Optional


class MonocularPoseGlobaliser:
    """Used to correct the scale of monocular poses and transform them into a global coordinate system."""

    # ENUMERATIONS

    class EState(int):
        pass

    UNTRAINED: EState = 0
    TRAINING: EState = 1
    ACTIVE: EState = 2

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False):
        """
        Construct a monocular pose globaliser.

        :param debug:   Whether or not to output debug messages.
        """
        self.__debug: bool = debug
        self.__fixed_height: Optional[float] = None
        self.__scale: float = 1.0
        self.__scale_count: int = 0
        self.__scale_sum: float = 0.0
        self.__state: MonocularPoseGlobaliser.EState = MonocularPoseGlobaliser.UNTRAINED
        self.__up: Optional[np.ndarray] = None

        # A metric transformation from reference space to world space, as estimated by the relocaliser.
        self.__relocaliser_w_t_r: Optional[np.ndarray] = None

        # A non-metric transformation from reference space to initial camera space, as estimated by the tracker.
        self.__tracker_i_t_r: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def apply(self, tracker_i_t_c: np.ndarray, *, suppress_scaling: bool = False) -> np.ndarray:
        """
        Convert a non-metric transformation from current camera space to initial camera space
        into a metric transformation from current camera space to world space.

        .. note::
            "World space" refers to the relocaliser's coordinate system.
        .. note::
            The transformations output by the relocaliser are assumed to be metric.
        .. note::
            The transformations output by the monocular tracker are assumed to be non-metric, and their scale
            may drift over time.

        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param suppress_scaling:    Whether to globalise the pose without scaling it (for debugging purposes).
        :return:                    A metric transformation from current camera space to world space.
        """
        # Determine the scale to use (this will be the scale we've estimated, unless we're suppressing scaling).
        scale: float = self.__scale if not suppress_scaling else 1.0

        # Make a metric transformation from reference space to initial camera space.
        # TODO: This could be calculated in advance.
        metric_tracker_i_t_r: np.ndarray = self.__tracker_i_t_r.copy()
        metric_tracker_i_t_r[0:3, 3] *= scale

        # Make a metric transformation from initial camera space to world space.
        # TODO: This could also be calculated in advance.
        # wTi = wTr . (iTr)^-1
        metric_tracker_w_t_i: np.ndarray = self.__relocaliser_w_t_r @ np.linalg.inv(metric_tracker_i_t_r)

        # Make a metric transformation from current camera space to initial camera space.
        metric_tracker_i_t_c: np.ndarray = tracker_i_t_c.copy()
        metric_tracker_i_t_c[0:3, 3] *= scale

        # Make a metric transformation from current camera space to world space.
        # wTc = wTi . iTc
        tracker_w_t_c: np.ndarray = metric_tracker_w_t_i @ metric_tracker_i_t_c

        # If a fixed height has been set and we're not suppressing scaling, use it to correct for any scale drift.
        if self.__fixed_height is not None and not suppress_scaling:
            height: float = vg.scalar_projection(tracker_w_t_c[0:3, 3], self.__up)
            tracker_w_t_c[0:3, 3] *= self.__fixed_height / height

            if self.__debug:
                print(f"Dynamic scaling factor: {self.__fixed_height} / {height} = {self.__fixed_height / height}")

        return tracker_w_t_c

    def finish_training(self) -> None:
        """
        Finish training the globaliser.
        """
        if self.__state == MonocularPoseGlobaliser.TRAINING:
            self.__state = MonocularPoseGlobaliser.ACTIVE
        else:
            raise RuntimeError("Cannot finish training a pose globaliser before starting to do so")

    def get_state(self) -> EState:
        """
        Get the state of the globaliser.

        :return:    The state of the globaliser.
        """
        return self.__state

    def has_fixed_height(self) -> bool:
        """
        Get whether or not a fixed height has been set.

        :return:    True, if a fixed height has been set, or False otherwise.
        """
        return self.__fixed_height is not None

    def set_fixed_height(self, tracker_w_t_c: np.ndarray, *, up: np.ndarray = np.array([0, -1, 0])) -> None:
        """
        Promise that the camera will stay at its current height from here on out.

        .. note::
            This can be used to mitigate scale drift.

        :param tracker_w_t_c:   A metric transformation from current camera space to world space.
        :param up:              The "up" direction of the camera (needed to determine the height).
        """
        self.__fixed_height = vg.scalar_projection(tracker_w_t_c[0:3, 3], up)
        self.__up = up

        if self.__debug:
            print(f"Setting fixed height to: {self.__fixed_height}")

    def start_training(self, tracker_i_t_c: np.ndarray, relocaliser_w_t_c: np.ndarray) -> None:
        """
        Start training the globaliser.

        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
        """
        # Set the reference space to the current camera space.
        self.__relocaliser_w_t_r = relocaliser_w_t_c
        self.__tracker_i_t_r = tracker_i_t_c

        # Reset the scale.
        self.__scale = 1.0
        self.__scale_count = 0
        self.__scale_sum = 0.0

        # Clear any fixed height that was being used.
        self.__fixed_height = None
        self.__up = None

        # Set the globaliser's state.
        self.__state = MonocularPoseGlobaliser.TRAINING

    def train(self, tracker_i_t_c: np.ndarray, relocaliser_w_t_c: np.ndarray, *, min_dist: float = 0.1) -> None:
        """
        Train the globaliser using a sample consisting of transformations estimated by the tracker
        and the relocaliser at the same point in time.

        .. note::
            Training involves estimating the scale by dividing the distance moved from the reference pose as estimated
            by the relocaliser by the distance moved from the reference pose as estimated by the tracker.

        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
        :param min_dist:            The minimum (metric) distance that the camera must be from the reference pose
                                    to add a new scale estimate.
        """
        # Calculate the non-metric distance moved from the reference pose as estimated by the tracker.
        tracker_offset: np.ndarray = tracker_i_t_c[0:3, 3] - self.__tracker_i_t_r[0:3, 3]
        tracker_dist: float = np.linalg.norm(tracker_offset)

        # Calculate the metric distance moved from the reference pose as estimated by the relocaliser.
        relocaliser_offset: np.ndarray = relocaliser_w_t_c[0:3, 3] - self.__relocaliser_w_t_r[0:3, 3]
        relocaliser_dist: float = np.linalg.norm(relocaliser_offset)

        # If the camera is far enough from the reference pose to reliably estimate the scale:
        if relocaliser_dist >= min_dist and tracker_dist > 0:
            # Make a new scale estimate.
            scale_estimate: float = relocaliser_dist / tracker_dist

            # If the scale estimate's at least somewhat reasonable:
            # FIXME: This is a hack to work around the fact that we're using the mean to average the scale estimates,
            #        and we don't want to include obvious outliers in that calculation. A better fix would be to use
            #        a more robust averaging process in the first place though - we should fix that ultimately.
            if scale_estimate <= 5.0:
                # Use it to update our overall estimate of the scale.
                self.__scale_sum += scale_estimate
                self.__scale_count += 1
                self.__scale = self.__scale_sum / self.__scale_count

                # Output a debug message if asked.
                if self.__debug:
                    print(
                        f"Added scale estimate: "
                        f"{relocaliser_dist}, {tracker_dist * self.__scale}, {scale_estimate}, {self.__scale}"
                    )
