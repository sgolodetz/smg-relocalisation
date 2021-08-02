import numpy as np
import vg

from typing import Optional


class HeightMonocularPoseGlobaliser:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False):
        """
        TODO

        :param debug:   Whether or not to output debug messages.
        """
        self.__debug: bool = debug
        self.__height_movement_sum: float = 0.0
        self.__last_height: Optional[float] = None
        self.__last_tracker_pos: Optional[np.ndarray] = None
        self.__reference_height: Optional[float] = None
        self.__scale: float = 1.0
        self.__tracker_i_t_r: Optional[np.ndarray] = None
        self.__tracker_movement_sum: float = 0.0
        self.__up: Optional[np.ndarray] = None
        self.__up_count: int = 0
        self.__up_sum: np.ndarray = np.zeros(3)

    # PUBLIC METHODS

    def apply(self, tracker_i_t_c: np.ndarray, height: Optional[float], *,
              suppress_scaling: bool = False) -> np.ndarray:
        # Determine the scale to use (this will be the scale we've estimated, unless we're suppressing scaling).
        scale: float = self.__scale if not suppress_scaling else 1.0

        # Make a metric transformation from reference space to initial camera space.
        # TODO: This could be calculated in advance.
        metric_tracker_i_t_r: np.ndarray = self.__tracker_i_t_r.copy()
        metric_tracker_i_t_r[0:3, 3] *= scale

        # Make a metric transformation from initial camera space to world space.
        # TODO: This could also be calculated in advance.
        # wTi = wTr . (iTr)^-1
        # w_t_r: np.ndarray = np.eye(4)
        # w_t_r[0:3, 3] = np.array([0, self.__reference_height, 0])
        # metric_tracker_w_t_i: np.ndarray = w_t_r @ np.linalg.inv(metric_tracker_i_t_r)
        metric_tracker_w_t_i: np.ndarray = np.linalg.inv(metric_tracker_i_t_r)
        metric_tracker_w_t_i[0:3, 3] += self.__reference_height * self.__up

        # Make a metric transformation from current camera space to initial camera space.
        metric_tracker_i_t_c: np.ndarray = tracker_i_t_c.copy()
        metric_tracker_i_t_c[0:3, 3] *= scale

        # Make a metric transformation from current camera space to world space.
        # wTc = wTi . iTc
        tracker_w_t_c: np.ndarray = metric_tracker_w_t_i @ metric_tracker_i_t_c

        # TODO: Height fixing

        return tracker_w_t_c

    def reset(self) -> None:
        self.__height_movement_sum = 0.0
        self.__last_height = None
        self.__last_tracker_pos = None
        self.__reference_height = None
        self.__scale = 1.0
        self.__tracker_i_t_r = None
        self.__tracker_movement_sum = 0.0
        self.__up = None
        self.__up_count = 0
        self.__up_sum = np.zeros(3)

    def train(self, tracker_i_t_c: np.ndarray, height: float) -> None:
        tracker_pos: np.ndarray = tracker_i_t_c[0:3, 3]

        if self.__last_height is None:
            self.__reference_height = height
            self.__tracker_i_t_r = tracker_i_t_c.copy()
            self.__last_height = height
            self.__last_tracker_pos = tracker_pos
        else:
            height_movement: float = abs(height - self.__last_height)
            if height_movement >= 0.02:
                tracker_offset: np.ndarray = tracker_pos - self.__last_tracker_pos
                tracker_movement: float = np.linalg.norm(tracker_offset)

                self.__height_movement_sum += height_movement
                self.__tracker_movement_sum += tracker_movement

                if self.__tracker_movement_sum > 0.1:
                    self.__scale = self.__height_movement_sum / self.__tracker_movement_sum

                    up_estimate: np.ndarray = tracker_offset.copy()
                    if up_estimate[1] > 0:
                        up_estimate *= -1
                    self.__up_sum += up_estimate
                    self.__up_count += 1
                    self.__up = vg.normalize(self.__up_sum / self.__up_count)

                    if self.__debug:
                        print(
                            height_movement, tracker_movement, tracker_offset,
                            self.__height_movement_sum, self.__tracker_movement_sum,
                            self.__scale, self.__up
                        )

                self.__last_height = height
                self.__last_tracker_pos = tracker_pos
