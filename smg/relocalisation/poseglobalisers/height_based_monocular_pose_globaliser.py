import numpy as np

from typing import Optional


class HeightBasedMonocularPoseGlobaliser:
    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False):
        self.__debug: bool = debug
        self.__height_movement: float = 0.0
        self.__last_height: Optional[float] = None
        self.__last_tracker_pos: Optional[np.ndarray] = None
        self.__reference_height: Optional[float] = None
        self.__scale: float = 1.0
        self.__tracker_i_t_r: Optional[np.ndarray] = None
        self.__tracker_movement: float = 0.0

    # PUBLIC METHODS

    def apply(self, tracker_i_t_c: np.ndarray, height: float, *, suppress_scaling: bool = False) -> np.ndarray:
        # Determine the scale to use (this will be the scale we've estimated, unless we're suppressing scaling).
        scale: float = self.__scale if not suppress_scaling else 1.0

        # Make a metric transformation from reference space to initial camera space.
        # TODO: This could be calculated in advance.
        metric_tracker_i_t_r: np.ndarray = self.__tracker_i_t_r.copy()
        metric_tracker_i_t_r[0:3, 3] *= scale

        # Make a metric transformation from initial camera space to world space.
        # TODO: This could also be calculated in advance.
        # wTi = wTr . (iTr)^-1
        w_t_r: np.ndarray = np.eye(4)
        w_t_r[0:3, 3] = np.array([0, self.__reference_height, 0])
        metric_tracker_w_t_i: np.ndarray = w_t_r @ np.linalg.inv(metric_tracker_i_t_r)

        # Make a metric transformation from current camera space to initial camera space.
        metric_tracker_i_t_c: np.ndarray = tracker_i_t_c.copy()
        metric_tracker_i_t_c[0:3, 3] *= scale

        # Make a metric transformation from current camera space to world space.
        # wTc = wTi . iTc
        tracker_w_t_c: np.ndarray = metric_tracker_w_t_i @ metric_tracker_i_t_c

        # TODO: Height fixing

        return tracker_w_t_c

    def reset(self) -> None:
        self.__height_movement = 0.0
        self.__last_height = None
        self.__last_tracker_pos = None
        self.__reference_height = None
        self.__scale = 1.0
        self.__tracker_i_t_r = None
        self.__tracker_movement = 0.0

    def train(self, tracker_i_t_c: np.ndarray, height: float) -> None:
        tracker_pos: np.ndarray = tracker_i_t_c[0:3, 3]

        if self.__last_height is None:
            self.__reference_height = height
            self.__tracker_i_t_r = tracker_i_t_c.copy()
        else:
            self.__height_movement += abs(height - self.__last_height)
            self.__tracker_movement += np.linalg.norm(tracker_pos - self.__last_tracker_pos)
            self.__scale = self.__height_movement / self.__tracker_movement

            if self.__debug:
                print(f"Current Scale Estimate: {self.__scale}")

        self.__last_height = height
        self.__last_tracker_pos = tracker_pos
