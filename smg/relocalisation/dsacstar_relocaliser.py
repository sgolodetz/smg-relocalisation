import numpy as np
import torch

from typing import Optional

from dsacstar.network import Network


class DSACStarRelocaliser:
    """A wrapper around Eric Brachmann's DSAC* relocaliser."""

    # CONSTRUCTOR

    def __init__(self, *, network_filename: str, tiny: bool = False):
        self.__network: Network = Network(torch.zeros(3), tiny)
        self.__network.load_state_dict(torch.load(network_filename))
        self.__network = self.__network.cuda()
        self.__network.eval()

    # PUBLIC METHODS

    def estimate_pose(self, image: np.ndarray) -> Optional[np.ndarray]:
        pass


def main() -> None:
    relocaliser: DSACStarRelocaliser = DSACStarRelocaliser(
        network_filename="D:/dsacstarmodels/rgb/7scenes_heads.net"
    )


if __name__ == "__main__":
    main()
