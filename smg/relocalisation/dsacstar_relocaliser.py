import cv2
import numpy as np
import torch
import dsacstar  # must be imported after torch

from typing import Optional

from torchvision import transforms

from network import Network


class DSACStarRelocaliser:
    """A wrapper around Eric Brachmann's DSAC* relocaliser."""

    # CONSTRUCTOR

    def __init__(self, *, hypothesis_count: int = 64, image_height: int = 480, inlier_alpha: float = 100.0,
                 inlier_threshold: float = 10.0, max_pixel_error: float = 100.0, network_filename: str,
                 tiny: bool = False):
        self.__hypothesis_count: int = hypothesis_count
        self.__inlier_alpha: float = inlier_alpha
        self.__inlier_threshold: float = inlier_threshold
        self.__max_pixel_error: float = max_pixel_error

        # TODO: Comment here.
        self.__network: Network = Network(torch.zeros(3), tiny)
        self.__network.load_state_dict(torch.load(network_filename))
        self.__network = self.__network.cuda()
        self.__network.eval()

        # TODO: Comment here.
        self.__image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_height),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4], std=[0.25])  # as per Eric's code
        ])

    # PUBLIC METHODS

    def estimate_pose(self, image: np.ndarray, focal_length: float) -> Optional[np.ndarray]:
        # Transform the image from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # TODO: Comment here.
            tensor: torch.Tensor = self.__image_transform(image).unsqueeze(0).cuda()

            # TODO: Comment here.
            scene_coordinates: torch.Tensor = self.__network(tensor).cpu()

            # TODO: Comment here.
            out_pose: torch.Tensor = torch.zeros((4, 4))
            dsacstar.forward_rgb(
                scene_coordinates,
                out_pose,
                self.__hypothesis_count,
                self.__inlier_threshold,
                focal_length,
                float(tensor.size(3) / 2),  # principal point
                float(tensor.size(2) / 2),
                self.__inlier_alpha,
                self.__max_pixel_error,
                self.__network.OUTPUT_SUBSAMPLE
            )

        return out_pose.cpu().numpy()
