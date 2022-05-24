import cv2
import numpy as np
import torch
import dsacstar  # must be imported after torch

from typing import Optional

from torchvision import transforms

from smg.external.dsacstar.network import Network


class DSACStarRelocaliser:
    """A wrapper around Eric Brachmann's DSAC* relocaliser."""

    # CONSTRUCTOR

    def __init__(self, network_filename: str, *, hypothesis_count: int = 64, image_height: int = 480,
                 inlier_alpha: float = 100.0, inlier_threshold: float = 10.0, max_pixel_error: float = 100.0,
                 tiny: bool = False):
        """
        Construct a wrapper around Eric Brachmann's DSAC* relocaliser.

        :param network_filename:    The name of the file containing the DSAC* network.
        :param hypothesis_count:    The number of RANSAC hypotheses to consider.
        :param image_height:        The height to which the colour images will be rescaled.
        :param inlier_alpha:        The alpha parameter to use for soft inlier counting.
        :param inlier_threshold:    The inlier threshold to use when sampling RANSAC hypotheses (in pixels).
        :param max_pixel_error:     The maximum reprojection error to use when checking pose consistency (in pixels).
        :param tiny:                Whether to load a tiny network to massively reduce the memory footprint.
        """
        self.__hypothesis_count: int = hypothesis_count
        self.__inlier_alpha: float = inlier_alpha
        self.__inlier_threshold: float = inlier_threshold
        self.__max_pixel_error: float = max_pixel_error

        # Load in the DSAC* network, copy it across to the GPU, and put it into evaluation mode.
        self.__network: Network = Network(torch.zeros(3), tiny)
        self.__network.load_state_dict(torch.load(network_filename))
        self.__network = self.__network.cuda()
        self.__network.eval()

        # Set up the sequence of transformations that will be applied to each image before passing it to the network.
        self.__image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_height),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4], std=[0.25])  # as per Eric's code
        ])

    # PUBLIC METHODS

    def estimate_pose(self, image: np.ndarray, focal_length: float) -> Optional[np.ndarray]:
        """
        Estimate the 6D pose of the specified image.

        .. note::
            The image is assumed to be in BGR format (i.e. what OpenCV uses).

        :param image:           The image whose pose is to be estimated.
        :param focal_length:    The focal length of the camera (in pixels).
        :return:                The estimated 6D pose, as a 4x4 matrix.
        """
        # Transform the image from BGR to RGB.
        # noinspection PyUnresolvedReferences
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # Convert the image into an appropriately shaped PyTorch tensor on the GPU.
            tensor: torch.Tensor = self.__image_transform(image).unsqueeze(0).cuda()

            # Run the tensor through the network to get the scene coordinate image.
            scene_coordinates: torch.Tensor = self.__network(tensor).cpu()

            # Make an empty tensor in which to store the camera pose.
            out_pose: torch.Tensor = torch.zeros((4, 4))

            # Use the predicted scene coordinates to estimate the camera pose.
            # noinspection PyUnresolvedReferences
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

            # Copy the estimated camera pose back across to a NumPy array on the CPU, and return it.
            return out_pose.cpu().numpy()
