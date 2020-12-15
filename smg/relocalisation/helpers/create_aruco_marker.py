import cv2
import numpy as np

from argparse import ArgumentParser


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument("--output_filename", "-o", type=str, required=True, help="the output filename")
    # DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, DICT_4X4_1000=3,
    # DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7,
    # DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11,
    # DICT_7X7_50=12, DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15,
    # DICT_ARUCO_ORIGINAL=16
    parser.add_argument("--dictionary", "-d", type=int, default=10, help="the ArUco dictionary to use")
    parser.add_argument("--id", type=int, required=True, help="the marker ID in the dictionary")
    parser.add_argument("--size_pixels", "-s", type=int, default=200, help="the marker size in pixels")
    parser.add_argument("--border_bits", "-bb", type=int, default=1, help="the number of bits in the marker borders")
    parser.add_argument("--show", action="store_true", help="whether to show the generated image")
    args: dict = vars(parser.parse_args())

    # Make the marker image.
    dictionary = cv2.aruco.getPredefinedDictionary(args["dictionary"])
    marker_img: np.ndarray = cv2.aruco.drawMarker(
        dictionary, args["id"], args["size_pixels"], borderBits=args["border_bits"]
    )

    # Show the marker image if requested.
    if args["show"]:
        cv2.imshow("Marker", marker_img)
        cv2.waitKey()

    # Save the marker image to the output file.
    cv2.imwrite(args["output_filename"], marker_img)


if __name__ == "__main__":
    main()
