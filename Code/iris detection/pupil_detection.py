import numpy as np
import cv2
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
from Evaluate import Evaluation
from HaarFeature import HaarFeature
from acwe import ACWE
import ellipse_methods as em
import matplotlib.pyplot as plt

"""
This script provides two main functions:
    - pupil_detection_image: Use this for image processing.
    - pupil_detection_video: Use this for video processing.

Parameters:
    - path (str): The path to the image or video you want to process.
    - scaling (float): Scaling factor for processing (default: 1).

Additional Parameters:
    - clahe_clip (float): CLAHE clip limit (default: 2.0).
    - clahe_grid (tuple): Grid size for CLAHE (default: (11, 11)).
    - maxradius (int): Maximum radius for Haar feature detection (default: 9).
    - minradius (int): Minimum radius for Haar feature detection (default: 3).
    - iterations_smoothing (int): Number of iterations for smoothing (default: 3).
    - iterations_acwe (int): Number of iterations for ACWE (default: 1000).
    - lambda1 (float): ACWE parameter lambda1 (default: 2).
    - lambda2 (float): ACWE parameter lambda2 (default: 0.3).
    - radius (int): Radius for processing (default: 10).
    - convergence_threshold (float): Convergence threshold for ACWE (default: 0.0002).
    - ransac_iterations (int): Number of RANSAC iterations (default: 1000).
    - ransac_threshold (float): RANSAC threshold (default: 0.01).
    - callback (bool): Enable callback for visualization (default: False).

Note: The requirements.txt file is not currently up to date and will not be automatically updated. 
Please ensure you have the required packages installed, and for detailed explanations of these parameters, refer to my thesis.
"""


def load_image(path, clahe_clip=2.0, clahe_grid=(11, 11), scaling=1):
    """
    Load an image, preprocess it, and return a Pupil object.

    Parameters:
        - path (str): The path to the image.
        - clahe_clip (float): CLAHE clip limit (default: 2.0).
        - clahe_grid (tuple): Grid size for CLAHE (default: (11, 11)).
        - scaling (float): Scaling factor for the image (default: 1).

    Returns:
        - Pupil: An object representing the processed image.
    """

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(
        img,
        (int(img.shape[1] * scaling), int(img.shape[0] * scaling)),
        interpolation=cv2.INTER_LINEAR,
    )
    pupil_obj = Pupil()
    pupil_obj.set_img(img)
    gray_eye_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pupil_obj.set_gray(gray_eye_img.copy())
    clahe = cv2.createCLAHE(clahe_clip, clahe_grid)
    gray_eye_img = clahe.apply(gray_eye_img)
    pupil_obj.set_processing(gray_eye_img.copy())

    return pupil_obj


def load_video_frame(video_path):
    """
    Generator function to load frames from a video.

    Parameters:
        - video_path (str): The path to the video.

    Yields:
        - numpy.ndarray: A frame from the video.
    """

    cap = cv2.VideoCapture(video_path)
    end = False
    if not cap.isOpened():
        print("EOF or Video not opened correctly")
        end = True
        return True

    count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        count += 1

        yield frame

    cap.release()
    cv2.destroyAllWindows()


def pupil_detection_image(params):
    """
    Process a single image for pupil detection.

    Parameters:
        - params (dict): A dictionary of parameters for processing.

    Returns:
        - tuple: The parameters (center, axes, angle) of the ellipse that fits the pupil.
    """

    path = params["path"]
    scaling = params["scaling"]
    clahe_clip = params["clahe_clip"]
    clahe_grid = params["clahe_grid"]
    maxradius = params["maxradius"]
    minradius = params["minradius"]
    iterations_smoothing = params["iterations_smoothing"]
    iterations_acwe = params["iterations_acwe"]
    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    radius = params["radius"]
    convergence_threshold = params["convergence_threshold"]
    ransac_iterations = params["ransac_iterations"]
    ransac_threshold = params["ransac_threshold"]
    callback = params["callback"]

    pupil_obj = load_image(
        path, clahe_clip=clahe_clip, clahe_grid=clahe_grid, scaling=scaling
    )

    coords, roi, roi_coords = em.haar_roi_extraction(
        pupil_obj.get_processing(), maxradius=maxradius, minradius=minradius, plot=False
    )

    # Prepare parameters for acew and ransac
    center = (coords[1] - roi_coords[0][1], coords[0] - roi_coords[0][0])
    acwe = ACWE()
    acwe.start(
        center=center,
        radius=radius,
        image=roi,
        iterations_smoothing=iterations_smoothing,
        iterations_ACWE=iterations_acwe,
        lambda1=lambda1,
        lambda2=lambda2,
        convergence_threshold=convergence_threshold,
        ransac_iterations=ransac_iterations,
        ransac_threshold=ransac_threshold,
        callback_bool=callback,
    )

    BOOL_PUPIL = acwe.result()
    ellipse = acwe.get_result_ellipse()
    if callback:
        acwe.plot_ellipse()
    if BOOL_PUPIL is True:
        pupil_obj.set_ellipse(ellipse, roi_coords, roi.shape)

    else:
        print(f"no ellipse found in this frame")

    return pupil_obj.get_ellipse()


def pupil_detection_video(params):
    """
    Process a video for pupil detection.

    Parameters:
        - params (dict): A dictionary of parameters for processing.

    Yields:
        - tuple: A tuple containing the frame number and the parameters (center, axes, angle) of the ellipse that fits the pupil.
    """

    path = params["path"]
    scaling = params["scaling"]
    clahe_clip = params["clahe_clip"]
    clahe_grid = params["clahe_grid"]
    maxradius = params["maxradius"]
    minradius = params["minradius"]
    iterations_smoothing = params["iterations_smoothing"]
    iterations_acwe = params["iterations_acwe"]
    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    radius = params["radius"]
    convergence_threshold = params["convergence_threshold"]
    ransac_iterations = params["ransac_iterations"]
    ransac_threshold = params["ransac_threshold"]
    callback = params["callback"]

    for count, frame in enumerate(load_video_frame(video_path=path)):
        pupil_obj = Pupil()
        frame = cv2.resize(
            frame.copy(),
            (int(frame.shape[1] * scaling), int(frame.shape[0] * scaling)),
            interpolation=cv2.INTER_LINEAR,
        )
        pupil_obj.set_img(frame.copy())
        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil_obj.set_gray(gray_eye_image.copy())
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray_eye_image = clahe.apply(gray_eye_image)
        pupil_obj.set_processing(gray_eye_image.copy())

        coords, roi, roi_coords = em.haar_roi_extraction(
            pupil_obj.get_processing(),
            maxradius=maxradius,
            minradius=minradius,
            plot=False,
        )

        # Prepare parameters for acew and ransac
        center = (coords[1] - roi_coords[0][1], coords[0] - roi_coords[0][0])

        acwe = ACWE()
        acwe.start(
            center=center,
            radius=radius,
            image=roi,
            iterations_smoothing=iterations_smoothing,
            iterations_ACWE=iterations_acwe,
            lambda1=lambda1,
            lambda2=lambda2,
            convergence_threshold=convergence_threshold,
            ransac_iterations=ransac_iterations,
            ransac_threshold=ransac_threshold,
            callback_bool=callback,
        )

        BOOL_PUPIL = acwe.result()
        ellipse = acwe.get_result_ellipse()
        if callback:
            acwe.plot_ellipse()
            key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image
            if key == 27:  # if is pressed, exit loop
                cv2.destroyAllWindows()
            break
        if BOOL_PUPIL is True:
            pupil_obj.set_ellipse(ellipse, roi_coords, roi.shape)

        else:
            print(f"no ellipse found in this frame")
            pupil_obj.set_ellipse(((0, 0), (0, 0), 0), roi_coords, roi.shape)

        yield count, pupil_obj.get_ellipse()


if __name__ == "__main__":
    params = {
        "path": "Code/algorithm demo/test.jpg",
        "scaling": 1,
        # CLAHE
        "clahe_clip": 2.0,
        "clahe_grid": (11, 11),
        # Haar Feature
        "maxradius": 9,
        "minradius": 3,
        # ACWE
        "iterations_smoothing": 3,
        "iterations_acwe": 1000,
        "lambda1": 2,
        "lambda2": 0.3,
        "radius": 10,
        "convergence_threshold": 0.0002,
        # RANSAC
        "ransac_iterations": 1000,
        "ransac_threshold": 0.01,
        # Visualize
        "callback": False,
    }
    # Image processing example
    ellipse = pupil_detection_image(params)
    print(ellipse)

    params["path"] = "D:/data_set/LPW/1/1.avi"

    # Video processing example
    for count, ellipse in pupil_detection_video(params):
        print(f"Frame number: {count} --- Ellipse parameters: {ellipse}")
