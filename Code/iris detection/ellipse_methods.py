import numpy as np
import cv2
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
from Evaluate import Evaluation
from HaarFeature import HaarFeature
from acwe import ACWE

""" 
--------------------------------------
Setup: 
Pupil, Iris, ImageObserver, Evaluation
Connect Observer with all instances
Return those Objects for Pupil tracking 

"""


def setup(path):
    file = path.split("/")
    data_set = file[-3]
    index1 = file[-2]
    index2 = file[-1].split(".")[0]
    name = data_set + "_" + index1 + "_" + index2
    pupil_obj = Pupil()
    iris_obj = Iris()
    observer = ImageObserver()
    evaluation_obj = Evaluation(pupil_obj, name, "Code/iris detection/results/")

    observer.add_img_obj(pupil_obj)
    return pupil_obj, iris_obj, observer, evaluation_obj, name


def get_video_frame(video_file, txt_file):
    print(video_file, txt_file)
    with open(txt_file) as f:
        lines = f.readlines()

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("EOF or Video not opened correctly")
        return True

    count = 0
    while True and count <= len(lines) - 1:
        ret, frame = cap.read()

        center = lines[count].split(" ")
        x_center_label = round(float(center[0].strip()))
        y_center_label = round(float(center[1].strip("\n")))
        center_label = (x_center_label, y_center_label)

        if not ret:
            break
        count += 1

        # resize frame:
        # frame = cv2.resize(frame, dsize=None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
        # center_label = (int(center_label[0]*0.7), int(center_label[1]*0.7))

        yield frame, center_label

    cap.release()
    cv2.destroyAllWindows()

    # for frame in get_video_frames(video_path):
    #    process_frame(frame)  # Call your custom frame processing method


def split_path(path):
    video_file, txt_file = path.split(",")
    return video_file, txt_file


def haar_roi_extraction(image, maxradius=9, minradius=3, plot=False):
    # print(f'plot: {plot}')
    # print(f'image: {image}')
    Haar_kernel = HaarFeature(maxradius, minradius, image)
    position_pupil, roi, roi_coords = Haar_kernel.find_pupil_ellipse(plot)
    # print('roi', roi)
    return position_pupil, roi, roi_coords


def threshold_ellipse(roi, intensity):
    thresholded = cv2.inRange(roi, int(intensity / 5), int(intensity * 2.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    return thresholded


def main_detection(path, scaling=1):
    video_file, txt_file = split_path(path)

    pupil_obj, iris_obj, observer, evaluation_obj, name = setup(video_file)

    # Load frame by frame to process
    for frame, label_center in get_video_frame(
        video_file=video_file, txt_file=txt_file
    ):
        frame = cv2.resize(
            frame.copy(),
            (int(frame.shape[1] * scaling), int(frame.shape[0] * scaling)),
            interpolation=cv2.INTER_LINEAR,
        )

        pupil_obj.set_img(frame.copy())

        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil_obj.set_gray(gray_eye_image.copy())

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
        gray_eye_image = clahe.apply(gray_eye_image)
        pupil_obj.set_processing(gray_eye_image.copy())

        coords, roi, roi_coords = haar_roi_extraction(
            pupil_obj.get_processing(), plot=True
        )

        center = (coords[1] - roi_coords[0][1], coords[0] - roi_coords[0][0])
        radius = 10
        acwe = ACWE()
        acwe.start(center, radius, roi, 3, 1000, 2, 0.3, 0.0002, 1000, 0.01)
        BOOL_PUPIL = acwe.result()
        ellipse = acwe.get_result_ellipse()

        if BOOL_PUPIL is True:
            pupil_obj.set_ellipse(ellipse, roi_coords, roi.shape)

        else:
            filename = (
                name + "_" + str(label_center[0]) + "_" + str(label_center[1]) + ".png"
            )
            cv2.imwrite(
                "Code/iris detection/results/failed_eval/" + name + "/" + filename,
                frame,
            )
            # print(f'pupil not found')

        # pupil_obj.set_ellipse(pupil, coords)
        # print(label_center, pupil_obj.get_center())

        evaluation_obj.add_frame(
            BOOL_PUPIL, label_center, pupil_obj.get_center(), roi_coords
        )

        observer.plot_pupil(pupil_obj)
        # observer.plot_imgs('original')

        key = cv2.waitKey(1)
        if key == ord("q"):  # Press 'q' to exit
            break
    evaluation_obj.create_log(scaling)


def main_Haar_image():
    frame = cv2.imread("eye_img_22.png", cv2.IMREAD_GRAYSCALE)

    gray_eye_image = cv2.resize(
        frame,
        (int(frame.shape[1] * 60 / 100), int(frame.shape[0] * 60 / 100)),
        interpolation=cv2.INTER_LINEAR,
    )
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11, 11))
    gray_eye_image = clahe.apply(gray_eye_image)
    coords, roi, roi_coords = haar_roi_extraction(gray_eye_image, plot=True)

    intensity = gray_eye_image[coords[1], coords[0]]

    cv2.imshow("frame", gray_eye_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main_detection("D:/data_set/LPW/1/1.avi,D:/data_set/LPW/1/1.txt", scaling=1)

    cv2.waitKey(0)
