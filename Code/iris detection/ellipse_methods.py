import numpy as np 
import cv2
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
from Evaluate import Evaluation
import matplotlib.pyplot as plt
import seaborn as sns
from HaarFeature import HaarFeature
import ellipse_detection_algo as eda
import create_plots as cp
from acwe import ACWE

''' 
--------------------------------------
Setup: 
Pupil, Iris, ImageObserver, Evaluation
Connect Observer with all instances
Return those Objects for Pupil tracking 

'''


def setup():
    
    pupil_obj = Pupil()
    iris_obj = Iris()
    observer = ImageObserver()
    evaluation_obj = Evaluation(pupil_obj,'results.csv','Code/iris detection/results/')

    observer.add_img_obj(pupil_obj)   
    return pupil_obj, iris_obj, observer, evaluation_obj

def get_video_frame(path):
    video_file, txt_file = path.split(',')
    print(video_file, txt_file)


    with open (txt_file) as f:
        lines = f.readlines()
   
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print("EOF or Video not opened correctly")
        return True
    
    count = 0
    while True and count <= len(lines)-1:
        ret, frame = cap.read()
        
        
        center = lines[count].split(' ')
        x_center_label = round(float(center[0].strip()))
        y_center_label = round(float(center[1].strip('\n')))
        center_label = (x_center_label, y_center_label)
        
        if not ret:
            break
        count += 1
        yield frame, center_label

    cap.release()
    cv2.destroyAllWindows()                    
        
    #for frame in get_video_frames(video_path):
    #    process_frame(frame)  # Call your custom frame processing method
    

def haar_roi_extraction( image, plot):
    print(f'plot: {plot}')
    print(f'image: {image}')
    Haar_kernel = HaarFeature(8, 3, image)
    position_pupil, roi, roi_coords = Haar_kernel.find_pupil_ellipse(plot)
    #print('roi', roi)
    return  position_pupil, roi, roi_coords

def threshold_ellipse(roi, intensity):

    thresholded = cv2.inRange(roi, int(intensity/5), int(intensity*2.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN, kernel)

    return thresholded

def main_detection():               
    pupil_obj, iris_obj, observer, evaluation_obj = setup()
    
    #Load frame by frame to process
    for frame, label_center in get_video_frame('D:/data_set/LPW/1/1.avi,D:/data_set/LPW/1/1.txt'):
        setup()
        pupil_obj.set_img(frame.copy())

        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil_obj.set_gray(gray_eye_image.copy())
        
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
        gray_eye_image = clahe.apply(gray_eye_image)
        pupil_obj.set_processing(gray_eye_image.copy())
        
        coords, roi, roi_coords = haar_roi_extraction(pupil_obj.get_processing(), plot= True)


    
        #pupil = eda.best_ellipse(edges)
        #merged_regions = MSER.mser(roi)
        #print("Number of contours found:", len(merged_regions))
#
#
        #regions = roi.copy()
        #for r in merged_regions:
        #    regions = cv2.drawContours(regions, r, -1, (0, 255, 0), 2)
        #
        #cv2.imshow('regions', regions)
       
   
        
        xy_1 = (int(coords[0]- 110), int(coords[1]-110))
        xy_2 = (int(coords[0]+110), int(coords[1]+110))
        
        cv2.rectangle(frame,xy_1, xy_2, (255,255,50), 1 )
        cv2.imshow('result', frame)
        cv2.imshow('roi', roi)
        
        center = (roi.shape[0]//2, roi.shape[1]//2)
        radius = 15
        acwe = ACWE()
        acwe.start(center, radius, roi, 3, 100, 1, 1, 0.01)
        BOOL_PUPIL = acwe.result()
        ellipse = acwe.get_result_ellipse()
        #acwe.plot_ellipse()
        pupil_obj.set_ellipse(ellipse, coords)
        #
        #BOOL_PUPIL = pupil is not None
        if BOOL_PUPIL is not True:
            print(f'pupil not found')
            continue
        #pupil_obj.set_ellipse(pupil, coords)
        print(label_center, pupil_obj.get_center())
        evaluation_obj.add_frame(BOOL_PUPIL,label_center ,pupil_obj.get_center(), roi_coords )
        #cp.plot_ellipse(roi, pupil)

        observer.plot_pupil(pupil_obj)
        #observer.plot_imgs('original')
        
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break
        
    evaluation_obj.create_log()
    
def main_Haar_image():
    frame = cv2.imread('eye_img_22.png', cv2.IMREAD_GRAYSCALE)
    print(frame.shape)
    gray_eye_image=cv2.resize(frame,(int(frame.shape[1]*60/100),int(frame.shape[0]*60/100)), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
    gray_eye_image = clahe.apply(gray_eye_image)
    coords, roi, roi_coords = haar_roi_extraction(gray_eye_image, plot= True)
    print(coords)
    intensity = gray_eye_image[coords[1], coords[0]]


    cv2.imshow('frame',gray_eye_image)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main_detection()
    #main_Haar_image()q
    cv2.waitKey(0)