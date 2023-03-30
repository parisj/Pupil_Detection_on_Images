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
    evaluation_obj = Evaluation(pupil_obj,'results.csv','results/')

    observer.add_img_obj(pupil_obj)   
    return pupil_obj, iris_obj, observer, evaluation_obj

def get_video_frame(video_path):
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("EOF or Video not opened correctly")
        return True
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        yield frame
        
    cap.release()
    cv2.destroyAllWindows()                    
        
    #for frame in get_video_frames(video_path):
    #    process_frame(frame)  # Call your custom frame processing method
    

def haar_roi_extraction( image, plot):
    
    Haar_kernel = HaarFeature(8, 3, image)
    coords, roi= Haar_kernel.find_pupil_ellipse(plot)
    #print('roi', roi)
    return  coords, roi

def otsu(roi):
    _, otsu = cv2.threshold(roi,0,255,cv2.THRESH_OTSU)
    return otsu

def threshold_ellipse(roi, intensity):
    #print('intensity',intensity)
    #print(roi.shape)
    thresholded = cv2.inRange(roi, int(intensity/5), int(intensity*2.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN, kernel)
    #thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('thresholded', thresholded)

    return thresholded

def main_Haar():               
    
    #Load frame by frame to process
    for frame in get_video_frame('D:/data_set/LPW/1/4.avi'):
        
        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
        gray_eye_image = clahe.apply(gray_eye_image)
        
        coords, roi = haar_roi_extraction(gray_eye_image, plot= True)
        #print('coords :', coords  )
        #print('roi', roi)
        intensity = gray_eye_image[coords[1], coords[0]]
    
        edges = threshold_ellipse(roi, intensity)
        cv2.imshow('edges', edges)
        pupil = eda.best_ellipse(edges)
        #print(roi, pupil)
        if pupil is None:

            continue
        cp.plot_ellipse(roi, pupil)

        #kmean(roi)

        
        
        xy_1 = (int(coords[0]- 90), int(coords[1]-90))
        xy_2 = (int(coords[0]+90), int(coords[1]+90))
        
        cv2.rectangle(frame,xy_1, xy_2, (255,255,50), 1 )
        cv2.imshow('result', frame)
        cv2.imshow('roi', roi)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break
    cv2.waitKey(0)
    
def main_Haar_image():
    frame = cv2.imread('eye_img_22.png', cv2.IMREAD_GRAYSCALE)
    print(frame.shape)
    gray_eye_image=cv2.resize(frame,(int(frame.shape[1]*60/100),int(frame.shape[0]*60/100)), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
    gray_eye_image = clahe.apply(gray_eye_image)
    coords, roi = haar_roi_extraction(gray_eye_image, plot= True)
    print(coords)
    intensity = gray_eye_image[coords[1], coords[0]]
    
    extract_ellipse(roi, intensity)
    kmean(roi)
    cv2.imshow('frame',gray_eye_image)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main_Haar()
    #main_Haar_image()
    cv2.waitKey(0)