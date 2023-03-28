import numpy as np 
import cv2
import create_plots as cp 
import math 
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
from Evaluate import Evaluation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from HaarFeature import HaarFeature
'''
IMPORTANT FLAG FOR DEBUGGING
----------------------------------------
'''
DEBUG = False
'''
---------------------------------------
'''

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


def creat_histogram(image, bins = 256):
    hist = cv2.calcHist([image], [0], None, [bins], [0,256])
    hist = cv2.normalize(hist,hist).flatten()
    return hist


def kmean(roi, clusters = 3):
    #hist = creat_histogram(roi)
    hist, bins = np.histogram(roi.ravel(), bins = range(256))
    bins = bins[:-1]
    #data = np.column_stack((bins,hist))
    data = bins.reshape(-1,1)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(data)
    roi = otsu(roi)
    cv2.imshow('otsu', roi )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    roi = cv2.morphologyEx(roi,cv2.MORPH_OPEN, kernel)
    roi = cv2.morphologyEx(roi,cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closed', roi )
    roi = cv2.Canny(roi,10, 20)
    cv2.imshow('canny', roi )

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    if centroids[0,0] > centroids[1,0]:
        if centroids[1,0] > centroids[2,0]:
            light_cluster = 0
            dark_cluster = 2
            gray_cluster = 1
        else: 
            light_cluster = 0
            dark_cluster = 1
            gray_cluster = 2

    elif centroids[1,0] > centroids[0,0]:
        if centroids[0,0] > centroids[2,0]:
            light_cluster = 1
            dark_cluster = 2
            gray_cluster = 0
        else:
            light_cluster = 1
            dark_cluster = 0
            gray_cluster = 2
    else: 
        light_cluster = 2
        if centroids[0,0] > [centroids[1,0]]:
            gray_cluster = 0
            dark_cluster = 1
            
        else: 
            gray_cluster = 1
            dark_cluster = 0 
    
    light_data = hist[kmeans.labels_ == light_cluster]
    dark_data =  hist[kmeans.labels_ == dark_cluster]
    light_bins = bins[kmeans.labels_ == light_cluster]
    dark_bins = bins[kmeans.labels_ == dark_cluster]
    
    gray_data = hist[kmeans.labels_ == gray_cluster]
    gray_bins = bins[kmeans.labels_ == gray_cluster]
    
    
    #plt.figure(figsize=(10,5))
    #plt.bar(bins,hist,color='gray', label='Original Histogram')
    #plt.bar(light_bins,light_data,color='green', label='light')
    #plt.bar(dark_bins,dark_data,color='yellow', label='dark')
    #plt.bar(gray_bins,gray_data,color='red', label='grey')
    #plt.legend()
    #plt.show()
    #plot_histogram(roi, clustered_hist, cluster_labels, hist_reshaped, clusters )
    return 


def plot_histogram(roi, clustered_hist, cluster_labels, flat_image, n_clusters):
    # Define colors for each cluster (BGR format for OpenCV)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Add more colors if you have more clusters

    # Create a lookup table (LUT) for the labels
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(n_clusters):
        lut[cluster_labels == i] = colors[i]

    # Convert the roi image to a 3-channel image
    roi_colored = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Apply the LUT to the roi image
    colored_image = cv2.LUT(roi_colored, lut)

    # Show the original image and the clustered image

    cv2.imshow("Clustered Image", colored_image)
    cv2.imshow('thresholded', thresh)
     # Plot the histogram with colored bins
    bin_colors = [colors[i] for i in cluster_labels]
    bin_colors = [(c[2] / 255, c[1] / 255, c[0] / 255) for c in bin_colors]  # Convert BGR to RGB
    bin_edges = np.linspace(0, 256, 257)

    fix, (ax1, ax2) = plt.subplots(1,2)
    sns.histplot(roi.flatten(), ax=ax1, element='bars', bins=bin_edges, color='gray', alpha=0.5)
    sns.barplot(x=np.arange(0, 256),ax=ax2, y=clustered_hist, palette=bin_colors)
    
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax1.set_xlabel('Intensity ROI')
    ax1.set_ylabel('Frequency')
    ax1.set_title('ROI ')
    ax2.set_title('Result Clustering')
    
    plt.tight_layout()
    plt.show()

def create_haar_kernel(radius, image, plot):
    
    Haar_kernel = HaarFeature(8, 3, image)
    coords, roi= Haar_kernel.find_pupil_ellipse(plot)
    return  coords, roi

def otsu(roi):
    _, otsu = cv2.threshold(roi,0,255,cv2.THRESH_OTSU)
    return otsu

def extract_ellipse(roi, intensity):
    print('intensity',intensity)
    thresholded = cv2.inRange(roi, int(intensity/5), int(intensity*2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN, kernel)
    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_CLOSE, kernel)
    cv2.imshow('thresholded', thresholded)
    thresholded = cv2.Canny(thresholded,10, 20)
    cv2.imshow('canny', thresholded )

    
    
def main_Haar():
    for frame in get_video_frame('E:/data_set/LPW/7/18.avi'):
        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
        gray_eye_image = clahe.apply(gray_eye_image)
        coords, roi = create_haar_kernel(10,gray_eye_image, plot= True)
        print(coords)
        intensity = gray_eye_image[coords[1], coords[0]]
    
        extract_ellipse(roi, intensity)
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
    


if __name__ == '__main__':
    main_Haar()
    cv2.waitKey(0)