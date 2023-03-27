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
    hist = creat_histogram(roi)
    data = np.arange(256).reshape(-1,1)
    hist_reshaped = hist.reshape(-1,1)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(hist_reshaped)
    cluster_labels = kmeans.predict(hist_reshaped)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    clustered_hist = np.zeros_like(hist)
    for i in range(clusters):
        clustered_hist[labels == i ] = centroids[i]
        
    plot_histogram(roi, clustered_hist, cluster_labels, hist_reshaped, clusters )
    return clustered_hist
    
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
    cv2.imshow("Original Image", roi)
    cv2.imshow("Clustered Image", colored_image)
     # Plot the histogram with colored bins
    bin_colors = [colors[i] for i in cluster_labels]
    bin_colors = [(c[2] / 255, c[1] / 255, c[0] / 255) for c in bin_colors]  # Convert BGR to RGB

    plt.figure()
    sns.barplot(x=np.arange(0, 256), y=clustered_hist, palette=bin_colors)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title("K-means Clustered Histogram")
    plt.show()

def create_haar_kernel(radius, image, plot):
    
    Haar_kernel = HaarFeature(8, 3, image)
    coords, roi= Haar_kernel.find_pupil_ellipse(plot)
    return  coords, roi



    
def main_Haar():
    for frame in get_video_frame('D:/data_set/LPW/1/4.avi'):
        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
        gray_eye_image = clahe.apply(gray_eye_image)
        coords, roi = create_haar_kernel(10,gray_eye_image, plot= True)
        print(coords)
        kmean(roi)
       
        
        
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