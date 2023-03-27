import numpy as np 
import cv2
#import create_plots as cp 
import math 
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
from Evaluate import Evaluation
from sklearn.cluster import KMeans

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
    observer.add_img_obj(iris_obj)
    
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
        

'''
--------------------------------------------------
   ______________  (6*Radius, 6*Radius)
  | 1            |
  |     ____     |
  |    |-8  |    | Center (2*Radius, 2*Radius) 
  |    |____|    |
  |              |
  |______________|
  
'''
def create_haar_kernel(radius):
    
    diameter = 6 * radius 
    kernel = np.ones((diameter, diameter), dtype=np.float32)

    #kernel = np.ones((diameter, diameter), dtype=np.float32)
    kernel[2*radius :4*radius , 2*radius:4*radius] = -6
    total = (6*radius)**2-(6*(2*radius)**2)
    kernel = kernel/total
    
    if DEBUG:
        print(kernel)
        
    return kernel 

def calculate_haar_convolution(image, radius):
    #image = cv2.GaussianBlur(image, (7,7),0)
    kernel = create_haar_kernel(radius)
    convolution = cv2.filter2D(image, -1, kernel)

    
    if DEBUG: 
        cv2.imshow('convolution', convolution)

    return convolution
 
def otsu_threshold(image,p1= 255, p2= 255 ):

    # Apply the binary threshold to the image
    _, thresholded_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    

    if DEBUG:
        cv2.imshow('otsu', thresholded_image)

        
    return thresholded_image


def kmeans_image_clutering(image, n_cluster):
    
    image_norm = image.astype(np.float32) /255.0
    pixels = image_norm.reshape((-1,1))
    kmeans = KMeans(n_clusters =n_cluster, random_state=42).fit(pixels)
    
    clustered_image = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered_image.reshape(image_norm.shape)
    
    clustered_image = (clustered_image *255).astype(np.uint8)
    
    return clustered_image, kmeans

def visualize_clusters(clustered_image, kmeans):
    # Generate random colors for each cluster
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(kmeans.cluster_centers_), 3), dtype=np.uint8)

    # Create a color image and set the pixel colors according to the cluster labels
    color_image = np.zeros((*clustered_image.shape, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        color_image[clustered_image == i] = color

    return color_image

def extract_connected(mask):
    mean_brightness = []
    
    num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        component_pixels = cv2.bitwise_and(mask, mask, mask= component_mask)
        num_pixels = np.sum(component_mask)
        total_brightness = np.sum(component_pixels)
        mean_brightness.append(total_brightness / num_pixels)
    
    if DEBUG: 
        print(f"Number of connected components: {num_labels}")
        print("Labels:")
        print(labels)
        print("Statistics:")
        print(stats)
        print("Centroids:")
        print(centroids)
        
        print("component intesities:")
        print(component_intensities)
        print("brightest shape")
        print(brightest_shape)
        
        cv2.imshow(brightest_shape_mask)
    
    return num_labels, labels,stats, centroids, mean_brightness

def calculate_integral_image(image):
    return  cv2.integral(image)

def haar_feature_value(integral_image, feature_rects):
    value = 0

    for rect in feature_rects:
        x, y, w, h, weight = rect
        a = integral_image[y - 1, x - 1]
        b = integral_image[y - 1, x + w - 1]
        c = integral_image[y + h - 1, x - 1]
        d = integral_image[y + h - 1, x + w - 1]

        value += weight * (a + d - b - c)

    return value

# Usage example
integral_image = calculate_integral_image("path/to/your/image.jpg")
print("Integral Image:\n", integral_image)

# Define a simple Haar-like feature (two vertical rectangles)
feature_rects = [
    (50, 50, 20, 40, 1),  # (x, y, width, height, weight) for the first rectangle (white region)
    (50, 90, 20, 40, -1)  # (x, y, width, height, weight) for the second rectangle (black region)
]

# Compute the feature value
feature_value = haar_feature_value(integral_image, feature_rects)
print("Haar-like feature value:", feature_value)
    
def display_connected_shapes(image):

    # Find connected components in the thresholded image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8))

    # Draw bounding boxes around each connected shape on a copy of the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):  # Skip the first component (background)
        x, y, w, h, area = stats[i]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image
    
    


def main_video():
    for frame in get_video_frame('1.avi'):
        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #gray_eye_image = clahe.apply(gray_eye_image)
        
        convolution = calculate_haar_convolution(gray_eye_image,30)
        cv2.imshow('gray_eye afert Clahe and normal',gray_eye_image)
        otsu = otsu_threshold(convolution)
        cv2.imshow('convolution', convolution)
        colormap = cv2.applyColorMap(convolution,cv2.COLORMAP_HOT)
        con_img = display_connected_shapes(convolution)


        cv2.imshow('colormap', colormap )
        
    
 
        cv2.imshow('otsu', otsu )
        # num_labels, labels, stats, centroids, bmask = extract_connected(otsu)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break
    cv2.waitKey(0)
    
def main_image():
    
    frame = cv2.imread('eye_img_22.png')

    gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    convolution = calculate_haar_convolution(gray_eye_image,30)
    
    otsu = otsu_threshold(convolution)
    cv2.imshow('convolution', convolution)
    colormap = cv2.applyColorMap(convolution,cv2.COLORMAP_HOT)
    con_img = display_connected_shapes(convolution)
    
    cv2.imshow('otsu', otsu )
    cv2.imshow('colormap', colormap )
    #num_labels, labels, stats, centroids, bmask = extract_connected(otsu)
    #cv2.imshow('mask', bmask)

    cv2.waitKey(0)
    

if __name__ == '__main__':
    main_video()
    #test()
    cv2.waitKey(0)