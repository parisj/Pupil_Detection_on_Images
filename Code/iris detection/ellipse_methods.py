import numpy as np 
import cv2
import create_plots as cp 
import math 
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
from Evaluate import Evaluation
from sklearn.cluster import KMeans

DEBUG = False

'''
   ______________  (6*Radius, 6*Radius)
  | 1            |
  |     ____     |
  |    |-8  |    | Center (2*Radius, 2*Radius) 
  |    |____|    |
  |              |
  |______________|
  
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
        


def create_haar_kernel(radius):
    
    diameter = 6 * radius 
    kernel = np.ones((diameter, diameter), dtype=np.float32)
    kernel[2*radius :4*radius , 2*radius:4*radius] = -8
    total = (6*radius)**2-(8*(2*radius)**2)
    kernel = kernel/total
    
    if DEBUG:
        print(kernel)
        
    return kernel 

def calculate_haar_convolution(image, radius):

    kernel = create_haar_kernel(radius)
    convolution = cv2.filter2D(image, -1, kernel)
    convolution = cv2.GaussianBlur(convolution, (11,11), 0)
    
    if DEBUG: 
        cv2.imshow('convolution', convolution)

    return convolution
 
def otsu_threshold(image,p1= 255, p2= 255 ):
      # Apply Otsu's thresholding method to obtain the threshold value
    _, otsu_threshold_value = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Set the binary threshold to the Otsu's threshold value
    binary_threshold = np.float64(otsu_threshold_value)

    # Apply the binary threshold to the image
    _, thresholded_image = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY)

    

    if DEBUG:
        cv2.imshow('otsu', otsu_threshold)

        
    return thresholded_image

def extract_connected(mask):
    num_labels, labels,stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    component_intensities = [np.mean(mask[labels == i]) for i in range(1, num_labels)]
    brightest_shape = np.argmax(component_intensities)+1
    brightest_shape_mask = (labels == brightest_shape).astype(np.uint8) * 255
     
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
    
    return num_labels, labels,stats, centroids, brightest_shape_mask


    
def display_connected_shapes(image):

    # Find connected components in the thresholded image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8))

    # Draw bounding boxes around each connected shape on a copy of the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):  # Skip the first component (background)
        x, y, w, h, area = stats[i]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image
    
    


def main():
    for frame in get_video_frame('D:/data_set/LPW/1/1.avi'):
        
        gray_eye_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        convolution = calculate_haar_convolution(gray_eye_image, 30)
        otsu = otsu_threshold(convolution)
        cv2.imshow('convolution', convolution)
        con_img = display_connected_shapes(convolution)
        
    
 
        cv2.imshow('otsu', otsu )
        num_labels, labels, stats, centroids, bmask = extract_connected(otsu)
        cv2.imshow('mask', bmask)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break
    cv2.waitKey(0)
    
def test():


    # Create a binary image (use your own binary image or threshold a grayscale image)
    binary_image = np.array([
        [0, 0, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 0, 0, 0],
        [0, 255, 255, 255, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    # Find connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Print results
    print(f"Number of connected components: {num_labels}")
    print("Labels:")
    print(labels)
    print("Statistics:")
    print(stats)
    print("Centroids:")
    print(centroids)
    
if __name__ == '__main__':
    main()
    #test()
    cv2.waitKey(0)