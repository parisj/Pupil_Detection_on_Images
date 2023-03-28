import cv2
import numpy as np
from sklearn.cluster import KMeans
import ellipse_detection_algo as ellip

def create_haar_kernel(radius):
    diameter = 6 * radius 
    kernel = np.ones((diameter, diameter), dtype=np.float32)
    kernel[2*radius :4*radius , 2*radius:4*radius] = -8
    total = (6*radius)**2-(8*(2*radius)**2)
    kernel = kernel/total
    return kernel

radius = 35 # Define the radius for the Haar-like center-surround feature
haar_kernel = create_haar_kernel(radius)
print(haar_kernel)

# Load the eye image and convert it to grayscale
eye_image = cv2.imread("test.jpg")
gray_eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)

# Convolve the image with the Haar-like kernel
convolved_image = cv2.filter2D(gray_eye_image, -1, haar_kernel)
cv2.imshow('convolved image', convolved_image)
contours,_ = cv2.findContours(convolved_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
max_simi = -1
max_circularity = -1
max_brightnes =-1
max_contour = None

for contour in contours:
    canvis = np.zeros_like(gray_eye_image)
    cv2.drawContours(canvis, contour, -1,(255,255,255),-1)
    cv2.imshow('canvis', canvis)
    BOOL_ELLIPSE, contourMask, ellipse, similarity, circularity= ellip.is_contour_ellipse(gray_eye_image, contour, 1)
    if contourMask is None :
        break 
    masked_img = cv2.bitwise_and(contourMask, convolved_image)
    cv2.imshow('masked image', masked_img)
    brightnes = masked_img[masked_img > 0].mean()
    #cluster = np.argwhere(contourMask)
    #cv2.imshow('cluster argwhere', cluster)
    #brightnes = np.mean(cluster)
    print(brightnes)
    if max_brightnes < brightnes or (max_simi> similarity and max_circularity < circularity):
        max_simi = similarity
        max_circularity = circularity
        max_brightnes = brightnes
        max_contour = contour
        
    cv2.imshow('contourmaks', contourMask)
    cv2.waitKey(0)
    
# Apply Otsu's thresholding on the convolved image
#_, otsu_thresholded = cv2.threshold(convolved_image, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#cv2.imshow('thresholded', otsu_thresholded)
# Multiply the thresholded image with the mask to filter out the less intense white spots
#filtered_mask = cv2.bitwise_and(otsu_thresholded, convolved_image)
#cv2.waitKey(0)
# Find the connected components in the filtered mask
#num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_mask.astype(np.uint8))

# Find the label with the largest area, excluding the background (label 0)
#largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

# Create a new mask with only the largest connected component
#refined_mask = (labels == largest_component).astype(np.uint8)

# Display the refined mask
#cv2.imshow("Refined Mask", refined_mask * 255)

data = convolved_image.reshape(-1,1)
data = data.astype(np.float32) / 255.0


#kmeans = KMeans(n_clusters=2)
#labels = kmeans.fit_predict(data)
#cluster_intensities = kmeans.cluster_centers_.squeeze()
#print('cluster intensities', cluster_intensities)
#index_max_intensities = np.argmax(cluster_intensities)
#print(index_max_intensities)


# Find the approximate pupil region
#(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(convolved_image)
#pupil_center = maxLoc
#
border_size = 20  # Define the size of the border

x, y, w, h = cv2.boundingRect(max_contour)
x1_hat = x - border_size
y1_hat = y - border_size
x2_hat = x + w + border_size
y2_hat = y + h + border_size
green_color = (0, 255, 0)
line_thickness = 2
cv2.rectangle(eye_image, (x1_hat, y1_hat), (x2_hat, y2_hat), green_color, line_thickness)

#mask = (labels == 1).reshape(convolved_image.shape)

# Apply the binary mask to the original image
#filtered_image = np.zeros_like(gray_eye_image)
#filtered_image[mask] = gray_eye_image[mask]

# Display the images
cv2.imshow("Original Image", gray_eye_image)
cv2.imshow("Convolved Image", convolved_image)
#cv2.imshow("Filtered Image", filtered_image)
cv2.imshow("Image with Green Border", eye_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
