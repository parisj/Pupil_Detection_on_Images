import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import ellipse_detection_algo as ellip

def plot_histogram(image, threshhold = None  ):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image = clahe.apply(image)
    masked_small, gray_small, coords,ellipse, peak_threshold =  ellip.calc_roi(image, 50, 30)
    #cv2.imshow('masked_small',masked_small )
    #cv2.imshow('gray_small',gray_small )
    #cv2.imshow('coords',coords )
    #cv2.waitKey(0)
    #print('peak_threshold', peak_threshold)
    intensity_range= (peak_threshold-10, peak_threshold +10)
    flat_image_array = image.flatten()

    # Create a mask for the intensity range in the image
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[(image >= intensity_range[0]) & (image <= intensity_range[1])] = 255

    # Color the original image with the mask (highlight the intensity range)
    colored_image = cv2.merge((image, image, image))
    colored_image[mask == 255] = (0, 255, 0)

  # Calculate histograms for both images
    if image.size==0 or gray_small.size==0: 
        #print('uups')
        return 0
    max_intensity = max(image.max(), gray_small.max())
    bin_edges = np.linspace(0, max_intensity, 257)
 

    # Create a figure with 2 subplots in a single row
    gridsize = (2,2)
    fig = plt.figure(figsize=(9,6))
    ax3 = plt.subplot2grid(gridsize,(0,0), colspan= 2, rowspan = 1)
    ax1 = plt.subplot2grid(gridsize,(1,0))
    ax2 = plt.subplot2grid(gridsize,(1,1))

    
    # Plot the gray image with the ROI highlighted
    ax1.imshow(colored_image)
    ax1.set_title('Gray Image with ROI')
    ax1.axis('off')

    # Plot the ROI image
    ax2.imshow(gray_small, cmap='gray')
    ax2.set_title('ROI Image')
    ax2.axis('off')

 
    # Plot the histograms of both images
    sns.histplot(image.flatten(), ax=ax3, element='bars', bins=bin_edges, color='gray', alpha=0.5,edgecolor=None, label='Original Image' )
    sns.histplot(gray_small.flatten(), ax=ax3, element='bars', bins=bin_edges, color='black', alpha=1,edgecolor=None, label='ROI')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('Histogram')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Count')
    ax3.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    test = cv2.imread('test.jpg')
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('teest,', test) 
    mask = np.zeros_like(test, dtype=np.uint8)
    mask[(test >=  100)] = 255
    # Color the original image with the mask (highlight the intensity range)
    colored_image = cv2.merge((test, test, test))
    colored_image[mask == 255] = (0, 255, 0)
    #cv2.imshow('colored_image', colored_image)

    plot_histogram(test)
    #cv2.waitKey(0)
   