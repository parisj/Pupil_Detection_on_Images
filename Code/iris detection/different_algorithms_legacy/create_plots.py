import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import ellipse_detection_algo as ellip


def plot_ellipse(image, ellipse):

    if image is None:
        print('image none', image)
        return False
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pupil_center = (round(ellipse[0][0]),round(ellipse[0][1]))
    pupil_axis = (round(ellipse[1][0]/2),round(ellipse[1][1]/2))
    pupil_angle = int(ellipse[2])
    
    #print ('plot',ellipse)
    
    result = cv2.ellipse(image, pupil_center,pupil_axis, pupil_angle, 0,360,(0,255,0),1)

    cv2.imshow('result ellipse', result)

    return True
    
def plot_histogram(image, roi, peak_threshold ):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image = clahe.apply(image)
    
    #cv2.imshow('masked_small',masked_small )
    #cv2.imshow('roi',roi )
    #cv2.imshow('coords',coords )
    #cv2.waitKey(0)
    #print('peak_threshold', peak_threshold)
    intensity_range= (peak_threshold-10, peak_threshold +10)
    flat_image_array = image.flatten()

    # Create a mask for the intensity range in the image
    mask = np.zeros_like(image, dtype=np.uint8)


    # Color the original image with the mask (highlight the intensity range)
    colored_image = cv2.merge((image, image, image))
    colored_image[mask == 255] = (0, 255, 0)

  # Calculate histograms for both images
    if image.size==0 or roi.size==0: 
        #print('uups')
        return 0
    max_intensity = max(image.max(), roi.max())
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
    ax2.imshow(roi, cmap='gray')
    ax2.set_title('ROI Image')
    ax2.axis('off')

 
    # Plot the histograms of both images
    sns.histplot(image.flatten(), ax=ax3, element='bars', bins=bin_edges, color='gray', alpha=0.5,edgecolor=None, label='Original Image' )
    sns.histplot(roi.flatten(), ax=ax3, element='bars', bins=bin_edges, color='black', alpha=1,edgecolor=None, label='ROI')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('Histogram')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Count')
    ax3.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    
    plt.show()
    fig.savefig('Latex/thesis/plots/histogram_with_roi.png')
    
def plot_canny(img):
    img_copy = img.copy()
    img_orig = img.copy()
    img_copy = cv2.GaussianBlur(img_copy, (5,5), 0)
    img_copy = cv2.Canny(img_copy, 20, 60)
    cv2.imshow('canny', img_copy)
    cv2.imshow('orig', img_orig)
    
    cv2.imwrite('Latex/thesis/plots/canny_eyelids.png', img_copy)
    cv2.imwrite('Latex/thesis/plots/orig_canny_eyelids.png', img_orig)
    cv2.waitKey(0)
    
def plot_clahe(img):
    img_copy = img.copy()
    img_clahe = img.copy()
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(11,11))
    img_clahe = clahe.apply(img_clahe)

    gridsize = (2,3)
    fig = plt.figure(figsize=(9,6))
    ax1 = plt.subplot2grid(gridsize,(0,0))
    ax2 = plt.subplot2grid(gridsize,(0,1), colspan=2, rowspan = 1)
    ax3 = plt.subplot2grid(gridsize,(1,0))
    ax4 = plt.subplot2grid(gridsize,(1,1), colspan= 2, rowspan = 1)
 
    ax1.set_title('Original Image', fontsize= 10)
    ax3.set_title('CLAHE Image',fontsize= 10)
    ax2.set_title('Histogram Original Image',fontsize= 10)
    ax4.set_title('Histogram CLAHE Image',fontsize= 10)

    ax1.imshow(img_copy, cmap='gray')
    ax3.imshow(img_clahe, cmap='gray')
    sns.histplot(img_copy.flatten(), element='bars', bins=256, color='black', alpha=0.8, edgecolor=None, label='Original Image', ax= ax2)
    sns.histplot(img_clahe.flatten(), element='bars', bins=256, color='black', alpha=0.8, edgecolor=None, label='Original Image', ax= ax4)
    
    ax1.set_axis_off() 
    ax3.set_axis_off()
    
    axes = [ax2,ax4]
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_ylim([0,6000])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Intensity', fontsize= 8)
        ax.set_ylabel('Count Pixels', fontsize= 8)
    plt.tight_layout()
    
    plt.show()
    
    
    
if __name__ == '__main__':
    test = cv2.imread('test_frame_eyelids.png')
    roi = cv2.imread('test_roi.png')
    
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('teest,', test) 
    test = cv2.resize(test, (round(test.shape[1]/2), round(test.shape[0]/2)), interpolation = cv2.INTER_LINEAR)
    mask = np.zeros_like(test, dtype=np.uint8)
    mask[(test >=  100)] = 255
    # Color the original image with the mask (highlight the intensity range)
    colored_image = cv2.merge((test, test, test))
    colored_image[mask == 255] = (0, 255, 0)
    #cv2.imshow('colored_image', colored_image)
    #plot_clahe(test)
    #plot_histogram(test, roi, 0)
    #cv2.waitKey(0)
    plot_canny(test)