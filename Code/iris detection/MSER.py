import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

def different_thresholds(image):
    result = []
    for i in range(0,80, 10):
        result.append(cv2.threshold(image, i, 255, type = cv2.THRESH_BINARY_INV))
        
    for i, img in enumerate(result):
        print(i)
        print(img[1])
        cv2.imshow('img'+str(i),img[1])
        
    cv2.waitKey(0)


def mser(image):
    vis =image.copy()
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(image)

    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(vis, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    return vis
    
    
    
def vector_field(image):
    img_x = cv2.Sobel(image,cv2.CV_64F, 1, 0 )
    img_y = cv2.Sobel (image,cv2.CV_64F, 0, 1)

    direction = np.arctan2(img_y,img_x)*180/np.pi % 180
    direction = cv2.convertScaleAbs(direction)
    magnitude = np.sqrt(img_x**2 + img_y**2, dtype = np.float32)
    magnitude = cv2.convertScaleAbs(magnitude)
    #direction = cv2.cvtColor(direction,cv2.COLOR_GRAY2BGR)
    #magnitude = cv2.cvtColor(magnitude,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('direction', direction)
    #cv2.imshow('magnitude,',magnitude)
    
    
    
    # initialize a figure to display the input grayscale image along with
    # the gradient magnitude and orientation representations, respectively
    (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    # plot each of the images
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(magnitude, cmap="hot")
    axs[2].imshow(direction, cmap="hot")
    # set the titles of each axes
    axs[0].set_title("Grayscale")
    axs[1].set_title("Gradient Magnitude")
    axs[2].set_title("Gradient Orientation [0, 180]")
    # loop over each of the axes and turn off the x and y ticks
    for i in range(0, 3):
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])
    # show the plots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    image = cv2.imread('test.jpg')
    coords = (334,305)
    x = coords[0]
    y = coords[1]
    size = 110
    top_corner =  (x - size, y - size)
    bottom_corner = (x + size, y + size)
    
    image = image[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    vector_field(image)
    #different_thresholds(image)
    #cv2.imshow('mser',mser(image))
    cv2.waitKey(0)