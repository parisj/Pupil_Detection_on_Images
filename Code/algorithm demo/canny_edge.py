import numpy as np
import cv2 

def gaussian_kernel(size, sigma):
    """create a gaussian kernel 

    Args:
        size (int): size kernel
        sigma (float): sigma of gaussian curve

    Returns:
        Hij (np.array): Kernel for gaussian blur
    """
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    scaling = 1 / (2 * np.pi * sigma ** 2)
    Hij = scaling * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
       
    return Hij

def sobel(img, val_x, val_y):
    """Use Sobel filter on img and return the magnitude and edge direction

    Args:
        img (np.array): contains values of img, greyscale
        val_x (np.array): ndim= (2,1), contains values for sobel in x
        val_y (np.array): ndim= (2,1), contains values for sobel in y

    Returns:
        Mag (np.array): Sobel filter applied in x and y direction magnitude
        theta(np.array): Edge Direction of Sobel filter
    """
    # Define kernel for Sobel in x direction 
    s_x = np.array([[-val_x[0], 0, val_x[0]],
                   [-val_x[1], 0, val_x[1]],
                   [-val_x[0], 0, val_x[0]]])
    print(s_x)
    # Define kernel for Sobel in y direction 
    s_y = np.array([[val_y[0], val_y[1], val_y[0]],
                   [0, 0, 0],
                   [-val_y[0], -val_y[1], -val_y[0]]])
    print(s_y)
    # Convolve img with Sobel kernel in x and y direction 
    S_x = cv2.filter2D(img,-1, s_x)
    cv2.imshow("s_x",S_x)
    
    
    S_y = cv2.filter2D(img,-1, s_y)
    cv2.imshow("s_y",S_y)
   
    # Calculate Magnitude
    Mag = np.sqrt(np.square(S_x) + np.square(S_y), dtype=np.float32)

    # Bring into value space [0, 255]
    Mag = Mag / Mag.max() * 255

    # Edge direction matrix
    theta = np.arctan2(S_y, S_x)
    
    return (Mag, theta)


def non_max_surpression(Mag, theta):
    
    M, N = Mag.shape
    Surp_M = np.zeros((M, N), dtype = np.float32)
    
    
    theta= theta * 180 / np.pi
    
    # make the angle positive, doesn't matter the direction, look at both neighbors 
    theta[theta < 0] += 180 
    
    for i in range(1,M-1):
        for j in range(1,N-1):
            cell1 = None
            cell2 = None
            phi= theta[i,j]
            
            # Horizontal
            if (0 <= phi <= 22.5) or (157.5 <= phi < 180):
                cell1= Mag[i, j-1]
                cell2= Mag[i,j+1]
                
            # Diagonal 1
            elif (22.5 <= phi < 67.5):
                cell1 = Mag[i+1, j-1]
                cell2 = Mag[i-1, j+1]
            
            # Vertical
            elif (67.5<= phi < 127.5):
                cell1 = Mag[i-1, j]
                cell2 = Mag[i+1, j]
            
            # Diagonal 2
            elif (127.5 <= phi < 157.5):
                cell1 = Mag[i-1, j+1]
                cell2 = Mag[i+1, j-1]
            
            if cell1 <= Mag[i,j] and cell2 <= Mag[i,j]:
                Surp_M[i, j] = Mag[i, j]
            
            else:
                Surp_M[i, j] = 0
    return Surp_M

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.float32)
    
    weak = np.float32(25)
    strong = np.float32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

if __name__ == "__main__":
    img = cv2.imread("Code/data_set/MMU-Iris-Database/8/right/eugenehor3.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_0 = img.copy()
    print(type(img))
    
    cv2.imshow("img",img)
   
    #Own Gaussian BLur
    #G_K = gaussian_kernel(11,1)
    #print(G_K)
    #img_blured = cv2.filter2D(img,-1, G_K)



    #opencv Gaussian blur
    img_blured = cv2.GaussianBlur(img, (5,5),0)
    
    cv2.imshow("img_blured",img_blured)
    
    Sx_y = cv2.Sobel(img_blured,-1,1,1,ksize=3)
    #Sy = cv2.Sobel(img_blured, -1, dx=0, dy=1, ksize=3)
    #S_total= np.sqrt(np.square(Sx) + np.square(Sy), dtype=np.float32)
    #S_tricked = np.uint8(Sx_y)

    #cv2.imshow("sobel opencv added",S_tricked)

    img_sobel, theta = sobel(img_blured, [1,2],[1,2])
    #cv2.imshow("img_sobel",img_sobel)
    img_canny = cv2.Canny(img_blured, 5, 70, 3)
    cv2.imshow("canny", img_canny)
    #circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 16, 70, param1=200, param2=250, minRadius= 0, maxRadius= 100)
    circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 2, 80, param1=30, param2=50, minRadius= 0, maxRadius= 30)

    if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        print(circles)
	    # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            
            # draw the circle in the output image, then draw a rectangle
		    # corresponding to the center of the circle
            cv2.circle(img_0, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img_0, (x, y), r, (0, 0, 255), 2)

    cv2.imshow("img_0", img_0)
    
 
    #surpr = non_max_surpression (S_tricked, theta)
    #cv2.imshow("non max surpression", surpr)
    
    
    #res, weak, strong = threshold (S_tricked)
    #cv2.imshow("threshold res",res)

    
 
    cv2.waitKey(0)