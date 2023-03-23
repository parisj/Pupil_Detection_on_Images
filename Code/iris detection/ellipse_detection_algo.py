import cv2
import numpy as np
import matplotlib.pyplot as plt
import create_plots as cp
import math 

def findEllipses(edges):
    # Create empty masks
    ellipseMask = np.zeros(edges.shape, dtype=np.uint8)
    contourMask = np.zeros(edges.shape, dtype=np.uint8)
    pi_4 = np.pi * 4
    ellipse_info = None
    
    # Find contours in preprocessed image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # iterate over all contours found
    for contour in contours:

        # filter out too short
        if len(contour) < 20:
            continue

        area = cv2.contourArea(contour)
        
        # filter out too small
        if area <= 500:  
            continue

        arclen = cv2.arcLength(contour, True)
        
        # Kullback–Leibler divergence
        circularity = (pi_4 * area) / (arclen * arclen)
        
        ellipse = cv2.fitEllipse(contour)
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)

        # if contour is circular enough
        if circularity > 0.6:
            cv2.fillPoly(ellipseMask, [poly], 255)
            ellipse_info = ellipse
            continue

        # if contour has enough similarity to an ellipse
        similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, 2, 0)
        if similarity <= 0.5:
            cv2.fillPoly(contourMask, [poly], 255)

    return ellipseMask, contourMask, ellipse_info

def extractEdges(gray, hsv):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closed', closed_mask)
    cv2.imshow('hsv', hsv)
    cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    _, thresh = cv2.threshold(gray.copy(), 50, 150, cv2.THRESH_BINARY_INV)
    
    gray_roi_mask, gray_roi, coords, peak_intensity = calc_roi(gray.copy(),hsv.copy(), 50, 30)
    
    if gray_roi_mask.size != 0:
        hsv_roi = gray_roi.copy()
        hsv_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_GRAY2BGR)
        hsv_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.GaussianBlur(gray_roi, (5,5), 0)
        
        h,s,v = cv2.split(hsv_roi)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        v = cv2.GaussianBlur(v, (3,3), 0)
        v_hat = cv2.Canny(v, 200,150)

        
        #cv2.imshow('mask', gray_roi_mask)
        #cv2.imshow('roi gray', gray_roi)
        edges = cv2.Canny(gray_roi, 45,120)
        combined_edges = cv2.bitwise_and(v_hat, edges)
        _, mask = cv2.threshold(combined_edges, 90, 170, cv2.THRESH_BINARY)
    else: 
        
        #print('not in threshhold')
        cp.plot_histogram(gray)
        edges = cv2.Canny(thresh, 70, 90)
        coords = None
    
        #cv2.imshow('edges1', edges)
        hsv_edges = cv2.Canny(cv2.GaussianBlur(hsv, (13, 13), 0), 130, 170)
    
        #cv2.imshow('hue_edges', hsv_edges)
 
   
        #combined_edges = cv2.bitwise_and(v_hat, edges)
    
  
        _, mask = cv2.threshold(edges, 90, 170, cv2.THRESH_BINARY)
    
    #cv2.imshow('mask threshold', mask)
    #cv2.imshow('erode',cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=1) )
    
    return cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=2), coords
    
def calc_roi(image, hsv, intensity_threshold=80, border = 20):
    # Convert the image to grayscale if it's not already
    #print('Start calc Roi')
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
        
    #cv2.imshow('gray_Image start ROI', gray_image)
    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Find the highest peak in the lower intensity region
    lower_intensity_hist = hist[:intensity_threshold]
    peak_intensity = np.argmax(lower_intensity_hist)
    #print('peak before',peak_intensity)
    # Create a binary mask with pixels around the peak intensity
    lower_bound = np.array([max(0, peak_intensity - 15)], dtype=np.uint8)
    upper_bound = np.array([min(255, peak_intensity + 20)], dtype=np.uint8)

        
    mask = cv2.inRange(gray_image, lower_bound, upper_bound)

    kernel = np.ones((3, 3), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('mask', closed_mask)
    #cv2.waitKey(0)
    # Get the contours of the ROI
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_similarity = 10
    max_circularity = 0
    max_contour = None
    best_ellipse = ((0,0),(0,0),0,0)
    for contour in contours:
        canvis = np.zeros_like(image)
        cv2.drawContours(canvis, contour, -1,(255,255,255),-1)
        cv2.imshow('canvis', canvis)
        #cv2.waitKey(0)
        BOOL_ELLIPSE, contourMask, ellipse, similarity, circularity = is_contour_ellipse(image, contour, 1)
        if BOOL_ELLIPSE: 
            if max_similarity > similarity and max_circularity < circularity:
                max_contour = contourMask
                best_ellipse = ellipse
                cv2.imshow('contourMask', contourMask)


        # Check if max_contour is None, and return appropriate values
    if max_contour is None:
        print('None found ')
        return np.array([]), np.array([]),(0,0,0,0,0), best_ellipse, 29
    
    # Get the bounding rectangle of the ROI
    x, y, w, h = cv2.boundingRect(max_contour)

    # Create a masked image with the ROI
    #roi_mask = np.zeros(gray_image.shape, dtype=np.uint8)
    #cv2.drawContours(roi_mask, [max_contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(gray_image, max_contour)

 
    #cv2.imshow('gray_image return calc', gray_image[y-border:y+h+border, x-border:x+w+border])
    #cv2.imshow('masked image return calc',masked_image[y-border:y+h+border, x-border:x+w+border] )
    #cv2.waitKey(0)
    #print('after',peak_intensity)
    # Return the masked ROI
    return masked_image[y-border:y+h+border, x-border:x+w+border], gray_image[y-border:y+h+border, x-border:x+w+border],(y,x,w,h, border), best_ellipse, peak_intensity

def is_contour_ellipse(gray_image, contour, threshold=0.05):
    
    
    contourMask = np.zeros(gray_image.shape, dtype=np.uint8)
    if len(contour) < 20:  # A minimum of 5 points is required to fit an ellipse
        return False, contourMask, ((0,0),(0,0),0,0), 10, 10
    
    img_gray = np.zeros_like(gray_image)
    ellipse = cv2.fitEllipse(contour)
    poly = cv2.ellipse2Poly((round(ellipse[0][0]), round(ellipse[0][1])), (round(ellipse[1][0] / 2), round(ellipse[1][1] / 2)), round(ellipse[2]), 0, 360, 1)
    similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, 2, 0)

    area = cv2.contourArea(contour)
     
     # filter out too small
    if area <= 500:  
        return False,contourMask,((0,0),(0,0),0,0), 10, 10
    arclen = cv2.arcLength(contour, True)
     
     # Kullback–Leibler divergence
    circularity = (4*np.pi * area) / (arclen * arclen)
    print('similarity',similarity, 'circularity', circularity)
    if similarity <= 0.6 and circularity >= 0.5:
        cv2.fillPoly(contourMask, [poly], 255)
        #cv2.imshow('contourMaks accepted', contourMask)
        #cv2.waitKey(0)
        return True, contourMask,ellipse, similarity, circularity
    

    
    #cv2.ellipse(img_gray, ellipse, (255,255,255), 1)
    #cv2.imshow('img_gray not accepted', img_gray)
  
   
    return  False,contourMask,((0,0),(0,0),0,0), 10, 10

def is_valid_ellipse(ellipse):
    center, axes, angle = ellipse
    if axes[0] <= 0 or axes[1] <= 0 or math.isnan(axes[0]) or math.isnan(axes[1]):
        return False
    if angle < 0 or angle > 360:
        return False
    return True

def haar_transformation (img, region ):
    kernel = [[]]