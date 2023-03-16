import cv2
import numpy as np


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
        
        # try to fit contour to ellipse
        ellipse = cv2.fitEllipse(contour)
        
        #create poly from ellipse for mask 
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)

        # if contour is circular enough
        if circularity > 0.6:
            cv2.fillPoly(ellipseMask, [poly], 255)
            ellipse_info = ellipse
            continue

        # if contour has enough similarity to an ellipse
        similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, 2, 0)
        if similarity <= 0.1:
            cv2.fillPoly(contourMask, [poly], 255)

    return ellipseMask, contourMask, ellipse_info

def extractEdges(gray, hsv):
    
    # generate Kernel for eroding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gray_copy = gray.copy()
    
    # Binary threshold onto the gray copy
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(11,11))
    gray_copy = clahe.apply(gray_copy)
    
    cv2.imshow('clahe', gray_copy)
    hsv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    _, gray_thresh = cv2.threshold(gray_copy, 50,200, cv2.THRESH_BINARY_INV) # (50, 150)
    
    gray_thresh = cv2.morphologyEx(gray_thresh, cv2.MORPH_CLOSE, kernel2)
    cv2.imshow('gray_thresh',gray_thresh)
    # Canny with focus on values to find pupil 
    edges = cv2.Canny(gray_thresh, 10, 95) #(65, 95)
    
    cv2.imshow('edges1', edges)
    hsv_edges = cv2.Canny(cv2.GaussianBlur(hsv, (13, 13), 0), 130, 170)
    cv2.imshow('hue_edges', hsv_edges)
    combined_edges = cv2.bitwise_or(hsv_edges, edges)
    
    # Threshold again 
    _, mask = cv2.threshold(combined_edges, 60, 190, cv2.THRESH_BINARY) # (90, 170)
    cv2.imshow('mask threshold', mask)
    #cv2.imshow('erode',cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=1) )
    return cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=1)
    