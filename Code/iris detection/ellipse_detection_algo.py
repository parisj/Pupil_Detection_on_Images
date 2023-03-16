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
        
        ellipse = cv2.fitEllipse(contour)
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
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    working = gray.copy()
    _, working = cv2.threshold(working, 50, 150, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(working, 70, 90)
    #cv2.imshow('edges1', edges)
    hsv_edges = cv2.Canny(cv2.GaussianBlur(hsv, (13, 13), 0), 130, 170)
    #cv2.imshow('hue_edges', hsv_edges)
    combined_edges = cv2.bitwise_or(hsv_edges, edges)
    _, mask = cv2.threshold(edges, 90, 170, cv2.THRESH_BINARY)
    #cv2.imshow('mask threshold', mask)
    #cv2.imshow('erode',cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=1) )
    return cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=1)
    