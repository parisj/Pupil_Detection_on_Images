import numpy as np 
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.filters as filters
import skimage.segmentation as seg


def _init_circle(center,radius, image_shape):
    mask = np.zeros(image_shape)
    circle = cv2.circle(mask, center, radius, 255, thickness = 1)
    points = np.argwhere(mask > 0 )
    return mask, points

def _create_P(N, alpha, beta, gamma):
    a = gamma*(2*alpha+6*beta)+1
    b = gamma*(-alpha-4*beta)
    c = gamma*beta

    P = np.diag(np.full(N, a))
    P += np.diag(np.full(N-1, b), 1) + np.diag(np.full(1, b), -N+1)
    P += np.diag(np.full(N-1, b), -1) + np.diag(np.full(1, b), N-1)
    P += np.diag(np.full(N-2, c), 2) + np.diag(np.full(2, c), -N+2)
    P += np.diag(np.full(N-2, c), -2) + np.diag(np.full(2, c), N-2)
    print(P)
    P_inv = np.linalg.inv(P)
    return P_inv

def _gradient(image):
    g_x, g_y = np.gradient(image)




    return g_x, g_y



def F_ex(points, g_x, g_y):
    f_x = g_x[points[:, 0], points[:, 1]]
    f_y = g_y[points[:, 0], points[:, 1]]
    return f_x, f_y


def plot_points_on_image(points, image, color=(0, 255, 0), thickness=2):
    for point in points:
        cv2.circle(image, tuple(point[::-1]), thickness, color, -1)
    return image

def step(points, g_x, g_y, P, gamma):
    f_x, f_y = F_ex(points, g_x, g_y)

    # Update the coordinates of the points and apply the P matrix
    points[:, 0] = (P @ (points[:, 0] + gamma * f_x))
    points[:, 1] = (P @ (points[:, 1] + gamma * f_y))

    return points

def test():
    iterations = 100
    image = cv2.imread('test_roi.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3),0)
    circle, points = _init_circle((100,100), 15, gray.shape)
    print(f'points: {points.shape}, {points}, {type(points)}')
    N = np.count_nonzero(circle)
    P= _create_P(N, 0.1, 0.4, 1)
    g_x, g_y = _gradient(gray)
    


    
    cv2.imshow('g_x', g_x)
    cv2.imshow('g_y', g_y)
    
    cv2.imshow('u', circle)
    start = plot_points_on_image(points, image.copy())
    cv2.imshow('start',start)
    cv2.waitKey(0)
    for i in range(iterations):
        points = step(points, g_x, g_y, P, 1)
        result_image = plot_points_on_image(points, image.copy())

        # Show the updated image
        cv2.imshow('Result', result_image)
        cv2.waitKey(0)
        
    cv2.imshow('Original', image)

    cv2.waitKey(0)  
    cv2.destroyAllWindows()
if __name__ == '__main__':
    
    test()
    
