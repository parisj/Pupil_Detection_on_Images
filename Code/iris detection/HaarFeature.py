import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from concurrent.futures import ThreadPoolExecutor
from numba import njit
class HaarFeature: 
    def __init__ (self, maxRadius, minRadius, image):

        self.maxRadius = maxRadius
        self.minRadius = minRadius
        self.image = image
        self.roi = None
        self.coords = None

    def extract_roi(self,coords, border):
        self.coords = coords
        x = coords[0]
        y = coords[1]
        top_corner =  (x - border, y - border)
        bottom_corner = (x + border, y + border)
        self.roi = self.image[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]]
        return self.image[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]]
        
    def find_pupil_ellipse(self, plot=False):
        
        response_image = np.zeros(self.image.shape, dtype=np.float64)

        eye_integral = np.zeros((self.image.shape[0] + 2 * self.maxRadius + 1,
                                 self.image.shape[1] + 2 * self.maxRadius + 1), dtype=np.int32)

        padding = 2 * self.maxRadius
        eye_pad = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        eye_integral = cv2.integral(eye_pad, eye_integral)

        position_pupil = (0, 0)

        min_response = np.inf

        with ThreadPoolExecutor() as executor:
            args = [(r, eye_integral, padding, self.image.shape, response_image) for r in range(self.minRadius, self.maxRadius, 2)]
            results = list(executor.map(process_radius, args))

        for min_radius_response, min_radius_position, r in results:
            if min_radius_response < min_response:
                min_response = min_radius_response
                position_pupil = min_radius_position
        


        normalized_response = cv2.normalize(response_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        if plot:
            colormap = cv2.COLORMAP_HOT
            normalized_response = cv2.GaussianBlur(normalized_response, (3,3),0)

            response_image_color = cv2.applyColorMap(normalized_response, colormap)
            cv2.imshow('Response Image', response_image_color)
        roi = self.extract_roi(position_pupil, 110)
        return position_pupil, roi
@njit
def process_radius(args):
    r, eye_integral, padding, img_shape, response_img = args
    r_inner = r
    r_outer = 3 * r
    min_radius_response = np.inf
    min_radius_position = (0, 0)
    for y in range(r, img_shape[0] - r, 4):
        for x in range(r, img_shape[1] - r, 4):
            sum_inner = (eye_integral[y + padding - r_inner, x + padding - r_inner]
                         + eye_integral[y + padding + r_inner + 1, x + padding + r_inner + 1]
                         - eye_integral[y + padding - r_inner, x + padding + r_inner + 1]
                         - eye_integral[y + padding + r_inner + 1, x + padding - r_inner])
            sum_outer = (eye_integral[y + padding - r_outer, x + padding - r_outer]
                         + eye_integral[y + padding + r_outer + 1, x + padding + r_outer + 1]
                         - eye_integral[y + padding - r_outer, x + padding + r_outer + 1]
                         - eye_integral[y + padding + r_outer + 1, x + padding - r_outer]
                         - sum_inner)
            response = -r_inner * sum_inner + r_outer * sum_outer
            response_img[y,x] = response
            if response < min_radius_response:
                min_radius_response = response
                min_radius_position = (x, y)
    return min_radius_response, min_radius_position, r
