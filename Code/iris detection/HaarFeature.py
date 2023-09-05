import numpy as np 
import cv2
from concurrent.futures import ThreadPoolExecutor
from numba import njit
import exceptions



class HaarFeature: 
        
    def __init__ (self, maxRadius, minRadius, image):

        # maxRadius that will be considered
        self.maxRadius = maxRadius
        # minRadius that will be considered
        self.minRadius = minRadius
        self.image = image
        self.roi = None
        #coords of the location with the best response 
        self.coords = None

    def extract_roi(self, coords, size):
    
        self.coords = coords
        x = coords[0]
        y = coords[1]
        
        if x - size < 0:
            x = size
        if y - size < 0:
            y = size
        top_corner =  (x - size, y - size)
        bottom_corner = (x + size, y + size)
        roi_coords = (top_corner, bottom_corner)
        # FORMAT (y,x)
        self.roi = self.image[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]]
        
        if self.roi.size == 0:
            raise exceptions.EmptyImageError(f'ROI is empty, {coords} were coords')

        return self.image[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]], roi_coords
        
    def find_pupil_ellipse(self, plot=False):
 
        #Preprocess arguments for process_radius
        response_image = np.zeros(self.image.shape, dtype=np.float64)
        padding = 2 * self.maxRadius
        eye_integral = np.zeros((self.image.shape[0] + padding + 1,
                                 self.image.shape[1] + padding + 1), dtype=np.int32)
        
        #add padding, padding size is influenced by maxRadius 
        eye_pad = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        
        eye_integral = cv2.integral(eye_pad, eye_integral)
        
        #Set start values to compare to 
        position_pupil = (0, 0)
        min_response = np.inf
        
        #Handels creation and destruction of a pool of worker threads
        with ThreadPoolExecutor() as executor:
            # create list of all arguments used with the ThreadPoolExecuter Instance 
            args = [(r, eye_integral, padding, self.image.shape, response_image) for r in range(self.minRadius, self.maxRadius, 2)]
            #save all results in an list 
            results = list(executor.map(process_radius, args))

        #iterate over all results and pick the best response (lowes value)
        for min_radius_response, min_radius_position, r in results:
            if min_radius_response < min_response:
                min_response = min_radius_response
                position_pupil = min_radius_position
                        
        #Ploting
        normalized_response = cv2.normalize(response_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if plot:
            colormap = cv2.COLORMAP_HOT
            normalized_response = cv2.GaussianBlur(normalized_response, (3,3),0)

            response_image_color = cv2.applyColorMap(normalized_response, colormap)
            cv2.imshow('Response Image', response_image_color)
            #cv2.imwrite('Latex/thesis/plots/results/originalbest.png', self.image)
            #cv2.circle(response_image_color, position_pupil, 4, (255,0,0), -1)
            #cv2.imwrite('Latex/thesis/plots/results/responsehaarbest.png', response_image_color)
        #extract ROI
        roi, roi_coords = self.extract_roi(position_pupil, 110)
        
        return position_pupil, roi, roi_coords
    
    
'''
     __ __ __ __ __ __ __ __ 
   | PADDING                 |    
   |   1 __ __ __ __ __  2   |
   |   |                 |   |
   |   |     5__ __6     |   |
   |   |     |     |     |   |
   |   |     |__ __|     |   |
   |   |     7     8     |   |
   |   |__ __ __ __ __ __|   |
   |   3                 4   |
   | __ __ __ __ __ __ __ __ |
   
   
FORM (y,x)

OUTER
1 = y + padding + y - r_outer, x + padding - r_outer
2 = y + padding + y - r_outer, x + padding + r_outer
3 = y + padding + y + r_outer, x + padding- r_outer
4 = y + padding + y + r_outer, x + padding + r_outer

INNER
5 = y + padding - r_inner, x + padding - r_inner
6 = y + padding - r_inner, x + padding + r_inner + 1
7 = y + padding + r_inner + 1, x + padding - r_inner
8 = y + padding + r_inner + 1, x + padding + r_inner + 1


CALUCLATION OF THE INTEGRAL IMAGE
(i) sum_inner = (5 + 8) - (6 + 7)
(ii) sum_outer = (1 + 4) - (2 + 3) - sum_inner

RESONSE = -r_inner * sum_inner + router * sum_outer

Numba using JIT optimisation for process_radius
boosting performance

'''

@njit
def process_radius(args):
    r, eye_integral, padding, img_shape, response_img = args
    #Setup for comparing the sums and get the right coordinates
    r_inner = r
    r_outer = 3 * r
    min_radius_response = np.inf
    min_radius_position = (0, 0)
    # calculate the response, but not on the entire Image, only a subset of it (range(r,..., #Skip  ))
    for y in range(r, img_shape[0] - r, 3):
        for x in range(r, img_shape[1] - r, 3):
            x_hat = x + padding
            y_hat = y + padding
            # (i) Calculate sum_inner 
            sum_inner = (eye_integral[y_hat - r_inner, x_hat - r_inner]
                         + eye_integral[y_hat + r_inner + 1, x_hat + r_inner + 1]
                         - eye_integral[y_hat - r_inner, x_hat + r_inner + 1]
                         - eye_integral[y_hat + r_inner + 1, x_hat - r_inner])
            # (ii) Caluclate sum_outer, needs result of sum_inner
            sum_outer = (eye_integral[y_hat - r_outer, x_hat - r_outer]
                         + eye_integral[y_hat + r_outer + 1, x_hat + r_outer + 1]
                         - eye_integral[y_hat - r_outer, x_hat + r_outer + 1]
                         - eye_integral[y_hat + r_outer + 1, x_hat - r_outer]
                         - sum_inner)
            response = -r_inner * sum_inner + r_outer * sum_outer
            #create image of response pixel by pixel
            response_img[y,x] = response
            
            # Check if response was lower -> better Haar Feature response
            if response < min_radius_response:
                min_radius_response = response
                min_radius_position = (x, y)
                
    return min_radius_response, min_radius_position, r
