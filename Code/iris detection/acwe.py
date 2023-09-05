import numpy as np 
import cv2
from ransac import ransac

class ACWE: 
    def __init__(self): 
        
        #define neighborhood for erosion and dilation
        '''
        4 different neighborhood configurations are used:
        
        1 0 0    0 1 0    0 0 0    0 0 1
        0 1 0    0 1 0    1 1 1    0 1 0
        0 0 1    0 1 0    0 0 0    1 0 0
        
        '''
        self.image = None
        self.neighborhood = (np.eye(3,3).astype(np.uint8),
                             np.array([[0,1,0],
                                       [0,1,0],
                                       [0,1,0]]).astype(np.uint8),
                             np.array([[0,0,0],
                                       [1,1,1],
                                       [0,0,0]]).astype(np.uint8),
                             np.array([[0,0,1],
                                       [0,1,0],
                                       [1,0,0]]).astype(np.uint8))
        self.mask = None
        self.intensity = None
        self.lambda1 = None
        self.lambda2 = None
        self.convergence_threshold = None
        self.center_start = None
        self.result_ellipse = None
        self.iterations_safe = 0
        self.ransac_iterations = 0
        self.ransac_threshold = 0
        self.callback_bool = False
        
    def set_image(self, image):
        self.image = image
    
    def get_image(self):
        return self.image
    
    def set_intensity(self, intensity):
        self.intensity = intensity

    def get_intensity(self):
        return self.intensity
    
    def set_center_start(self, center_start):
        self.center_start = center_start
    
    def get_center_start(self):
        return self.center_start
    
    def set_mask(self, mask):
        self.mask = mask.astype(np.uint8)

    def get_mask(self):
        return self.mask   
    
    def get_neighborhood(self):
        return self.neighborhood
    
    def set_lambda1(self, lambda1):
        self.lambda1 = lambda1
    
    def get_lambda1(self):
        return self.lambda1
    
    def set_lambda2(self, lambda2):
        self.lambda2 = lambda2
        
    def get_lambda2(self):
        return self.lambda2
    
    def set_convergence_threshold(self, convergence_threshold):
        self.convergence_threshold = convergence_threshold

    def get_convergence_threshold(self):
        return self.convergence_threshold
    
    def set_result_ellipse(self, result_ellipse):
        self.result_ellipse = result_ellipse
    
    def get_result_ellipse(self):
        return self.result_ellipse
    
    def set_ransac_iterations(self, ransac_iterations):
        self.ransac_iterations = ransac_iterations
        
    def get_ransac_iterations(self):
        return self.ransac_iterations
    
    def set_ransac_threshold(self, ransac_threshold):
        self.ransac_threshold = ransac_threshold
    
    def get_ransac_threshold(self):
        return self.ransac_threshold
    
    def set_callback_bool(self, c_bool):
        
        self.callback_bool = c_bool

    def get_callback_bool(self):
        return self.callback_bool
    
    def _init_mask(self, center, radius):
        image_shape = self.get_intensity().shape
        mask = np.zeros(image_shape)
        cv2.circle(mask, (center[1],center[0]), radius, 1, -1)
        self.set_mask(mask)
        return True
        
    def _init_Intensity(self, image): 
        image_color = image.copy()
        image_color = cv2.cvtColor(image_color, cv2.COLOR_GRAY2BGR)
        self.set_image(image_color)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.set_intensity(image)
        return True
    
    
    #Page 19 of the paper, c1 is the mean intensity inside the contour
    def calc_c1(self, mask, intensity):
        '''
        c1 = integral of Intensity inside the contour/area inside the contour
        '''
 

        return (intensity*mask).sum()/float((mask).sum()+1e-16)    
    
    #Page 19 of the paper, c2 is the mean intensity outside the contour

    def calc_c2(self, mask, intensity):
        '''
        c2 = integral of Intensity outside the contour/area outside the contour
        '''       
        return (intensity*(1-mask)).sum()/float((1-mask).sum()+1e-16)
    
    # Balloon force
    def erode(self, mask):
        eroded = []
        for neighbor in self.neighborhood:
            mask = cv2.erode(mask, neighbor)
            eroded.append(mask)
        return np.stack(eroded, axis=0).max(0)
    
    def dilate(self, mask):
        dilated = []
        for neighbor in self.neighborhood:
            mask = cv2.dilate(mask, neighbor)
            dilated.append(mask)
        return np.stack(dilated, axis=0).min(0)
    
    
    def si_is(self):
        #  "sup-inf" and "inf-sup" operators
        mask = self.get_mask()
        mask = self.erode(mask)
        mask = self.dilate(mask)
        self.set_mask(mask)
        return True
            
    def is_si (self):
        mask = self.get_mask()
        mask = self.dilate(mask)
        mask = self.erode(mask)
        self.set_mask(mask)
        return True
    
    def smoothing (self, iterations):
        for i in range(iterations):
            self.si_is()
            self.is_si()
        return True
        

    def _ACWE(self, iterations_smoothing, iterations_ACWE):
        prev_mask = None
        l_1 = self.get_lambda1()
        l_2 = self.get_lambda2()
        intensity = self.get_intensity()
        callback_bool = self.get_callback_bool()
        for _ in range(iterations_ACWE):
            if prev_mask is not None:
                #convergence check
                mask_difference = np.abs(self.get_mask() - prev_mask).sum()
                total_pixels = self.get_mask().size
                percent_change = (mask_difference / total_pixels) * 100

                if percent_change < self.get_convergence_threshold():
                    #print("Convergence reached.")
                    break

            prev_mask = self.get_mask().copy()
            mask = self.get_mask()
            c1 = self.calc_c1(mask, intensity)
            c2 = self.calc_c2(mask, intensity)

            
            dm = np.gradient(mask)
            abs_dm = np.abs(dm).sum(0)
            #curvature flow calculation
            x = l_1 * abs_dm * (intensity - c1)**2 - l_2 * abs_dm * (intensity - c2)**2

            mask[x < 0] = 1
            mask[x > 0] = 0
            self.set_mask(mask)
            
            self.smoothing(iterations_smoothing)
            if callback_bool:
                self.callback()

    def start(self, center, radius, image, iterations_smoothing, iterations_ACWE, lambda1, lambda2, convergence_threshold, ransac_iterations, ransac_threshold, callback_bool= False):
        #init parameters and start ACWE
        self._init_Intensity(image)
        self.set_center_start(center)
        self._init_mask(center, radius)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.set_callback_bool(callback_bool)

        self.set_convergence_threshold(convergence_threshold)
        self._ACWE(iterations_smoothing, iterations_ACWE)
        self.set_ransac_iterations(ransac_iterations)
        self.set_ransac_threshold(ransac_threshold)

    def callback(self):
        #visualize the result every iteration for debugging
        mask = self.get_mask().copy()
        image = self.get_image().copy()
        color = (0, 0, 255)
        alpha = 0.3
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(f'contours active: {contours}')
        filled_mask = np.zeros_like(image)
        cv2.drawContours(filled_mask, contours, -1, color, cv2.FILLED)
        overlay = cv2.addWeighted(image, 1, filled_mask, alpha, 0)
        cv2.drawContours(overlay, contours, -1, color, 1)
        cv2.circle(overlay, self.get_center_start(), 2, (0, 255, 0), -1)
        cv2.circle(overlay, (self.get_center_start()[1],self.get_center_start()[0]), 2, (0, 0, 255), -1)
        filename = f'Latex/thesis/plots/acwe/iterationplot_{self.iterations_safe}.png'
        cv2.imshow('Result', overlay)
        #cv2.imwrite(filename, overlay)
        cv2.waitKey(1)
        self.iterations_safe += 1


    def plot_ellipse(self):
        
        image_copy = self.image.copy()
        ellipse = self.get_result_ellipse()
        color = (0, 0, 255)
        
        # Draw the ellipse on the image
        cv2.ellipse(image_copy, ellipse, color, 1)
        cv2.imshow('Result', image_copy)
        #cv2.imwrite('path/to/folder.png', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def result(self):
        mask = self.get_mask()
        
        #erode and dilate the mask to remove noise
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)


        # Find the contours in the mask
        contours, _ = cv2.findContours(eroded_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours is None or len(contours) == 0:
            return False
        # Find the contour with the largest area (in case there are multiple contours)
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 5:
            return False
        image = self.get_image().copy()

        # Use Ransac to fit an ellipse to the contour
        r_threshold = self.get_ransac_threshold()
        r_iterations = self.get_ransac_iterations()
        rans = ransac(largest_contour, r_iterations, r_threshold, self.callback_bool)
        a,b,c,d,stat = rans.ransac_start()
        
        #-------------------
        #useful for debugging
        #-------------------
        ##print(f'best_ellipse: {a} best_inliers: {b} best_area: {c} best_border: {d}')
        #print(f'leng points_contour: {len(rans.get_points_contour())}')
        #print(f'lenght of best_inliers: {len(b)}')
        #print(f'lenght of best_border: {len(d)}')
        #print(f'best stat: {stat}')
        #-------------------
        self.set_result_ellipse(a)

        return True
        
if __name__ == '__main__':
    #Test the ACWE algorithm
    image = cv2.imread('Latex/thesis/plots/acwe/robusfit2.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = (312, 298)
    radius = 5
    acwe = ACWE()
    acwe.start(center, radius, image, 3, 10000, 2.5, 0.4, 0.00005, 1000, 0.005)
    acwe.result()
    acwe.plot_ellipse()
    cv2.destroyAllWindows()