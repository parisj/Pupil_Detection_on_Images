import numpy as np 
import cv2 

class ransac: 
    def __init__(mask, coords_roi):
        self.mask = mask
        self.ellipses = []
        self.center = None
        self.axis = None
        self.angle = None
        self.contour = self.init_contour()
        self.best_ellipse = None
        self.fit_error_per_iteration = []
        self.coords_roi = coords_roi
        
    def set_ellipse(self, ellipse):
        self.ellipse = ellipse
    
    def get_ellipse(self):
        return self.ellipse
    
    def set_center(self, center):
        self.center = center
        
    def get_center(self):
        return self.center
    
    def get_mask(self):
        return self.mask
    
    def set_axis(self, axis):
        self.axis = axis
        
    def get_axis(self):
        return self.axis
    
    def set_angle(self, angle):
        self.angle = angle
    
    def get_angle(self):
        return self.angle
    
    def set_contour(self, contour):
        self.contour = contour
        
    def get_contour(self):
        return self.contour
    
    def set_best_ellipse(self, best_ellipse):
        self.set_best_ellipse = best_ellipse
    
    def get_best_ellipse(self):
        return self.best_ellipse
    
    def init_contour(self):
        mask = self.get_mask()
        
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key = cv2.contourArea)
        
        self.set_contour(largest_contour)
        
        return True
    
    def pick_random_point (self):
        contour = self.get_contour()
        random_point = np.random.randint(0, len(contour))
        return random_point
        
    
    def pick_random_points(self, number_of_points):
        random_points = []
        for i in range(number_of_points):
            random_point = self.pick_random_point()
            random_points.append(random_point)
        return random_points
    
    def fit_ellipse(self, points, )