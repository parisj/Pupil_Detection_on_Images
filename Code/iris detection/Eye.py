import numpy as np
import cv2
import math 
import exceptions
from ImageObserver import ImageObserver

class Eye:
    def __init__(self):

        # Image information
        self._img = None
        self._gray = None
        self._processing = None
        # hue, saturation, value
        self._hsv = None
        
        # Object Information
        self._center = None
        self._axis = None
        self._angle = None
        self._circles = None
        self._ellipse = None
        self._mask = None
        self._outline_mask = None
        
        # Observers
        self._observers = []
    

    
    # getter and setter methods
    def set_img(self, img):
        
        if (img is None or img.size == 0):
            raise exceptions.EmptyImageError("Image for set_img is empty")
        
        self._img = img
        self.notify_observers(attr_name="_img")
        
    def get_img(self):
        return self._img
    
    def set_gray(self, gray):
        if gray is None or gray.size == 0:
            raise exceptions.EmptyImageError("Image for gray is empty")
        self._gray = gray
        self.notify_observers('_gray')
        
    def get_gray(self):
        return self._gray
    
    def set_processing(self, processing):
        self._processing = processing
        self.notify_observers('_processing')
        
    def get_processing(self):
        return self._processing
    
    def set_center (self, center):
        self._center = center
    
    def get_center (self):
        return self._center
    
    def set_axis (self, axis):
        self._axis= axis
    
    def get_axis (self):
        return self._axis
    
    def set_angle (self, angle):
        self._angle = angle
    
    def get_angle (self):
        return self._angle
    
    def set_circles (self, circles):
        self._circles = circles
        self.notify_observers('_circles')
    
    def get_circles (self):
        return self._circles
    
    def set_ellipse (self, ellipse, coords):
        # Coords = (x,y,w,h, border)
        print('set_ellipse', ellipse)
        print('with coords', coords)
        y_off = coords[0]-coords[4]
        x_off = coords[1]-coords[4]
        self._center = (round(ellipse[0][0]),round(ellipse[0][1]))
        self._axis = (round(ellipse[1][0]/2),round(ellipse[1][1]/2))
        self._angle = ellipse[2]
        self._ellipse = (self._center, self._axis, self._angle)
        self.notify_observers('_ellipse')
    
    def get_ellipse (self):
        return self._ellipse
    
    def set_mask(self,mask):
        self._mask = mask
        self.notify_observers('_mask')
        
    def get_mask(self):
        return self._mask
    
    def set_outline_mask(self, outline):
        self._outline_mask = outline
        self.notify_observers('_outline_mask')
        
    def get_outline_mask(self):
        return self._outline_mask

    def set_hsv (self, hsv):
        self._hsv = hsv
        self.notify_observers('_hsv')
    
    def get_hsv (self):
        return self._hsv
    
    # observer methods
    def add_observer(self, observer):
        self._observers.append(observer)
        
    def notify_observers(self, attr_name):
        for observer in self._observers:
            observer.update(self, attr_name)
    
    # Image methods
    def load_image(self, path, gray=False):
        self._img = cv2.imread(path)
        if gray: 
            self._gray = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
    
    def distance_to_center(self, point):
        """
        Calculates the Euclidean distance between the center of the pupil and a given point (x,y)

        Args:
            point (np.array): form x,y
        """
        return math.sqrt((self._center[0] - point[0]) ** 2 + (self._center[1] - point[1]) ** 2)
