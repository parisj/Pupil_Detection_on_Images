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
        
        # Pupil information
        self._center = np.array((2,))
        self._radius = None
        self._circles = None
        self._mask = None
        
        
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
    
    def set_radius (self, radius):
        self._radius = radius
    
    def get_radius (self):
        return self._radius
    
    def set_circles (self, circles):
        self._circles = circles
        self.notify_observers('_circles')
    
    def get_circles (self):
        return self._circles
    
    def add_observer(self, observer):
        self._observers.append(observer)
        
    def notify_observers(self, attr_name):
        for observer in self._observers:
            observer.update(self, attr_name)
    
    def distance_to_point(self, point):
        """
        Calculates the Euclidean distance between the center of the pupil and a given point (x,y)

        Args:
            point (np.array): form x,y
        """
        return math.sqrt((self._center[0] - point[0]) ** 2 + (self._center[1] - point[1]) ** 2)
    
    def load_image(self, path, gray=False):
        self._img = cv2.imread(path)
        if gray: 
            self._gray = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
    
    def set_mask(self,mask):
        self._pupil_mask = mask 
        
    def get_mask(self):
        return self._pupil_mask
    
    def create_mask(self, circle):
        #Circle of the form [x,y,r]
        mask = np.zeros_like(self._img)
        

    