import numpy as np 
import cv2
import matplotlib.pyplot

class iris():
    
    def __init__(self, path):
        _center = None
        _radius = None
        _img = None
        _path = path