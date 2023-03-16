import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageObserver:
    def __init__(self):
        # Initialize an empty list to store the image objects being observed
        self._img_objs = []
        # Initialize empty dictionaries to store the numpy array attributes for each image object
        self._imgs = {}
        self._processings = {}
        self._hsvs = {}
        self._grays = {}
        self._circles = {}
        self._ellipses = {}
        self._masks = {}
        self._outline_masks = {}
        self._histograms = {}
       
  

    def add_img_obj(self, img_obj):
        # Add the new image object to the list of observed objects
        self._img_objs.append(img_obj)
        
        # Initialize empty dictionary entries for the new image object to store the numpy array attributes
        self._imgs[id(img_obj)] = None
        self._processings[id(img_obj)] = None
        self._hsvs[id(img_obj)] = None
        self._grays[id(img_obj)] = None
        self._circles[id(img_obj)] = None
        self._ellipses[id(img_obj)] = None
        self._masks[id(img_obj)] = None
        self._outline_masks[id(img_obj)] = None
        self._histograms[id(img_obj)] = None
        

        # Add the observer to the new image object's list of observers
        img_obj.add_observer(self)

    def update(self, img_obj, attr_name):
        # Update the corresponding dictionary entry with the new value when an attribute of an image object is updated
        if attr_name == '_img':
            self._imgs[id(img_obj)] = img_obj.get_img()
        elif attr_name == '_processing':
            self._processings[id(img_obj)] = img_obj.get_processing()
        elif attr_name == '_gray':
            self._grays[id(img_obj)] = img_obj.get_gray()
        elif attr_name == '_circles':
            self._circles[id(img_obj)] = img_obj.get_circles()
        elif attr_name == '_ellipse':
            self._ellipses[id(img_obj)] = img_obj.get_ellipse()
        elif attr_name == '_mask': 
            self._masks[id(img_obj)] = img_obj.get_mask()
        elif attr_name == '_outline_mask':
            self._outline_masks[id(img_obj)] = img_obj.get_outline_mask()
        elif attr_name == '_hsv': 
            self._hsvs[id(img_obj)] = img_obj.get_hsv()
        elif attr_name == '_historgram':
            self._histograms[id(img_obj)] = img_obj.get_histogram()
   


    def plot_imgs(self, img_type):
        
        # Determine which dictionary to use based on the requested image type
        if img_type == 'original':
            imgs = self._imgs
    
        elif img_type == 'processing':
            imgs = self._processings
            
        elif img_type == 'mask':
            imgs = self._masks
        
        elif img_type == 'outline_mask':
            imgs = self._outline_masks

        elif img_type == 'hsv':
            imgs = self._hsvs
            
        else:
            raise ValueError(f"Invalid image type: {img_type}")

        # Plot the images for each image object
        for img_obj in self._img_objs:
            img_id = id(img_obj)
            img = imgs[img_id]

            if img is not None:
                cv2.imshow(f"{img_type.capitalize()} Image {img_id}", img)



        # Wait for a key press before closing the windows
        cv2.waitKey(1) == ord('q')