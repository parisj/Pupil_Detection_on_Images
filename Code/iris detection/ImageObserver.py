import numpy as np
import cv2

class ImageObserver:
    def __init__(self):
        # Initialize an empty list to store the image objects being observed
        self._img_objs = []
        # Initialize empty dictionaries to store the numpy array attributes for each image object
        self._imgs = {}
        self._processings = {}
        self._grays = {}
        self._circles = {}
        # Initialize an empty dictionary to keep track of whether the circles have been plotted for each image object
        self._circles_plots = {}

    def add_img_obj(self, img_obj):
        # Add the new image object to the list of observed objects
        self._img_objs.append(img_obj)
        # Initialize empty dictionary entries for the new image object to store the numpy array attributes
        self._imgs[id(img_obj)] = None
        self._processings[id(img_obj)] = None
        self._grays[id(img_obj)] = None
        self._circles[id(img_obj)] = None
        # Initialize the _circles_plots entry for the new image object to None
        self._circles_plots[id(img_obj)] = None
        # Add the observer to the new image object's list of observers
        img_obj.add_observer(self)

    def update(self, img_obj, attr_name):
        # Update the corresponding dictionary entry with the new value when an attribute of an image object is updated
        print(attr_name,'attr_name')
        print(attr_name == "_img")
        if attr_name == '_img':
            print('HALLO',id(img_obj))
            self._imgs[id(img_obj)] = img_obj._img
        elif attr_name == '_processing':
            self._processings[id(img_obj)] = img_obj._processing
        elif attr_name == '_gray':
            self._grays[id(img_obj)] = img_obj._gray
        elif attr_name == '_circles':
            self._circles[id(img_obj)] = img_obj._circles
        # Plot the circles for the updated image object
        #self.plot_circles(img_obj)

    def plot_circles(self, img_obj):
        # Get the ID of the image object to plot
        img_id = id(img_obj)
        # Make a copy of the original image to draw the circles on
        img_with_circles = self._imgs[img_id].copy()
        # If the image object has circles detected, draw them on the copied image
        if self._circles[img_id] is not None:
            for circle in self._circles[img_id]:
                x, y, r = circle
                cv2.circle(img_with_circles, (x, y), r, (0, 255, 0), 2)
        # If the _circles_plots entry for the image object is None, create a new window and display the image with circles
        if self._circles_plots[img_id] is None:
            cv2.imshow(f'Image with circles {img_id}', img_with_circles)
            # Set the _circles_plots entry to True to indicate that the circles have been plotted
            self._circles_plots[img_id] = True
        # If the _circles_plots entry for the image object is not None, update the existing window with the new image with circles
        else:
            cv2.imshow(f'Image with circles {img_id}', img_with_circles)

    def plot_imgs(self, img_type):
        # Determine which dictionary to use based on the requested image type
        if img_type == 'original':
            imgs = self._imgs
            print(imgs,self._img_objs)
        elif img_type == 'processing':
            imgs = self._processings
        elif img_type == 'gray':
            imgs = self._grays
        else:
            raise ValueError(f"Invalid image type: {img_type}")

        # Plot the images for each image object
        for img_obj in self._img_objs:
            img_id = id(img_obj)
            img = imgs[img_id]

            # Create a window if one doesn't already exist
            if img is not None:
                cv2.imshow(f"{img_type.capitalize()} Image {img_id}", img)



        # Wait for a key press before closing the windows
        cv2.waitKey(0)