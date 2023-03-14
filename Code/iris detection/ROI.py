import numpy as np 
import cv2
import exceptions

class ROI:
    def __init__(self, img, region):
        self._img = img
        self._roi = None
        
        #[x1,y1][x2,y2]
        p1,p2 = self.convert_range(region)
        self._region = np.array([p1,p2])

    def set_img(self,img):
        self._img = img
        
    def get_img(self):
        return self._img
    
    def set_roi(self,roi):
        self._roi = roi
        
    def get_roi(self):
        return self._roi
    
    def set_region(self, region):
        p1,p2 = self.convert_range(region)
        self._region = np.array([p1,p2])
        
    def get_region(self):
        return self._region
    
    def create_roi(self):
        
        if self._img is None or self._img.size == 0:
            raise exceptions.EmptyImageError("Image for ROI is empty")
        
        if self._region is None or self._region.size != 4:
            raise exceptions.EmptyRegionError("Image region is empty or is not size of 4")
        
        print(self._region)
        print(self._region[0][0],self._region[1][0],self._region[0][1],self._region[1][1])
        self._roi = self._img[self._region[0][0]:self._region[1][0], self._region[0][1]:self._region[1][1]]
        print(type(self._roi))
        print(self._roi)
        
    def convert_range(self, region):
        p1 = np.array([region[0][1], region[0][0]])
        p2 = np.array([region[1][1], region[1][0]])
        return p1, p2

if __name__ == '__main__':
    # Check if ROI is created correctly
    # Center at 334 305
    img = cv2.imread('test.jpg')
    region = np.array([[324,295],[344,315]])

    cv2.imshow("img", img)
    roi = ROI(img, region)
    roi.create_roi()

    
    cv2.imshow('Roi', roi.get_roi())
    cv2.waitKey(0)