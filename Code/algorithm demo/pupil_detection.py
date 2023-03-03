import numpy as np 
import cv2 
import pandas as pd

class pupil: 
    """This class is used to process images of eyes to find the center and outline of the pupils aswell of the iris 
    """
    def __init__(self, path):
        self._path = path 
        self._center = np.array([0,0])
        self._radius = 0
        self._img = None
        self._processing = None
        self._center_iris = np.array([0,0])
        self._radius_iris = 0
    
    def load_image(self,visual= False):
        self._img = cv2.imread(self._path)
        self._processing = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        self._gray=self._processing.copy()
        if visual:
            self.visual_picture()

    def visual_picture(self, img=None):

        cv2.imshow("original", self._img)
        if img is None:
            cv2.imshow("processed", self._processing)
        else: 
            cv2.imshow("processed", img)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
        
    
    def preprocess_image(self, g_b_Kernel=(3, 3), sig_x=0, visual= False):
        img = cv2.GaussianBlur(self._gray, g_b_Kernel, sig_x)
        self._processing = img
        if visual:
            self.visual_picture()
        
    def canny(self, th1=5, th2=70, edge=3, visual = False):
        self._processing = cv2.Canny(self._processing, th1, th2, edge)
        
        if visual:
            self.visual_picture()
        
    def hough_circle(self, dp=2, mindist=80, p1=30, p2=50, r_min=0, r_max=30, visual= False):
        circles = cv2.HoughCircles(self._processing, cv2.HOUGH_GRADIENT, dp, minDist=mindist,
                                   param1=p1, param2=p2, minRadius=r_min,maxRadius=r_max)
        
        if visual: img_0 = self._img.copy()
        if circles is not None:
            
            circles = np.round(circles[0, :]).astype("int")
            if self._radius: 
                for (x,y,r) in circles:
                    
                    if ((x - self._center[0])**2 + (y - self._center[1])**2 < ( (self._radius/9)**2)):
                        
                        self._center_iris = np.array([x,y])
                        self._radius_iris = r
                        
                        if visual: 
                            cv2.circle(img_0, (x,y),0,(0,0,255),6)
                            cv2.circle(img_0,(x,y),r,(0,0,255),2)
                    
                    
            for (x,y,r) in circles:

                self._center = np.array([x,y])
                self._radius = r
                if visual:
                    cv2.circle  (img_0, (x, y), 0, (0, 255, 0), 6)
                    cv2.circle(img_0, (x, y), r, (0, 0, 255), 1)
            if visual: self.visual_picture(img_0)
            
if __name__ == '__main__':
    list= pd.read_csv("path_files_MMU_IRIS_DATABASE.csv")
    subset = []
    for i, row in list.iterrows():
        subset.append(row[0])
        
    for i in range(1,10):
        
        #create instance of pupil
        pupil_test = pupil(subset[i])
        
        # Load and preprocess image
        pupil_test.load_image()
        pupil_test.preprocess_image((15,15), 5,visual=True)
        
        # use canny and hough circle to find center and radius of the pupil
        pupil_test.canny(visual=True, th1=35,th2=40)
        pupil_test.hough_circle(visual=True, dp=2, r_max= 30, r_min=10)
        
        # get radius of the pupil
        radius = pupil_test._radius
        # preprocess image differently to achieve higher accuracy on the iris
        pupil_test.preprocess_image((11,11), 10,visual=True)
        pupil_test.canny(35,40, visual=True)  
        
        # detect irlis, use the radius of the pupil to calculate the min and max radius 
        # average diameter pupil = 2-9 mm 12/9, 12/2
        # average diameter iris = 12 mm 
        
        pupil_test.hough_circle(visual=True, dp=2, r_max=int(radius*6),r_min=int(radius*(11/9)))
    
                                    