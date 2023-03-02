import numpy as np 
import cv2 
import pandas as pd

class pupil: 
    def __init__(self, path):
        self._path = path 
        self._center = np.array([0,0])
        self._radius = 0 
        self._img = None
        self._processing = None
    
    def load_image(self,visual= False):
        self._img = cv2.imread(self._path)
        self._processing = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        print("loading")
        if visual:
            self.visual_picture()

    def visual_picture(self, img=None):

        cv2.imshow("original", self._img)
        if img is None:
            cv2.imshow("processed", self._processing)
        else: 
            cv2.imshow("processed", img)
        print("show")

        cv2.waitKey(0)
    
    def preprocess_image(self, g_b_Kernel=(3, 3), sig_x=0, visual= False):
        img = cv2.GaussianBlur(self._processing, g_b_Kernel, sig_x)
        self._processing = img
        if visual:
            self.visual_picture()
        print("processing")
        
    def canny(self, th1=5, th2=70, edge=3, visual = False):
        self._processing = cv2.Canny(self._processing, th1, th2, edge)
        
        if visual:
            self.visual_picture()
        print("canny")
        
    def hough_circle(self, dp=2, mindist=80, p1=30, p2=50, r_min=0, r_max=30, visual= False):
        circles = cv2.HoughCircles(self._processing, cv2.HOUGH_GRADIENT, dp, minDist=mindist,
                                   param1=p1, param2=p2, minRadius=r_min,maxRadius=r_max)
        
        if visual: img_0 = self._img.copy()
        print("maybe")
        print(circles)
        if circles is not None:
             
            circles = np.round(circles[0, :]).astype("int")
            for (x,y,r) in circles:
                self._center = np.array([x,y])
                self._radius = r
                if visual:
                    cv2.circle  (img_0, (x, y), 0, (0, 255, 0), 6)
                    cv2.circle(img_0, (x, y), r, (0, 0, 255), 1)
            if visual: self.visual_picture(img_0)
        print("hough")
             
if __name__ == '__main__':
    list= pd.read_csv("path_files_MMU_IRIS_DATABASE.csv")
    subset = []
    for i, row in list.iterrows():
        subset.append(row[0])
    for i in range(100,200):
        pupil_test = pupil(subset[i])
        pupil_test.load_image()
        pupil_test.preprocess_image((11,11), 5)
        pupil_test.canny()
        pupil_test.hough_circle(visual=True, dp=2, r_max= 30, r_min=10)
        radius=pupil_test._radius
        pupil_test.hough_circle(visual=True, dp=2, r_max=int(radius*3),r_min=int(radius*2))
    
                                    