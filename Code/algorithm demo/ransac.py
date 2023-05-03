import numpy as np 
import cv2 
import random
from scipy.linalg import null_space
class ransac: 
    def __init__(self, mask, coords_roi):
        self.mask = mask
        self.ellipses = []
        self.center = None
        self.axis = None
        self.angle = None
        self.contour = self.init_contour()
        self.best_ellipse = None
        self.fit_error_per_iteration = []
        self.coords_roi = coords_roi
        
        self.C = np.array([[0,0,2,0,0,0],
                           [0,-1,0,0,0,0],
                           [2,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0]])
        
        self.a = np.array([0,0,0,0,0,0])
        
        
        
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
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key = cv2.contourArea)
        
        self.set_contour(largest_contour)
        
        return True
    
    def pick_random_points(self, number_of_points):
        contour = self.get_contour()
        random_points = random.sample(list(contour), number_of_points)
        return random_points
    

    def direct_least_square_fitting(self, points):
        points = np.array(points)[:,0]
        print(f'points: {points}')
        print(f'len(points): {len(points)}')

        print(f'points[:,0]: {points[:,0]}')
        print(f'points[:,0]: {points[:,1]}')

        D = np.mat(np.vstack([points[:,0]**2, points[:,0]*points[:,1], points[:,1]**2, points[:,0], points[:,1], np.ones(len(points))])).T
        S = np.dot(D.T,D) # scatter matrix
        C = self.C
        Z = np.dot(np.linalg.inv(S),C)
        eigen_value, eigen_vec = np.linalg.eig(Z)
        print(f'eigen_value: {eigen_value}')
        eigen_value= eigen_value.reshape(1,-1)
        pos_r, pos_c = np.where(eigen_value >0 & ~np.isinf(eigen_value)) 
        a = eigen_vec[:,pos_c]
        return a
    
if __name__ == '__main__':
    mask = np.zeros((200,200),dtype = np.uint8)
    mask = cv2.ellipse(mask,(100,100),(50,25),0,0,360,255,1)
    cv2.imshow('mask',mask)
    
    rans = ransac(mask, (0,0))
    rans.init_contour()
    points = rans.pick_random_points(5)
    print (f'Points: {points}')
    a = rans.direct_least_square_fitting(points)
    print (f'a: {a}')
    test = np.zeros((200,200),dtype = np.uint8)
    cv2.ellipse(test,(int(a[0]),int(a[1])),(int(a[2]/2),int(a[3]/2)),int(a[4]),0,360,255,1)
    cv2.imshow('test',test)
    cv2.waitKey(0)