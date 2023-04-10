import numpy as np 
import cv2
import matplotlib.pyplot as plt

class active_contour: 
    def __init__(self):
        self.ellipse= None
        self.roi = None
        self.roi_gray = None
        self.gradient = None
        self.direction = None
        self.normalvector = None
        self.energy = None

    # Setters and Getters

    def set_ellipse (self, ellipse):
        self.ellipse = ellipse
        
    def get_ellipse (self):
        return self.ellipse
    
    def set_roi(self, roi):
        self.roi = roi
    
    def get_roi(self):
        return self.roi
    
    def set_roi_gray(self, roi_gray):
        self.roi_gray = roi_gray
    
    def get_roi_gray(self):
        return self.roi_gray
        
    
    def set_gradient(self, gradient):
        self.gradient = gradient
        
    def get_gradient(self):
        return self.gradient
    
    def set_normalvector(self, normalvector):
        self.normalvector = normalvector
        
    def get_normalvector(self):
        return self.normalvector
    
    def set_energy(self, energy):
        self.energy = energy
        
    def get_energy(self):
        return self.energy
    
    def set_direction(self, direction):
        self.direction = direction
    
    def get_direction(self):
        return self.direction
    
    
    def _init_roi(self,roi):
        self.set_roi(roi)
        self.set_roi_gray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        mag, direction = self._gradient(self.get_roi_gray())
        self.set_gradient(mag)
        self.set_direction(direction)
        return True
    
    
    # TODO:
    ''''
    create ellipse
    calculate gradient over image, 
    calulate normalvector over ellipse
    define energy function 
    use scipy.optimize to minimize energy function
    calculate threshold to spot 
    use balloon force to expand the ellipse towards the edge

    '''
    # Create Ellipse
    def _init_circle(self, center, axis, angle, image_shape, num_points= 20):
        t = np.linspace(0, 2*np.pi, num_points)
        x =  round(axis[0]//2) * np.cos(t)
        y = round(axis[1]/2) * np.sin(t)
        
        grad_x = -round(axis[0]/2) * np.sin(t) 
        grad_y = round(axis[1]/2) * np.cos(t)
        length = 1
        normal_x = -grad_y
        normal_y = grad_x
        print(f'x: {x}, y: {y}')
        print(f'grad_x: {grad_x}')
        print(f'grad_y: {grad_y}')
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        grad = np.matmul(R, -np.array([normal_x, normal_y]))
        ellipse = np.matmul(R, np.array([x, y])).astype(np.int32)
        ellipse[0,:] += center[0]
        ellipse[1,:] += center[1]

        print(f'ellipse: {ellipse}')
        
        mask = np.zeros(image_shape).astype(np.uint8)
        for point_x, point_y, nx, ny in zip(ellipse[0], ellipse[1],grad[0], grad[1]):
            cv2.circle(mask, (round(point_x), round(point_y)), 1, (255, 255, 255), -1)
            start_point = (round(point_x), round(point_y))
            end_point = (round(point_x + length*nx ), round(point_y +length* ny))
            cv2.arrowedLine(mask, start_point, end_point, (255, 255, 255), 1, tipLength=0.1)

        grad = np.array([element for element in zip(grad[0], grad[1])]).reshape(-1,2)
        print(f'grad: {grad}')
#        for point_x, point_y in zip (ellipse[0], ellipse[1]):
#            cv2.circle(mask, (round(point_x), round(point_y)), 2, (255,255,255), -1)
        cv2.imshow('mask', mask)
        self.set_ellipse(ellipse)
        self.set_normalvector(grad)
        return  ellipse, grad
    # Calculate Gradient over complete image
    def _gradient (self, img):
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        
        # Calculate Gradient Mag and direction
        mag = np.sqrt(sx**2 + sy**2)
        mag += cv2.GaussianBlur(mag, (5, 5), 0)
        mag += cv2.GaussianBlur(mag, (21, 21), sigmaX=10, sigmaY=10)
        mag += cv2.GaussianBlur(mag, (51, 51), sigmaX=20, sigmaY=20)

        mag = mag / np.max(mag)
    
        direction = np.arctan2(sy, sx) * 180 / np.pi

        # Normalize direction to the range [0, 1]
        direction_normalized = (direction + 360) % 360 / 360

        # Convert direction to hue value (0-180) for OpenCV
        hue = (direction_normalized * 180).astype(np.uint8)

        # Create HSV image with hue as direction, full saturation, and magnitude as value
        hsv = np.zeros((*img.shape, 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 255
        hsv[..., 2] = (mag * 255).astype(np.uint8)

        # Convert HSV to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('mag', mag)
        cv2.imshow('direction', bgr)
        self.set_gradient(mag)
        self.set_direction(direction)
        return mag, direction
    
    def _cross(self, center, axis, angle, img, num_points=20 ):
   
        points, normal = self._init_circle(center, axis, angle, img, num_points)
        print(f'points: {points}')
        print(f'normal: {normal}')
        grad = self.get_gradient()
        points.reshape(-1,2)
        direction = self.get_direction()
        print(f'grad: {grad}')
        print(f'direction: {direction}')
        
        grad_points = grad[points[0,:], points[1,:]]
        direction_points = direction [points[0,:], points[1,:]]
        normalv = self.get_normalvector()
        print(f'normalv: {normalv}')
        grad_info = np.array([e for e in zip(grad_points, direction_points)])
        self._setup_points(points, normalv, grad_info)
        for idx, cross in enumerate( zip( normalv, grad_info)):
            print(f'idx: {idx}, cross: {cross}')
            print(f'point: {points[:,idx]}')
    
    def _energy(self, center, axis, angle, img, num_points=20):
        return np.sum(self._cross((center[0], center[1]), (axis[0], axis[1]), angle, img, num_points))



    def _setup_points (self,points,normalv,grad_info):
        points_normal_vec = np.array([points[0,:]+normalv[:,0],points[1,:]+normalv[:,1]]).reshape(-1,2)
        print(f'grad_info[:,0]:{grad_info[:,0]}')
        print(f'grad_info[0,:]:{grad_info[0,:]}')

        dx = grad_info[:,0]*np.cos(np.deg2rad(grad_info[:,1]))
        dy = grad_info[:,0]*np.sin(np.deg2rad(grad_info[:,1]))
        points_gradient_vec = np.array([points[0,:]+dx, points[1,:]+dy]).reshape(-1,2)
        print (f'points_gradient_vec: {points_gradient_vec}')

        print (f'points_normal_vec: {points_gradient_vec}')
    
    
if __name__ == '__main__':
    image = cv2.imread('test_roi.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    contour = active_contour()
    contour._gradient(gray)
    contour._cross((100,100), (50, 25),np.pi/2,(200,200), 35 )

    contour._init_roi(image)
    cv2.waitKey(0)
