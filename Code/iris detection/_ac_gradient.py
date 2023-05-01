import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class active_contour: 
    def __init__(self):
        
        '''
        Initializes the active contour class
        '''
        
        
        # saves parameters of the points_ellipse
        self.points_ellipse= None
        # saves the region of interest in color
        self.roi = None
        # save the region of interest in gray
        self.roi_gray = None
        # saves the gradient of the region of interest of each point
        self.gradient = None
        # saves the direction of the gradient
        self.direction = None
        # saves the normalvector of the points_ellipse curve
        self.normalvector = None
        # save the energy of the energy function 
        self.energy = None

    # Setters and Getters

    def set_points_ellipse (self, points_ellipse):
        self.points_ellipse = points_ellipse
        
    def get_points_ellipse (self):
        return self.points_ellipse
    
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
        '''
        Initializes the region of interest with its properties
        '''
        
        self.set_roi(roi)
        self.set_roi_gray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        mag, direction = self._gradient(self.get_roi_gray())
        self.set_gradient(mag)
        self.set_direction(direction)
        return True
    
    
    # TODO:
    ''''
    create points_ellipse
    calculate gradient over image, 
    calulate normalvector over points_ellipse
    define energy function 
    use scipy.optimize to minimize energy function
    calculate threshold to spot 
    use balloon force to expand the points_ellipse towards the edge

    '''
    # Create points_ellipse
    def _init_circle(self, center, axis, angle, image_shape, num_points= 20):
        t = np.linspace(0, 2*np.pi, num_points)
        x =  round(axis[0]/2) * np.cos(t)
        y = round(axis[1]/2) * np.sin(t)
        
        grad_x_normal = -round(axis[0]/2) * np.sin(t) 
        grad_y_normal = round(axis[1]/2) * np.cos(t)
        length = 1
        normal_x = -grad_y_normal
        normal_y = grad_x_normal
        #print(f'x: {x}, y: {y}')
        #print(f'grad_x_normal: {grad_x_normal}')
        #print(f'grad_y_normal: {grad_y_normal}')
        
        # Rotation Matrix
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # Rotate points
        grad_normal = np.matmul(R, -np.array([normal_x, normal_y]))
        points_ellipse = np.matmul(R, np.array([x, y])).astype(np.float64)
        
        #centering the points_ellipse
        points_ellipse[0,:] += round(center[0])
        points_ellipse[1,:] += round(center[1])

        #print(f'points_ellipse: {points_ellipse}')
        
        mask = np.zeros(image_shape).astype(np.uint8)
        for point_x, point_y, nx, ny in zip(points_ellipse[0], points_ellipse[1],grad_normal[0], grad_normal[1]):
            cv2.circle(mask, (round(point_x), round(point_y)), 1, (255, 255, 255), -1)
            start_point = (round(point_x), round(point_y))
            end_point = (round(point_x + length*nx ), round(point_y +length* ny))
            cv2.arrowedLine(mask, start_point, end_point, (255, 255, 255), 1, tipLength=0.1)

        normal_curve = np.array([element for element in zip(grad_normal[0], grad_normal[1])]).reshape(-1,2)
        #print(f'normal_points_ellipse: {grad_normal}')
        for point_x, point_y in zip (points_ellipse[0], points_ellipse[1]):
            cv2.circle(mask, (round(point_x), round(point_y)), 2, (255,255,255), -1)
        cv2.imshow('mask', mask)
        
        self.set_points_ellipse(points_ellipse)
        self.set_normalvector(normal_curve)
        
        return  points_ellipse, normal_curve
    
    

    def _gradient (self, img):
        
        '''
        Calculates the gradient of the image
        '''
        img_copy =img.copy()
        #img_copy = cv2.GaussianBlur(img_copy, (91, 91), sigmaX=0, sigmaY=0)
        sx = cv2.Sobel(img_copy, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img_copy, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate Gradient Mag and direction
        mag = np.sqrt(sx**2 + sy**2)
        mag_blur = mag
        #sx_blur = cv2.GaussianBlur(sx, (51, 51), sigmaX=0, sigmaY=0)
        #sy_blur = cv2.GaussianBlur(sy, (51, 51), sigmaX=0, sigmaY=0)
        # Fade the Gradient to create force field
        #mag_blur = cv2.GaussianBlur(mag, (11, 11), sigmaX=10, sigmaY=10)
        #mag_blur += cv2.GaussianBlur(mag, (21, 21), sigmaX=10, sigmaY=10)
        #mag_blur = cv2.GaussianBlur(mag, (51, 51), sigmaX=0, sigmaY=0)
        
        # normalize the magnitude
        mag_blur = mag_blur / np.max(mag_blur)

        # Calculate direction of the gradient
        direction = np.arctan2(sy, sx)
        print(f'direction: {direction}')	
        print(f'direction.max: {direction.max()}')
        # Normalize direction to the range [0, 1]
        direction_normalized = ((direction)*180/np.pi) +180
        print(f'direction_normalized: {direction_normalized}')
        # Convert direction to hue value (0-180) for OpenCV
        hue = (direction_normalized).astype(np.uint8)

        # Create HSV image with hue as direction, full saturation, and magnitude as value
        hsv = np.zeros((*img.shape, 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 255
        hsv[..., 2] = (mag_blur * 255).astype(np.uint8)

        # Convert HSV to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('mag', mag_blur*255)
        cv2.imshow('direction', bgr)
        cv2.imwrite('Latex/thesis/plots/eye_dataset/mag.png', mag_blur*255)
        cv2.imwrite('Latex/thesis/plots/eye_dataset/direction.png', bgr)
        cv2.imwrite('Latex/thesis/plots/eye_dataset/roi..png', img)
        self.set_gradient(mag_blur)
        self.set_direction(direction)
        return mag_blur, direction
    
    
    def _setup_points (self,points,normalv,grad_info):
        points_normal_vec = np.array([normalv[:,0],normalv[:,1]]).reshape(-1,2)
        #print(f'grad_info[:,0]:{grad_info[:,0]}')
        #print(f'grad_info[0,:]:{grad_info[0,:]}')

        # Split into x and y components of the gradient
        # grad_info [0] = magnitude
        # grad_info [1] = direction
        dx = grad_info[:,0]*np.cos((grad_info[:,1]))
        dy = grad_info[:,0]*np.sin((grad_info[:,1]))
        
        
        gradient_vec = np.array([dx, dy]).reshape(-1,2)
        points_gradient_vec = np.array([gradient_vec[:,0],gradient_vec[:,1]]).reshape(-1,2)
      
        magnitudes_normal = np.linalg.norm(points_normal_vec, axis=1)
        points_normal_vec = points_normal_vec / magnitudes_normal[:, np.newaxis]
        
        #print (f'points_gradient_vec: {points_gradient_vec}')

        #print (f'points_normal_vec: {points_gradient_vec}')
        self.set_normalvector(points_normal_vec)

        return points_normal_vec, points_gradient_vec
    

    
    def _cross(self, center, axis, angle, img_shape, num_points=20 ):
   
        points, _ = self._init_circle(center, axis, angle, img_shape, num_points)
        #print(f'points: {points}')
        #print(f'normal: {normal}')
        
        grad = self.get_gradient()
        points.reshape(-1,2)
        direction = self.get_direction()
        #print(f'grad: {grad}')
        #print(f'direction: {direction}')
        #print(f'grad.shape: {grad.shape}')
        
        grad_points = grad[np.round(points[0,:]).astype(int), np.round(points[1,:]).astype(int)]
        direction_points = direction [np.round(points[0,:]).astype(int), np.round(points[1,:]).astype(int)]
        normalv = self.get_normalvector()
        #print(f'normalv: {normalv}')
        grad_info = np.array([e for e in zip(grad_points, direction_points)])
        #print(f'grad_info: {grad_info}')
        points_normal_vec, points_gradient_vec = self._setup_points(points, normalv, grad_info)
        
        elementwise_product = np.multiply(points_normal_vec, points_gradient_vec)
        
        print(f'elementwise_product: {elementwise_product}')
        for idx, cross in enumerate( zip( points_normal_vec, points_gradient_vec)):
            print(f'points normal and gradient: {cross}, elementwise_product: {elementwise_product[idx]}')

        return elementwise_product
    

    
    def _energy(self, center, axis, angle, img, num_points=20, alpha=5, beta=2, gamma=2, delta=1, zeta= 0):
        #size_penalty = penalty_weight * (1 / (axis[0] * axis[1]))
        
        # Set up the constraints
        size_penalty = 1 / (axis[0] * axis[1])


        size_reward = 1 * (axis[0]//2 * axis[1]//2)
        print(f'cross_product: {np.sum(self._cross((center[0], center[1]), (axis[0], axis[1]), angle, img, num_points), axis=1)}')
        cross_product= np.sum(self._cross((center[0], center[1]), (axis[0], axis[1]), angle, img, num_points), axis = 1 )
        
        cross_product_reward = -np.sum(cross_product[cross_product > 0])
        cross_product_penalty =-np.sum(cross_product[cross_product < 0])
        cross_product_non = np.count_nonzero(cross_product == 0)
        
        cross_product_total = np.sum(cross_product, axis=0)
        print(f'cross_product_non: {cross_product_non}')
        print(f'cross_product_reward: {cross_product_reward}')
        print(f'cross_product_penalty: {cross_product_penalty}')
        print(f'size_reward: {size_reward}')
        print(f'energy: {alpha * cross_product_reward + beta * cross_product_penalty -  delta * size_reward}')
        
        #return  alpha * cross_product_reward + beta * cross_product_penalty+ gamma * size_penalty -  delta * size_reward + zeta * cross_product_non
        return - alpha * cross_product_total + beta *cross_product_penalty + gamma * size_penalty 


    
    def _optimize(self, center, axis, angle, img_shape, num_points=20 ):
        initial_guess = np.array([*center, *axis, angle])
        energy_func = lambda x: self._energy(x[:2], x[2:4], x[4], img_shape, num_points)
        print(f'img_shape: {img_shape}')
        # Set up the constraints

        
        #center_bounds = [(0, img_shape[0]), (0, img_shape[1])]
        #axis_bounds = [(1, (img_shape[0] )), (1, (img_shape[1]))]
        #angle_bounds = [(0, 360)]
        #print(f'bounds: {bounds}')
    
        bounds= ((0, img_shape[0]-1), (0, img_shape[1]-1), (1, (img_shape[0]-1 )), (1, (img_shape[1]-1)), (0, 360))
        constraints =   [{'type': 'ineq', 'fun': center_constraint1, 'args': (img_shape,)},
                        {'type': 'ineq', 'fun': center_constraint2, 'args': (img_shape,)},
                        {'type': 'ineq', 'fun': center_constraint3, 'args': (img_shape,)},
                        {'type': 'ineq', 'fun': center_constraint4, 'args': (img_shape,)}]
        
        
        
        #print(f'bounds: {bounds}')        
        # other possible methods: 'SLSQP', 'COBYLA', 'TNC', 'L-BFGS-B', 'SLSQP', 'trust-constr'
        result = minimize(energy_func, initial_guess, method='SLSQP',
                        bounds=bounds, constraints= constraints, tol = 1e-10,
                          options = {'maxiter': 1000}, callback=callback(self.get_roi(),(*center, *axis, angle)))
        
        optimized_params = result.x
        print(f'result.success: {result.success}')
        print(f'result.message: {result.message}')
        print(f' result.fun: {result.fun}')
        print(f'result.nfev: {result.nfev}')
        print(f'result.nit: {result.nit}')
        print(f'result.x: {result.x}')
        return optimized_params[:2], optimized_params[2:4], optimized_params[4]
    
    

    
'''
------------------------------------------------
Constraints:

min_x, min_y < center ellipse < max_x, max_y
   
------------------------------------------------
'''

def center_constraint1(x, img_shape):
    # x[0] is the center x coordinate
    # x[2] is the major axis length
    min_x = x[2] // 2
    return x[0] - min_x 

def center_constraint2(x, img_shape):
    # x[0] is the center x coordinate
    # x[2] is the major axis length
    max_x = img_shape[0] - x[2] // 2 
    return max_x - x[0] 

def center_constraint3(x, img_shape):
    # x[1] is the center y coordinate
    # x[3] is the minor axis length
    min_y = x[3] // 2

    return x[1] - min_y 

def center_constraint4(x, img_shape):
    # x[1] is the center y coordinate
    # x[3] is the minor axis length
    max_y = img_shape[1] - x[3] // 2
    return max_y - x[1] 
    


def plot(img, center, axis, angle):
    img = img.copy()
    cv2.ellipse(img, center, axis, angle, 0, 360, (255, 255, 0), 2)
    cv2.waitKey(1)
    return True

def callback(img,x):
    plot(img, (x[0], x[1]), (x[2], x[3]), x[4])
    

if __name__ == '__main__':
    image = cv2.imread('test_roi.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_copy = image.copy()
    contour = active_contour()
    contour._gradient(gray)
    contour._init_roi(image)
    img_shape = image.shape
    optimized_center, optimized_axis, optimized_angle = contour._optimize((img_shape[0]//2, img_shape[1]//2), (30, 25), 0, img_shape, 20)
    print(optimized_center, optimized_axis, optimized_angle)
    optimized_center = optimized_center.astype(np.int32)
    optimized_axis = np.array([optimized_axis[0]/2, optimized_axis[1]/2]).astype(np.int32)
    
    cv2.ellipse(image_copy, optimized_center, optimized_axis, optimized_angle, 0, 360, (0, 255, 0), 1)
    cv2.imshow('image', image_copy)
    cv2.waitKey(0)
    img_plot = cv2.imread('test_roi.png')
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2GRAY)
    
    contour._gradient(img_plot)
    cv2.waitKey(0)