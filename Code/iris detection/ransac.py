import numpy as np 
import cv2
from skimage.measure import EllipseModel
import exceptions
from numba import njit

class ransac:
    def __init__(self, mask, iterations, threshold):
        self.mask = mask
        self.iterations = iterations
        self.threshold = threshold
        self.points_contour = None
        
    def set_mask(self, contour):
        self.contour = contour

    def get_mask(self):
        return self.mask

    def set_iterations(self, iterations):
        self.iterations = iterations

    def get_iterations(self):
        return self.iterations

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_threshold(self):
        return self.threshold
    
    def set_points_contour(self, points_contour):
        self.points_contour = points_contour
 
    def get_points_contour(self):
        return self.points_contour
    
    
       
    def init_points_contour(self):
        mask = self.get_mask()
        
        if mask is None:
            raise exceptions.EmptyImageError("Mask for init_points_contour is empty")

        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if len(contour[0]) == 0 or contour is None:
            raise exceptions.EmptyImageError("Contour for init_points_contour is empty")
        
        self.set_points_contour(np.squeeze(contour[0],axis=1))
        #print (f'contour: {contour[0]}')
        return True 
    
    def calc_area(self, params):
        return np.pi * params[2] * params[3]
    


    
    def fit(self):
        best_inliers = np.empty((0, 2), dtype=np.int32)


        best_ellipse = None
        best_area = 0
        ellipse_fit = EllipseModel()
        points = np.array(self.get_points_contour())
        threshold = self.get_threshold()
        #print(f'points.shape: {points.shape}')
        #print(f'points[0]: {points[0]}')
        #print(f'points[0].shape: {points[0].shape}')

        for i in range (self.get_iterations()):
            # Select 5 random points from the contour
            idx = np.random.choice(len(points), 5, replace = False)
            #print(f'i: {i} idx: {idx}')
            # Fit an ellipse to the selected points
            #print(f'points[idx,:]: {points[idx,:]}')
            #print(f'points[idx,:].shape: {points[idx,:].shape}')
            if not ellipse_fit.estimate(points[idx,:]):
                continue
            params = np.array([ellipse_fit.params])[0]
            area = self.calc_area(params)
            #print(f'points: {points}, params: {params}, threshold: {threshold}, area: {area}, best_area: {best_area}')
            best_ellipse, best_inliers, best_area = evaluate(points, params, threshold, area, best_area, best_inliers,best_ellipse)

            
        return best_ellipse, best_inliers, best_area

    def ransac_start(self):
        self.init_points_contour()
        best_ellipse, best_inliers, best_area = self.fit()
        
        x, y, a, b, theta = best_ellipse
        x = round(x)
        y = round(y)
        a = round(a)
        b = round(b)
    
        e = (x,y,a,b,theta)
        return e, best_inliers, best_area
    
@njit(nogil=True)
def distance(point, params,threshold):
   # x, y = point
   # xc, yc, a, b, theta = params
   # 
   # A = (a**2) * (np.sin(theta)**2) + (b**2) * (np.cos(theta)**2)
   # B = 2 * (b**2 - a**2) * np.sin(theta) * np.cos(theta)
   # C = (a**2) * (np.cos(theta)**2) + (b**2) * (np.sin(theta)**2)
   # D = -2 * A * xc - B * yc
   # E = -2 * C * yc - B * xc
   # F = A * xc**2 + B * xc * yc + C * yc**2 - a**2 * b**2
   # 
   # ellipse_val = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
   # 
   # 
   #  # Compute the distance between the point and the center of the ellipse
   # dist_to_center = norm(point - center)
#
   # # Find the closest point on the ellipse boundary to the given point
   # angle = np.arctan2(y - yc, x -xc)
   # closest_point_on_boundary = center + np.array([axes[0] * np.cos(angle), axes[1] * np.sin(angle)])
   # 
   # # Compute the distance between the closest point on the boundary and the center of the ellipse
   # dist_to_boundary = norm(closest_point_on_boundary - center)
#
   # # Check if the point is inside the ellipse
   # if dist_to_center < dist_to_boundary:
   #     return True
   # else:
   #     return False
   
    # Unpack the ellipse parameters
    xc, yc, a, b, angle = params
    
    # Convert the angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Compute the rotation matrix
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    # Transform the point and the ellipse to a coordinate system where the ellipse is centered
    # and has axes aligned with the x and y axes
    point_centered = point - np.array([xc, yc])
    point_transformed = rot_matrix @ point_centered
    a_transformed = a
    b_transformed = b
    ellipse_centered = (0, 0, a_transformed, b_transformed, 0)
    
    # Compute the distance to the center
    center_dist = np.linalg.norm(point_transformed)
    
    # Compute the distance to the boundary
    boundary_dist = ((point_transformed[0] / a_transformed) ** 2 +
                     (point_transformed[1] / b_transformed) ** 2) ** 0.5
    
    # Check if the point is on, inside or outside the ellipse
    if (boundary_dist <1+ threshold/2 and boundary_dist> 1-threshold):
        return 0

    else:
        return np.abs(1-boundary_dist)
    
#@njit(nogil=True)
def evaluate(points, params, threshold, area, best_area, best_inliers, best_ellipse):
    # Find the inliers
    inliers = np.zeros((0, 2), dtype=np.int32)     # initialize best_inliers to an empty numpy array
    for point in points:
        dist = distance(point, params, threshold)
        if dist < threshold:
            inliers = np.vstack((inliers,  point))
                     
    if inliers.shape[0] > best_inliers.shape[0]:
        best_inliers =inliers
        best_ellipse = params
        best_area = area
    params[4] = params[4] * 180 / np.pi
    return best_ellipse, best_inliers, best_area


if __name__ == '__main__':
    mask = np.zeros((200,200),dtype = np.uint8)
    mask = cv2.ellipse(mask,(100,100),(50,25),20,0,360,255,1)
    cv2.imshow('mask',mask)
    
    rans = ransac(mask, 500 ,0.0001 )
    a,b,c = rans.ransac_start()
    print(f'best_ellipse: {a} best_inliers: {b} best_area: {c}')
    print(f'lenght of best_inliers: {len(b)}')
    print(f'leng points_contour: {len(rans.get_points_contour())}')
    test = np.zeros((200,200),dtype = np.uint8)
    mask = cv2.ellipse(mask, (a[0],a[1]), (a[2],a[3]), a[4],0,360,200,1)
    cv2.imshow('test',mask)
    cv2.waitKey(0)