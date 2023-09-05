import numpy as np 
import cv2
import exceptions
import math
from numba import njit



class ransac:
    def __init__(self, mask, iterations, threshold, callback_bool = False):
        self.mask = mask
        #print(f'mask: {mask}')
        #print(f'mask.shape: {mask.shape}')
        self.iterations = iterations
        self.threshold = threshold
        self.points_contour = None
        self.test_mask = np.zeros((385,640), dtype=np.uint8)
        self.callback_bool = callback_bool
    
    # Getter and Setter
        
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
    

    
    # Convert mask into points (contour)    
    def init_points_contour(self):
        mask = self.get_mask()
        
        if mask is None:
            raise exceptions.EmptyImageError("Mask for init_points_contour is empty")

        contour = mask
        
        if len(contour[0]) == 0 or contour is None:
            raise exceptions.EmptyImageError("Contour for init_points_contour is empty")
        
        self.set_points_contour(np.squeeze(contour,axis=1))
        #print (f'contour: {contour[0]}')
        return True 
    
    #Calculate the area of the ellipse
    def calc_area(self, params):
        return np.pi * params[1][0]/2 * params[1][1]/2
    

    def fit(self):
        
        best_inliers = np.empty((0, 2), dtype=np.int32)
        best_border = np.empty((0, 2), dtype=np.int32)
        best_ellipse = None
        best_area = 1000000
        best_distance = 1000000
        best_stat = -np.inf
        points = np.array(self.get_points_contour())
        threshold = self.get_threshold()
        contour = self.get_mask()
        
        for i in range (self.get_iterations()):
            self.test_mask = np.zeros((220,220), dtype=np.uint8)
            
            mask_points = self.test_mask
            mask_points = cv2.cvtColor(mask_points, cv2.COLOR_GRAY2BGR)
            # Select 5 random points from the contour
            idx = np.random.choice(len(points), 5, replace = False)
            points_idx = points[idx,:]
            points_idx[:, 0], points_idx[:, 1] = points_idx[:, 1], points_idx[:, 0].copy()
            
            #fit ellipse to the 5 points
            params = cv2.fitEllipse(points[idx,:])

            if params is None :
                continue

            #plot_points_and_circle(params,points[idx,:], self.get_mask())

            area = self.calc_area(params)
            if params[1][0] == math.inf or params[1][1] == math.inf:
                continue
            best_ellipse, best_inliers, best_area, best_border, best_distance, best_stat = evaluate(points,
                                                                                                    params,
                                                                                                    threshold,
                                                                                                    area,
                                                                                                    best_area,
                                                                                                    best_inliers,
                                                                                                    best_border,best_ellipse, 
                                                                                                    best_distance,
                                                                                                    mask_points,
                                                                                                    best_stat, 
                                                                                                    self.callback_bool
                                                                                                    )
            
            if self.callback_bool:                                                     
                cv2.ellipse(mask_points, (round(params[0][0]),round(params[0][1])), (round(params[1][0] / 2),round(params[1][1] / 2)), params[2], 0, 360, (0,255,0), 1)
                cv2.imshow('test_mask', mask_points)
                cv2.waitKey(1)
        return best_ellipse, best_inliers, best_area, best_border, best_stat

    def ransac_start(self):
        self.init_points_contour()
        best_ellipse, best_inliers, best_area, best_border, best_stat = self.fit()
        
        center, axis, theta = best_ellipse
        x,y = center
        a,b = axis
        x = round(x)
        y= round(y)
        a = round(a)
        b = round(b)
    
        e = ((x,y),(a,b),theta)
        return e, best_inliers, best_area, best_border, best_stat


@njit
def distance_to_ellipse_eigen(x, y, h, k, a, b, theta):
    """Calculate the distance from a point (x,y) to an ellipse using eigenvalues and eigenvectors"""
    # Convert theta to deg and rotate it 180 degrees
    theta = np.deg2rad(-theta+180)
    
    #((x-h)cosθ + (y-k)sinθ)^2/a^2 + ((x-h)sinθ - (y-k)cosθ)^2/b^2 = 1
    #Ax^2 + Bxy + Cy^2 = 1
    #[x, y] * [[A, B/2], [B/2, C]] * [x, y] = 1
    # Construct the matrix of the ellipse
    
    ellipse_matrix = np.array([[a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2, (a**2 - b**2) * np.sin(theta) * np.cos(theta)],
                               [(a**2 - b**2) * np.sin(theta) * np.cos(theta), a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2]])

    # Compute the eigenvalues and eigenvectors of the ellipse matrix
    eigenvalues, eigenvectors = np.linalg.eig(ellipse_matrix)
    # Eigenvectors need to be sorted!
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    #Check if valid covariance matrix
    if np.isnan(np.max(eigenvalues)) or np.max(eigenvalues) < 0:
        return 100
    if np.isnan(np.min(eigenvalues)) or np.min(eigenvalues) < 0:
        return 100
  
    # The semi-major and semi-minor axes of the ellipse in the rotated coordinate system are given by the square roots of the eigenvalues
    a_eigen = np.sqrt(np.max(eigenvalues))
    b_eigen = np.sqrt(np.min(eigenvalues))
    
    #check if valid eigenvalues
    if a_eigen == 0 or b_eigen == 0:
        return 100
    
    # The rotation matrix is given by the eigenvectors
    R = eigenvectors
    
    # Translate and rotate the point to the new coordinate system
    point = np.dot(R.T, np.array([x - h, y - k]))
    
    # Calculate the distance to the ellipse in the new coordinate system
    r = np.hypot(point[0] / a_eigen, point[1] / b_eigen)
    distance = r - 1  # Subtract 1 because the ellipse in the new coordinate system has a radius of 1
    return distance

#@njit
def evaluate(points, params, threshold, area, best_area, best_inliers,best_border, best_ellipse, best_distance, mask_points, best_stat, callback_bool):
    
    center, axis, angle = params
    xc, yc = center
    a, b = axis
    total_distance = 0
    
    # check if the ellipse is valid
    if math.isnan(a) or math.isinf(a):
        return best_ellipse, best_inliers, best_area, best_border, best_distance, best_stat

    if math.isnan(b) or math.isinf(b):
        return best_ellipse, best_inliers, best_area, best_border, best_distance, best_stat


    
    # Find the inliers
    inliers = np.zeros((0, 2), dtype=np.int32) 
    border = np.zeros((0,2), dtype=np.int32)# initialize best_inliers to an empty numpy array
    
    for point in points:
        
        dist= distance_to_ellipse_eigen(point[0],point[1],xc,yc,round(a/2),round(b/2),angle)
        total_distance += np.abs(dist)**2/points.shape[0]

        # if border
        if dist > -threshold and dist < threshold:
            reshaped_point = np.empty((1, 2), dtype=np.int32)
            reshaped_point[0] = point
            border = np.append(border, reshaped_point, axis=0)
            if callback_bool:
                cv2.circle(mask_points, (point[0],point[1]), 2, (255,0,0), -1)
        # if inlier
        elif dist < 0:
            reshaped_point = np.empty((1, 2), dtype=np.int32)
            reshaped_point[0] = point
            inliers = np.append(inliers, reshaped_point, axis=0)
            if callback_bool:
                cv2.circle(mask_points, (point[0],point[1]), 1, (0,0,255), -1)

        #if outlier    
        else:
            if callback_bool:
                cv2.circle(mask_points, (point[0],point[1]), 1, (255,255,255), -1)
            continue   
    
    # fit score calculation and comparison
    n_inliers = inliers.shape[0]/points.shape[0]
    n_border = border.shape[0]/points.shape[0]
    area = area
    stat = 170* n_inliers + 280* n_border - 1*area/(points.shape[0])*np.pi - (total_distance*border.shape[0])
    if stat > best_stat :

        best_stat = stat
        best_inliers = inliers
        best_border = border
        best_ellipse = params
        best_area = area
        best_distance = total_distance
        
    return best_ellipse, best_inliers, best_area, best_border, best_distance, best_stat


def plot_points_and_circle(params,points, contour):
    center, axis, angle = params
    xc, yc = center
    a, b = axis
    mask = np.zeros((385,640), dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask, contour, -1, (0,255,0), cv2.FILLED)


    mask = cv2.ellipse(mask, (round(xc),round(yc)), (round(a / 2),round(b / 2)), angle, 0, 360, (255,0,0), 1)
    for p in points:
        mask = cv2.circle(mask, (p[0],p[1]), 1, (0,0,255), 1)
    
    cv2.imshow('mask', mask)
    cv2.waitKey(1)


    
