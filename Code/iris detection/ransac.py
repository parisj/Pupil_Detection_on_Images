import numpy as np 
import cv2
import exceptions
#from numba import njit


class ransac:
    def __init__(self, mask, iterations, threshold):
        self.mask = mask
        #print(f'mask: {mask}')
        #print(f'mask.shape: {mask.shape}')
        self.iterations = iterations
        self.threshold = threshold
        self.points_contour = None
        self.test_mask = np.zeros((220,220), dtype=np.uint8)
    
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
        #print(f'mask: {mask}')
        #print(f'mask.shape: {mask.shape}')
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
        points = np.array(self.get_points_contour())
        threshold = self.get_threshold()
        contour = self.get_mask()
        for i in range (self.get_iterations()):
            # Select 5 random points from the contour
            idx = np.random.choice(len(points), 5, replace = False)
            points_idx = points[idx,:]
            points_idx[:, 0], points_idx[:, 1] = points_idx[:, 1], points_idx[:, 0].copy()
            
            params = cv2.fitEllipse(points[idx,:])
            #plot_points(params[0][0],params[0][1],params[1][0],params[1][1], params[2],points[idx,:],'test1', contour)

            #print(f'get_params: {params}')
            if params is None :
                continue
            #print(f'params: {params}')

            plot_points_and_circle(params,points[idx,:], self.get_mask())

            area = self.calc_area(params)
            #print(f'points: {points}, params: {params}, threshold: {threshold}, area: {area}, best_area: {best_area}')
            best_ellipse, best_inliers, best_area, best_border, best_distance = evaluate(points, params, threshold, area, best_area,
                                                             best_inliers,best_border,best_ellipse, best_distance)

            
        return best_ellipse, best_inliers, best_area, best_border

    def ransac_start(self):
        self.init_points_contour()
        best_ellipse, best_inliers, best_area, best_border = self.fit()
        
        center, axis, theta = best_ellipse
        x,y = center
        a,b = axis
        x = round(x)
        y= round(y)
        a = round(a)
        b = round(b)
    
        e = ((x,y),(a,b),theta)
        return e, best_inliers, best_area, best_border


#@njit
def distance_to_ellipse_eigen(x, y, h, k, a, b, theta):
    """Calculate the distance from a point (x,y) to an ellipse using eigenvalues and eigenvectors"""
    # Construct the matrix of the ellipse
    
    #((x-h)cosθ + (y-k)sinθ)^2/a^2 + ((x-h)sinθ - (y-k)cosθ)^2/b^2 = 1
    #Ax^2 + Bxy + Cy^2 = 1
    #[x, y] * [[A, B/2], [B/2, C]] * [x, y] = 1
    # Construct the matrix of the ellipse
    ellipse_matrix = np.array([[a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2, (b**2 - a**2) * np.sin(theta) * np.cos(theta)],
                               [(b**2 - a**2) * np.sin(theta) * np.cos(theta), a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2]])

    # Compute the eigenvalues and eigenvectors of the ellipse matrix
    eigenvalues, eigenvectors = np.linalg.eig(ellipse_matrix)
    if np.isnan(np.sqrt(np.max(eigenvalues))) or np.isnan(np.sqrt(np.min(eigenvalues))):
        return 100
    # The semi-major and semi-minor axes of the ellipse in the rotated coordinate system are given by the square roots of the eigenvalues
    a_eigen = round(np.sqrt(np.max(eigenvalues)))
    b_eigen = round(np.sqrt(np.min(eigenvalues)))
    if a_eigen == 0 or b_eigen == 0:
        return 100
    # The rotation matrix is given by the eigenvectors
    R = eigenvectors
    #print(f'x: {x}, y: {y}, h: {h}, k: {k}, a: {a}, b: {b}, theta: {theta}, a_eigen: {a_eigen}, b_eigen: {b_eigen}, R: {R}')
    # Translate and rotate the point to the new coordinate system
    point = np.dot(R.T, np.array([x - h, y - k]))
    # Calculate the distance to the ellipse in the new coordinate system
    r = np.hypot(round(point[0]) / a_eigen, round(point[1]) / b_eigen)
    distance = r - 1  # Subtract 1 because the ellipse in the new coordinate system has a radius of 1
    #print(distance)
    return distance

#@njit(nogil=True)
def evaluate(points, params, threshold, area, best_area, best_inliers,best_border, best_ellipse, best_distance):
    
    center, axis, angle = params
    xc, yc = center
    a, b = axis
    total_distance = 0
        
    # Find the inliers
    #print(f'points: {points}, params: {params}, threshold: {threshold}, area: {area}, best_area: {best_area}')
    inliers = np.zeros((0, 2), dtype=np.int32) 
    border = np.zeros((0,2), dtype=np.int32)# initialize best_inliers to an empty numpy array
    for point in points:
        dist= distance_to_ellipse_eigen(point[0],point[1],xc,yc,round(a/2),round(b/2),angle)
        total_distance += np.abs(dist)**2/points.shape[0]
        #print(f'dist: {dist}')
        if np.isclose(dist, 0, atol=threshold):
            border = np.vstack((border,  point))
        
        if dist < 0 or np.isclose(dist, 0, atol=threshold) :
            inliers = np.vstack((inliers,point))
                  
        else:
            continue   
    BOOL_AREA =  area < best_area
    BOOL_DISTANCE =  total_distance < best_distance
    BOOL_INLIERS = inliers.shape[0] >=best_inliers.shape[0]
    
    if BOOL_INLIERS and BOOL_DISTANCE:


        best_inliers = inliers
        best_border = border
        best_ellipse = params
        best_area = area
        best_distance = total_distance
        print('found better ellipse')
    return best_ellipse, best_inliers, best_area, best_border, best_distance


def plot_points_and_circle(params,points, contour):
    center, axis, angle = params
    xc, yc = center
    a, b = axis
    mask = np.zeros((220,220), dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask, contour, -1, (0,255,0), 1)


    mask = cv2.ellipse(mask, (round(xc),round(yc)), (round(a / 2),round(b / 2)), angle, 0, 360, (255,0,0), 1)
    for p in points:
        mask = cv2.circle(mask, (p[0],p[1]), 1, (0,0,255), 1)
    
    cv2.imshow('mask', mask)
    cv2.waitKey(1)


        


def plot_points(xc,yc,a,b,angle, point, rot_point):
    
    mask = np.zeros((200,200), dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.ellipse(mask, (round(xc),round(yc)), (round(a / 2),round(b / 2)), angle, 0, 360, (255,0,0), 1)
    mask = cv2.ellipse(mask, (0 + 100,0 + 100),(round(a / 2),round(b / 2)),0 , 0, 360, (0,255,255), 1)
    mask = cv2.circle(mask, (round(rot_point[0]) + 100,round(rot_point[1]) + 100), 3, (0,150,0), 1)
    mask = cv2.circle(mask, (point[0],point[1]), 3, (0,0,255), -1)
    mask = cv2.ellipse(mask,(100,100), (50,25), 20, 0, 360, (255,255,255), 1)
    cv2.drawContours(mask, contours, -1, (255,0,0), 1)

    for p in points_idx: 
        mask = cv2.circle(mask, (p[0],p[1]), 3, (0,255,0), -1)
    
    cv2.imshow(name, mask)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    mask = np.zeros((200,200),dtype = np.uint8)
    #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    mask = cv2.ellipse(mask,(100,100),(50,25),50,0,360,(255,0,0),1)
    cv2.imshow('mask',mask)
    
    rans = ransac(mask, 300 , 0.1)
    a,b,c,d = rans.ransac_start()
    print(f'best_ellipse: {a} best_inliers: {b} best_area: {c} best_border: {d}')
    print(f'leng points_contour: {len(rans.get_points_contour())}')
    print(f'lenght of best_inliers: {len(b)}')
    print(f'lenght of best_border: {len(d)}')
    test = np.zeros((200,200),dtype = np.uint8)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    mask = cv2.ellipse(mask, (a[1],a[0]), (round(a[2]/2),round(a[3]/2)), a[4],0,360,(0,255,0),1)
    cv2.imshow('test',mask)
    cv2.waitKey(0)