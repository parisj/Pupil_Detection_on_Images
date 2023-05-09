import numpy as np 
import cv2
from skimage.measure import EllipseModel
import exceptions
from scipy import optimize

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
    

    
    # Convert mask into points (contour)    
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
    
    #Calculate the area of the ellipse
    def calc_area(self, params):
        return np.pi * params[1][0] * params[1][1]
    


    
    def fit(self):
        best_inliers = np.empty((0, 2), dtype=np.int32)
        best_border = np.empty((0, 2), dtype=np.int32)


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
            #if not ellipse_fit.estimate(np.flip(points[idx,:])):
            #   continue
            params = cv2.fitEllipse(points[idx,:])
            
            #print(f'get_params: {params}')
            if params is None:
                continue
            #print(f'params: {params}')

            #plot_points_and_circle(params,points[idx,:])

            area = self.calc_area(params)
            #print(f'points: {points}, params: {params}, threshold: {threshold}, area: {area}, best_area: {best_area}')
            best_ellipse, best_inliers, best_area, best_border = evaluate(points, params, threshold, area, best_area,
                                                             best_inliers,best_border,best_ellipse)

            
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
    
        e = (x,y,a,b,theta)
        return e, best_inliers, best_area, best_border


            
    

def boundary_dist(phi, point, params, opti=True):
    #print(f'point boundary dist: {point}')
    #print(f'params boundary dist: {params}')
    #print(f'phi boundary dist: {phi}')
    center, axis, angle = params
    xc, yc = center
    a, b = axis
    #coordination transformation 
    #new center
    #print(f'xr: {xr}')
    #print(f'yc: {yc}')
    
    #rotate over new center 
    x0 =xc + a * np.cos(angle) * np.cos(phi) - b * np.sin(angle) * np.sin(phi)
    y0 =yc + a * np.sin(angle) * np.cos(phi) + b * np.cos(angle) * np.sin(phi)
    #print(f'x0: {x0}')
    x_error = point[0] - x0
    y_error = point[1] - y0
    dist = np.sqrt(x_error**2 + y_error**2)
    
    if opti == False:
        return x_error, y_error
    
    return dist



def dist_boundary(point, params, threshold):
    center, axis, angle = params
    xc, yc = center
    a, b = axis
    
    if a < b: 
        c = a
        b = a
        a = c
    #phi = np.arctan2(point[1]-yc, point[0]-xc)
    #print(f'phi: {phi}')
    #phi_o, _ = optimize.leastsq(boundary_dist,phi, args=(point,params))
    #print(f'phi_o: {phi_o}')
    
    #x_error,y_error = boundary_dist(phi_o,point, params, opti= False)
    
    #print(f'distance: {distance}')
    #distance = np.sqrt(x_error**2 + y_error**2)
    dist = is_inside_ellipse(point, params)
    print(f'dist: {dist}')
    print(f'is around: {dist is np.around(2*a,decimals=1)}')
    if dist is np.around(a, decimals=2):
        return 0
    
    elif dist < a:
        return 1
    
    else: 
        return 2
    
def plot_points(xc,yc,a,b,angle, point, rot_point):
    
    mask = np.zeros((200,200), dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.ellipse(mask, (round(yc),round(xc)), (round(a / 2),round(b / 2)), angle, 0, 360, (255,0,0), 1)
    mask = cv2.ellipse(mask, (0 + 100,0 + 100),(round(a / 2),round(b / 2)),0 , 0, 360, (0,255,255), 1)
    mask = cv2.circle(mask, (round(rot_point[0]) + 100,round(rot_point[1]) + 100), 3, (0,150,0), 1)
    mask = cv2.circle(mask, (point[0],point[1]), 3, (0,0,255), -1)
    mask = cv2.ellipse(mask,(100,100), (50,25), 20, 0, 360, (255,255,255), 1)
    mask = cv2.circle(mask, (round(rot_point[0])+ 100, round(rot_point[1]) + 100), 3, (0,150,0), 1)
    
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    
    
def is_inside_ellipse(point,params):
    
    xc,yc = params[0]
    a,b = params[1]
    print(f'a: {a} b:{b}')
    if b > a: 
        c = a
        b = a
        a = c
    angle = params[2]
    
    xp = point[0]
    yp = point[1]
    
    xn = xp-xc
    yn = yp-yc
    
    
    #[x'] = [c, -s] [x]
    #[y'] = [s,  c] [y]
    
    x_rot = xn*np.cos(angle) - yn*np.sin(angle)
    y_rot = xn*np.sin(angle) + yn*np.cos(angle)
    
    plot_points(xc,yc,a,b,angle,point,(x_rot,y_rot))
    foci = np.sqrt((a/2)**2 - (b/2)**2)
    print(f'foci: {foci}')
    foci = np.array([-foci,foci])
    dist = np.sqrt((x_rot-foci[0])**2 + y_rot**2) + np.sqrt((x_rot-foci[1])**2 + y_rot**2)
    return dist

#@njit(nogil=True)
def evaluate(points, params, threshold, area, best_area, best_inliers,best_border, best_ellipse):

    # Find the inliers
    #print(f'points: {points}, params: {params}, threshold: {threshold}, area: {area}, best_area: {best_area}')
    inliers = np.zeros((0, 2), dtype=np.int32) 
    border = np.zeros((0,2), dtype=np.int32)# initialize best_inliers to an empty numpy array
    for point in points:
        dist = dist_boundary(point, params, threshold)
        #print(f'dist: {dist}')
        if dist == 0:
            border = np.vstack((border,  point))
        
        elif dist == 1:
            inliers = np.vstack((inliers,point))
                  
        else:
            continue   
                 
    if inliers.shape[0] >= best_inliers.shape[0]:
        if border.shape[0] >= best_border.shape[0]:
            
            best_inliers = inliers
            best_border = border
            best_ellipse = params
            best_area = area

    return best_ellipse, best_inliers, best_area, best_border


if __name__ == '__main__':
    mask = np.zeros((200,200),dtype = np.uint8)
    #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    mask = cv2.ellipse(mask,(100,100),(50,25),20,0,360,(255,0,0),1)
    cv2.imshow('mask',mask)
    
    rans = ransac(mask, 110 ,0.01 )
    a,b,c,d = rans.ransac_start()
    print(f'best_ellipse: {a} best_inliers: {b} best_area: {c} best_border: {d}')
    print(f'lenght of best_inliers: {len(b)}')
    print(f'leng points_contour: {len(rans.get_points_contour())}')
    print(f'lenght of best_border: {len(d)}')
    test = np.zeros((200,200),dtype = np.uint8)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    mask = cv2.ellipse(mask, (a[1],a[0]), (round(a[2]/2),round(a[3]/2)), a[4],0,360,(0,255,0),1)
    cv2.imshow('test',mask)
    cv2.waitKey(0)