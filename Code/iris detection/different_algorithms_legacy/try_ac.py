import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ActiveContour: 
    def __init__(self):
        """
        Initializes the ActiveContour class with properties such as points_ellipse, 
        roi, roi_gray, gradient, direction, normalvector, and energy.
        """
        self.points_ellipse = None
        self.roi = None
        self.roi_gray = None
        self.gradient = None
        self.direction = None
        self.normalvector = None
        self.energy = None

    # Setters and Getters
    def set_points_ellipse(self, points_ellipse):
        self.points_ellipse = points_ellipse

    def get_points_ellipse(self):
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
    
    def _init_roi(self, roi):
        """
        Initializes the region of interest with its properties such as roi, roi_gray, gradient, and direction.
        """
        self.set_roi(roi)
        self.set_roi_gray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        mag, direction = self._gradient(self.get_roi_gray())
        self.set_gradient(mag)
        self.set_direction(direction)
        return True
    
    def _init_circle(self, center, axis, angle, image_shape, num_points=20):
        """
        Initializes the circle points ellipse using center, axis, and angle parameters
        """
        t = np.linspace(0, 2*np.pi, num_points)
        x = round(axis[0] // 2) * np.cos(t)
        y = round(axis[1] / 2) * np.sin(t)
        grad_x_normal = -round(axis[0] / 2) * np.sin(t) 
        grad_y_normal = round(axis[1] / 2) * np.cos(t)
        length = 1
        normal_x = -grad_y_normal
        normal_y = grad_x_normal
        
        # Rotation Matrix
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        
        grad_normal = np.matmul(R, -np.array([normal_x, normal_y]))
        points_ellipse = np.matmul(R, np.array([x, y])).astype(np.float64)
        points_ellipse[0,:] += round(center[0])
        points_ellipse[1,:] += round(center[1])

        mask = np.zeros(image_shape).astype(np.uint8)
        for point_x, point_y, nx, ny in zip(points_ellipse[0], points_ellipse[1], grad_normal[0], grad_normal[1]):
            cv2.circle(mask, (round(point_x), round(point_y)), 1, (255, 255, 255), -1)
            start_point = (round(point_x), round(point_y))
            end_point = (round(point_x + length * nx), round(point_y + length * ny))
            cv2.line(mask, start_point, end_point, (255, 255, 255), 1)

        self.set_points_ellipse(points_ellipse)
        self.set_normalvector(grad_normal)
        return True
    
    
    def _balloon_force(self, points, expansion_speed, img):
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        edges = cv2.Canny(blurred, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        kernel = np.ones((5,5),np.uint8)
        dilated_mask = cv2.dilate(mask,kernel,iterations = 3)
        eroded_mask = cv2.erode(dilated_mask,kernel,iterations = 2)
        boundary = cv2.Canny(eroded_mask, 100, 200)
        speed = expansion_speed * boundary
        force_field = np.zeros(img.shape)
        for point in points.T:
            x, y = point
            if x < img.shape[0] and y < img.shape[1]:
                force_field[int(x), int(y)] = 1
        balloon_force_field = speed * force_field
        return balloon_force_field



    def _optimize(self, center, axis, angle, image_shape):
        # Define optimization function
        def _energy(x, points, img):
            energy = 0
            for point in points.T:
                x, y = point
                if x < img.shape[0] and y < img.shape[1]:
                    energy += img[int(x), int(y)]
            return energy
        
        # Run optimization
        optimization_result = minimize(_energy, [center[0], center[1], axis[0], axis[1], angle],
                                       args=(self.get_points_ellipse(), self.get_roi_gray()),
                                       method='L-BFGS-B', bounds=((0, image_shape[0]), (0, image_shape[1]),
                                                                   (0, None), (0, None), (0, 2*np.pi)))

        optimized_center = (optimization_result.x[0], optimization_result.x[1])
        optimized_axis = (optimization_result.x[2], optimization_result.x[3])
        optimized_angle = optimization_result.x[4]
        
        return optimized_center, optimized_axis, optimized_angle


    def run(self, img):
        img_shape = img.shape
        center = (img_shape[0]//2, img_shape[1]//2)
        axis = (30, 30)
        angle = 0
        self._init_roi(img)
        while True:
            print(f"Current center: {center}")
            print(f"Current axis: {axis}")
            print(f"Current angle: {angle}")
            optimized_center, optimized_axis, optimized_angle = self._optimize(center, axis, angle, img_shape)
            balloon_force_field = self._balloon_force(self.get_points_ellipse(), 0.05, img)
            self.set_points_ellipse(self.get_points_ellipse() + balloon_force_field)
            center, axis, angle = optimized_center, optimized_axis, optimized_angle
            if abs(center[0] - img_shape[0] // 2) < 5 and abs(center[1] - img_shape[1] // 2) < 5:
                break

if __name__ == '__main__':
    image = cv2.imread('test_roi.png')
    contour = ActiveContour()
    contour.run(image)