import cv2
import numpy as np
from scipy.linalg import solve
import math
from threading import Lock, Thread
from collections import namedtuple
import matplotlib.pyplot as plt


PI = 3.142
Ellipse = namedtuple('Ellipse', ['center', 'major_axis', 'minor_axis', 'angle'])

class EllipseFinderRansac:
    def __init__(self):
        self.points_per_iteration = []
        self.fit_error_per_iteration = []
        self.m_lock = Lock()
        self.ellipses = []
        self.best_fit_ellipses = []
        self.border = None
        self.coordinates_ROI = None

    def get_random_point(self, pts):
        idx = np.random.randint(len(pts))
        return pts[idx]

    def get_n_random_points(self, pts, n_points):
        random_points = []

        while len(random_points) <(n_points):
            
            random_points.append(self.get_random_point(pts))

        return random_points

   
        
    def process(self, data_points, min_radius, max_radius, n_iterations, image):
        if len(data_points) < 5:
            print("\n Not enough points to fit ellipse. Minimum 5 points required \n")
            return False

        self.ellipses.clear()
        self.best_fit_ellipses.clear()
        threads = []

        for _ in range(n_iterations):
            one_fit_points = self.get_n_random_points(data_points, 30)
            fit_error = 0

            # plot_chosen_points(image, one_fit_points, self.border, self.coordinates_ROI)
            t = Thread(target=self.fit_ellipse, args=(one_fit_points, min_radius, max_radius, fit_error))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if len(self.fit_error_per_iteration) == 0:
            return False

        min_error_idx = np.argmin(self.fit_error_per_iteration)
        min_error = self.fit_error_per_iteration[min_error_idx]

        self.best_fit_ellipses.append(self.ellipses[min_error_idx])

        return True

    def fit_ellipse(self, data_pts, min_radius, max_radius, fit_error):
        if len(data_pts) == 0:
            return False

        A = np.zeros((len(data_pts), 5))
        B = np.zeros(len(data_pts))

        for i in range(len(data_pts)):
            A[i, 0] = data_pts[i][0] * data_pts[i][1]
            A[i, 1] = data_pts[i][1] * data_pts[i][1]
            A[i, 2] = data_pts[i][0]
            A[i, 3] = data_pts[i][1]
            A[i, 4] = 1

            B[i] = -(data_pts[i][0] * data_pts[i][0])

        x = solve(np.dot(A.T, A), np.dot(A.T, B))

        b, c, d, e, f = x

        ellipse_check = (b * b) - (4 * c)

        if ellipse_check < 0:
            print(" set of points converges to an ellipse")
        else:
            print(" Points does't satisfy ellipse condition (B^2 - 4AC < 0)")
            return False

        m0 = np.array([[f, d / 2, e / 2], [d / 2, 1, b / 2], [e / 2, b / 2, c]])
        m1 = np.array([[1, b / 2], [b / 2, c]])

        eigenvalues, _ = np.linalg.eig(m1)
        lambda1, lambda2 = eigenvalues

        h = ((b * e) - (2 * c * d)) / ((4 * c) - (b * b))
        k = ((b * d) - (2 * e)) / ((4 * c) - (b * b))

        major_axis = np.sqrt(-np.linalg.det(m0) / (np.linalg.det(m1)) * lambda1)
        minor_axis = np.sqrt(-np.linalg.det(m0) / (np.linalg.det(m1)) * lambda2)

        alpha = (PI / 2) - np.arctan(((1 - c) / b) / 2)

        if min_radius <= major_axis <= max_radius and min_radius <= minor_axis <= max_radius:
            center = (h, k)

            with self.m_lock:
                self.ellipses.append(Ellipse(center, major_axis, minor_axis, alpha))

        else:
            return False

        errors = 0
        dist1 = 0
        dist2 = 0
        error1 = 0
        error2 = 0

        for i in range(len(data_pts)):
            x_pt = data_pts[i][0]
            y_pt = data_pts[i][1]

            xx = x_pt * x_pt
            mul = np.sqrt((b * b * xx) + (2 * e * b * x_pt) - (4 * c * xx) - (4 * c * d * x_pt) + (e * e) - (4 * c * f))
            yhat = -((e / 2) + ((b * x_pt) / 2) + (mul / 2)) / c
            yhat1 = -((e / 2) + ((b * x_pt) / 2) - (mul / 2)) / c

            di1 = (yhat) - data_pts[i][1]
            dist1 = di1 * di1

            di2 = yhat1 - data_pts[i][1]
            dist2 = di2 * di2

            if dist1 <= 5:
                error1 = error1 + dist1

            if dist2 <= 5:
                error2 = error2 + dist2

        with self.m_lock:
            self.fit_error_per_iteration.append(error1 + error2)

        return True

    def draw(self, vis):
        if vis is None:
            print(" Input image is empty ")
            return False

        #amount of spacing added
        border = self.border
        #coordinate of roi in full image 
        coordinates = self.coordinates_ROI
        x,y,w,h = coordinates
        x_offset = x-border
        y_offset = y-border
        
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for ellipse in self.best_fit_ellipses:
            for i in range(360):
                pt = (int(ellipse.center[0] + x_offset + (ellipse.major_axis * np.cos(i))), 
                     int(ellipse.center[1] + y_offset + (ellipse.minor_axis * np.sin(i))))
                center_point = (int(ellipse.center[0] + x_offset), int(ellipse.center[1] + y_offset))
                cv2.circle(vis, center_point, 1, (0, 0, 255), -1)
                cv2.circle(vis, pt, 1, (0, 0, 255), 1)
        return True
    
def plot_chosen_points(image, points, border, coordinates):
    plt.figure()
    plt.imshow(image, cmap='gray')
    border = border
    (x,y,w,h) = coordinates
    x_offset = x-border
    y_offset = y-border
    # Convert list of points to numpy array
    np_points = np.array(points)

    # Plot the chosen points
    plt.scatter(np_points[:, 1]+x_offset, np_points[:, 0]+ y_offset, c='red', marker='x')

    plt.title('Chosen Points for Ellipse Fitting')
    plt.show()
    
def get_roi(image, intensity_threshold=80, border = 10):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Find the highest peak in the lower intensity region
    lower_intensity_hist = hist[:intensity_threshold]
    peak_intensity = np.argmax(lower_intensity_hist)

    # Create a binary mask with pixels around the peak intensity
    lower_bound = np.array([max(0, peak_intensity - 10)], dtype=np.uint8)
    upper_bound = np.array([min(255, peak_intensity + 10)], dtype=np.uint8)
    mask = cv2.inRange(gray_image, lower_bound, upper_bound)

    # Get the contours of the ROI
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Get the bounding rectangle of the ROI
    x, y, w, h = cv2.boundingRect(max_contour)

    # Create a masked image with the ROI
    roi_mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, [max_contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(gray_image, roi_mask)

    # Return the masked ROI
    return masked_image[y-border:y+h+border, x-border:x+w+border], gray_image[y-border:y+h+border, x-border:x+w+border],(x,y,w,h, border)



def plot_histogram(image, row= None):
    # If the image is not grayscale, convert it to grayscale
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if row:
        if row < 0 or row >= gray_image.shape[0]:
            raise ValueError('Row number is out of the image bounds.')

        # Get the row from the image
        gray_image = gray_image[row, :]
        
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])


    # Plot the histogram
    plt.figure()
    plt.title('Histogram of the Image')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def filter_points_by_gradient_direction(points, gradient_directions, ellipse_center, angle_threshold):
    """
    Filter out points based on the gradient direction.

    :param points: A list of points (x, y).
    :param gradient_directions: A 2D array representing the Sobel gradient directions.
    :param ellipse_center: The center of the ellipse (x, y).
    :param angle_threshold: The angle threshold for filtering points.
    :return: A list of points with gradient directions pointing towards or away from the ellipse center.
    """

    filtered_points = []
    for (x, y) in points:
        gradient_direction = gradient_directions[y, x]
        direction_to_center = np.arctan2(ellipse_center[1] - y, ellipse_center[0] - x)

        angle_diff = np.rad2deg(np.abs(gradient_direction - direction_to_center)) % 180

        if angle_diff <= angle_threshold or 180 - angle_diff <= angle_threshold:
            filtered_points.append((x, y))

    return filtered_points




def main():
    # Load input image and convert to grayscale
    input_image_path = 'test.jpg'
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot_histogram(gray)

    # Calculate the gradient in x and y directions using the Sobel operator
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude and direction of the gradient
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_directions_rad = np.arctan2(sobel_y, sobel_x)

    # Convert the gradient directions from radians to degrees
    gradient_directions_deg = np.rad2deg(gradient_directions_rad)

    # Normalize the gradient directions to the range of 0 to 255
    gradient_directions_normalized = cv2.normalize(gradient_directions_deg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)



    # Normalize the gradient magnitude to the range [0, 255] and convert to 8-bit unsigned integers
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Gradient Y', cv2.convertScaleAbs(sobel_y))
    cv2.imshow('Magnitude', normalized_magnitude)
    cv2.imshow('Direction', gradient_directions_normalized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Display the gradient images
    cv2.imshow('Gradient X', cv2.convertScaleAbs(sobel_x))
    
    # Get the region of interest
    masked, gray_mask, (x,y,w,h, border) = get_roi(gray)
    
    print ('coords reciefed',x,y,w,h,border)
    # Display the ROI
    cv2.imshow('gray_masked', gray_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Apply edge detection (Canny) to extract points for RANSAC
    edges = cv2.Canny(gray_mask, 50, 150)
    cv2.imshow('canny', edges)
    # Extract edge points (non-zero pixels) for RANSAC input
    edge_points = np.argwhere(edges > 0)

    # Define parameters for RANSAC ellipse fitting
    min_radius = 30
    max_radius = 100
    n_iterations = 500

    # Create an instance of EllipseFinderRansac and run RANSAC
    ellipse_finder = EllipseFinderRansac()
    ellipse_finder.border = border
    ellipse_finder.coordinates_ROI = (x,y,w,h)
    
    success = ellipse_finder.process(edge_points, min_radius, max_radius, n_iterations, img)

    # If successful, draw the detected ellipses on the input image
    if success:
        ellipse_finder.draw(img)
        cv2.imshow('Detected Ellipses', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No ellipses found.")


if __name__ == '__main__':
    main()
    