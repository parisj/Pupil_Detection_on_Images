import cv2
import numpy as np
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt


def get_gradients(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobel_x, sobel_y

def internal_energy(point, prev_point, next_point, alpha, beta):
    dp = point - prev_point
    dn = next_point - point
    curvature = dn - dp
    return alpha * np.linalg.norm(dp) + beta * np.linalg.norm(curvature)

def external_energy(point, gradients_x, gradients_y):
    x, y = int(point[0]), int(point[1])
    return -(gradients_x[x, y]**2 + gradients_y[x, y]**2)

def total_energy(point, prev_point, next_point, gradients_x, gradients_y, alpha, beta):
    E_int = internal_energy(point, prev_point, next_point, alpha, beta)
    E_ext = external_energy(point, gradients_x, gradients_y)
    return E_int + E_ext

def update_point(point, prev_point, next_point, gradients_x, gradients_y, alpha, beta):
    result = minimize(total_energy, point, args=(prev_point, next_point, gradients_x, gradients_y, alpha, beta))
    return result.x

def update_contour(points, gradients_x, gradients_y, alpha, beta):
    updated_points = np.zeros_like(points)
    num_points = points.shape[0]

    for i in range(num_points):
        prev_point = points[i - 1]
        point = points[i]
        next_point = points[(i + 1) % num_points]

        updated_point = update_point(point, prev_point, next_point, gradients_x, gradients_y, alpha, beta)
        updated_points[i] = updated_point

    return updated_points

def generate_initial_contour(center, radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points)
    return np.array([[center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)] for angle in angles])

def active_contour(image, initial_contour, alpha, beta, iterations):
    gradients_x, gradients_y = get_gradients(image)
    contour = initial_contour

    for _ in range(iterations):
        contour = update_contour(contour, gradients_x, gradients_y, alpha, beta)

    return contour


def plot_active_contour(image, contour_points, title):
    plt.imshow(image, cmap="gray")
    plt.plot(contour_points[:, 1], contour_points[:, 0], '-r', linewidth=2)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    image = cv2.imread("test_roi.png", cv2.IMREAD_GRAYSCALE)
    center = (150, 100)
    radius = 30
    num_points = 30
    initial_contour = generate_initial_contour(center, radius, num_points)

    alpha = 4
    beta = 3
    iterations = 100
    final_contour = active_contour(image, initial_contour, alpha, beta, iterations)
    print("Final contour: ", final_contour)
    plot_active_contour(image, final_contour, 'Result')
    
    