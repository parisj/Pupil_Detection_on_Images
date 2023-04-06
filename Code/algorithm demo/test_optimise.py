import cv2
import numpy as np
from scipy.optimize import minimize

class Ellipse:
    def __init__(self, center, major_axis, minor_axis, angle):
        self.center = center
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.angle = angle

def compute_gradient(image):
    # Berechne den Gradienten des Bildes
    gradient = np.gradient(image)
    abs_grad = np.absolute(np.sum(gradient, axis=0))
    return abs_grad

def is_on_ellipse_border(point, ellipse, tolerance=1e-3):
    x, y = point
    x0, y0 = ellipse.center
    a, b = ellipse.major_axis, ellipse.minor_axis
    theta = np.radians(ellipse.angle)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    ellipse_eq = (((x - x0) * cos_theta + (y - y0) * sin_theta) ** 2 / a ** 2) + \
                 (((x - x0) * sin_theta - (y - y0) * cos_theta) ** 2 / b ** 2)

    return abs(ellipse_eq - 1) < tolerance

def objective_function(params, image, maximize=True):
    # Definiere eine Zielfunktion
    center = (params[0], params[1])
    major_axis = params[2]
    minor_axis = params[3]
    angle = params[4]
    ellipse = Ellipse(center, major_axis, minor_axis, angle)

    # Zielfunktion basierend auf dem Gradienten und der Ellipse
    value = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if is_on_ellipse_border((j, i), ellipse):
                value += image[i, j]

    if maximize:
        value = -value

    return value

def optimize_ellipse_params(image, initial_params, maximize=True):
    # Optimiere die Parameter, um die Zielfunktion zu minimieren oder maximieren
    result = minimize(objective_function, initial_params, args=(image, maximize), method='Nelder-Mead')
    return result

def main():
    # Lade ein Bild und konvertiere es in Graustufen
    image = cv2.imread('test_roi.png', cv2.IMREAD_GRAYSCALE)

    # Berechne den Gradienten des Bildes
    gradient_image = compute_gradient(image)

    # Definiere die Anfangsparameter für die Ellipse (Mittelpunkt, Major-Achse, Minor-Achse und Winkel)
    initial_params = [image.shape[1] // 2, image.shape[0] // 2, 30,20, 0]

    # Optimiere die Parameter der Ellipse
    result = optimize_ellipse_params(gradient_image, initial_params, maximize=True)

    optimized_params = result.x
    print(f'Optimierte Parameter: Mittelpunkt = ({optimized_params[0]}, {optimized_params[1]}), Major-Achse = {optimized_params[2]}, Minor-Achse = {optimized_params[3]}, Winkel = {optimized_params[4]}')

    cv2.ellipse(image,(round(optimized_params[0]), round(optimized_params[1])),(round(optimized_params[2]/2), round(optimized_params[3]/2)),round(optimized_params[4]),0,360,(0,255,0), 2)
    cv2.imshow('image Result', image)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
