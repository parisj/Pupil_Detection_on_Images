import cv2
import numpy as np

def clip_contour(contour, img_shape):
    clipped_contour = np.clip(contour, (0, 0), (img_shape[1] - 1, img_shape[0] - 1))
    return clipped_contour

def compute_gradient(img_gray, contour, alpha, beta, gamma):
    energy_grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    energy_grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    contour_2d = contour.reshape((-1, 2))

    internal_energy_grad_x = alpha * np.gradient(contour_2d[:, 0], axis=0)
    internal_energy_grad_y = alpha * np.gradient(contour_2d[:, 1], axis=0)

    external_energy_grad_x = beta * energy_grad_x[contour_2d[:, 1], contour_2d[:, 0]]
    external_energy_grad_y = beta * energy_grad_y[contour_2d[:, 1], contour_2d[:, 0]]

    balloon_energy_grad_x = gamma * np.gradient(contour_2d[:, 0], axis=0)
    balloon_energy_grad_y = gamma * np.gradient(contour_2d[:, 1], axis=0)

    gradient_x = internal_energy_grad_x - external_energy_grad_x + balloon_energy_grad_x
    gradient_y = internal_energy_grad_y - external_energy_grad_y + balloon_energy_grad_y

    return np.concatenate((gradient_x[:, None], gradient_y[:, None]), axis=1)

def create_ellipse_contour(center, axes, angle, num_points=100):
    angle_rad = np.deg2rad(angle)
    t = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + axes[0] * np.cos(t) * np.cos(angle_rad) - axes[1] * np.sin(t) * np.sin(angle_rad)
    y = center[1] + axes[0] * np.cos(t) * np.sin(angle_rad) + axes[1] * np.sin(t) * np.cos(angle_rad)
    return np.column_stack((x, y)).astype(np.int32).reshape((-1, 1, 2))

# Load the image
img = cv2.imread('test.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (11, 11), 0)

# Initialize the contour as an ellipse
# Initialize the ellipse contour
center = (335, 305)
axes = (25, 35)
angle = 0
ellipse_points = cv2.ellipse2Poly(center, axes, angle, 0, 360, 5)
contour = ellipse_points.reshape((-1, 1, 2)).astype(np.int32)

# Set up the parameters for the balloon algorithm
alpha = 0.125
beta = 0.02
gamma = 0.2
iterations = 200

# Iterate the algorithm
for i in range(iterations):
    # Clip the contour points within the image bounds
    contour = clip_contour(contour, img.shape)

    # Compute the gradient of the energy
    gradient = compute_gradient(blur, contour, alpha, beta, gamma)

    # Move the contour
    contour += gradient.astype(np.int32).reshape((-1, 1, 2))

    # Display the result
    result = cv2.drawContours(img.copy(), [contour], 0, (0, 255, 0), 2)
    cv2.imshow('result', result)
    cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()

