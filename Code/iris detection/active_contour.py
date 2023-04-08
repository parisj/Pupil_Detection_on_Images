import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
'''
N=4
[[-1,  1,  0,  0],
 [ 0, -1,  1,  0],
 [ 0,  0, -1,  1],
 [ 1,  0,  0, -1]]
'''
def _diff_elastic(N): 
    Melast = np.zeros((N,N))
    np.fill_diagonal(Melast, -1)
    Melast +=np.eye(Melast.shape[0], k=1)
    Melast[N-1][0]=1
    print(Melast)
    return Melast

'''
N=4
[[-2,  1,  0,  1],
 [ 1, -2,  1,  0],
 [ 0,  1, -2,  1],
 [ 1,  0,  1, -2]]
'''
    
def _diff_smooth(N):
    Msmooth= np.eye(N)*-2
    Msmooth[N-1][0] = 1
    Msmooth[0][N-1] = 1
    Msmooth +=np.eye(Msmooth.shape[0], k=1)
    Msmooth +=np.eye(Msmooth.shape[0], k=-1)
    print(Msmooth)
    return Msmooth
    
'''
Calculate Gradient over complete image    
'''
def _gradient(image):
    
    
    g = np.gradient(image)
    G = np.abs(g).sum(axis = 0, dtype = np.uint8) 
    # Normalize
    G = G / np.max(G)
    G = (G*255).astype(np.uint8)

    
    return G


'''
Create Circle points
'''
def _init_circle(center,radius, image_shape):
    mask = np.zeros(image_shape)
    circle = cv2.circle(mask, center, radius, 255, thickness = 1)
    points = np.argwhere(mask > 0 )
    return mask, points
    
    
def create_circle_points(center, radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.array([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]).T
    return points

'''
Use different Kernels to Blur the image, creates a broather Gradient trail 
sort of like a force field
'''
def _blur_G(G):
    #Calculate different strong blurings 
    G_blurred = np.zeros_like(G, dtype=np.float64)

    G2 = cv2.GaussianBlur(G, (301, 301), sigmaX=10, sigmaY=10)
    G_blurred += G2

    G2 = cv2.GaussianBlur(G, (201, 201), sigmaX=10, sigmaY=10)
    G_blurred += G2
#
    #G2 = cv2.GaussianBlur(G, (101, 101), sigmaX=10, sigmaY=10)
    #G_blurred += G2
#
    #G2 = cv2.GaussianBlur(G, (61, 61),0)
    #G_blurred += G2
    #G2 = cv2.GaussianBlur(G, (15, 15),0)
    #G_blurred += G2
    # normalize
    G_normalized = G_blurred / np.max(G_blurred)
    G_normalized = (G_normalized * 255).astype(np.uint8)
    cv2.imshow('grad_blured', G_normalized)
    return G_normalized

'''
Calculates Energies in the window with dimension (size x size)

E_Elastic = ||x(i-1)-x(i)||^2
E_Smooth = ||x(i-1)-2x(i)-x(i+1)||^2
'''

def _window_energy(size,idx, point, G, img, M_e, M_s, points,
                   alpha, beta, gamma):
    
    window = G[point[0] - size: point[0] + size, point[1] - size: point[1] + size]
    points_energy = points.copy()
    print(f'window.shape: {window.shape}')
    print(f'range(-size, size+1): {len(range(-size, size+1))}')
    # create Arrays of the same shape as window to save each energy level 
    E_Elastic = np.zeros(window.shape)
    print(f'E_Elasit.shape: {E_Elastic.shape}')
    E_Smooth = np.zeros(window.shape)
    print(f'E_Smooth.shape: {E_Smooth.shape}')
    total_energy = np.zeros_like(window, dtype=np.float64)
    cv2.imshow('window', window)
    
    cv2.waitKey(1)
    # iterate over window and calculate all the E_contour for each point
    # O(n^2) VERY SLOW
    for x, i in enumerate(range(-size, size)):
        for y, j in enumerate(range(-size, size)):
            print(f'x : {x}, y: {y}')
            # form [idx][x,y]
            print(f'point_energy[idx] {points_energy[idx]}') 
            start_point_x, start_point_y = points_energy[idx]
            points_energy[idx] = np.array([start_point_x+i, start_point_y +j])
            print(f'NEW point_energy[idx] {points_energy[idx]} i: {i}, j: {j}') 
        
            energy_elastic_x = np.square(M_e[idx] @ points_energy[:,0])
            energy_elastic_y =  np.square(M_e[idx] @ points_energy[:,1])
            E_Elastic[x,y] = energy_elastic_x + energy_elastic_y
            energy_smooth_x = np.square(M_s[idx] @ points_energy[:,0])
            energy_smooth_y =  np.square(M_s[idx] @ points_energy[:,1])
            E_Smooth[x,y] = energy_smooth_x + energy_smooth_y
        
    print (f'energy_elastic: {E_Elastic}, energy_smooth: {E_Smooth}')
        
    total_energy +=( alpha * E_Elastic + beta * E_Smooth) * np.ones_like(window)
    total_energy -= gamma * np.square(window)
    min_energy = np.argmin(total_energy)
    min_energy = np.unravel_index(min_energy, total_energy.shape)
    new_point = (min_energy[0] + point[0] - size, min_energy[1] + point[1] - size)

    return new_point


def total_energy(points, G, M_e, M_s, alpha, beta, gamma):
    points = points.reshape((-1, 2)).astype(int)
    print(f'points: {points}')
    energy_elastic_x = np.sum(np.square(M_e @ points[:, 0]))
    energy_elastic_y = np.sum(np.square(M_e @ points[:, 1]))
    E_Elastic = energy_elastic_x + energy_elastic_y

    energy_smooth_x = np.sum(np.square(M_s @ points[:, 0]))
    energy_smooth_y = np.sum(np.square(M_s @ points[:, 1]))
    E_Smooth = energy_smooth_x + energy_smooth_y

    E_External = -gamma * np.sum(np.square(G[points[:, 0], points[:, 1]]))

    energy = -alpha * E_Elastic + beta * E_Smooth + E_External
    #energy = E_External
    print(f'E_Elastic: {E_Elastic},  alpha * E_Elastic: {-alpha* E_Elastic}\n E_Smooth: {E_Smooth}, beta * E_Smooth: {beta *E_Smooth} \n E_External: {E_External}, gamma * E_External : {gamma * E_External},\n energy {energy}')
    return energy


def callback(xk, img, G, M_e, M_s, alpha, beta, gamma):
    iteration_points = xk.reshape((-1, 2)).astype(int)
    print(f'xk: {xk}')
    energy_value = total_energy(xk, G, M_e, M_s, alpha, beta, gamma)
    
    plt.imshow(img, cmap='gray')
    plt.scatter(iteration_points[:, 1], iteration_points[:, 0], c='blue', marker='x')
    plt.title(f'Iteration points\n Energy value: {energy_value}')
    plt.draw()
    plt.pause(0.1)
    plt.clf()

def gradient_energy(points, G, M_e, M_s, alpha, beta, gamma):
    points = points.reshape((-1, 2)).astype(int)

    grad_E_Elastic_x = 2 * (-alpha) * (M_e.T @ (M_e @ points[:, 0]))
    grad_E_Elastic_y = 2 * (-alpha) * (M_e.T @ (M_e @ points[:, 1]))

    grad_E_Smooth_x = 2 * beta * (M_s.T @ (M_s @ points[:, 0]))
    grad_E_Smooth_y = 2 * beta * (M_s.T @ (M_s @ points[:, 1]))

    grad_E_External_x = -gamma * G[points[:, 0], points[:, 1]]
    grad_E_External_y = -gamma * G[points[:, 0], points[:, 1]]
    
    grad_x = grad_E_Elastic_x + grad_E_Smooth_x + grad_E_External_x
    grad_y = grad_E_Elastic_y + grad_E_Smooth_y + grad_E_External_y

    gradient = np.column_stack((grad_x, grad_y)).flatten()

    return gradient

def update_points(points, G, img, M_e, M_s, alpha, beta, gamma):
    height, width = img.shape
    bounds = [(0, height - 1) for _ in points] + [(0, width - 1) for _ in points]
    bounds = [(min_y, max_y) for min_y, max_y in bounds[:len(points)]] + [(min_x, max_x) for min_x, max_x in bounds[len(points):]]
    print(f'bounds.shape : {len(bounds)}')


    #result = minimize(total_energy, 
    #                points, 
    #                args=(G, M_e, M_s, alpha, beta, gamma),
    #                bounds=bounds, method='L-BFGS-B', 
    #                callback=lambda xk: callback(xk, img,G, M_e, M_s, alpha, beta, gamma), 
    #                options={'disp': True})



    result = minimize(total_energy, 
                points.ravel(), 
                args=(G, M_e, M_s, alpha, beta, gamma),
                method='BFGS',
                jac=gradient_energy,
                callback=lambda xk: callback(xk, img,G, M_e, M_s, alpha, beta, gamma), 
                options={'disp': True})

    updated_points = result.x.reshape((-1, 2)).astype(int)
    
    print(f'result.success: {result.success}')
    plot_result(img, updated_points)
    return updated_points



def update_points_old(points, size, G, img, M_e, M_s,
                  alpha, beta, gamma):
    updated_points = []

    for idx, point in enumerate(points):
        new_point = _window_energy(size,idx, point, G, img, M_e, M_s, points,
                                   alpha, beta, gamma)
        updated_points.append(new_point)
    print(f'updated points: {updated_points}')
    return np.array(updated_points)



def _initialize_algo(path, center, alpha, beta, gamma):
    image= cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = create_circle_points(center,30,15)
    print(f'points: {points}, shape {points.shape}')
    N = points.shape[0]
    

    M_e = _diff_elastic(N)
    print(f'M_e.shape: {M_e.shape}')
    M_s = _diff_smooth(N)
    print(f'M_s.shape: {M_s.shape}')
    G = _gradient(gray)
    G_blur = _blur_G(G.copy())

    updated_points = update_points(points, G_blur, gray, M_e, M_s, 
                                       alpha, beta, gamma)
    points = updated_points       
    plt.show()
    return updated_points
    

def plot_progress(iteration, initial_points, updated_points, img):
    plt.clf()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    initial_points = np.array(initial_points)
    updated_points = np.array(updated_points)

    plt.scatter(initial_points[:, 1], initial_points[:, 0], c='green', marker='o', label='Initial Points')
    plt.scatter(updated_points[:, 1], updated_points[:, 0], c='blue', marker='x', label='Updated Points')

    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.draw()
    plt.pause(0.5)


def plot_result(image, points):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 1], points[:, 0], c='red', marker='x', label='Updated Points')
    plt.title("Active Contour Result")
    plt.legend()
    plt.show()


def active_contour(path,center,  alpha, beta, gamma):
    points = _initialize_algo(path, center,  alpha, beta ,gamma)
    plt.figure()
    
    return points



if __name__ == '__main__':
    path = 'test_roi.png'
  
    center = (120,100)
    alpha = 0.001
    beta = 0.1
    gamma = 0.1
    points = active_contour(path, center,  alpha, beta, gamma)

