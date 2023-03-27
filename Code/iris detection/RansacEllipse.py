import cv2
import numpy as np
import random
from typing import List
from multiprocessing import Process, Manager

N_POINTS_ELLIPSE_FIT = 6
PI = 3.142

pointsPerIteration = []
fitErrorPerIteration = []

# RANSAC ELLIPSE FIT
class EllipseFinderRansac:
    def __init__(self):
        self.ellipses = []
        self.bestFitEllipses = []

    def __del__(self):
        pass

    def GetRandomPoint(self, pts: List[np.ndarray]) -> np.ndarray:
        idx = random.randint(0, len(pts)-1)
        return pts[idx]

    def GetNrandomPoints(self, pts: List[np.ndarray], nPoints: int) -> List[np.ndarray]:
        randomPoints = []

        for i in range(nPoints):
            randomPoints.append(self.GetRandomPoint(pts))

        return randomPoints

    def Process(self, dataPoints: List[np.ndarray], minRadius: float, maxRadius: float, nIterations: int) -> bool:
        if len(dataPoints) < N_POINTS_ELLIPSE_FIT:
            print(f"\n Not enough points to fit ellipse. Minimum {N_POINTS_ELLIPSE_FIT} points required \n")
            return False

        self.ellipses.clear()
        self.bestFitEllipses.clear()

        threadPool = []

        for iter in range(nIterations):
            oneFitPoints = self.GetNrandomPoints(dataPoints, N_POINTS_ELLIPSE_FIT)

            if PUSH_TO_THREADS:
                p = Process(target=self.FitEllipse, args=(oneFitPoints, minRadius, maxRadius))
                threadPool.append(p)
                p.start()
            else:
                fitError = 0
                if not self.FitEllipse(oneFitPoints, minRadius, maxRadius, fitError):
                    print("\n Error in ellipse fit \n")
                    return False

        if PUSH_TO_THREADS:
            # Make sure all threads are completed
            for thread in threadPool:
                thread.join()

        # No ellipse detected
        if not fitErrorPerIteration:
            return False

        # Get an ellipse with minimum fit error
        minErrorIdx = fitErrorPerIteration.index(min(fitErrorPerIteration))
        minError = min(fitErrorPerIteration)

        # Get one best fit ellipse
        self.bestFitEllipses.append(self.ellipses[minErrorIdx])

        return True


    def FitEllipse(dataPts, minRadius, maxRadius, fitError):
        if len(dataPts) == 0:
            return False

        mA = np.zeros((len(dataPts), 5))
        mB = np.zeros((len(dataPts), 1))
        mC = np.zeros((len(dataPts), 1))

        # Populate the coefficient matrices to solve the least square equation
        for i in range(len(dataPts)):
            mA[i, 0] = dataPts[i].x * dataPts[i].y
            mA[i, 1] = dataPts[i].y * dataPts[i].y
            mA[i, 2] = dataPts[i].x
            mA[i, 3] = dataPts[i].y
            mA[i, 4] = 1

            mB[i, 0] = -(dataPts[i].x * dataPts[i].x)

            mC[i, 0] = dataPts[i].x

        # Solve the least square equation ( A'Ax = A'B)
        x = np.linalg.solve(mA.T @ mA, mA.T @ mB)

        # Get the ellipse coefficients
        b = x[0, 0]
        c = x[1, 0]
        d = x[2, 0]
        e = x[3, 0]
        f = x[4, 0]

        # Check for condition satisfying an ellipse
        ellipseCheck = (b * b) - (4 * c)
        if ellipseCheck < 0:
            print("set of points converges to an ellipse")
        else:
            print("Points does't satisfy ellipse condition (B^2 - 4AC < 0)")
            return False

        m0 = np.zeros((3, 3))
        m1 = np.zeros((2, 2))

        # Represent general equation of ellipse in matrix form
        m0[0, 0] = f
        m0[0, 1] = d / 2
        m0[0, 2] = e / 2

        m0[1, 0] = d / 2
        m0[1, 1] = 1
        m0[1, 2] = b / 2

        m0[2, 0] = e / 2
        m0[2, 1] = b / 2
        m0[2, 2] = c

        m1[0, 0] = 1
        m1[0, 1] = b / 2
        m1[1, 0] = b / 2
        m1[1, 1] = c

        majorAxis, minorAxis, h, k, alpha = 0, 0, 0, 0, 0
        eigen = cv2.eigen(m1)
        lambda1, lambda2 = eigen[1][0], eigen[0][0]

        # Get ellipse parameters from obtained coefficients
        # Center
        h = ((b * e) - (2 * c * d)) / ((4 * c) - (b * b))
        k = ((b * d) - (2 * e)) / ((4 * c) - (b * b))

        # Major and Minor axis radius
        
        m0_det = np.linalg.det(m0)
        m1_det = np.linalg.det
        majorAxis = np.sqrt(-(m0_det/(m1_det)*lambda1))
        minorAxis = np.sqrt(-(m0_det/(m1_det*lambda2)))
        
        # Rotation 
        alpha = (np.pi/2)- np.arctan(((1-c)/b)/2)
        
        if (majorAxis >= minRadius and majorAxis <= maxRadius and
            minorAxis >= minRadius and minorAxis <= maxRadius):
            center = (int(h),int(k))
            self.ellipses.append((center, majorAxis, minorAxis, alpha))
        else:
            return False
        

        error, dist1, dist2, error1, error2, dist, err = 0,0,0,0,0,0,0
        
               # Check if the ellipse fits the data within the specified error
        for i in range(len(dataPts)):
            xPt = dataPts[i].x 
            yPt = dataPts[i].y
            
            xx = xPt * xPt
            mul = np.sqrt((b * b * xx) + (2 * e * b * xPt) - (4 * c * xx) - (4 * c * d * xPt) + (e * e) - (4 * c * f))
            yhat = -((e / 2) + ((b * xPt) / 2) + (mul / 2)) / c
            yhat1 = -((e / 2) + ((b * xPt) / 2) - (mul / 2)) / c

            di1 = (yhat) - yPt
            dist1 = di1 * di1

            di2 = yhat1 - yPt
            dist2 = di2 * di2
            if (dist1 <= 5):
                error1 = error1 + dist1
                
            if (dist2 <= 5):
                error2 = error2 + dist2
	
	

            val = (x * np.cos(alpha) + y * np.sin(alpha))**2 / (majorAxis**2) + \
                  (x * np.sin(alpha) - y * np.cos(alpha))**2 / (minorAxis**2)

            if abs(val - 1) > fitError:
                return False

            with mLock:
                fitErrorPerIteration.append(error1 + error2)
           
            
        # Return the ellipse parameters if the fit is successful
        return True # (center, (majorAxis, minorAxis), alpha)
    
    
if __name__ == '__main__':
        finder = EllipseFinderRansac()
        pointmap = np.random.randint(0,high = 256, size= (256,256), dtype = int)

        print(pointmap)
        print(finder.Process(pointmap,20 ,100, 20))
        