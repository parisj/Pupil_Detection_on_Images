from Pupil import Pupil
from Iris import Iris
import ROI
from ImageObserver import ImageObserver


import numpy as np 
import cv2

img = cv2.imread('test.jpg')

pupil_obj= Pupil()

iris_obj = Iris()

observer = ImageObserver()

observer.add_img_obj(pupil_obj)

observer.add_img_obj(iris_obj)

print('_img'=='_img')

pupil_obj.set_img(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
pupil_obj.set_gray(gray)


observer.plot_imgs('gray')
observer.plot_imgs('original')

