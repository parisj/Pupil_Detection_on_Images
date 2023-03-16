import numpy as np 
import cv2 
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver



path = 'D:/data_set/LPW/1/4.avi,D:/data_set/LPW/1/4.txt'
video_file, txt_file = path.split(',')
print(video_file, txt_file)

cap = cv2.VideoCapture(video_file)

with open (txt_file) as f:
    lines = f.readlines()


# Set up objects and connect to observer
pupil_obj = Pupil()
iris_obj = Iris()
observer = ImageObserver()

observer.add_img_obj(pupil_obj)
observer.add_img_obj(iris_obj)
count = 0
while cap.isOpened() and count <= len(lines)-1:

    center = lines[count].split(' ')
    #print(center)
    x_center_label = round(float(center[0].strip()))
    y_center_label = round(float(center[1].strip('\n')))
    
    #print(center)
    ret, frame = cap.read()
    pupil_obj.set_img(frame)
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    pupil_obj.set_gray(gray)
    
        
    
    cv2.circle(frame, (x_center_label,y_center_label),0,(0,0,255),3)
    # apply binary thresholding
    thresh = cv2.adaptiveThreshold(pupil_obj.get_gray(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 231, 25)
    thresh_contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=frame, contours= thresh_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # visualize the binary image
    pupil_obj.set_processing(thresh)
    
    observer.plot_imgs("processing")


    cv2.imshow('frame', frame)
    count += 1
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()