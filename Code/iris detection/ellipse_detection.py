import numpy as np 
import cv2 
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
import ellipse_detection_algo as ellip
from Evaluate import Evaluation 


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
evaluation_obj = Evaluation(pupil_obj,'results.csv','results/')

observer.add_img_obj(pupil_obj)
observer.add_img_obj(iris_obj)
count = 0

while cap.isOpened() and count <= len(lines)-1:

    # extract center from the labels
    center = lines[count].split(' ')
    x_center_label = round(float(center[0].strip()))
    y_center_label = round(float(center[1].strip('\n')))
    center_label = (x_center_label, y_center_label)

    # Start reading file and extract frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # save frame as current img and conver it 
    pupil_obj.set_img(frame)
    hsv = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    
    # set object informations
    pupil_obj.set_gray(gray)
    pupil_obj.set_hsv(hsv)
    
    # draw the labled center for this frame
    cv2.circle(frame, (x_center_label,y_center_label),0,(0,0,255),3)

    # Extract Edges and same them as the current processing information 
    pupil_obj.set_processing(ellip.extractEdges(pupil_obj.get_gray(),hsv))
 
    # Calculate ellipse informations, try to find ellipses and crutial informations
    ellipseMask, contourMask, ellipse_info = ellip.findEllipses(pupil_obj.get_processing())
    pupil_obj.set_mask(ellipseMask)
    
 

    # Plot with observer
    observer.plot_imgs("original")  
    # observer.plot_imgs("hsv")  
    observer.plot_imgs("processing")
    observer.plot_imgs("mask")
    
    # if ellipse_info (measurement) was found then update parameter of the ellipse
    if ellipse_info:
        pupil_obj.set_ellipse(ellipse_info)
        center = pupil_obj.get_center()
        axis = pupil_obj.get_axis()
        angle = pupil_obj.get_angle()
        result = frame.copy()
        # draw and show ellipse 
        # TODO create observer function for this task 
        cv2.ellipse(result, center, axis, angle, 0, 360, (0,255,0), 1)
        cv2.imshow('result', result)
    #cv2.imshow('hsv',hsv)
    #cv2.imshow("edges",edges)
    #cv2.imshow('frame', frame)
    
    # TODO: find better way than counting to keep track of number of iterations
    count += 1
    # If in this frame a measurement was possible
    if ellipse_info:
        BOOL_FOUND = True
    else:
        BOOL_FOUND = False
    # Evaluate Obj
    evaluation_obj.add_frame(BOOL_FOUND, center_label, pupil_obj.get_center())
    # Exit
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
# Create result log of this session 
evaluation_obj.create_log()