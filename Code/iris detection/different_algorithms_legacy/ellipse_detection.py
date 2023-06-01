import numpy as np 
import cv2 
from Pupil import Pupil
from Iris import Iris
from ImageObserver import ImageObserver
import ellipse_detection_algo as ellip
from Evaluate import Evaluation 
import create_plots 


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
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    
    h,s,v = cv2.split(hsv)
 
    #v = clahe.apply(v)
    v = cv2.GaussianBlur(v, (3,3), 0)
    v_hat = cv2.Canny(v, 200,150)

    #cv2.imshow('v', v)
    #cv2.imshow('v Canny', v_hat)
    
    h_hat = cv2.Canny(h, 250,300)
    #cv2.imshow('h', h)
    #cv2.imshow('h canny', h_hat)
    s_hat = cv2.Canny(s, 250,200)
    #cv2.imshow('s canny',s_hat )
    #cv2.imshow('s',s )
    
  
    
    # set object informations
    pupil_obj.set_gray(gray)
    pupil_obj.set_hsv(hsv)

    hsv = cv2.merge([h,s,v])
    change = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow('mchangee', change)
    
    #cv2.imshow('merge', hsv)
    # draw the labled center for this frame
    #cv2.circle(frame, (x_center_label,y_center_label),0,(0,0,255),3)
    #cv2.imshow('gray before passing ROI',gray)
    #cv2.imshow('hsv before passing ROI',hsv)
    #cv2.waitKey(0)
    # Extract Edges and set them as the current processing information 
    masked_img, masked_gray, coords, best_ellipse, peakintensity = ellip.calc_roi(gray, hsv, 80,30)
    
    pupil_obj.set_processing(masked_gray)

    #cv2.imshow('masked_img',masked_img)
    #cv2.imshow('masked_gray',masked_gray)
    #cv2.waitKey(0)
    # Calculate ellipse informations, try to find ellipses and crutial informations

    if coords is None or len(coords) == 0:
        center = pupil_obj.get_center()
        if center == None:
            center = (0,0)

        x = center[0]
        y = center[1]
        
        w = 0
        h = 0
        border = 0
        coords = (y,x,w,h,border)

    # Plot with observer
    observer.plot_imgs("original")  
    # observer.plot_imgs("hsv")  
    if pupil_obj.get_processing().size != 0:
        observer.plot_imgs("processing")
    #observer.plot_imgs("mask")
    
    # if ellipse_info (measurement) was found then update parameter of the ellipse
    if best_ellipse:
        print('best_ellipse', best_ellipse)
        print('coords', coords)
        pupil_obj.set_ellipse(best_ellipse, coords)
        center = pupil_obj.get_center()
        print('center', center)
        axis = pupil_obj.get_axis()
        print('axis', axis)
        
        angle = pupil_obj.get_angle()
        print('angle', angle)
        result = frame.copy()
        # draw and show ellipse 
        # TODO create observer function for this task 
        cv2.ellipse(result, center, axis, angle, 0, 360, (0,255,0), 1)
        cv2.imshow('result', result)
    #cv2.imshow('hsv',hsv)
    #cv2.imshow("masked_gray",masked_gray)
    #cv2.imshow('frame', frame)
    
    # TODO: find better way than counting to keep track of number of iterations
    count += 1

    # If in this frame a measurement was possible
    if best_ellipse:
        BOOL_FOUND = True
    else:
        BOOL_FOUND = False
    # Evaluate Obj
    evaluation_obj.add_frame(BOOL_FOUND, center_label, pupil_obj.get_center())
    # Exit
    if cv2.waitKey(1) == ord('q'):
        break
    if count == 1 :
        create_plots.plot_histogram(pupil_obj.get_gray())
cap.release()
cv2.destroyAllWindows()


# Create result log of this session 
evaluation_obj.create_log()