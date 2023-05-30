
import math 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np 

class Evaluation:
    def __init__(self, eval_obj,  name_file, path_file):
        self._name_file = name_file
        self._path_file = path_file
        self._eval_obj = eval_obj
        self._label_centers = []
        self._coords_roi = []
        self._center = []
        self._error = []
        self._x_error = []
        self._y_error = []
        self._header = ['frame number', 'data set label center', 'measured center','data set center inside Roi', 'euclidean distance label - measured', 'x_error', 'y_error']
        self._df = pd.DataFrame(columns = self._header)
     
    def create_log(self, scaling):

        total_lines = len(self._label_centers)
        
        # iterate over all center labels and append the measured center if it exists
        for i, label in enumerate(self._label_centers):
            label = (round(scaling* label[0]), round(scaling* label[1]))
            inside_roi = self.pupil_in_roi(self._coords_roi[i], label)
            # if  measurement was possible
            if self._center[i] != 'None':
                error = self.calculate_error(self._center[i], label)
                x_error = float(self._center[i][0]) - float(label[0])
                y_error = float(self._center[i][1]) - float(label[1])
                self._error.append((i,error))
                self._x_error.append((i,x_error))
                self._y_error.append((i,y_error))
            else: 
                error = self._center[i]
            new_row = [i,label, self._center[i], inside_roi, error, x_error, y_error]
            self._df.loc[len(self._df)] = new_row
            
        self._df.to_excel(self._path_file + self._name_file +'_s_'+str(int(scaling*100)) +'.xlsx', index = False)
    

     #Calculate the euclidean distance between the label and the measured center
    def calculate_error(self, center, label):
        return math.sqrt((float(center[0]) - float(label[0])) ** 2 + (float(center[1]) - float(label[1])) ** 2)
    
    def calculate_failed(self):
        return self._center.count('None')
        
    
    def add_frame(self ,BOOL_FOUND, label, measured, coords):
        self._label_centers.append(label)
        self._coords_roi.append(coords)
        
        if BOOL_FOUND:
            self._center.append(measured)

        else: 
            self._center.append('None')

            
            

    
    def pupil_in_roi(self, coords, label):
        
        '''
        coords = (x1,y1),(x2,y2)
        top_1 = (x1,y1)
        top_2 = (x1,y2)
        bottom_1 = (x2,y1)
        bottom_2 = (x2,y2)
        '''

        x1 = coords[0][0]
        x2 = coords[1][0]
        y1 = coords[0][1]
        y2 = coords[1][1]
        x = 0.5 *label[0]
        y = 0.5 *label[1]
        
        return (x1 < x < x2 and y1 < y < y2)
    
    

            
    
# Change area, change path, change 