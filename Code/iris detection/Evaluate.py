
import math 
import pandas as pd
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
        self._header = ['frame number', 'data set label center', 'measured center','data set center inside Roi', 'euclidean distance label - measured']
        self._df = pd.DataFrame(columns = self._header)
     
    def create_log(self):

        total_lines = len(self._label_centers)
        
        # iterate over all center labels and append the measured center if it exists
        for i, label in enumerate(self._label_centers):
            inside_roi = self.pupil_in_roi(self._coords_roi[i], label)
            # if  measurement was possible
            if self._center[i] != 'None':
                error = self.calculate_error(self._center[i], label)
                self._error.append((i,error))
            else: 
                error = self._center[i]
            new_row = [i,label, self._center[i], inside_roi, error]
            self._df.loc[len(self._df)] = new_row
            
        self._df.to_csv(self._path_file + self._name_file + '.csv', index = False)
        analyze_dataframe(self._df, 'euclidean distance label - measured')

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
            #print('Measured')
        else: 
            self._center.append('None')
            #print('None')
            
            

    
    def pupil_in_roi(self, coords, label):
        
        '''
        coords = (x1,y1),(x2,y2)
        top_1 = (x1,y1)
        top_2 = (x1,y2)
        bottom_1 = (x2,y1)
        bottom_2 = (x2,y2)
        '''
        print(f'coords: {coords}')
        x1 = coords[0][0]
        x2 = coords[1][0]
        y1 = coords[0][1]
        y2 = coords[1][1]
        x = label[0]
        y = label[1]
        
        return (x1 < x < x2 and y1 < y < y2)
    
    
def analyze_dataframe(df, column):
    # Load the saved dataframe
    df = df.copy()
    
    # Calculate the average mean of the chosen column
    mean = df[column].mean()
    
    # Identify outliers and calculate the z-score for each
    z_scores = zscore(df[column])
    threshold = 5
    outliers = np.where(np.abs(z_scores) > threshold)[0]
    num_outliers = len(outliers)
    
    # Create a new dataframe to store the results
    result_df = pd.DataFrame({
        'Column': [column],
        'Mean': [mean],
        'Num Outliers': [num_outliers],
        'Z-Scores': [z_scores]
    })
    
    # Show the result in a bar plot
    result_df.plot(kind='bar', x='Column', y=['Mean', 'Num Outliers'])
    plt.show()
    
    # Return the result dataframe
    return result_df
        
            
    
# Change area, change path, change 