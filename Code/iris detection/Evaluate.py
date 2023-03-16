import csv 
import math 

class Evaluation:
    def __init__(self, eval_obj,  name_file, path_file):
        self._name_file = name_file
        self._path_file = path_file
        self._eval_obj = eval_obj
        self._label_centers = []
        self._center = []
        self._error = []
        self._header = ['frame', 'label', 'measured', 'error']
        print('Init')
     
    def create_log(self):

        with open (self._name_file, 'w', newline='') as file:

            writer = csv.writer(file, delimiter= ',' )
            # Start with header
            writer.writerow(self._header)
            
            total_lines = len(self._label_centers)
            # iterate over all center labels and append the measured center if it exists
            for i, label in enumerate(self._label_centers):
                # if it measurement was possible
                if self._center[i] is not 'None': 

                    error = self.calculate_error(self._center[i], label)
                    self._error.append((i,error))
                    
                else: 
                    error = self._center[i] 
                    
                writer.writerow([i,label, self._center[i], error])
 
            failed = self.calculate_failed()
            total_measurements = total_lines - failed
            
            sum_error = 0 
            for i, e in self._error:
                sum_error += e
            average_error =  sum_error/total_measurements 
                  
            writer.writerow(['Total Frames','Total Measurements', 'Failed','Average Error' ])
            writer.writerow([total_lines, total_measurements, failed, average_error ])       
     
    def calculate_error(self, center, label):
        return math.sqrt((float(center[0]) - float(label[0])) ** 2 + (float(center[1]) - float(label[1])) ** 2)
    
    def calculate_failed(self):
        return self._center.count('None')
        
    
    def add_frame(self,BOOL_FOUND, label, measured):
        self._label_centers.append(label)
        if BOOL_FOUND:
            self._center.append(measured)
            print('Measured')
        else: 
            self._center.append('None')
            print('None')
            
    

    
     
        
            
    
