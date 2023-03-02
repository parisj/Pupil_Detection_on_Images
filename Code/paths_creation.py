import glob

import os


data_set = glob.glob('Code/data_set/eyes_color//**//*.jpg', recursive=True)
CSV_FILE_NAME = 'path_files_EYE_COLOR.csv'
with open(CSV_FILE_NAME, 'w') as f:
    for path in data_set:
        path = '/'.join(path.split('\\'))
        f.write(path)     # write the path in the first column
        f.write(',')      # separate first and second item by a comma
        f.write('eye_image') # write the label in the second column
        f.write('\n')   