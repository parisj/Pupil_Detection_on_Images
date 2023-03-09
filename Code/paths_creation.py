import glob

import os

#YAML file evtl c
def write_paths_img_only(path, name):
    data_set = glob.glob(path + '//**//*.jpg', recursive=True)
    CSV_FILE_NAME = name
    with open(CSV_FILE_NAME, 'w') as f:
        for path in data_set:
            path = '/'.join(path.split('\\'))
            f.write(path)     # write the path in the first column
            f.write(',')      # separate first and second item by a comma
            f.write('eye_image') # write the label in the second column
            f.write('\n')   
            
            
def write_videos_and_labels(path, name,drive, file_ending):
    data_set = glob.glob( drive + path + '//**//*.'+ file_ending, recursive=True)
    CSV_FILE_NAME = name
    with open(CSV_FILE_NAME, 'w') as f:
        for path in data_set:
            path = '/'.join(path.split('\\'))
            f.write(path)     # write the path in the first column
            list = path.split('.')
            f.write(',')
                        
            f.write(list[0]+'.txt')
            f.write('\n')   
            
if __name__ == '__main__':
    write_videos_and_labels('\data_set\LPW',"LPW",'E:','avi')