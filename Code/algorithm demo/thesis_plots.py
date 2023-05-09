import cv2
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_image(path, gray=False):
    img = cv2.imread(path)
    if gray: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def extract_row(image, row):
    return image[row,:]
    
def extract_col(image, col):
    return image[:,col-2:col+2]
    
def plot_intensity (image, row):
    row = extract_row(image, row)
    row_data = np.array(row)
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,3))
    print(f'row_data.shape: {row_data.shape}')
    print(f'row', row)
    print(f'row_data', row_data)
    color_labels = ['low', 'medium', 'high']
    color_bins = np.linspace(row_data.min(), row_data.max(), 4)  # Divide data into 3 intervals
    color_categories = pd.cut(row_data, bins=color_bins, labels=color_labels)

    # Map categories to colors
    color_map = {'low': 'blue', 'medium': 'gray', 'high': 'red', np.nan: 'white'}  # Added 'np.nan': 'white'
    colors = [color_map[label] for label in color_categories]

    ax1 = plt.subplot2grid((100, 1), (0, 0), rowspan=90)
    ax2 = plt.subplot2grid((100, 1), (92, 0), rowspan=4)
    sns.lineplot(x=np.arange(row_data.shape[0]), y=row_data, ax=ax1, color= 'black', linewidth = 1.2)  # Use lineplot instead of barplot
    #sns.barplot(x=np.arange(row_data.shape[0]), y=row_data, ax=ax1, palette=colors)  # Use palette argument
    
    ax1.set_title('Cutout of a row with the intensity values of the pixels')
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel('Intensity value')

    # Create the rowplot on the second subplot
    ax2.imshow(row_data[np.newaxis, :], cmap='gray', aspect='auto')
    sns.despine(ax=ax1, bottom= True)
    ax2.get_xaxis().set_visible(True)  # Hide x-axis
    ax2.get_yaxis().set_visible(False)  # Hide y-axis
 # Adjust the space between subplots
    ax1.set_ylim(0, 255)
    ax1.set_xlim(0, row_data.shape[0]) # Set the y-axis limits
    ax2.set_xlim(0, row_data.shape[0]) # Set the y-axis limits
    ax2.set_xlabel('x coordinate and intensity of the pixel')
    plt.tight_layout()
    plt.show()



img = cv2.imread('Latex/thesis/plots/orig_canny_eyelids.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, fx = 0.5, fy = 0.5, dsize=(0,0))

img_copy = img.copy()



for i in range(0,15):
    ret, img_copy = cv2.threshold(img.copy(), 65+i*6, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('img'+str(i), img_copy)
    cv2.imwrite('Latex/thesis/plots/thresholding/th'+str(i)+'.jpg', img_copy)
    print(65+6*i)

print(f'img.shape: {img}')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('Latex/thesis/plots/thresholding/thresholded_eyelid.jpg', img)


#plot_intensity(img, 310)

#row = extract_row(img, 310)