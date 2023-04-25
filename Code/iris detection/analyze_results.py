import math 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import zscore
import numpy as np 

def analyze_dataframe(path, column, x_error_column, y_error_column):
    # Load the saved dataframe
    df = pd.read_excel(path, sheet_name='Sheet1', index_col=None, header=0)
    print(df)
    
    # Calculate the average mean of the chosen column
    mean = df[column].mean()
    
    # Identify outliers and calculate the z-score for each
    z_scores = zscore(df[column])
    threshold = 1
    outliers = np.where(np.abs(z_scores) > threshold)[0]
    num_outliers = len(outliers)
    
    # Create a new dataframe to store the results
    result_df = pd.DataFrame({
        'Column': [column],
        'Mean': [mean],
        'Num Outliers': [num_outliers],
        'Z-Scores': [z_scores]
    })
    
    sns.set_theme(style="darkgrid")
    sns.set_style("ticks")


    euclidean_distance = np.linalg.norm(df[['x_error', 'y_error']], axis=1)
    filtered_df = df[euclidean_distance < 20]
    x_min, x_max = np.percentile(filtered_df['x_error'], [0, 100])
    y_min, y_max = np.percentile(filtered_df['y_error'], [0, 100]) 
    
    
    
    #f, ax = plt.subplots(figsize=(6, 6))
    #sns.scatterplot(data= filtered_df, x= y_error_column, y=x_error_column, s=5, color=".15")
    #sns.histplot(data= filtered_df,x=y_error_column, y=x_error_column, bins=50, pthresh=.1, cmap="mako")
    #sns.kdeplot(data= filtered_df,x= y_error_column, y=x_error_column, levels=5, color="w", linewidths=1)
    
    # Create plot
    cmap = sns.cubehelix_palette(rot=-.2,light=1, as_cmap=True)
    ax = sns.kdeplot(data=df, x=x_error_column, y=y_error_column, 
                cmap=cmap, fill=True, clip=(-25, 25), cut=5,
                thresh=0, levels=20
               )
    plt.grid(True, linestyle='--', color='grey', alpha=0.5)
    plt.xticks(np.linspace(x_min, x_max, 5))
    plt.xlabel('X Error')
    plt.yticks(np.linspace(y_min, y_max, 5))
    plt.ylabel('Y Error')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    
    plt.tight_layout()
#
    plt.show()
    #
    # Return the result dataframe
    return result_df

if __name__ == '__main__':
    path= 'Code/iris detection/results/LPW_1_4.xlsx'
    analyze_dataframe(path, 'euclidean distance label - measured', 'x_error', 'y_error')
