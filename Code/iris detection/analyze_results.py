
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Soure: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def analyze_dataframe(path,name_file, column, x_error_column, y_error_column):
    # Load the saved dataframe
    df = pd.read_excel(path, sheet_name='Sheet1', index_col=None, header=0)
    #print(df)
    name = 'File: ' + name_file
    # Calculate the average mean of the chosen column
    mean = df[column].mean()
    
    # Identify outliers and calculate the z-score for each
    z_scores = zscore(df[column])
    threshold = 3
    outliers = np.where(np.abs(z_scores) > threshold)[0]
    num_outliers = len(outliers)
    
    # Create a new dataframe to store the results
    result_df = pd.DataFrame({
        'Column': [column],
        'Mean': [mean],
        'Num Outliers': [num_outliers],
        'Z-Scores': [z_scores]
    })
    


    euclidean_distance = np.linalg.norm(df[['x_error', 'y_error']], axis=1)
    filtered_df = df[euclidean_distance < 20]
    x_min, x_max = np.percentile(filtered_df['x_error'], [2.5, 97.5])
    y_min, y_max = np.percentile(filtered_df['y_error'], [2.5, 97.5]) 
    min_limit = min(x_min,y_min)
    max_limit = max(x_max,y_max)
    array_error = filtered_df[['x_error', 'y_error']].to_numpy()

    # Create plot
    cmap = sns.cubehelix_palette(rot=-.2,light=1, as_cmap=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.grid(True, which='major', axis='both')
    cbar_kwargs = { 'label': 'error density','shrink': 0.8, 'drawedges': False}
    ax = sns.kdeplot(data=df, x=x_error_column, y=y_error_column, 
                cmap=cmap, fill=True, clip=(-25, 25), cut=0,
                thresh=0, levels=20, legend = True, cbar= True, cbar_kws=cbar_kwargs)
    #, cbar= True, cbar_kws=cbar_kwargs
    

    sns.set_theme()



    sns.despine(left=True, bottom=True)

    # mappable = ax.collections[0]
    # mappable.set_cmap(cmap)
    # cbar = fig.colorbar(ax=ax, mappable=mappable,**cbar_kwargs)
    # cbar.outline.set_visible(False)
    # cbar.ax.tick_params(labelsize=10)
    # cbar.ax.set_ylabel(cbar_kwargs['label'], fontsize=10)
    plt.xticks(range(round(min_limit)-2, round(max_limit)+2, 1))
    plt.xlabel('x error [pixel]')
    plt.yticks(range(round(min_limit)-2, round(max_limit)+2, 1))
    plt.ylabel('y error [pixel]')
    plt.xlim(x_min-0.5, x_max+0.5)
    plt.ylim(y_min-0.5, y_max+0.5)

    general_info = '\n'.join((
        r'z-score threshold: $%.2f$' % (threshold, ),
        r'number of outliers: $%d$' % (num_outliers, ),
        r'mean error: $%.2f$' % (mean, )))
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.7,edgecolor='gray')
    
    ax.text(0.95, 0.95, general_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',horizontalalignment='right', bbox=props)
    
    ax.text(0.95,0.84, name, transform=ax.transAxes, fontsize=8, 
            verticalalignment='top',horizontalalignment='right', bbox=props)

    confidence_ellipse(array_error[:,0], array_error[:,1], ax, n_std=1,
                       label=r'$1\sigma$', edgecolor='black')
    
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc= 'upper left', fontsize=10)
    ax.set_title('Error distribution in x and y ')

    plt.tight_layout()
    
    
    plt.savefig('Code/iris detection/results/figures/'+name_file+'.png', dpi=300)
    #
    # Return the result dataframe
    return result_df


def analyse(path):
    name = path.split('/')[-1].split('.')[0]
    analyze_dataframe(path, name,'euclidean distance label - measured', 'x_error', 'y_error')

if __name__ == '__main__':
    path= 'Code/iris detection/results/LPW_2_13_s_100.xlsx'
    analyse(path)