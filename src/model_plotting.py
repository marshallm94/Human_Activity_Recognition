import numpy as np
from random import sample

import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
plt.style.use('ggplot')


def cv_error_comparison_plot(df, x_label, y_label, title, filename=None):
    '''
    Plots multiple violinplots on the same figure to compare the distribution
    of cross validation scores.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame whose column names will be used as the labels for the
        various violinplots and whose columns are the array's of cross
        validation errors.
    x_label : (str)
        A label for the x axis.
    y_label : (str)
        A label for the y axis.
    title : (str)
        A title for the plot.
    filename : (str)
        The filename to which the figure should be saved. If None (default),
        plt.show() will be called and the image will be shown.

    Returns:
    ----------
    None : (None)
        No object is returned; the image is either shown or saved to filename.
    '''
    fig, ax = plt.subplots(figsize=(12, 9))

    df = df[df.mean(axis=0).sort_values(ascending=False).index]
    parts = ax.violinplot(df.T,
                          showmeans=True,
                          showmedians=False,
                          showextrema=False)

    color_sample = sns.color_palette('husl', len(df.columns))
    for x, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_sample[x])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # setting y-axis to percentage
    full_perc_range = np.arange(0, 1.1, 0.1)
    plt.yticks(full_perc_range,
               labels=['{:,.2%}'.format(x) for x in full_perc_range])

    plt.xticks(range(1, len(df.columns)+1), labels=df.columns)

    plt.xlabel(x_label, fontweight='bold', fontsize=12)

    plt.xticks(rotation=-30, ha='left')

    plt.ylabel(y_label, fontweight='bold', fontsize=12)
    plt.suptitle(title, fontweight='bold', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if not filename:
        plt.show()
    elif filename:
        plt.savefig(filename)

def plot_feature_importances(df, importances_col, model_names_col, x_label,
                             y_label, title, filename=None):

    fig, ax = plt.subplots(figsize=(12, 9))

    palette = sns.color_palette("RdBu_r", df.shape[0])[::-1]
    ax = sns.barplot(df[importances_col],
                     df[model_names_col],
                     palette=palette)

    plt.xlabel(x_label, fontweight='bold', fontsize=12)
    plt.ylabel(y_label, fontweight='bold', fontsize=12)
    plt.suptitle(title, fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if not filename:
        plt.show()
    elif filename:
        plt.savefig(filename)

