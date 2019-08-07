import numpy as np
from random import sample

import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
plt.style.use('ggplot')

all_colors =  [k for k, v in pltc.cnames.items()]


def cv_error_comparison_plot(df, x_label, y_label, title, long_x_names=False,
                             filename=None):
    '''
    Creates a violin plot that shows the CV error rates for various models.
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
    if long_x_names:
        plt.xticks(rotation=-30, ha='left')

    plt.xlabel(x_label, fontweight='bold', fontsize=12)
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

