import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def dimension_plot(df, variable, y_tick_range, color, title):
    '''
    Used to plot each activity across a given dimension.
    Stacks multiple line plots on top of one another in order to compare how
    different activities vary. Pass in a dataframe with only one subject.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame that contains a numeric column named variable.
    variable : (str)
        The name of the numerical column to be plotted.
    y_tick_range : (list)
        A list of length = 2 that determines the y-tick range.
    color : (str)
        The color to for the line plot.
    title : (str)
        The title for the plot. 

    Returns:
    ----------
    None : (None)
        No object is returned; the image is shown.
    '''
    fig = plt.figure(figsize=(12, 8))

    activities = df['label'].unique()
    for x, activity in enumerate(activities):

        array = df.loc[df['label'] == activity, variable]

        plt.subplot(len(activities), 1, x+1)
        plt.plot(array, c=color)
        plt.yticks(y_tick_range)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.title(label=activity, loc='center', fontsize=10)

    plt.suptitle(title, fontweight='bold', fontsize=14, y=1.025)
    plt.tight_layout()
    plt.show()

def subject_comparison_plot(df, variable, color, title):
    '''
    Stacks multiple line plots on top of one another in order to compare how
    different subjects "behave" across a given activity and dimension. Pass
    in a dataframe with only one activity.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame that contains a numeric column named variable.
    variable : (str)
        The name of the numerical column to be plotted.
    color : (str)
        The color to for the line plot.
    title : (str)
        The title for the plot. 

    Returns:
    ----------
    None : (None)
        No object is returned; the image is shown.
    '''
    fig = plt.figure(figsize=(12, 20))

    subjects = df['subject'].unique()
    for x, subject in enumerate(subjects):

        array = df.loc[df['subject'] == subject, variable]

        plt.subplot(len(subjects), 1, x+1)
        plt.plot(array, c=color)
        #plt.yticks(y_tick_range)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.title(label=f'Participant {subject}', loc='center', fontsize=10)

    plt.suptitle(title, fontweight='bold', fontsize=14, y=1.025)
    plt.tight_layout()
    plt.show()

def barplot(df, values_col, labels_col, x_label, y_label, title,
            long_x_names=False, filename=False):
    '''
    Create a barplot using the values in values_col to dictate the bar heights
    and the labels in labels_col to dictate the labels for those values.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame that contains a numeric column that contains the values
        to be used for bar heights (named values_col) and a column to be
        used as labels for those values (named labels_col).
    values_col : (str)
        The name of the column in df to be used to dictate the bar heights.
        Must be numeric.
    labels_col : (str)
        The name of the column in df to be used for the labels of the bars.
    x_label : (str)
        A label for the x axis.
    y_label : (str)
        A label for the y axis.
    title : (str)
        A title for the plot.
    long_x_names : (bool)
        Indicates whether the labels (keys in dictionary) for the x axis
        are long enough to warrant rotating them. Default=False.
    filename : (str)
        The filename to which the figure should be saved. If None (default),
        plt.show() will be called and the image will be shown.

    Returns:
    ----------
    None : (None)
        No object is returned; the image is either shown or saved to filename.
    '''
    fig = plt.figure(figsize=(12, 9))

    palette = sns.color_palette('husl', df.shape[0])
    sns.barplot(x=df[labels_col],
                y=df[values_col],
                order=df[labels_col],
                palette=palette)

    # setting y-axis to percentage
    locs, _ = plt.yticks()
    plt.yticks(locs, labels=['{:,.2%}'.format(x) for x in locs])

    if long_x_names:
        plt.xticks(rotation=-30, ha='left')

    plt.xlabel(x_label, fontweight='bold', fontsize=12)
    plt.ylabel(y_label, fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=16, y=1.025)

    plt.tight_layout()

    if not filename:
        plt.show()
    elif filename:
        plt.savefig(filename)
