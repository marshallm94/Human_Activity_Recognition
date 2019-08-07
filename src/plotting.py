import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def dimension_plot(df, variable, y_tick_range, color, title):
    '''
    Used to plot each activity across a given dimension.
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
    Used to compare how different subjects "behave" across a given activity and dimension.
    Pass in a dataframe with only one activity.
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


def bar_plot(df, values_col, label_col, x_label, y_label, title,
             long_x_names=False):
    '''
    Used to show the distribution of participant and class data.
    '''
    fix, ax = plt.subplots(figsize=(12, 9))

    palette = sns.color_palette('husl', df.shape[0])
    ax = sns.barplot(df[label_col],
                     df[values_col],
                     order=df[label_col],
                     palette=palette)

    if long_x_names:
        plt.xticks(rotation=-30, ha='left')

    plt.xlabel(x_label, fontweight='bold', fontsize=12)
    plt.ylabel(y_label, fontweight='bold', fontsize=12)
    plt.suptitle(title, fontweight='bold', fontsize=16, y=1.025)

    plt.tight_layout()

    plt.show()
