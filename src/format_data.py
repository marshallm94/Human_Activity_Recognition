import pandas as pd
import numpy as np


def format_csv(filepath):
    '''
    Formats each CSV according to the problem definition.

    Parameters:
    ----------
    filepath : (str)
        The (relative or absolute) filepath to the CSV file to be imported.

    Returns:
    ----------
    df : (Pandas DataFrame)
        A data frame with an intuitively labeled target attribute. 
    '''
    columns = ['seq', 'x_acc', 'y_acc', 'z_acc', 'label']
    col_types = {'seq':int, 'x_acc':float, 'y_acc':float, 'z_acc':float,'label':object}

    df = pd.read_csv(filepath, names=columns, dtype=col_types)

    df.loc[df['label'] == '0', 'label'] = '7'

    label_dict = {'1': 'Working at Computer',
                  '2': 'Standing up, Walking & Going Up/Down Stairs',
                  '3': 'Standing',
                  '4': 'Walking',
                  '5': 'Going Up/Down Stairs',
                  '6': 'Walking & Talking w/ Someone',
                  '7': 'Talking while Standing'}
    
    for k, v in label_dict.items():
        
        mask = df['label'] == k

        df.loc[mask, 'label'] = v


    return df

# defining default csv files to import
data_dir = '../data/'
subjects = [str(i) for i in range(1, 16)]
files = [data_dir + subject + '.csv' for subject in subjects]

def aggregate_subjects(files=files, subject_names=subjects):
    '''
    Aggregates all files in files into one dataframe.

    Parameters:
    ----------
    files : (list/iterable)
        An iterable object that contains the relative or absolute filenames
        of the CSV files to be aggregated. Default is a list of 15 numerically
        labeled files in the ../data/ directory.
    subject_names : (list/iterable)
        An iterable object containing an identifier for each subject.
        Default = [str(i) for i in range(1, 16)].

    Returns:
    ----------
    total_data_set : (Pandas DataFrame)
        A vertically stacked data frame of all data in files.
    '''
    frames = []
    for i in range(len(files)):

        frame = format_csv(files[i])

        frame['subject'] = subject_names[i]

        frames.append(frame)

    total_data_set = pd.concat(frames)

    return total_data_set

def standardize_df(df, cols_to_scale, cols_to_keep, col_dtype_dict):
    '''
    Standardizes the columns in cols_to_scale to have mean 0 and variance 1.
    Adds on additional columns in cols_to_keep to final output.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame that contains all columns in the cols_to_scale and
        cols_to_keep lists.
    cols_to_scale : (list)
        A list of the column names that should be scaled.
    cols_to_keep : (list)
        A list of column names to keep that should not be scaled.
    col_dtype_dict : (dict)
        A dictionary whose keys are the names of df's columns and whose values
        are the dtypes those columns should be.

    Returns:
    ----------
    standard_df : (Pandas DataFrame)
        A data frame that contains the standardized columns in cols_to_scale
        and the columns in cols_to_keep (un-standardized).
    '''
    stand_X = ((df[cols_to_scale].values - df[cols_to_scale].values.mean(axis=0)) 
            / df[cols_to_scale].values.std(axis=0))

    X = np.hstack((stand_X, df[cols_to_keep].values.reshape(-1, len(cols_to_keep))))

    cols_to_scale.extend(cols_to_keep)

    standard_df = pd.DataFrame(X, columns=cols_to_scale)

    for col, dtype in col_dtype_dict.items():
        standard_df[col] = standard_df[col].astype(dtype)

    return standard_df

def create_lagged_features(df, columns, shift, row_time_steps,
                           remove_imputed_zeros=True, verbose=True):
    '''
    Creates shift lagged columns for each column specified in columns.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame that contains the columns listed in columns.
    columns : (list)
        A list of strings that are the columns for which shifted variables
        should be created. len(columns) * shift = number of new columns created.
    shift : (int)
        The number of sequence steps that each instance should be able to
        "look back." len(columns) * shift = number of new columns created.
    row_time_steps : (int)
        The number of time steps between instances (i.e. if the time between
        instance 0 and instance 1 is 5 minutes, this should be 5).
    remove_imputed_zeros : (bool)
        Whether the observations that have zeros which are necessarily imputed
        during this process should be removed. Default = True. The total
        number of instances that will be removed = shift.
    verbose : (bool)
        Whether the number of observations that are removed should be printed.

    Returns:
    ----------
    out : (Pandas DataFrame)
        A copy of the data frame that was passed in with the lagged columns
        added.
    '''
    out = df.copy()
    for col in columns:

        feature_names = [f"{col}_T_minus_{row_time_steps*i}" for i in range(1, shift + 1)]
        base = out[col].copy().values
        for x, new_col in enumerate(feature_names):
            x += 1
            values = np.insert(base, np.repeat(0, x), np.repeat(0, x))
            out[new_col] = values[:-x]

    if remove_imputed_zeros:
        cols = [col for col in out.columns if f'T_minus_{shift}' in col]

        masks = []
        for col in cols:
            mask = out[col] != 0
            masks.append(mask)

        total_mask = np.logical_and.reduce(masks)
        out = out.loc[total_mask,:]

        if verbose:
            print(f'{df.shape[0]-out.shape[0]} observations removed from data set.')
    
    return out

def create_lagged_df(df, activity_col, subject_col, columns, shift,
                     row_time_steps=1, remove_imputed_zeros=True,
                     verbose=True):
    '''
    Splits df into subsets based on the different activities in activity_col
    and calls create_lagged_features() with columns as the columns from which
    lagged features should be created. Note that there will NOT be any overlap
    of activities when creating the lagged features.
    
    Parameters:
    ----------
    df : (Pandas DataFrame)
        A data frame that contains the columns listed in columns.
    activity_col : (str)
        The name of the column that will be used to stratify the activity in
        the data set.
    subject_col : (str)
        The name of the column that will be used to stratify the subjects in
        the data set.
    columns : (list)
        A list of strings that are the columns for which shifted variables
        should be created. len(columns) * shift = number of new columns created.
    shift : (int)
        The number of sequence steps that each instance should be able to
        "look back." len(columns) * shift = number of new columns created.
    row_time_steps : (int)
        The number of time steps between instances (i.e. if the time between
        instance 0 and instance 1 is 5 minutes, this should be 5). Default = 1.
    remove_imputed_zeros : (bool)
        Whether the observations that have zeros which are necessarily imputed
        during this process should be removed. Default = True. The total
        number of instances that will be removed = shift.
    verbose : (bool)
        Whether the number of observations that are removed should be printed.

    Returns:
    ----------
    out : (Pandas DataFrame)
        A copy of the data frame that was passed in with the lagged columns
        added.
    '''
    frames = []

    for subject in np.unique(df[subject_col]):
        for activity in np.unique(df[activity_col]):

            mask = df[subject_col] == subject
            mask1 = df[activity_col] == activity
            subset = df.loc[mask & mask1, :].copy()

            lagged_activity_df = create_lagged_features(subset,
                                                        columns,
                                                        shift,
                                                        row_time_steps,
                                                        remove_imputed_zeros,
                                                        verbose)

            if remove_imputed_zeros:
                # adding the activity column back on to the data frame,
                # ensuring the column "lines up" since the observations with
                # zeros were removed.
                lagged_activity_df[activity_col] = subset.loc[shift:, activity_col].copy()

            frames.append(lagged_activity_df)

    return pd.concat(frames)
