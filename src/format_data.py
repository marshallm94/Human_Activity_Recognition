import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ActivitySequenceAverager(object):
    '''
    A class that is used to average each subjects data by matching on activity and sequence.
    '''

    def __init__(self, dataframes):

        self.original_dataframes = dataframes

    def transform(self):

        for dataframe in self.original_dataframes:

            self.resequence_dataframe(dataframe, 'label', 'seq')

        self.aggregated_df = self.average_dfs()

    def resequence_dataframe(self, df, activity_col, sequence_col):

        for activity in np.unique(df[activity_col]):

            self.resequence_activity(df, activity_col, activity, sequence_col)

    def resequence_activity(self, df, activity_col, activity, sequence_col):

        mask = df[activity_col] == activity

        seq_len = df.loc[mask, sequence_col].shape[0]

        df.loc[mask, sequence_col] = np.arange(seq_len)

    def merge_dfs(self, df1, df2, dimensions=set(['x_acc','y_acc','z_acc'])):
        
        out = pd.merge(df1, df2, on=['seq', 'label'], how='inner', suffixes=('_1','_2'))
        
        updated_cols = []
        for col in out.columns:
            
            if col[:-2] in dimensions:
                
                updated_cols.append(col[:-2])
            
            else:
                
                updated_cols.append(col)
        
        out.columns = updated_cols
        
        return out

    def average_dimensions(self, df, dimensions=['x_acc','y_acc','z_acc'], weight=1/15, start=True):
        
        for dimension in dimensions:
            
            # if the first two dataframes are being used, we want to weight them evenly in the overall average
            dim_array = df[dimension].values
            if start:

                dim_array = (dim_array[:,0] * weight) + (dim_array[:,1] * weight)

            # if not the start, then then df1[dimension] has already been weighted appropriately and we don't want to
            # reweight it
            elif not start:

                dim_array = dim_array[:, 0] + (dim_array[:,1] * weight)

            # dropping both dimension columns
            df.drop(dimension, inplace=True, axis=1)

            # replacing with one dimension column
            df[dimension] = dim_array
            
    def average_dfs(self, dfs=None):
       
        if not dfs:
            dfs = self.original_dataframes

        for x, dataframe in enumerate(dfs):

            if x == 0:

                cached_df = dataframe
                continue

            elif x == 1:
                
                merged_df = self.merge_dfs(cached_df, dataframe)
                self.average_dimensions(merged_df, start=True)

            elif x >= 2:

                merged_df = self.merge_dfs(merged_df, dataframe)
                self.average_dimensions(merged_df, start=False)
                

        return merged_df

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
    standardizes the columns in cols to scale to have mean 0 and variance 1.
    Adds on additional columns in cols_to_keep to final output.
    '''
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(df[cols_to_scale])

    X = np.hstack((scaled_data, df[cols_to_keep].values.reshape(-1, len(cols_to_keep))))

    cols_to_scale.extend(cols_to_keep)

    new_df = pd.DataFrame(X, columns=cols_to_scale)

    for col, dtype in col_dtype_dict.items():
        new_df[col] = new_df[col].astype(dtype)

    return new_df

def create_lagged_features(df, columns, shift, row_time_steps,
                           remove_imputed_zeros=True, verbose=True):
    '''
    Creates shift lagged columns for each column specified in columns.
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

def create_lagged_df(df, activity_col, columns, shift, row_time_steps=1,
                     remove_imputed_zeros=True, verbose=True):
    '''
    Iterates through the dataframe that is the output of
    ActivitySequenceAverager and creates lagged variables.
    '''
    frames = []

    for activity in np.unique(df[activity_col]):

        mask = df[activity_col] == activity
        subset = df.loc[mask, :].copy()

        lagged_activity_df = create_lagged_features(subset,
                                                    columns,
                                                    shift,
                                                    row_time_steps,
                                                    remove_imputed_zeros,
                                                    verbose)

        lagged_activity_df[activity_col] = subset.loc[shift:, activity_col].copy()
        frames.append(lagged_activity_df)

    return pd.concat(frames)


if __name__ == "__main__":

    data_dir = '../data/'

    # leaving the last subject (participant 15) to use as the testing data
    files = [data_dir + f'{i}.csv' for i in range(1, 15)]
    subject_names = [str(i) for i in range(1, 15)]

    # importing data for EDA
    df = aggregate_subjects(files, subject_names)

#    # importing data for base estimator
#    subject_dfs = {}
#    for x, filename in enumerate(files):
#        subject = str(x+1)
#        subject_dfs[subject] = format_csv(filename)
#
#    # averaging all subjects accelerometer data on sequence and activity
#    averager = ActivitySequenceAverager(subject_dfs.values())
#    averager.transform()
#
#    # removing the sequence number column (seq) to prevent any "leakage" - 
#    # since some activities were performed for longer sequences, this could be
#    # accidentally be recognized as useful information by the model
#    averager.aggregated_df.drop('seq', axis=1, inplace=True)
#
#    # standardizing the training data and saving the scaler object
#    scaler = StandardScaler()
#    averager.aggregated_df[['x_acc','y_acc','z_acc']] = scaler.fit_transform(averager.aggregated_df[['x_acc','y_acc','z_acc']].values)
#    data_dir = "../data"
#    joblib.dump(scaler, f"{data_dir}/scaler.joblib")
#
#    # base estimator data - no sequential nature taken into account
#    base_estimator_X = averager.aggregated_df[['x_acc','y_acc','z_acc']].values
#    base_estimator_y = averager.aggregated_df['label'].values



    # creating lagged variables for final data set
#    lag_5_df = create_lagged_df(averager.aggregated_df,
#                                activity_col='label',
#                                columns=['x_acc','y_acc','z_acc'],
#                                shift=5,
#                                verbose=False)
    frames = []
    for subject in np.unique(df['subject']):
        mask = df['subject'] == subject
        subset = df.loc[mask, df.columns]
        subject_lag_df = create_lagged_df(subset,
                                          activity_col='label',
                                          columns=['x_acc','y_acc','z_acc'],
                                          shift=5,
                                          verbose=False)
        frames.append(subject_lag_df)

    lag_5_df = pd.concat(frames)
    
    # create rolling average over previous 5 time-steps for each dimension
    cols = list(lag_5_df.columns)
    cols.remove('label')

    for dimension in ['x','y','z']:
        dimension_columns = [col for col in cols if dimension in col]
        lag_5_df[f'rolling_{dimension}_average'] = np.mean(lag_5_df[dimension_columns], axis=1)

    X_columns = lag_5_df.columns[~lag_5_df.columns.isin(['label','seq'])]

    lag_5_df[X_columns].to_csv(f"{data_dir}/lag_5_X.csv",
                               index=False,
                               header=True)
    lag_5_df['label'].to_csv(f"{data_dir}/lag_5_y.csv",
                             index=False,
                             header=True)

    # creating 15 time-lagged variables
#    lag_15_df = create_lagged_df(averager.aggregated_df,
#                                 activity_col='label',
#                                 columns=['x_acc','y_acc','z_acc'],
#                                 shift=15,
#                                 verbose=False)

    # creating 15 time-lagged variables
    frames = []
    for subject in np.unique(df['subject']):
        mask = df['subject'] == subject
        subset = df.loc[mask, df.columns]
        subject_lag_df = create_lagged_df(subset,
                                          activity_col='label',
                                          columns=['x_acc','y_acc','z_acc'],
                                          shift=15,
                                          verbose=False)
        frames.append(subject_lag_df)

    lag_15_df = pd.concat(frames)

    # create 5, 10 & 15 minute rolling average and variance columns for
    # each dimension
    cols = list(lag_15_df.columns)
    cols.remove('label')

    for dimension in ['x','y','z']:
        for lag in [5, 10, 15]:
        
            # creating column subset for mean() & var() calculation
            input_cols = []
            for col in cols:
                if col in input_cols:
                    continue
                # if dim in col and ...
                elif dimension in col:
                    # if col != <dim>_acc col and the time lag of the col is <= lag
                    if len(col.split("_")) > 2 and int(col.split("_")[-1]) <= lag:
                        input_cols.append(col)
                    # if the col == <dim>_acc col
                    elif len(col.split("_")) == 2:
                        input_cols.append(col)

            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_average'] = np.mean(lag_15_df[input_cols], axis=1)
            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_variance'] = np.var(lag_15_df[input_cols], axis=1)

    # removing columns whose time lag isn't divisible by 5, a statistic column
    # or one of the original variables given
    cols_to_keep = ['x_acc','y_acc','z_acc', 'label','subject']
    for col in lag_15_df.columns[~lag_15_df.columns.isin(['seq'])]:
        if col in cols_to_keep:
            continue
        elif 'roll' in col or 'var' in col:
            cols_to_keep.append(col)
        elif int(col.split("_")[-1]) % 5 == 0:
            cols_to_keep.append(col)

    lag_15_df = lag_15_df[cols_to_keep]

    X_columns = lag_15_df.columns[~lag_15_df.columns.isin(['label','seq'])]

    lag_15_df[X_columns].to_csv(f"{data_dir}/lag_15_X.csv",
                               index=False,
                               header=True)
    lag_15_df['label'].to_csv(f"{data_dir}/lag_15_y.csv",
                             index=False,
                             header=True)
