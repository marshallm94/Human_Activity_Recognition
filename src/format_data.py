import pandas as pd
import numpy as np
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

def aggregate_subjects(files, subject_names):
    '''
    Aggregates all files in files into one dataframe.
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
    Adds and additional columns in cols_to_keep to final output.
    '''
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(df[cols_to_scale])

    X = np.hstack((scaled_data, df[cols_to_keep].values.reshape(-1, len(cols_to_keep))))

    cols_to_scale.extend(cols_to_keep)

    new_df = pd.DataFrame(X, columns=cols_to_scale)

    for col, dtype in col_dtype_dict.items():
        new_df[col] = new_df[col].astype(dtype)

    return new_df


def create_lagged_features(df, columns, shift, row_time_steps):
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

    return out


def create_lagged_df(df, activity_col, columns, shift=5, row_time_steps=1):
    '''
    Iterates through the dataframe that is the output of ActivitySequenceAverager and
    creates lagged variables.
    '''

    frames = []

    for activity in np.unique(df[activity_col]):

        mask = df[activity_col] == activity
        subset = df.loc[mask, :].copy()

        lagged_activity_df = create_lagged_features(subset, columns, shift, row_time_steps)

        frames.append(lagged_activity_df)

    return pd.concat(frames)


if __name__ == "__main__":

    ################ START DATA TRANSFORMATION FOR EDA SECTION ################ 
    data_dir = '../data/'

    files = [data_dir + f'{i}.csv' for i in range(1, 16)]
    subject_names = [str(i) for i in range(1, 16)]

    # importing data for EDA
    df = aggregate_subjects(files, subject_names)

    ################# END DATA TRANSFORMATION FOR EDA SECTION ################# 

    ################ START DATA TRANSFORMATION FOR ML SECTION ################# 
    # importing data for base estimator
    subject_dfs = {}
    for x, filename in enumerate(files):
        subject = str(x+1)
        subject_dfs[subject] = format_csv(filename)

    averager = ActivitySequenceAverager(subject_dfs.values())
    averager.transform()

    # base estimator - no sequential nature taken into account
    X = averager.aggregated_df[['x_acc','y_acc','z_acc']].values
    y = averager.aggregated_df['label'].values

    # train test split for base estimator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # creating lagged variables for final data set
    lagged_df = create_lagged_df(averager.aggregated_df,
                                 'label',
                                 ['x_acc','y_acc','z_acc'])

    # adding the activity labels back on
    lagged_df['label'] = averager.aggregated_df['label']

    # removing imputed zero's from lagged data transformation
    mask = lagged_df['x_acc_T_minus_5'] != 0
    mask1 = lagged_df['y_acc_T_minus_5'] != 0
    mask2 = lagged_df['z_acc_T_minus_5'] != 0

    trimmed_df = lagged_df[mask & mask1 & mask2].copy()

    # create rolling average
    cols = list(trimmed_df.columns)
    cols.remove('label')

    for dimension in ['x','y','z']:
        
        dimension_columns = [col for col in cols if dimension in col]
        trimmed_df[f'rolling_{dimension}_average'] = np.mean(trimmed_df[dimension_columns], axis=1)


    X_lag = trimmed_df.loc[:,~trimmed_df.columns.isin(['label'])].values
    y_lag = trimmed_df['label'].values

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_lag,
                                                        y_lag,
                                                        test_size=0.25)

    # standardizing the values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # saving x_train, x_test, y_train, y_test and StandardScaler object
    # for modeling phase
    data_dir = "../data"
    pd.DataFrame(X_train_scaled).to_csv(f"{data_dir}/X_train_scaled.csv",
                                        index=False,
                                        header=False)
    pd.DataFrame(X_test).to_csv(f"{data_dir}/X_test_unscaled.csv",
                                index=False,
                                header=False)
    pd.DataFrame(y_train).to_csv(f"{data_dir}/y_train.csv",
                                 index=False,
                                 header=False)
    pd.DataFrame(y_test).to_csv(f"{data_dir}/y_test.csv",
                                index=False,
                                header=False)
    ################# END DATA TRANSFORMATION FOR ML SECTION ##################
