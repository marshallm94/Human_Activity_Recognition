from collections import Counter

# the below import assume that Human_Activity_Recognition have been added to
# the PYTHONPATH environment variable
from src.format_data import *
from src.modeling import *
from src.model_plotting import *
from src.plotting import *

if __name__ == "__main__":

    # 'stacking' all 15 subject's data on top of one another
    df = aggregate_subjects()

    # creating lagged variables for each subject/label subset
    lag_5_df = create_lagged_df(df=df,
                                activity_col='label',
                                subject_col='subject',
                                columns=['x_acc','y_acc','z_acc'],
                                shift=5,
                                verbose=False)
    
    # create various statistical rolling statistical features
    cols = list(lag_5_df.columns)
    cols.remove('label')

    for dimension in ['x','y','z']:
        dimension_cols = [col for col in cols if dimension in col]
        lag_5_df[f'rolling_{dimension}_average'] = np.mean(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_variance'] = np.var(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_min'] = np.min(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_max'] = np.max(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_kurtosis'] = kurtosis(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_skewness'] = skew(lag_5_df[dimension_cols], axis=1)

    # removing columns that would lead to prediction leakage given the approach
    X_columns = lag_5_df.columns[~lag_5_df.columns.isin(['label','seq','subject'])]

    # training each model in model_dict on each subject
    for subject in np.unique(lag_5_df['subject']):

        # subset to include data of one subject
        mask = lag_5_df['subject'] == subject
        lag_5_X = lag_5_df.loc[mask, X_columns].values
        lag_5_y = lag_5_df.loc[mask, 'label'].values

        # undersample majority classes randomly
        under_sampler = RandomUnderSampler(sampling_strategy='not minority',
                                           random_state=5,
                                           replacement=False)

        lag_5_X, lag_5_y = under_sampler.fit_resample(lag_5_X, lag_5_y)

        x_train, x_test, y_train, y_test = train_test_split(lag_5_X,
                                                            lag_5_y,
                                                            test_size=0.25)


        # model_dict originally defined in src/modeling.py
        model_dict = cross_validate_multiple_models(x_train,
                                                    np.ravel(y_train))

        cv_error_dict = {}
        for name, sub_dict in model_dict.items():
            cv_error_dict[name] = sub_dict['CV Scores']

        filename=f"images/Approach2_InitivalCV_Lag5_Subject{subject}.png"
        cv_error_comparison_plot(pd.DataFrame(cv_error_dict),
                                 x_label="Model Type",
                                 y_label="Accuracy",
                                 title=f'Subject {subject}: 8 Fold CV Accuracy with 5 Step Time Lag',
                                 filename=filename)

    # creating 15 time-lagged variables
    lag_15_df = create_lagged_df(df=df,
                                 activity_col='label',
                                 subject_col='subject',
                                 columns=['x_acc','y_acc','z_acc'],
                                 shift=15,
                                 verbose=False)

    # create 5, 10 & 15 minute rolling statistical features for each dimension
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
            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_min'] = np.min(lag_15_df[input_cols], axis=1)
            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_max'] = np.max(lag_15_df[input_cols], axis=1)
            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_kurtosis'] = kurtosis(lag_15_df[input_cols], axis=1)
            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_skewness'] = skew(lag_15_df[input_cols], axis=1)

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

    # removing columns that would lead to prediction leakage given the approach
    X_columns = lag_15_df.columns[~lag_15_df.columns.isin(['label','seq','subject'])]

    # training each model in model_dict on each subject
    for subject in np.unique(lag_15_df['subject']):

        # subset to include data of one subject
        mask = lag_15_df['subject'] == subject
        lag_15_X = lag_15_df.loc[mask, X_columns].values
        lag_15_y = lag_15_df.loc[mask, 'label'].values

        # undersample majority classes randomly
        under_sampler = RandomUnderSampler(sampling_strategy='not minority',
                                           random_state=5,
                                           replacement=False)

        lag_15_X, lag_15_y = under_sampler.fit_resample(lag_15_X, lag_15_y)

        x_train, x_test, y_train, y_test = train_test_split(lag_15_X,
                                                            lag_15_y,
                                                            test_size=0.25)


        # model_dict originally defined in src/modeling.py
        model_dict = cross_validate_multiple_models(x_train,
                                                    np.ravel(y_train))

        cv_error_dict = {}
        for name, sub_dict in model_dict.items():
            cv_error_dict[name] = sub_dict['CV Scores']

        filename=f"images/Approach2_InitivalCV_Lag15_Subject{subject}.png"
        cv_error_comparison_plot(pd.DataFrame(cv_error_dict),
                                 x_label="Model Type",
                                 y_label="Accuracy",
                                 title=f'Subject {subject}: 8 Fold CV Accuracy with 15 Step Time Lag',
                                 filename=filename)


