import random
from collections import Counter

# the below imports assume that Human_Activity_Recognition have been added to
# the PYTHONPATH environment variable
from src.format_data import *
from src.modeling import *
from src.model_plotting import *
from src.plotting import *

def train_test_split(X, y, test_subject_id):
    '''
    Creates a train test split based on a subject id. The training data will
    not include any observations from the test subject in an effort to exlude
    any "behavioral" data leakage.

    Note that X must have a column named 'subject' in order to define the
    split, however the matrices that are returned will exlude the subject
    column.

    Parameters:
    ----------
    X : (Pandas DataFrame)
        A data frame that has a column named 'subject' that contains the
        subject id of each observations.
    y : (Pandas DataFrame - array-like)
        An array-like object that holds the target variable.
    test_subject_id : (int)
        The id of the subject to use as the test data.
        
    Returns:
    ----------
    x_train : (numpy.ndarray)
        The X matrix for training.
    x_test : (numpy.ndarray)
        The y vector for training.
    y_train : (numpy.ndarray)
        The X matrix for testing.
    y_test : (numpy.ndarray)
        The y vector for testing
    '''
    indices = X['subject'] == test_subject_id

    x_test = X.loc[indices, X.columns[~X.columns.isin(['subject'])]].values
    y_test = y[indices].values
    
    x_train = X.loc[~indices, X.columns[~X.columns.isin(['subject'])]].values
    y_train = y[~indices].values

    return x_train, x_test, y_train, y_test

def CV_multiple_models(X, y, model_dict, cv=8, verbose=True):
    '''
    Cross validates various models in model_dict using X and y.

    Parameters:
    ----------
    X : (Pandas DataFrame)
        A data frame that has a column named 'subject' that contains the
        subject id of each observations.
    y : (Pandas DataFrame - array-like)
        An array-like object that holds the target variable.
    model_dict : (dict)
        A dictionary of dictionaries. The keys of the model_dict should be the
        name of the model framework (i.e. 'Random Forest', 'SVM', etc.). The
        values should be another dict, with a key named 'Model' whose value
        is the model that has a fit() and predict() method.
    cv : (int)
        Analogous to the number of "folds" to make in K-Fold CV, although the
        method used in this function isn't exactly the same.
    verbose : (bool)
        Whether the testing set accuracy should be printed.
        
    Returns:
    ----------
    model_dict : (dict)
        The same model_dict that was used as an argument, however the inner
        dictionary will have a new key value pair of the cross validation
        scores (key = 'CV Scores').
    '''
    # randomly selecting subjects to use as testing subjects - doing this
    # within the function ensures that the ith element in
    # model_sub_dict['CV Scores'] is using the same testing subject for each
    # model.
    subjects_to_test = random.choices(np.unique(X['subject']), k=cv) 

    for model_name, model_sub_dict in model_dict.items():

        model_sub_dict['CV Scores'] = []

    for test_subject in subjects_to_test:

        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_subject)

        # undersample majority classes randomly
        under_sampler = RandomUnderSampler(sampling_strategy='not minority',
                                           random_state=5,
                                           replacement=False)

        x_train, y_train = under_sampler.fit_resample(x_train, y_train)

        for model_name, model_sub_dict in model_dict.items():

            if verbose:
                print(f"Time = {time.ctime()} | Training a {model_name} on X and y")

            model_sub_dict['ModelPipeline'].fit(x_train,
                                                np.ravel(y_train))
            y_hat = model_sub_dict['ModelPipeline'].predict(x_test)

            acc = np.mean(y_hat == y_test)

            model_sub_dict['CV Scores'].append(acc)

            if verbose:
                print(f"Time = {time.ctime()} | {model_name} Accuracy = {acc}")

    # converting CV scores from list to np.array
    for _, model_sub_dict in model_dict.items():

        model_sub_dict['CV Scores'] = np.array(model_sub_dict['CV Scores'])

    return model_dict

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
        dimension_columns = [col for col in cols if dimension in col]
        lag_5_df[f'rolling_{dimension}_average'] = np.mean(lag_5_df[dimension_columns], axis=1)
        lag_5_df[f'rolling_{dimension}_variance'] = np.var(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_min'] = np.min(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_max'] = np.max(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_kurtosis'] = kurtosis(lag_5_df[dimension_cols], axis=1)
        lag_5_df[f'rolling_{dimension}_skewness'] = skew(lag_5_df[dimension_cols], axis=1)


    # removing columns that would lead to prediction leakage given the approach
    X_columns = lag_5_df.columns[~lag_5_df.columns.isin(['label','seq'])]

    lag_5_X = lag_5_df[X_columns]
    lag_5_y = lag_5_df['label']

    # undersample majority classes randomly
    under_sampler = RandomUnderSampler(sampling_strategy='not minority',
                                       random_state=5,
                                       replacement=False)

    lag_5_X, lag_5_y = under_sampler.fit_resample(lag_5_X, lag_5_y)

    # model_dict originally defined in src/modeling.py
    model_dict = CV_multiple_models(X=lag_5_X,
                                    y=lag_5_y,
                                    model_dict=model_dict)
                 

    cv_error_dict = {}
    for name, sub_dict in model_dict.items():
        cv_error_dict[name] = sub_dict['CV Scores']

    filename=f"images/Approach3_CV_Lag5.png"
    cv_error_comparison_plot(pd.DataFrame(cv_error_dict),
                             x_label="Model Type",
                             y_label="Accuracy",
                             title=f'8 Subject Train 1 Subject Test Accuracy with 5 Step Time Lag',
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
    cols_to_keep = ['x_acc','y_acc','z_acc', 'label']
    for col in lag_15_df.columns[~lag_15_df.columns.isin(['seq','subject'])]:
        if col in cols_to_keep:
            continue
        elif 'roll' in col or 'var' in col:
            cols_to_keep.append(col)
        elif int(col.split("_")[-1]) % 5 == 0:
            cols_to_keep.append(col)

    # removing columns that would lead to prediction leakage given the approach
    X_columns = lag_15_df.columns[~lag_15_df.columns.isin(['label'])]

    lag_15_X = lag_15_df[X_columns]
    lag_15_y = lag_15_df['label']

    # undersample majority classes randomly
    under_sampler = RandomUnderSampler(sampling_strategy='not minority',
                                       random_state=5,
                                       replacement=False)

    lag_15_X, lag_15_y = under_sampler.fit_resample(lag_15_X, lag_15_y)

    # model_dict originally defined in src/modeling.py
    model_dict = CV_multiple_models(X=lag_15_X,
                                    y=lag_15_y,
                                    model_dict=model_dict)
                 

    cv_error_dict = {}
    for name, sub_dict in model_dict.items():
        cv_error_dict[name] = sub_dict['CV Scores']

    filename=f"images/Approach3_CV_Lag5.png"
    cv_error_comparison_plot(pd.DataFrame(cv_error_dict),
                             x_label="Model Type",
                             y_label="Accuracy",
                             title=f'8 Subject Train 1 Subject Test Accuracy with 15 Step Time Lag',
                             filename=filename)
