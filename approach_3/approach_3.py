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
    # randomly selecting subjects to use as testing subjects
    subjects_to_test = random.choices(np.unique(X['subject']), k=cv) 

    for model_name, model_sub_dict in model_dict.items():

        model_sub_dict['CV Scores'] = []

    # cross validation - note that each model_sub_dict['CV Scores'] array will
    # be tested on the same testing subjects
    for test_subject in subjects_to_test:

        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_subject)

        for model_name, model_sub_dict in model_dict.items():

            if verbose:
                print(f"\nCross validating a {model_name} on X and y\n")

            model_sub_dict['Model'].fit(x_train, np.ravel(y_train))
            y_hat = model_sub_dict['Model'].predict(x_test)

            acc = np.mean(y_hat == y_test)

            model_sub_dict['CV Scores'].append(acc)

            if verbose:
                print(f"\n{model_name} Accuracy = {acc}\n")

    # converting CV scores from list to np.array
    for model_name, model_sub_dict in model_dict.items():

        model_sub_dict['CV Scores'] = np.array(model_sub_dict['CV Scores'])

    return model_dict

if __name__ == "__main__":

    # 'stacking' all 15 subject's data on top of one another
    df = aggregate_subjects()

    # removing the SVM model since the training time is absurd for data of this
    # size (check src.modeling for model_dict)
    del model_dict['SVM']

    # creating lagged variables for each subject/label subset
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

    # removing columns that would lead to prediction leakage given the approach
    X_columns = lag_5_df.columns[~lag_5_df.columns.isin(['label','seq'])]

    lag_5_X = lag_5_df[X_columns]
    lag_5_y = lag_5_df['label']
