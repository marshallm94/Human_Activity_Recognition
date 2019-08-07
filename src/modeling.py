import pandas as pd
import numpy as np
from sklearn.externals import joblib

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from RNNClassifier import RNNClassifier
from model_plotting import *
from format_data import create_lagged_features, create_lagged_df 

def cross_validate_multiple_models(X, y, model_dict, verbose=True):
    '''
    Performs K-fold cross validation where k=8 by fitting all model in
    model_dict.values() to X and y.
    '''
    for model_name, model_sub_dict in model_dict.items():

        if verbose:
            print(f"Cross validating a {model_name} on X and y...")

        if "fit_params" in model_sub_dict.keys():

            model_scores = cross_val_score(model_sub_dict['Model'],
                                           X,
                                           y,
                                           cv=8,
                                           scoring='accuracy',
                                           n_jobs=-1,
                                           verbose=int(verbose),
                                           fit_params=model_sub_dict['fit_params'])
        elif "fit_params" not in model_sub_dict.keys():

            model_scores = cross_val_score(model_sub_dict['Model'],
                                           X,
                                           y,
                                           cv=8,
                                           scoring='accuracy',
                                           n_jobs=-1,
                                           verbose=int(verbose))

        model_sub_dict['CV Scores'] = model_scores

        if verbose:
            print(f"... {model_name} Accuracy = {np.mean(model_scores)}")

    return model_dict

if __name__ == "__main__":

#    data_dir = "../data"
#    X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
#    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
#
#    n_steps = 1
#    X_train.values.reshape(-1, n_steps, X_train.shape[1])
#
#    # training parameters
#    learning_rate = 0.01
#    n_epochs = 50
#    batch_size = 50
#
#    # network architecture
#    hidden_layer_array = [100, 100]
#    n_outputs = len(np.unique(y_train))
#
#    model_dict = {
#                  "RNN":
#                      {"Model": RNNClassifier(),
#                          "fit_params": {"hidden_layer_architecture": hidden_layer_array,
#                              "n_outputs": n_outputs,
#                              "n_steps": n_steps,
#                              "learning_rate": learning_rate,
#                              "batch_size": batch_size,
#                              "n_epochs": n_epochs}
#                       },          
#            "Random Forest": 
#                      {"Model": RandomForestClassifier(n_estimators=500,
#                          random_state=5)},
#                  "AdaBoost Classifer":
#                      {"Model": AdaBoostClassifier(n_estimators=500)},
#                  "Multinomial Logistic Regression":
#                      {"Model": LogisticRegression(multi_class="ovr",
#                                                   solver='saga',
#                                                   max_iter=500)},
#                  "Support Vector Machine":
#                      {"Model": svm.SVC(kernel='poly',
#                                        degree=5,
#                                        gamma='auto')},
#                  "MLP":
#                      {"Model": MLPClassifier(hidden_layer_sizes=(100, 100),
#                                              batch_size=500)}
#                 }
#
#    model_dict = cross_validate_multiple_models(X_train.values,
#                                                np.ravel(y_train.values),
#                                                model_dict,
#                                                verbose=False)
#
#    cv_error_dict = {}
#    for name, sub_dict in model_dict.items():
#        cv_error_dict[name] = sub_dict['CV Scores']
#
#    # getting feature importances for fully trained Random Forest
#    model_dict['Random Forest']['Model'].fit(X_train.values, np.ravel(y_train.values))
#
#    feature_imps = model_dict['Random Forest']['Model'].feature_importances_
#    feat_imp_df = pd.DataFrame({"Importances": feature_imps,
#                                "Variable Names": X_train.columns})
#    feat_imp_df.sort_values('Importances', ascending=False, inplace=True)
#
#    # creating 15 time-lagged variables
#    reconstructed_df = pd.concat([X_train, y_train], axis=1)
#    cols_to_drop = [col for col in reconstructed_df.columns if 'minus' in col or 'rolling' in col]
#    reconstructed_df.drop(cols_to_drop, inplace=True, axis=1)
#
#    lag_15_df = create_lagged_df(reconstructed_df,
#                                 activity_col='Activity',
#                                 columns=['x_acc','y_acc','z_acc'],
#                                 shift=15)
#
#    # removing rows that will have imputed zeros
#    mask = lag_15_df['x_acc_T_minus_15'] != 0
#    mask1 = lag_15_df['y_acc_T_minus_15'] != 0
#    mask2 = lag_15_df['z_acc_T_minus_15'] != 0
#
#    lag_15_df = lag_15_df[mask & mask1 & mask2]
#
#    # create 5, 10 & 15 minute rolling average columns for each dimension
#    cols = list(lag_15_df.columns)
#    cols.remove('Activity')
#
#    for dimension in ['x','y','z']:
#        for lag in [5, 10, 15]:
#        
#            # creating column subset for mean() & var() calculation
#            input_cols = []
#            for col in cols:
#                if col in input_cols:
#                    continue
#                # if dim in col and ...
#                elif dimension in col:
#                    # if col != <dim>_acc col and the time lag of the col is <= lag
#                    if len(col.split("_")) > 2 and int(col.split("_")[-1]) <= lag:
#                        input_cols.append(col)
#                    # if the col == <dim>_acc col
#                    elif len(col.split("_")) == 2:
#                        input_cols.append(col)
#
#            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_average'] = np.mean(lag_15_df[input_cols], axis=1)
#            lag_15_df[f'rolling_T_minus_{lag}_{dimension}_variance'] = np.var(lag_15_df[input_cols], axis=1)
#
#    # removing columns whose time lag isn't divisible by 5, a statistic column
#    # or one of the original variables given
#    cols_to_keep = ['x_acc','y_acc','z_acc', 'Activity']
#    for col in lag_15_df.columns:
#        if col in cols_to_keep:
#            continue
#        elif 'roll' in col or 'var' in col:
#            cols_to_keep.append(col)
#        elif int(col.split("_")[-1]) % 5 == 0:
#            cols_to_keep.append(col)
#
#    lag_15_df = lag_15_df[cols_to_keep]

    # Grid Searching for best Random Forest Parameters
    rf_param_grid = {'n_estimators': [500, 1000, 3000, 5000],
            'max_features': ['sqrt','log2', None],
            'class_weight': ['balanced', None]}

    model_dict['Random Forest']['GridSearchModel'] = RandomForestClassifier(random_state=5, n_jobs=-1)
    model_dict['Random Forest']['GridSearch'] = GridSearchCV(model_dict['Random Forest']['GridSearchModel'],
            param_grid=rf_param_grid,
            scoring='accuracy',
            cv=8)

    X = lag_15_df.loc[:, ~lag_15_df.columns.isin(['Activity'])].values
    y = lag_15_df['Activity'].values

    model_dict['Random Forest']['GridSearch'].fit(X, y)

    grid_search_results_df = pd.DataFrame(model_dict['Random Forest']['GridSearch'].cv_results_)

    cols_to_keep = []
    for col in grid_search_results_df.columns:
        if col.endswith("_score") or col == 'params':
            cols_to_keep.append(col)
    
    grid_search_results_df = grid_search_results_df[cols_to_keep]

    # training the grid search's best estimator on the entire data set
    # and then saving the model to a file.
    final_model = model_dict['Random Forest']['GridSearch'].best_estimator_
    final_model.fit(X, y)
    joblib.dump(final_model, 'final_model.joblib')
