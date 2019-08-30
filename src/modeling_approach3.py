import random
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


if __name__ == "__main__":

    data_dir = "../data"
    lag_5_X = pd.read_csv(f"{data_dir}/lag_5_X.csv")
    lag_5_y = pd.read_csv(f"{data_dir}/lag_5_y.csv")

    model_dict = {
            "Random Forest": 
                {"Model": RandomForestClassifier(n_estimators=500,
                                                 random_state=5)},
            "AdaBoost Classifer":
                {"Model": AdaBoostClassifier(n_estimators=500)},
            "Multinomial Logistic Regression":
                {"Model": LogisticRegression(multi_class="multinomial",
                                             solver='saga',
                                             max_iter=500)},
            "Support Vector Machine":
                {"Model": svm.SVC()},
            "MLP":
                {"Model": MLPClassifier(hidden_layer_sizes=(100, 100),
                                        batch_size=500)}
                 }

    model_dict = CV_multiple_models(lag_5_X,
                                    lag_5_y,
                                    model_dict,
                                    verbose=True)

#    n_steps = 1
#    lag_5_X.values.reshape(-1, n_steps, lag_5_X.shape[1])
#
#    # training parameters
#    learning_rate = 0.01
#    n_epochs = 50
#    batch_size = 50
#
#    # network architecture
#    hidden_layer_array = [100, 100]
#    n_outputs = len(np.unique(lag_5_y))
#
#    model_dict["RNN"] = {"Model": RNNClassifier(),
#            "fit_params": {"hidden_layer_architecture": hidden_layer_array,
#                "n_outputs": n_outputs,
#                "n_steps": n_steps,
#                "learning_rate": learning_rate,
#                "batch_size": batch_size,
#                "n_epochs": n_epochs}} 

    cv_error_dict = {}
    for name, sub_dict in model_dict.items():
        cv_error_dict[name] = sub_dict['cv_scores']

    cv_error_comparison_plot(pd.DataFrame(cv_error_dict),
                             x_label="Model Type",
                             y_label="Accuracy",
                             title='Non-blended 8 Subject Cross Validation Accuracy',
                             long_x_names=True,
                             filename='../images/non_blended_cv_model_comparison.png')

#    # getting feature importances for fully trained Random Forest
#    model_dict['Random Forest']['Model'].fit(lag_5_X.values, np.ravel(lag_5_y.values))
#
#    feature_imps = model_dict['Random Forest']['Model'].feature_importances_
#    feat_imp_df = pd.DataFrame({"Importances": feature_imps,
#                                "Variable Names": lag_5_X.columns})
#    feat_imp_df.sort_values('Importances', ascending=False, inplace=True)
#
#    # Grid Searching for best Random Forest Parameters on lag_15 data
#    rf_param_grid = {'n_estimators': [250, 500, 1000, 1500],
#            'max_features': ['sqrt','log2', None],
#            'class_weight': ['balanced', None]}
#
#    model_dict['Random Forest']['GridSearchModel'] = RandomForestClassifier(random_state=5, n_jobs=-1)
#    model_dict['Random Forest']['GridSearch'] = GridSearchCV(model_dict['Random Forest']['GridSearchModel'],
#            param_grid=rf_param_grid,
#            scoring='accuracy',
#            cv=8)
#
#    lag_15_X = pd.read_csv(f"{data_dir}/lag_15_X.csv")
#    lag_15_y = pd.read_csv(f"{data_dir}/lag_15_y.csv")
#
#    model_dict['Random Forest']['GridSearch'].fit(lag_15_X.values,
#                                                  np.ravel(lag_15_y.values))
#
#    grid_search_results_df = pd.DataFrame(model_dict['Random Forest']['GridSearch'].cv_results_)
#
#    cols_to_keep = []
#    for col in grid_search_results_df.columns:
#        if col.endswith("_score") or col == 'params':
#            cols_to_keep.append(col)
#    
#    grid_search_results_df = grid_search_results_df[cols_to_keep]
#
#    # training the grid search's best estimator on the entire data set
#    # and then saving the model to a file.
#    final_model = model_dict['Random Forest']['GridSearch'].best_estimator_
#    final_model.fit(lag_15_X.values, np.ravel(lag_15_y.values))
#    joblib.dump(final_model, 'final_model.joblib')
