import time
import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, train_test_split

model_dict = {"Random Forest": 
                  {"ModelPipeline": Pipeline([
                       ('scaler', StandardScaler()),
                       ('model', RandomForestClassifier(n_estimators=500,
                                                        random_state=5,
                                                        n_jobs=-1,
                                                        verbose=1))
                       ])
                   },
              "AdaBoost Classifier":
                  {"ModelPipeline": Pipeline([
                       ('scaler', StandardScaler()),
                       ('model', AdaBoostClassifier(n_estimators=500,
                                                    random_state=5))          
                       ])
                   },
              "GradientBoost Classifier":
                  {"ModelPipeline": Pipeline([
                       ('scaler', StandardScaler()),
                       ('model', GradientBoostingClassifier(n_estimators=500,
                                                            random_state=5,
                                                            verbose=1)),
                       ])
                   },
              "Multinomial Logistic Regression":
                  {"ModelPipeline": Pipeline([
                       ('scaler', StandardScaler()),
                       ('model', LogisticRegression(multi_class="ovr",
                                                    solver='saga',
                                                    max_iter=500,
                                                    n_jobs=-1,
                                                    verbose=1)),
                       ])
                   },
              "MLP":
                  {"ModelPipeline": Pipeline([
                       ('scaler', StandardScaler()),
                       ('model', MLPClassifier(hidden_layer_sizes=(100, 100),
                                               batch_size=500,
                                               verbose=1))
                                 
                       ])
                   }
             }

def cross_validate_multiple_models(X, y, model_dict=model_dict, cv=8,
                                   verbose=True):
    '''
    Cross validates mutliple ModelPipeline's that are composed of a
    StandardScaler and a model.

    Parameters:
    ----------
    X : (2D numpy ndarray)
        A two dimensional array of data to which the model pipeline should be
        fit.
    y : (1D numpy ndarray)
        A vector of response values that line up with the data provided to the
        X parameter.
    model_dict : (dict)
        A dictionary of dictionaries. The key's of the dictionary should be
        the names of the model ('Random Forest', 'MLP', 'SVM', etc) and the
        values should be another dictionary. Within this subdictionary, there
        should be a key-value pair whose key is 'ModelPipeline' and whose value
        is a valid sklearn Pipeline() object.
    cv : (int)
        A integer specifying 'K' in K-Fold cross validation.
    verbose : (bool)
        The verbosity level.

    Returns:
    ----------
    model_dict : (dict)
        The same model dict, however there will be a new key-value pair in each
        model's subdictionary that contains an array of the CV scores. The key
        for this array is 'CV Scores'.
    '''
    for model_name, model_sub_dict in model_dict.items():

        if verbose:
            print(f"Time = {time.ctime()} | Cross validating a {model_name} on X and y")

        model_scores = cross_val_score(model_sub_dict['ModelPipeline'],
                                       X,
                                       y,
                                       cv=cv,
                                       scoring='accuracy',
                                       n_jobs=-1,
                                       verbose=int(verbose))

        model_sub_dict['CV Scores'] = model_scores

        if verbose:
            print(f"Time = {time.ctime()} | {model_name} Accuracy = {np.mean(model_scores)}")

    return model_dict
