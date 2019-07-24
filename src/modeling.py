from random import sample
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
all_colors =  [k for k, v in pltc.cnames.items()]

def cross_validate_multiple_models(X, y, model_dict, verbose=True):
    '''
    Performs K-fold cross validation where k=8 by fitting all model in
    model_dict.values() to X and y.
    '''
    for model_name, model_sub_dict in model_dict.items():

        if verbose:
            print(f"Cross validating a {model_name} on X and y...")

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


def cv_error_comparison_plot(df, x_label, y_label, title, filename=None):
    '''
    Creates a box plot that shows the CV error rates for various models.
    '''
    fig, ax = plt.subplots(figsize=(12, 9))

    df = df[df.mean(axis=0).sort_values(ascending=False).index]
    parts = ax.violinplot(df.T,
                          showmeans=True,
                          showmedians=False,
                          showextrema=False)

    color_sample = sample(all_colors, len(df.columns))
    for x, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_sample[x])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # setting y-axis to percentage
    full_perc_range = np.arange(0, 1.1, 0.1)
    plt.yticks(full_perc_range,
               labels=['{:,.2%}'.format(x) for x in full_perc_range])

    plt.xticks(range(1, len(df.columns)+1), labels=df.columns)

    plt.xlabel(x_label, fontweight='bold', fontsize=12)
    plt.xticks(rotation=30)
    plt.ylabel(y_label, fontweight='bold', fontsize=12)
    plt.suptitle(title, fontweight='bold', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if not filename:
        plt.show()
    elif filename:
        plt.savefig(filename)


if __name__ == "__main__":

    model_dict = {"Random Forest": 
                      {"Model": RandomForestClassifier(n_estimators=500,
                                                       random_state=5)},
                  "AdaBoost Classifer":
                      {"Model": AdaBoostClassifier(n_estimators=500)},
                  "Multinomial Logistic Regression":
                      {"Model": LogisticRegression(multi_class="ovr",
                                                   solver='saga',
                                                   max_iter=500)},
                  "Support Vector Machine":
                      {"Model": svm.SVC(kernel='poly',
                                        degree=5,
                                        gamma='auto')},
                  "MLP":
                      {"Model": MLPClassifier(hidden_layer_sizes=(100, 100),
                                              batch_size=500)}
                 }

    data_dir = "../data"
    X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv",
                          header=None)
    y_train = pd.read_csv(f"{data_dir}/y_train.csv",
                          header=None)

    model_dict = cross_validate_multiple_models(X_train,
                                                np.ravel(y_train.values),
                                                model_dict)

    cv_error_dict = {}
    for name, sub_dict in model_dict.items():
        cv_error_dict[name] = sub_dict['CV Scores']

    cv_error_comparison_plot(pd.DataFrame(cv_error_dict),
                             x_label="Model Type",
                             y_label="Accuracy",
                             title='8 Fold Cross Validation Accuracy',
                             filename="../images/ModelComparison.png")
