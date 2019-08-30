import sys
from argparse import ArgumentParser

import pandas as pd
from sklearn.externals import joblib

from format_data 

def create_boolean(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def transform(data, scaler_filename='../data/scaler.obj'):

    # loading StandardScaler object that holds the means and standard
    # deviations from training data
    scaler = joblib.load(scaler_filename)

    # standardizing incoming data
    data = scaler.transform(data)

    # creating lagged variables




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-f',
                        '--filename',
                        help=,
                        default='final_model.joblib')
    parser.add_argument('-t',
                        '--transform',
                        default=True,
                        type=create_boolean,
                        help='Whether the data should be transformed
                        (i.e. scaled and time-lag features created)'.)
    parser.add_argument('-m',
                        '--model_file',
                        default='../data/X_test.csv',
                        help='The CSV filename of the data to use for 
                        prediction. Note that if '.)
    args = parser.parse_args()

    # load data
    X = pd.read_csv(args.filename).values

    # tranform data if necessary
    if args.transform:
        X = transform(X)
        
    # load model
    model = joblib.load(args.filename)

    # predict
    y_hat = model.predict(X)

    # return prediction to standard out
    sys.stdout.write(y_hat)
