import os
import argparse
from collections import Counter
import numpy as np
from pathlib import Path
import pandas as pd
import json
import copy
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="""Predict property using descriptor (geometric features and chemical features).
Before running machine learning, chemical-descriptor code should be running (mof_features/MOFFeatures.py).
reference : 10.1021/acsami.1c18521""")

parser.add_argument(
    '--downstream', '-d', type=str, help='(str) Downstream task in DATA folder.'
)
parser.add_argument(
    '--path', '-p', type=str, help='(str) Downstream path'
)
parser.add_argument(
    '--mean', '-m', type=float, default=None, help='(str) Mean of dataset (default : None)'
)
parser.add_argument(
    '--std', '-s', type=float, default=None, help='(str) Standard deviation of dataset (default : None)'
)
parser.add_argument(
    '--path-chemical-descriptor', '-pc', type=str, default=None,
    help='(str) Path to chemical-descriptor csv file from mof_features/chemical_features.py.'
)
parser.add_argument(
    '--path-geometric-descriptor', '-pg', type=str, default=None,
    help='(str) Path to geometric-descriptor csv file from mof_features/geometric_features.py.'
)
parser.add_argument(
    '--output', '-o', type=str, default=None, help='(str) Output File'
)
parser.add_argument(
    '--outdir', type=str, default=None, help='(str) Model save directory'
)

parser.add_argument(
    '--load-model', type=str, default=None, help='(str) Model Load'
)


parser.add_argument(
    '--test-only', action='store_true'
)

args = parser.parse_args()


class Scaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, tensor):
        return (tensor - self.mean) / self.std

    def inverse_transform(self, tensor):
        return (tensor * self.std) + self.mean


def get_data(df, split):
    global args

    path = Path(args.path)
    downstream = args.downstream

    with open(path / f'{split}_{downstream}.json') as f:
        data = json.load(f)

    cifs = data.keys()
    df_split = df[df.index.isin(cifs)]
    index = df_split.index

    x = df_split.to_numpy('float')
    y = np.array([data[i] for i in index])

    return x, y


def predict(x, y, model, x_scaler, y_scaler):
    xs_test = x_scaler.transform(x)
    ys_pred = model.predict(xs_test)
    y_pred = y_scaler.inverse_transform(ys_pred)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return r2, mae


def main():
    global args

    # Get dataframe of descriptor
    df_geo = pd.read_csv(args.path_geometric_descriptor, index_col=0)
    df_chem = pd.read_csv(args.path_chemical_descriptor, index_col='MOF')
    df = pd.concat([df_geo, df_chem], axis=1).dropna()

    # change metal type to index
    metal_type = Counter(df['metal type'])
    for i, metal in enumerate(metal_type):
        df = df.replace(metal, i)

    x_train, y_train = get_data(df, 'train')
    x_val, y_val = get_data(df, 'val') # Validation
    x_test, y_test = get_data(df, 'test')

    mean = args.mean
    std = args.std
    if mean is None or std is None:
        mean = np.mean(y_train)
        std = np.std(y_train)

    x_scaler, y_scaler = StandardScaler(), Scaler(mean, std)
    xs_train = x_scaler.fit_transform(x_train)
    ys_train = y_scaler.transform(y_train)

    best_r2 = 0
    best_model = None
    
    if args.load_model:
        if not os.path.exists(args.load_model):
            raise ValueError('model does not exists!')

        with open(args.load_model, 'rb') as f:
            load_model = pickle.load(f)

        load_model.fit(xs_train, ys_train)
        best_model = load_model

    else:
        for n_estimators in [10, 20, 30, 50, 100, 200, 300, 500, 1000]:
            model = RandomForestRegressor(n_estimators=n_estimators)
            model.fit(xs_train, ys_train)

            r2, mae = predict(x_val, y_val, model, x_scaler, y_scaler)
            if r2 > best_r2:
                best_r2 = r2
                best_model = copy.deepcopy(model)

    test_r2, test_mae = predict(x_test, y_test, best_model, x_scaler, y_scaler)

    if args.outdir:
        save_path = Path(args.outdir)
        save_path.mkdir(exist_ok=True, parents=True)
        with open(save_path/'best_model.pickle', 'wb') as f:
            pickle.dump(best_model, f)


    with open(args.output, 'a') as f:
        f.write(f"Model:{args.path}\tDownstream:{args.downstream}\tR2:{test_r2}\tMAE:{test_mae}\n")


if __name__ == '__main__':
    main()
