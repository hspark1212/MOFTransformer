import argparse
import copy
import numpy as np
from pathlib import Path
import pickle

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from model.data import Dataset

parser = argparse.ArgumentParser(description='Predict property using Energy histogram')
parser.add_argument(
    '--downstream', '-d', type=str, help='(str) Downstream task in DATA folder.'
)
parser.add_argument(
    '--path', '-p', type=str, help='(str) Downstream path'
)

parser.add_argument(
    '--model-direc', '-m', type=str, default=None, dest='model', help='(str) Model directory'
)

parser.add_argument(
    '--output', type=str, default=None, help='(str) Save log'
)

args = parser.parse_args()


class Scaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, tensor):
        return (tensor * self.std) + self.mean


def evaluate(model, scaler, x, y):
    ys_pred = model.predict(x)
    y_pred = scaler.denorm(ys_pred)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return r2, mae


def main():
    global args

    path, downstream = args.path, args.downstream
    
    model_path = Path(args.model)/'best_model.pickle'
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    best_model = data['model']
    mean = data['mean']
    std = data['std'] 
    scaler = Scaler(mean, std)

    test_db = Dataset(path, downstream, 'test')
    x_test, y_test, cif_test = test_db()


    scaler = Scaler(mean, std)

    ys_test = scaler.norm(y_test)

    test_r2, test_mae = evaluate(best_model, scaler, x_test, y_test)


    with open(args.output, 'a') as f:
        f.write(f"Model:{path}\tDownstream:{downstream}\tR2:{test_r2}\tMAE:{test_mae}\n")
    

if __name__ == '__main__':
    main()
