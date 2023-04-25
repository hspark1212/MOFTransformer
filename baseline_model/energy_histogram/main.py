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
    '--mean', '-m', type=float, default=None, help='(str) Mean of dataset (default : None)'
)
parser.add_argument(
    '--std', '-s', type=float, default=None, help='(str) Standard deviation of dataset (default : None)'
)
parser.add_argument(
    '--output', '-o', type=str, default=None, help='(str) Output File'
)

parser.add_argument(
    '--outdir', type=str, default=None, help='(str) Save model path'
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
    mean, std = args.mean, args.std

    train_db = Dataset(path, downstream, 'train')
    val_db = Dataset(path, downstream, 'val')
    test_db = Dataset(path, downstream, 'test')

    x_train, y_train, cif_train = train_db()
    x_val, y_val, cif_val = val_db()
    x_test, y_test, cif_test = test_db()

    if mean is None or std is None:
        mean = np.mean(y_train)
        std = np.std(y_train)

    scaler = Scaler(mean, std)

    ys_train, ys_val, ys_test = scaler.norm(y_train), scaler.norm(y_val), scaler.norm(y_test)

    best_r2 = -1000
    best_model = None

    for alpha in [0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
        model = Lasso(alpha=alpha)
        model.fit(x_train, ys_train)

        r2, mae = evaluate(model, scaler, x_val, y_val)
        if r2 > best_r2:
            best_r2 = r2
            best_model = copy.deepcopy(model)

    if not best_r2:  # best_r2 equals 0:
        raise ValueError('R2 does not be renewed!')

    test_r2, test_mae = evaluate(best_model, scaler, x_test, y_test)

    with open(args.output, 'a') as f:
        f.write(f"Model:{path}\tDownstream:{downstream}\tR2:{test_r2}\tMAE:{test_mae}\n")
    
    if args.outdir:
        save_path = Path(args.outdir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        data = {
            'model': best_model,
            'mean': mean,
            'std': std,
        }
        with open(save_path/'best_model.pickle', 'wb') as f:
            
            pickle.dump(data, f)

if __name__ == '__main__':
    main()
