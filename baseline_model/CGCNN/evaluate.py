import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error


def main(direc, output=None):

    direc = Path(direc)
    csv_file = direc/'test_results.csv'
    
    if not csv_file.exists():
        raise ValueError(f'{csv_file} must be exists')
    df = pd.read_csv(csv_file, names= ['cif_id', 'y_true', 'y_pred'])
    
    y_true = df['y_true']
    y_pred = df['y_pred']

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    if output is None:
        output = direc/'evaluate.txt'

    with open(output, 'a') as f:
        f.write(f"Model:{direc}\t\tR2:{r2}\tMAE:{mae}\n")


    print (f'mae : {mae}\nr2 : {r2}')

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d',)
    parser.add_argument('--output', '-o', default=None)
    
    args = parser.parse_args()

    
    main(args.dir, args.output)
