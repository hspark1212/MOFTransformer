import numpy as np
from pathlib import Path
import json

from model.utils import get_grid, get_griddata, get_energy_histogram


class Dataset(object):
    """ 
    Dataset for Energy grid histogram
    """
    griddata_format = '.griddata16'
    grid_format = '.grid'

    def __init__(self, path, downstream, split):
        self.path = Path(path)
        self.downstream = downstream

        self.cifs, self.values = self.get_data(downstream, split)
        self.histogram = self.get_histogram(split, self.cifs)

    def __call__(self):
        return self.histogram, self.values, self.cifs

    def get_histogram(self, split, cifs, path=None, ):
        if path is None:
            path = self.path

        ls_hist = []
        for cif in cifs:
            f_griddata = path / f'{split}/{cif}{self.griddata_format}'
            f_grid = f_griddata.with_suffix(self.grid_format)

            grid = get_grid(f_grid)
            griddata = get_griddata(f_griddata)

            try:
                griddata.reshape(*grid['GRID_NUMBERS'].astype('int'))
            except ValueError:
                raise ValueError(grid.stem)

            histogram = get_energy_histogram(griddata)
            ls_hist.append(histogram)

        ls_hist = np.array(ls_hist, dtype='float32')
        return ls_hist

    def get_data(self, downstream, split, path=None):
        if path is None:
            path = self.path

        with open(path / f"{split}_{downstream}.json") as f:
            data = json.load(f)

        cifs = np.array(list(data.keys()), dtype=object)
        values = np.fromiter(data.values(), dtype='float32')

        return cifs, values
