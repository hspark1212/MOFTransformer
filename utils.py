import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


def visualize_grid(grid_data, cell=None, zero_index=101, sign="=", path_cif=None):
    """
    :param grid_data:
    :param cell: (x, y, z) cell size, List
    :param zero_index: zero index, Int
    :param sign: sign for zero index, String (>, =, <)
    :param path_cif: (optional) path for cif to visualize with ase
    """
    grid_data = np.array(grid_data)
    if len(grid_data.shape) == 1:
        assert cell
        grid_data = np.reshape(grid_data, cell, order="F")
    elif len(grid_data.shape) > 3:
        raise Exception("make grid_data to 3-dimension")

    assert sign in ("=", ">", "<")
    if sign == "=":
        mask = grid_data == zero_index
    elif sign == ">":
        mask = grid_data > zero_index
    else:
        mask = np.logical_and(grid_data < zero_index, grid_data > 0)

    x, y, z = np.where(mask)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    p = ax.scatter(
        xs=x,
        ys=y,
        zs=z,
        c=grid_data[mask],
        cmap=plt.cm.Greens,
        s=1,
        alpha=0.5)

    if path_cif:
        atoms = read(path_cif)
        pos = atoms.positions.astype(int)
        ax.scatter(
            xs=pos[:, 0],
            ys=pos[:, 1],
            zs=pos[:, 2],
            c=atoms.numbers,
            alpha=1.0,
        )
    if cell:
        _x, _y, _z = cell
    else:
        _x, _y, _z = (60, 60, 60)

    ax.set_xlim(0, _x)
    ax.set_ylim(0, _y)
    ax.set_zlim(0, _z)

    ax.view_init(30, 300)
    fig.colorbar(p, ax=ax)
    plt.show()
