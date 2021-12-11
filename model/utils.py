from ase.io import read
from ase.visualize import view
import numpy as np
import matplotlib.pyplot as plt


def visualize_grid(grid_data, cell, normalize=False, cif_path=None):
    """
    Args:
        grid_data: Tensor, [H, W, D] or [H*W*D]
        cell: list # H, W, D
        normalize: bool, if True, normalization min_max with 5000.
        cif_path: str, visualize cif file
    """
    if cif_path:
        atoms = read(cif_path)
        view(atoms)

    if len(grid_data.shape) > 1:
        grid_data = grid_data.reshape(-1)
    assert len(grid_data.shape) == 1

    if normalize:
        grid_data[grid_data >= 5000.] = 5000.
        grid_data[grid_data < -5000.] = -5000.
        grid_data = grid_data / (5000. * 2) + 0.5

    hist = np.histogram(grid_data, range=(0, 2))
    print(hist)

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')

    _x, _y, _z = cell
    xz = np.arange(0, _x)
    yz = np.arange(0, _y)
    zz = np.arange(0, _z)
    list_xyz = []
    for z in zz:
        for y in yz:
            for x in xz:
                list_xyz.append([x, y, z])
    xyz = np.array(list_xyz)

    mask = grid_data > 0.5
    new_xyz = xyz[mask]
    new_griddata = grid_data[mask]

    p = ax.scatter(
        xs=new_xyz[:, 0],
        ys=new_xyz[:, 1],
        zs=new_xyz[:, 2],
        c=new_griddata,
        cmap=plt.cm.Greens,
        alpha=0.1)

    ax.set_xlim(0, _x)
    ax.set_ylim(0, _y)
    ax.set_zlim(0, _z)

    fig.colorbar(p, ax=ax)
    plt.show()