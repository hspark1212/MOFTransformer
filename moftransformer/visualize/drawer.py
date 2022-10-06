import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from ase.data import covalent_radii
from moftransformer.assets.colors import cpk_colors
from moftransformer.visualize.utils import plot_cube


def draw_colorbar(fig, ax, cmap, minatt, maxatt, **cbar_kwargs):
    norm = Normalize(0., 1.)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(smap, ax=ax, fraction=cbar_kwargs['fraction'], shrink=cbar_kwargs['shrink'])
    cbar.ax.tick_params(labelsize=cbar_kwargs['fontsize'])
    ticks_loc = np.linspace(0, 1, cbar_kwargs['num_ticks'])
    ticks_label = np.round(np.linspace(minatt, maxatt, cbar_kwargs['num_ticks']),
                           decimals=cbar_kwargs['decimals'])
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    cbar.ax.set_yticklabels(ticks_label)

    cbar.ax.set_ylabel('Attention score', rotation=270, labelpad=cbar_kwargs['labelpad'],
                       fontdict={"size": cbar_kwargs['labelsize']})


def draw_line(ax, pos1, pos2, **kwargs):
    """
    Draw line from position 1 to position 2
    :param ax: <matplotlib.axes> figure axis
    :param pos1: <np.array> starting point position
    :param pos2: <np.array> end point position
    :param kwargs: matplotlib plot3D kwargs
    :return:
    """
    ax.plot3D(*zip(pos1, pos2), **kwargs)


def draw_cell(ax, lattice, s_point=None, **kwargs):
    """
    Draw unit-p_lattice p_lattice using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param lattice: <np.array> p_lattice vectors (3 X 3 matrix)
    :param s_point: <np.array> start point of p_lattice
    :param kwargs: matplotlib plot3D kwargs
    """
    vec1, vec2, vec3 = lattice
    if s_point is None:
        s_point = np.zeros(3)

    opp_vec = vec1 + vec2 + vec3 + s_point

    for v1, v2 in combinations([vec1, vec2, vec3], 2):
        draw_line(ax, s_point, s_point + v1, **kwargs)
        draw_line(ax, s_point, s_point + v2, **kwargs)
        draw_line(ax, s_point + v1, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v2, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v1 + v2, opp_vec, **kwargs)


def draw_atoms(ax, atoms, atomic_scale):
    """
    Draw p_atoms using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param atoms: <ase.p_atoms> Target p_atoms for drawing
    :param atomic_scale: <float> scaling factor for draw_atoms.
    """
    coords = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers])
    atomic_colors = np.array([cpk_colors[i] for i in atomic_numbers])
    ax.scatter(
        xs=coords[:, 0],
        ys=coords[:, 1],
        zs=coords[:, 2],
        c=atomic_colors,
        s=atomic_sizes * atomic_scale,
        marker="o",
        edgecolor="black",
        linewidths=0.8,
        alpha=1.0,
    )


def draw_heatmap_grid(ax, positions, colors, lattice, num_patches, alpha=0.5, **kwargs):
    cubes = plot_cube(positions, colors, lattice=lattice,
                      num_patches=num_patches, edgecolor=None, alpha=alpha)
    ax.add_collection3d(cubes, **kwargs)


def draw_heatmap_graph(ax, atoms, uni_idx, colors, atomic_scale, alpha):
    coords = atoms.get_positions()
    for i, idxes in enumerate(uni_idx):
        uni_coords = coords[idxes]
        # att = heatmap_graph[i]
        # c = cmap(scaler(att, minatt, maxatt))
        ax.scatter(
            xs=uni_coords[:, 0],
            ys=uni_coords[:, 1],
            zs=uni_coords[:, 2],
            color=colors[i],
            s=atomic_scale,
            marker='o',
            linewidth=0,
            alpha=alpha,
        )
