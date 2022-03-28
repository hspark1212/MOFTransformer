import math
from itertools import combinations

import torch

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from ase.io import read
from ase.data import covalent_radii

from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import seaborn as sns

from model.assets.colors import cpk_colors


def visualize_grid(grid_data, cell=None, zero_index=51, sign=">", path_cif=None):
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
        cmap=plt.cm.plasma,
        s=10,
        alpha=0.5)

    if path_cif:
        atoms = read(path_cif)
        pos = atoms.positions.astype(int)
        ax.scatter(
            xs=pos[:, 0],
            ys=pos[:, 1],
            zs=pos[:, 2],
            c=atoms.numbers,
            s=30,
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


def cuboid_data2(o, size=(1, 1, 1)):
    """
    help function for 3d plot
    """

    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plot_cube_at2(positions, sizes=None, colors=None, **kwargs):
    """
    help function for 3d plot
    """
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors, 6), **kwargs)


def plot_cube_at3(positions, sizes=None, colors=None, **kwargs):
    """
    help function for 3d plot
    """
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))

    return Poly3DCollection(np.concatenate(g),
                            facecolors=colors,
                            **kwargs)


def get_heatmap(out, batch_idx, graph_len=300, grid_len=6 * 6 * 6):
    """
    attention rollout  in "Qunatifying Attention Flow in Transformers" paper.

    :param out: output of model.infer(batch)
    :param batch_idx: batch index
    :param graph_len: the length of graph embedding
    :param grid_len: the length of grid embedding
    :return: heatmap_graph, heatmap_grid
    """
    # out = model.infer(batch)

    attn_weights = torch.stack(out["attn_weights"])  # [num_layers, B, num_heads, max_len, max_len]
    att_mat = attn_weights[:, batch_idx]  # [num_layers, num_heads, max_len, max_len]

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)  # [num_layers, max_len, max_len]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att

    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # [num_layers, max_len, max_len]

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())  # [num_layers, max_len, max_len]
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]  # [max_len, max_len]

    # Don't drop class token when normalizing
    cost_graph = v[0, :graph_len + 1] / v[0].max()
    cost_grid = v[0, graph_len + 1:] / v[0].max()

    heatmap_graph = cost_graph[1:].detach().numpy()  # omit cls token
    heatmap_grid = cost_grid[1:-1].reshape(6, 6, 6).detach().numpy()  # omit cls + volume tokens

    return heatmap_graph, heatmap_grid


class Visualize(object):
    def __init__(self, path_cif,
                 interpolate=False,
                 show_cell=False,
                 show_uni_idx=False,
                 show_colorbar=False,
                 atomic_scale=200,
                 ):
        self.path_cif = path_cif
        self.interpolate = interpolate
        self.atomic_scale = atomic_scale
        self.show_cell = show_cell
        self.show_uni_idx = show_uni_idx
        self.show_colorbar = show_colorbar

        self.fig = plt.figure()
        self._ax = self.fig.add_subplot(projection='3d')
        self.default_setting()

    @property
    def ax(self):
        return self._ax

    @ax.setter
    def ax(self, d):
        self._ax.update(d)

    def default_setting(self):
        self._ax.view_init(0, 0)
        self._ax.grid(visible=False)
        self._ax.set_axis_off()
        self.color_palatte = sns.color_palette("rocket_r", as_cmap=True)

        # colorbar
        if self.show_colorbar:
            norm = matplotlib.colors.Normalize(0, 1)
            smap = plt.cm.ScalarMappable(cmap=self.color_palatte, norm=norm)
            cbar = self.fig.colorbar(smap, ax=self._ax, fraction=0.1, shrink=0.8)
            cbar.ax.tick_params(labelsize=11)
            cbar.ax.set_ylabel('attention score', rotation=270, labelpad=20, fontdict={"size": 20})

    @staticmethod
    def get_primitive_structure(path_cif):
        st = CifParser(path_cif, occupancy_tolerance=2.0).get_structures(primitive=True)[0]
        return st

    @staticmethod
    def get_interpolate_structure(st, max_length=60, min_length=30):
        # make super-cell
        scale_abc = []
        for l in st.lattice.abc:
            if l > max_length:
                print(f"primitive cell is larger than max_length {max_length}")
                break
            elif l < min_length:
                scale_abc.append(math.ceil(min_length / l))
            else:
                scale_abc.append(1)

        st.make_supercell(scale_abc)
        atoms = AseAtomsAdaptor().get_atoms(st)

        # interpolate min_length and orthogonal
        atoms.set_cell(np.identity(3) * min_length, scale_atoms=True)

        return atoms

    def draw_line(self, pos1, pos2, **kwargs):
        self.ax.plot3D(*zip(pos1, pos2), **kwargs)

    def draw_cell(self, lattice, center=None, **kwargs):
        """
        draw cell using matplotlib
        :param cell:  lattice vectors (3 X 3 matrix)
        """
        # draw cell
        vec1, vec2, vec3 = lattice
        if not center:
            center = np.zeros(3)

        opp_vec = vec1 + vec2 + vec3 + center

        for vec1, vec2 in combinations([vec1, vec2, vec3], 2):
            self.draw_line(center, vec1, **kwargs)
            self.draw_line(center, vec2, **kwargs)
            self.draw_line(vec1, vec1 + vec2, **kwargs)
            self.draw_line(vec2, vec1 + vec2, **kwargs)
            self.draw_line(vec1 + vec2, opp_vec, **kwargs)

    def draw_atoms(self, atoms, uni_idx):
        # draw atoms
        coords = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers])
        atomic_colors = np.array([cpk_colors[i] for i in atomic_numbers])

        self._ax.scatter(
            xs=coords[:, 0],
            ys=coords[:, 1],
            zs=coords[:, 2],
            c=atomic_colors,
            s=atomic_sizes * self.atomic_scale,
            marker="o",
            edgecolor="black",
            linewidths=0.8,
        )

        if uni_idx and self.show_uni_idx:
            # assert not self.interpolate, print("interpolate should be False to visualize uni_idx")
            for idxes in uni_idx:
                rand_rgb = np.random.randint(low=0, high=255, size=3) / 255
                uni_coords = coords[idxes]
                self._ax.scatter(
                    xs=uni_coords[:, 0],
                    ys=uni_coords[:, 1],
                    zs=uni_coords[:, 2],
                    c=[list(rand_rgb)] * len(uni_coords),
                    s=500,
                    marker="o",
                    edgecolor="black",
                    linewidths=0.5,
                    alpha=0.5,
                )

    def draw_heatmap_graph(self, atoms, heatmap_graph, uni_idx):
        assert uni_idx, print("uni_idx doesn't exist")

        coords = atoms.get_positions()
        cm_heatmap = self.color_palatte(heatmap_graph)

        for i, idxes in enumerate(uni_idx):
            uni_coords = coords[idxes]
            c = cm_heatmap[i]
            self._ax.scatter(
                xs=uni_coords[:, 0],
                ys=uni_coords[:, 1],
                zs=uni_coords[:, 2],
                color=c,
                s=1000,
                marker="o",
                linewidths=0.5,
                alpha=0.7,
            )

    def draw_heatmap_grid(self, heatmap_grid):
        # set colors
        cm_heatmap = self.color_palatte(heatmap_grid.flatten())
        colors = np.repeat(cm_heatmap[:, None, :], 6, axis=1).reshape(-1, 4)

        # set positions and size
        positions = [(5 * i, 5 * j, 5 * k) for i in range(6) for j in range(6) for k in range(6)]
        sizes = [[5, 5, 5]] * 6 * 6 * 6

        pc = plot_cube_at3(positions, sizes, colors=colors, edgecolor=None, alpha=0.1)
        self._ax.add_collection3d(pc)

    def draw(self, atoms, cell, heatmap_graph=False, heatmap_grid=False, uni_idx=False):
        # draw lattice
        if self.show_cell:
            self.draw_cell(cell, color="black")
            self._ax.set_xlim([0, 30])
            self._ax.set_ylim([0, 30])
            self._ax.set_zlim([0, 30])

        # draw atoms
        self.draw_atoms(atoms, uni_idx)
        if heatmap_graph is not False:
            self.draw_heatmap_graph(atoms, heatmap_graph, uni_idx)

        if heatmap_grid is not False:
            self.draw_heatmap_grid(heatmap_grid)
        # draw uni_idx

    def view(self, heatmap_graph=False, heatmap_grid=False, uni_idx=False):
        # get structure from cif
        st = self.get_primitive_structure(self.path_cif)
        atoms = AseAtomsAdaptor().get_atoms(st)

        # interpolate
        if self.interpolate:
            atoms = self.get_interpolate_structure(st)

        lattice = atoms.cell.array
        self.draw(atoms, lattice, heatmap_graph, heatmap_grid, uni_idx)
