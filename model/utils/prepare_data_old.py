import os
import math
import logging
import logging.handlers
import json
import subprocess
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa

from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cssr import Cssr

from ase.neighborlist import natural_cutoffs
from ase import neighborlist

GRIDAY_PATH = os.path.realpath(os.path.join(__file__, '../../../libs/GRIDAY/scripts/grid_gen'))
FF_PATH = os.path.realpath(os.path.join(__file__, '../../../libs/GRIDAY/FF'))


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_unique_atoms(atoms):
    # get graph
    cutoff = natural_cutoffs(atoms)
    neighbor_list = neighborlist.NeighborList(cutoff, self_interaction=True, bothways=True)
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix()

    # Get N, N^2
    numbers = atoms.numbers
    number_sqr = np.multiply(numbers, numbers)

    matrix_sqr = matrix.dot(matrix)
    matrix_cub = matrix_sqr.dot(matrix)
    matrix_sqr.data[:] = 1  # count 1 for atoms

    # calculate
    list_n = [numbers, number_sqr]
    list_m = [matrix, matrix_sqr, matrix_cub]

    arr = [numbers]
    for m in list_m:
        for n in list_n:
            arr.append(m.dot(n))

    arr = np.vstack(arr).transpose()
    u, unique_idx, unique_count = np.unique(arr, axis=0, return_index=True, return_counts=True)

    # sort
    final_unique_idx = unique_idx[np.argsort(-unique_count)].tolist()
    final_unique_count = unique_count[np.argsort(-unique_count)].tolist()

    return final_unique_idx, final_unique_count


def get_crystal_graph(st, radius=8, max_num_nbr=12):
    atoms = AseAtomsAdaptor().get_atoms(st)

    atom_num = list(atoms.numbers)  # [N]

    dist_matrix = atoms.get_all_distances(mic=True)

    nbr_idx = []
    nbr_dist = []

    for i, row in enumerate(dist_matrix):
        cond = np.logical_and(row > 0., row < radius)
        idx = np.where(cond)[0]
        sort = np.argsort(row[idx])
        sorted_idx = idx[sort]
        sorted_dist = row[sorted_idx]

        nbr_idx.extend(sorted_idx[:max_num_nbr])
        nbr_dist.extend(sorted_dist[:max_num_nbr])

    # get same-topo atoms
    uni_idx, uni_count = get_unique_atoms(atoms)

    return atom_num, nbr_idx, nbr_dist, uni_idx, uni_count


def calculate_scaling_matrix_for_orthogonal_supercell(cell_matrix, eps=0.01):
    """
    cell_matrix: contains lattice vector as column vectors.
                 e.g. cell_matrix[:, 0] = a.
    eps: when value < eps, the value is assumed as zero.
    """
    inv = np.linalg.inv(cell_matrix)

    # Get minimum absolute values of each row.
    abs_inv = np.abs(inv)
    mat = np.where(abs_inv < eps, np.full_like(abs_inv, 1e30), abs_inv)
    min_values = np.min(mat, axis=1)

    # Normalize each row with minimum absolute value of each row.
    normed_inv = inv / min_values[:, np.newaxis]

    # Calculate scaling_matrix.
    # New cell = np.dot(scaling_matrix, cell_matrix).
    scaling_matrix = np.around(normed_inv).astype(np.int32)

    return scaling_matrix


def get_energy_grid(structure, cif_id, root_dataset, split, eg_logger):
    global GRIDAY_PATH, FF_PATH

    eg_file = os.path.join(root_dataset, split) + f'/{cif_id}'

    if os.path.exists(f'{eg_file}.grid') and os.path.exists(f'{eg_file}.griddata'):
        eg_logger.info(f"{cif_id} energy grid already exists")
        return

    random_str = str(np.random.rand()).encode()
    tmp_file = "./{}.cssr".format(hashlib.sha256(random_str).hexdigest())

    Cssr(structure).write_file(tmp_file)  # write_file
    num_grid = [str(round(cell)) for cell in structure.lattice.abc]

    proc = subprocess.Popen(
        [GRIDAY_PATH, *num_grid, f'{FF_PATH}/UFF_Type.def', f'{FF_PATH}/UFF_FF.def', tmp_file, eg_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if err:
        eg_logger.info(f"{cif_id} energy grid failed")
    else:
        eg_logger.info(f"{cif_id} energy grid success")

    proc = subprocess.Popen(['rm', './*.cssr'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, _ = proc.communicate()


def prepare_data(root_cifs, root_dataset,
                 max_num_atoms=1000,
                 max_length=60.,
                 min_length=30.,
                 max_num_nbr=12,
                 calculate_energy_grid=True):
    """
    Args:
        root_cifs (str): root for cif files,
                        it should contains "train" and "test" directory in root_cifs
                        ("val" directory is optional)
        root_dataset (str): root for generated datasets
        max_num_atoms (int): max number atoms in primitive cell
        max_length (float) : maximum length of supercell
        min_length (float) : minimum length of supercell
        max_num_nbr (int) : maximum number of neighbors when calculating graph
        calculate_energy_grid (bool) : calculate energy grid using GRIDDAY or not
    """
    logger = get_logger(filename="prepare_data.log")
    eg_logger = get_logger(filename="prepare_energy_grid.log")

    assert {"train", "val"}.issubset(os.listdir(root_cifs)), \
        print("There is no train or val directories in the root_cifs")

    for split in ["test", "val", "train"]:

        root = os.path.join(root_cifs, split)
        if not os.path.exists(root):
            continue

        # make dataset
        os.makedirs(os.path.join(root_dataset, split), exist_ok=True)

        json_path = os.path.join(root, f"target_{split}.json")

        assert os.path.exists(json_path)

        with open(json_path, "r") as f:
            d = json.load(f)

        batches = []
        for i, (cif_id, target) in enumerate(tqdm(d.items())):

            # 0. check primitive cell and atom number < max_num_atoms
            p = os.path.join(root, cif_id + ".cif")
            try:
                st = CifParser(p, occupancy_tolerance=2.0).get_structures(primitive=True)[0]
            except Exception as e:
                print(e)
                continue

            if len(st.atomic_numbers) > max_num_atoms:
                logger.info(f"{cif_id} failed : more than max_num_atoms in primitive cell")
                continue
            # 1. get crystal graph
            atom_num, nbr_idx, nbr_dist, uni_idx, uni_count = get_crystal_graph(st, radius=8, max_num_nbr=max_num_nbr)
            if len(nbr_idx) % max_num_nbr > 0:
                print("please make radius larger")
                continue

            # 2. make orthogonal cell and supercell with min_length and max_length
            cell_matrix = st.lattice.matrix
            scaling_matrix = \
                calculate_scaling_matrix_for_orthogonal_supercell(cell_matrix, eps=0.01)

            st.make_supercell(scaling_matrix)

            scale_abc = []
            for l in st.lattice.abc:
                if l > max_length:
                    logger.info(f"{cif_id} failed : supercell have more than max_length")
                    break
                elif l < min_length:
                    scale_abc.append(math.ceil(min_length / l))
                else:
                    scale_abc.append(1)

            if len(scale_abc) != len(st.lattice.abc):
                continue

            st.make_supercell(scale_abc)

            # 3. calculate energy grid
            if calculate_energy_grid:
                get_energy_grid(st, cif_id, root_dataset, split, eg_logger)

            logger.info(f"{cif_id} succeed : supercell length {st.lattice.abc}")

            batches.append([cif_id, atom_num, nbr_idx, nbr_dist, uni_idx, uni_count, target])

        # save data using pyarrow
        df = pd.DataFrame(
            batches, columns=["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count", "target"]
        )

        table = pa.Table.from_pandas(df)
        os.makedirs(root_dataset, exist_ok=True)
        with pa.OSFile(
                os.path.join(root_dataset, f"{split}.arrow"), "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


if __name__ == "__main__":
    prepare_data("/home/data/pretrained_mof/ver2/cif", "/home/data/pretrained_mof/ver2/dataset")
