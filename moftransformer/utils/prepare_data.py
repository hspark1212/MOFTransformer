import os
import math
import logging
import logging.handlers
import json
import subprocess
import hashlib
import pickle
import shutil

import numpy as np

from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cssr import Cssr

from ase.neighborlist import natural_cutoffs
from ase import neighborlist

from moftransformer import __root_dir__


GRIDAY_PATH = os.path.join(__root_dir__, 'libs/GRIDAY/scripts/grid_gen')
FF_PATH = os.path.join(__root_dir__, 'libs/GRIDAY/FF')


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

    uni, unique_idx, unique_count = np.unique(arr, axis=0, return_index=True, return_counts=True)

    # sort
    final_uni = uni[np.argsort(-unique_count)].tolist()
    final_unique_count = unique_count[np.argsort(-unique_count)].tolist()

    arr = arr.tolist()
    final_unique_idx = []
    for u in final_uni:
        final_unique_idx.append([i for i, a in enumerate(arr) if a == u])

    return final_unique_idx, final_unique_count


def get_crystal_graph(st, radius=8, max_num_nbr=12):
    atom_num = list(st.atomic_numbers)

    all_nbrs = st.get_all_neighbors(radius)
    all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance)[:max_num_nbr] for nbrs in all_nbrs]

    nbr_idx = []
    nbr_dist = []
    for nbrs in all_nbrs:
        nbr_idx.extend(list(map(lambda x: x.index, nbrs)))
        nbr_dist.extend(list(map(lambda x: x.nn_distance, nbrs)))

    # get same-topo atoms
    atoms = AseAtomsAdaptor().get_atoms(st)
    uni_idx, uni_count = get_unique_atoms(atoms)

    # convert to small size
    atom_num = np.array(atom_num, dtype=np.int8)
    nbr_idx = np.array(nbr_idx, dtype=np.int16)
    nbr_dist = np.array(nbr_dist, dtype=np.float32)
    uni_count = np.array(uni_count, dtype=np.int16)
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


def make_float16_griddata(file_griddata):
    griddata = np.fromfile(file_griddata, dtype=np.float32)
    griddata[griddata > 6e4] = 6e4
    griddata[griddata < -6e4] = -6e4
    griddata = griddata.astype(np.float16)
    return griddata


def get_energy_grid(structure, cif_id, root_dataset, eg_logger):
    global GRIDAY_PATH, FF_PATH

    eg_file = os.path.join(root_dataset, cif_id)

    random_str = str(np.random.rand()).encode()
    tmp_file = os.path.join(root_dataset, f"{hashlib.sha256(random_str).hexdigest()}.cssr")
    # tmp_file = "{}/{}.cssr".format(, hashlib.sha256(random_str).hexdigest())

    Cssr(structure).write_file(tmp_file)  # write_file
    num_grid = [str(round(cell)) for cell in structure.lattice.abc]

    proc = subprocess.Popen(
        [GRIDAY_PATH, *num_grid, f'{FF_PATH}/UFF_Type.def', f'{FF_PATH}/UFF_FF.def', tmp_file, eg_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if err:
        eg_logger.info(f"{cif_id} energy grid failed {err}")
    else:
        eg_logger.info(f"{cif_id} energy grid success")

    try:
        os.remove(tmp_file)
    except Exception as e:
        print(e)

    if os.path.exists(eg_file + ".griddata"):
        grid_data = make_float16_griddata(eg_file + ".griddata")
        path_save = os.path.join(root_dataset, f"{cif_id}.griddata16")
        pickle.dump(grid_data, open(path_save, "wb"))
        eg_logger.info(f"{cif_id} energy grid changed to np16")

        try:
            os.remove(eg_file + ".griddata")
        except Exception as e:
            print(e)
    else:
        eg_logger.info(f"{cif_id} energy grid failed to change to np16")


def prepare_data(root_cifs,
                 root_dataset,
                 task,
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
        task (str) : name of downstream tasks
        max_num_atoms (int): max number atoms in primitive cell
        max_length (float) : maximum length of supercell
        min_length (float) : minimum length of supercell
        max_num_nbr (int) : maximum number of neighbors when calculating graph
        calculate_energy_grid (bool) : calculate energy grid using GRIDDAY or not
    """

    if not os.path.exists(GRIDAY_PATH):
        raise ImportError('GRIDAY must be installed. \n'
                          'Run the following code in bash, \n\n'
                          '$ moftransformer install-griday\n\n'
                          'or run the following code on Python\n\n'
                          '>>> from moftransformer.utils import install_griday\n'
                          '>>> install_griday()')

    # set logger
    logger = get_logger(filename="prepare_data.log")
    eg_logger = get_logger(filename="prepare_energy_grid.log")

    # automatically split data
    json_path = os.path.join(root_cifs, f"raw_{task}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            d = json.load(f)
            f.close()

        names = np.array(list(d.keys()))
        idxs = np.random.permutation(len(names))
        k = int(len(names) * 0.8)
        k_ = int(len(names) * 0.9)
        idx_train = idxs[:k]
        idx_val = idxs[k:k_]
        idx_test = idxs[k_:]
        split = ["train", "val", "test"]
        for i, idx_ in enumerate([idx_train, idx_val, idx_test]):
            target = {}
            for n in names[idx_]:
                target[n] = d[n]
            save_path = os.path.join(root_cifs, f"{split[i]}_{task}.json")
            json.dump(target, open(save_path, "w"))
            print(f" save {save_path}, the number is {len(target)}")

    for split in ["test", "val", "train"]:
        # check target json and make root_dataset
        json_path = os.path.join(root_cifs, f"{split}_{task}.json")

        assert os.path.exists(json_path)

        root_dataset_split = os.path.join(root_dataset, split)
        # make split directory in root_dataset
        os.makedirs(root_dataset_split, exist_ok=True)
        # copy target_{split}.json to root_data
        shutil.copy(json_path, root_dataset)

        with open(json_path, "r") as f:
            d = json.load(f)
            f.close()

        for i, (cif_id, target) in enumerate(tqdm(d.items())):
            # check file exist (removed in future)
            p_graphdata = os.path.join(root_dataset_split, f"{cif_id}.graphdata")
            p_griddata = os.path.join(root_dataset_split, f"{cif_id}.griddata16")
            p_grid = os.path.join(root_dataset_split, f"{cif_id}.grid")
            if os.path.exists(p_graphdata) and os.path.exists(p_griddata) and os.path.exists(p_grid):
                logger.info(f"{cif_id} graph data already exists")
                eg_logger.info(f"{cif_id} energy grid already exists")
                continue

            # 0. check primitive cell and atom number < max_num_atoms
            p = os.path.join(root_cifs, f"{cif_id}.cif")
            try:
                st = CifParser(p, occupancy_tolerance=2.0).get_structures(primitive=True)[0]
                # save primitive cif
                p_primitive_cif = os.path.join(root_dataset_split, f"{cif_id}.cif")
                st.to(fmt="cif", filename=p_primitive_cif)
            except Exception as e:
                logger.info(f"{cif_id} failed : {e}")
                continue

            if len(st.atomic_numbers) > max_num_atoms:
                logger.info(f"{cif_id} failed : more than max_num_atoms in primitive cell")
                continue
            # 1. get crystal graph
            atom_num, nbr_idx, nbr_dist, uni_idx, uni_count = get_crystal_graph(st, radius=8, max_num_nbr=max_num_nbr)
            if len(nbr_idx) < len(atom_num) * max_num_nbr:
                logger.info(f"{cif_id} failed : num_nbr is smaller than max_num_nbr")
                print("please make radius larger")
                continue

            # 2. make orthogonal cell and supercell with min_length and max_length

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
                get_energy_grid(st, cif_id, root_dataset_split, eg_logger)

            logger.info(f"{cif_id} succeed : supercell length {st.lattice.abc}")

            data = [cif_id, atom_num, nbr_idx, nbr_dist, uni_idx, uni_count, target]

            p = os.path.join(root_dataset_split, f"{cif_id}.graphdata")
            with open(p, "wb") as f:
                pickle.dump(data, f)
