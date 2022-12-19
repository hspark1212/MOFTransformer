import os
import math
import logging
import logging.handlers
import json
import subprocess
import hashlib
import pickle
import shutil
from pathlib import Path
from collections import namedtuple
from collections.abc import Iterable

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

    #stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)
    #logger.addHandler(stream_handler)

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
    # Before 1.1.1 version : num_grid = [str(round(cell)) for cell in structure.lattice.abc]
    # After 1.1.1 version : num_grid = [30, 30, 30]
    global GRIDAY_PATH, FF_PATH

    eg_file = os.path.join(root_dataset, cif_id)
    random_str = str(np.random.rand()).encode()
    tmp_file = os.path.join(root_dataset, f"{hashlib.sha256(random_str).hexdigest()}.cssr")

    try:
        Cssr(structure).write_file(tmp_file)
        num_grid = ['30', '30', '30']
        proc = subprocess.Popen(
            [GRIDAY_PATH, *num_grid, f'{FF_PATH}/UFF_Type.def', f'{FF_PATH}/UFF_FF.def', tmp_file, eg_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
    finally:
        # remove temp_file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    if err:
        eg_logger.info(f"{cif_id} energy grid failed {err}")
        return False
    else:
        eg_logger.info(f"{cif_id} energy grid success")

    if os.path.exists(eg_file + ".griddata"):
        grid_data = make_float16_griddata(eg_file + ".griddata")
        path_save = os.path.join(root_dataset, f"{cif_id}.griddata16")
        pickle.dump(grid_data, open(path_save, "wb"))
        eg_logger.info(f"{cif_id} energy grid changed to np16")

        try:
            os.remove(eg_file + ".griddata")
        except Exception as e:
            print(e)
        return True
    else:
        eg_logger.info(f"{cif_id} energy grid failed to change to np16")
        return False


def _split_dataset(root_dataset:Path, **kwargs):
    """
    make train_{task}.json, test_{task}.json, and val_{task}.json from raw_{task}.json
    :param root_cifs: root for cif files
    :param root_dataset: root for generated datasets
    :param single_task: name of downstream tasks
    :param kwargs:
        - overwrite_json (bool) : If True, overwrite {split}_task.json file when it exists. (default : False)
        - seed : (int) random seed for split data. (default : 42)
        - duplicate : (bool) If True, allow duplication of data in train, test, and validation. (default: False)
        - train_fraction : (float) fraction for train dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.8)
        - test_fraction : (float) fraction for test dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.1)

    :return:
    """
    # get argument from kwargs
    seed = kwargs.get('seed', 42)
    threshold = kwargs.get('threshold', 0.01)
    train_fraction = kwargs.get('train_fraction', 0.8)
    test_fraction = kwargs.get('test_fraction', 0.1)

    # get directories
    total_dir = root_dataset/'total'
    assert total_dir.exists()

    split_dir = {split : root_dataset/split for split in ['train', 'test', 'val']}
    for direc in split_dir.values():
        direc.mkdir(exist_ok=True)

    # get success prepare-data list
    cif_list = {}
    CifPath = namedtuple('CifPath', ['cif', 'graphdata', 'grid', 'griddata16'])
    for cif in total_dir.glob('*.cif'):
        cif_id = cif.stem
        graphdata = cif.with_suffix('.graphdata')
        grid = cif.with_suffix('.grid')
        griddata = cif.with_suffix('.griddata16')

        if cif.exists() and graphdata.exists() and grid.exists() and griddata.exists():
            cif_list[cif_id] = CifPath(cif, graphdata, grid, griddata)

    # get number of split
    if train_fraction + test_fraction > 1:
        raise ValueError(f'"train_fraction + test_fraction" must be smaller than 1.0, not {train_fraction + test_fraction}')

    n_total = len(cif_list.keys())
    n_train = int(n_total * train_fraction)
    n_test = int(n_total * test_fraction)
    n_val = n_total - n_train - n_test
    n_split = {'train':n_train, 'test':n_test, 'val':n_val}

    # remove already-divided values
    for split, direc in split_dir.items():
        split_cifs = {cif.stem for cif in direc.glob('*.cif')}
        for cif in split_cifs:
            if cif in cif_list:
                del cif_list[cif]
                n_split[split] -= 1

    assert sum(n_split.values()) == len(cif_list), 'Error! contact with code writer!'
    if not cif_list: # NO additional divided task
        return

    for split, n in n_split.items():
        if n < -n_total*threshold:
            raise ValueError("{split} folder's cif number is larger than {split}_fraction. change argument {split}_fraction.")

    # random split index
    cif_name = sorted(list(cif_list.keys()))
    split_idx = ['train']*n_split['train'] + ['test']*n_split['test'] + ['val']*n_split['val']
    np.random.seed(seed=seed)
    np.random.shuffle(split_idx)

    assert len(cif_name) == len(split_idx)

    for cif, split in zip(cif_name, split_idx):
        cifpath = cif_list[cif]
        for suffix in ['cif', 'graphdata', 'grid', 'griddata16']:
            src = getattr(cifpath, suffix)
            dest = root_dataset/split
            shutil.copy(src, dest)


def _split_json(root_cifs:Path, root_dataset:Path, downstream:str):
    with open(str(root_cifs/f'raw_{downstream}.json')) as f:
        src = json.load(f)

    for split in ['train', 'test', 'val']:
        cif_folder = root_dataset/split
        cif_list = [cif.stem for cif in cif_folder.glob('*.cif')]
        split_json = {i:src[i] for i in cif_list if i in src}
        with open(str(root_dataset/f'{split}_{downstream}.json'), 'w') as f:
            json.dump(split_json, f)


def make_prepared_data(cif:Path, root_dataset_total:Path, logger=None, eg_logger=None, **kwargs):
    if logger is None:
        logger = get_logger(filename="prepare_data.log")
    if eg_logger is None:
        eg_logger = get_logger(filename="prepare_energy_grid.log")

    if isinstance(cif, str):
        cif = Path(cif)
    if isinstance(root_dataset_total, str):
        root_dataset_total = Path(root_dataset_total)

    get_primitive = kwargs.get('get_primitive', 'True')
    max_num_atoms = kwargs.get('max_num_atoms', 1000)
    max_length = kwargs.get('max_length', 60.)
    min_length = kwargs.get('min_length', 30.)
    max_num_nbr = kwargs.get('max_num_nbr', 12)

    cif_id: str = cif.stem

    p_graphdata = root_dataset_total / f"{cif_id}.graphdata"
    p_griddata = root_dataset_total / f"{cif_id}.griddata16"
    p_grid = root_dataset_total / f"{cif_id}.grid"

    # Grid data and Graph data already exists
    if p_graphdata.exists() and p_griddata.exists() and p_grid.exists():
        logger.info(f"{cif_id} graph data already exists")
        eg_logger.info(f"{cif_id} energy grid already exists")
        return True

    # 0. check primitive cell and atom number < max_num_atoms
    try:
        st = CifParser(str(cif), occupancy_tolerance=2.0).get_structures(primitive=get_primitive)[0]

    except Exception as e:
        logger.info(f"{cif_id} failed : {e}")
        return False

    if len(st.atomic_numbers) > max_num_atoms:
        logger.info(f"{cif_id} failed : more than max_num_atoms in primitive cell")
        return False

    # 1. get crystal graph
    atom_num, nbr_idx, nbr_dist, uni_idx, uni_count = get_crystal_graph(st, radius=8, max_num_nbr=max_num_nbr)
    if len(nbr_idx) < len(atom_num) * max_num_nbr:
        logger.info(f"{cif_id} failed : num_nbr is smaller than max_num_nbr. please make radius larger")
        return False

    # 2. make orthogonal cell and supercell with min_length and max_length

    scale_abc = []
    for l in st.lattice.abc:
        if l > max_length:
            logger.info(f"{cif_id} failed : supercell have more than max_length")
            return False
        elif l < min_length:
            scale_abc.append(math.ceil(min_length / l))
        else:
            scale_abc.append(1)

    assert len(scale_abc) == len(st.lattice.abc), 'Error! contact to code writer!'

    st.make_supercell(scale_abc)

    # 3. calculate energy grid
    eg_success = get_energy_grid(st, cif_id, root_dataset_total, eg_logger)

    if eg_success:
        logger.info(f"{cif_id} succeed : supercell length {st.lattice.abc}")

        # save primitive files
        p_primitive_cif = root_dataset_total / f'{cif_id}.cif'
        st.to(fmt="cif", filename=str(p_primitive_cif))

        # save graphdata file
        data = [cif_id, atom_num, nbr_idx, nbr_dist, uni_idx, uni_count]
        with open(str(p_graphdata), "wb") as f:
            pickle.dump(data, f)
        return True
    else:
        return False


def prepare_data(root_cifs, root_dataset, downstream, **kwargs):
    """
    Args:
        root_cifs (str): root for cif files,
                        it should contains "train" and "test" directory in root_cifs
                        ("val" directory is optional)
        root_dataset (str): root for generated datasets
        downstream (str or list) : name of downstream tasks

    kwargs:
        - seed : (int) random seed for split data. (default : 42)
        - train_fraction : (float) fraction for train dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.8)
        - test_fraction : (float) fraction for test dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.1)

        - get_primitive (bool) : If True, use primitive cell in graph embedding
        - max_num_atoms (int): max number atoms in primitive cell
        - max_length (float) : maximum length of supercell
        - min_length (float) : minimum length of supercell
        - max_num_nbr (int) : maximum number of neighbors when calculating graph
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

    # directory to "Path"
    root_cifs = Path(root_cifs)
    root_dataset = Path(root_dataset)

    if not root_cifs.exists():
        raise ValueError(f'{root_cifs} does not exists.')

    # make prepare_data in 'total' directory
    root_dataset_total = Path(root_dataset) / 'total'
    root_dataset_total.mkdir(exist_ok=True, parents=True)

    # make *.grid, *.griddata16, and *.graphdata file
    for cif in tqdm(root_cifs.glob('*.cif'), total=sum(1 for _ in root_cifs.glob('*.cif'))):
        make_prepared_data(cif, root_dataset_total, logger, eg_logger, **kwargs)

    # automatically split data
    _split_dataset(root_dataset, **kwargs)

    # split json file
    if isinstance(downstream, str):
        _split_json(root_cifs, root_dataset, downstream)
    elif isinstance(downstream, Iterable):
        for single_downstream in downstream:
            _split_json(root_cifs, root_dataset, single_downstream)
    else:
        raise TypeError(f'task must be str or Iterable, not {type(downstream)}')