import os
import random
import json
import pickle

import numpy as np

import torch
from torch.nn.functional import interpolate


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            split: str,
            nbr_fea_len: int,
            draw_false_grid=True,
            downstream="",
            tasks=[],
            dataset_size=False,
    ):
        """
        Dataset for pretrained MOF.
        Args:
            data_dir (str): where dataset cif files and energy grid file; exist via model.utils.prepare_data.py
            split(str) : train, test, split
            draw_false_grid (int, optional):  how many generating false_grid_data
            nbr_fea_len (int) : nbr_fea_len for gaussian expansion
        """
        super().__init__()
        self.data_dir = data_dir
        self.draw_false_grid = draw_false_grid
        self.split = split

        assert split in {"train", "test", "val"}
        if downstream:
            path_file = os.path.join(data_dir, f"{split}_{downstream}.json")
        else:
            path_file = os.path.join(data_dir, f"{split}.json")
            # future removed (only experiments)
            if dataset_size:
                path_file = os.path.join(data_dir, f"{dataset_size}k", f"{split}.json")
        print(f"read {path_file}...")
        assert os.path.isfile(path_file), f"{path_file} doesn't exist in {data_dir}"

        dict_target = json.load(open(path_file, "r"))
        self.cif_ids, self.targets = zip(*dict_target.items())

        self.nbr_fea_len = nbr_fea_len

        self.tasks = {}

        for task in tasks:
            if task in ["mtp", "vfp", "moc", "bbp"]:
                path_file = os.path.join(data_dir, f"{split}_{task}.json")
                # future removed (only experiments)
                if dataset_size:
                    path_file = os.path.join(data_dir, f"{dataset_size}k", f"{split}_{task}.json")
                print(f"read {path_file}...")
                assert os.path.isfile(path_file), f"{path_file} doesn't exist in {data_dir}"

                dict_task = json.load(open(path_file, "r"))
                cif_ids, t = zip(*dict_task.items())
                self.tasks[task] = list(t)
                assert self.cif_ids == cif_ids, print("order of keys is different in the json file")

    def __len__(self):
        return len(self.cif_ids)

    @staticmethod
    def make_grid_data(grid_data, emin=-5000., emax=5000, bins=101):
        """
        make grid_data within range (emin, emax) and
        make bins with logit function
        and digitize (0, bins)
        ****
            caution : 'zero' should be padding !!
            when you change bins, heads.MPP_heads should be changed
        ****
        """
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax

        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1

        return new_grid_data

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(1 - a_ ** 2 - b_ ** 2 - c_ ** 2 + 2 * a_ * b_ * c_)

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cif_id):
        file_grid = os.path.join(self.data_dir, self.split, f"{cif_id}.grid")
        file_griddata = os.path.join(self.data_dir, self.split, f"{cif_id}.griddata16")

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    def get_grid_data(self, cif_id, draw_false_grid=False):

        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data
                }
            )
        return ret

    @staticmethod
    def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
        """
        Expands the distance by Gaussian basis
        (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        _filter = np.linspace(dmin, dmax, num_step)  # = np.arange(dmin, dmax + step, step) with step = 0.2

        return np.exp(-(distances[..., np.newaxis] - _filter) ** 2 /
                      var ** 2).float()

    def get_graph(self, cif_id):
        file_graph = os.path.join(self.data_dir, self.split, f"{cif_id}.graphdata")

        graphdata = pickle.load(open(file_graph, "rb"))
        # graphdata = ["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count", "target"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(self.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8))

        uni_idx = graphdata[4]
        uni_count = graphdata[5]

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
        }

    def get_tasks(self, index):
        ret = dict()
        for task, value in self.tasks.items():
            ret.update(
                {
                    task: value[index]
                }
            )

        return ret

    def __getitem__(self, index):

        ret = dict()
        cif_id = self.cif_ids[index]
        target = self.targets[index]

        ret.update(
            {
                "cif_id": cif_id,
                "target": target,
            }
        )
        ret.update(self.get_grid_data(cif_id, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cif_id))

        ret.update(self.get_tasks(index))

        return ret

    @staticmethod
    def collate(batch, img_size):
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell), target]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data, target]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        crystal_atom_idx = []
        base_idx = 0

        for bi in range(batch_size):
            n_i = len(batch_atom_num[bi])
            crystal_atom_idx.append(
                torch.arange(n_i) + base_idx
            )
            base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []
        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            orig = interpolate(orig[None, None, :, :, :],
                               size=[img_size, img_size, img_size],
                               mode="trilinear",
                               align_corners=True,
                               )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():

            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            new_false_grids = []
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                orig = interpolate(orig[None, None, :, :, :],
                                   size=[img_size, img_size, img_size],
                                   mode="trilinear",
                                   align_corners=True,
                                   )
                new_false_grids.append(orig)
            new_false_grids = torch.concat(new_false_grids, axis=0)
            dict_batch["false_grid"] = new_false_grids

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch
