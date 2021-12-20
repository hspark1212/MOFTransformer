import os
import random

import pyarrow as pa
import numpy as np

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            split: str,
            draw_false_grid=True,
    ):
        """
        Dataset for pretrained MOF.
        Args:
            data_dir (str): where dataset cif files and energy grid file; exist via model.utils.prepare_data.py
            split(str) : train, test, split
            draw_false_grid (int, optional):  how many generating false_grid_data
        """
        super().__init__()
        self.data_dir = data_dir
        self.draw_false_grid = draw_false_grid

        assert split in {"train", "test", "val"}
        path_file = os.path.join(data_dir, f"{split}.arrow")
        self.split = split

        assert os.path.isfile(path_file), print(f"{path_file} doesn't exist in {data_dir}")

        self.data = pa.ipc.RecordBatchFileReader(
            pa.memory_map(path_file, "r")
        ).read_all().to_pandas()

    def __len__(self):
        return len(self.data)

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

    def get_raw_grid_data(self, filename):
        file_grid = os.path.join(self.data_dir, self.split, f"{filename}.grid")
        file_griddata = os.path.join(self.data_dir, self.split, f"{filename}.griddata")

        # get grid
        with open(file_grid, "r") as f:
            cell = [int(i) for i in f.readlines()[2].split()[1:]]

        # get grid data
        grid_data = np.fromfile(file_griddata, dtype=np.float32)
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, grid_data

    def get_grid_data(self, index, draw_false_grid=False):
        filename = self.data["cif_id"][index]
        cell, grid_data = self.get_raw_grid_data(filename)
        ret = {
            "cell": cell,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.data) - 1)
            filename = self.data["cif_id"][random_index]
            cell, grid_data = self.get_raw_grid_data(filename)
            ret.update(
                {
                    "false_cell": cell,
                    "false_grid_data": grid_data
                }
            )
        return ret

    @staticmethod
    def get_gaussian_distance(distances, dmax, dmin=0, step=0.2, var=None):
        """
        Expands the distance by Gaussian basis
        (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        assert dmax - dmin > step
        _filter = np.arange(dmin, dmax + step, step)

        if var is None:
            var = step

        return np.exp(-(distances[..., np.newaxis] - _filter) ** 2 /
                      var ** 2).float()

    def get_graph(self, index):

        atom_num = torch.LongTensor(self.data["atom_num"][index].copy())
        nbr_idx = torch.LongTensor(self.data["nbr_idx"][index].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(self.data["nbr_dist"][index].copy()).view(len(atom_num), -1)
        nbr_fea = torch.FloatTensor(self.get_gaussian_distance(nbr_dist, dmax=8))

        uni_idx = self.data["uni_idx"][index].tolist()
        uni_count = self.data["uni_count"][index].tolist()

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
        }

    def __getitem__(self, index):

        ret = dict()
        ret.update(
            {
                "cif_id": self.data["cif_id"][index],
                "target": self.data["target"][index],
            }
        )
        ret.update(self.get_grid_data(index, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(index))

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
        new_grids = torch.zeros(batch_size, img_size, img_size, img_size)
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]

        for bi in range(batch_size):
            # griddata needs to be reshape with Fortran-like indexing style (ex. np.reshape( , , order="F")
            # with the first index changing fastest, and the last index changing slowest
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)

            new_grids[bi, :orig.shape[0], :orig.shape[1], :orig.shape[2]] = orig
        new_grids = new_grids[:, None, :, :, :]  # [B, 1, H, W, D]
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():
            new_false_grids = torch.zeros(batch_size, img_size, img_size, img_size)
            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                new_false_grids[bi, : orig.shape[0], : orig.shape[1], : orig.shape[2]] = orig
            new_false_grids = new_false_grids[:, None, :, :, :]  # [B, 1, H, W, D]
            dict_batch["false_grid"] = new_false_grids

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch
