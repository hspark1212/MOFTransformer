from tqdm import tqdm
from ase.io import read
import json

non_metals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
              'Si', 'P', 'S', 'Cl', 'Ar',
              'Ge', 'As', 'Se', 'Br', 'Kr',
              'Sb', 'Te', 'I', 'Xe',
              'Po', 'At', 'Rn']

j = json.load(open("/home/data/pretrained_mof/ver4/dataset/test_vfp.json"))
cif_ids, vf = zip(*j.items())

d = {}
for i, cif_id in enumerate(tqdm(cif_ids)):
    atoms = read(f"/home/data/pretrained_mof/ver4/primitive_cif/{cif_id}.cif")
    atomic_nums = atoms.get_chemical_symbols()
    d[cif_id] = list(set(atomic_nums) - set(non_metals))
json.dump(d, open("result_tsne/test_metal.json", "w"))


""" get tsne
import numpy as np
from sklearn.manifold import TSNE

total_cls = np.load("result_tsne/cls_mtp_moc_vfp.npz")
tsne = TSNE(n_components=2)
result_tsne = tsne.fit_transform(total_cls)
np.save(open("result_tsne/result_tsne.npz", "wb"), result_tsne)
print("finish")
"""

""" get cls vectors

import pytorch_lightning as pl

from model.datamodules.datamodule import Datamodule
from model.modules.module import Module
from model.datamodules.dataset import Dataset

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

from model.config import config, _loss_names
_config = config()
_config["visualize"] = False
_config["per_gpu_batchsize"] = 20
_config["data_root"] = "/home/data/pretrained_mof/ver4/dataset/"
_config["exp_name"] = "task_mtp"
_config["log_dir"] = "result_visualization"
_config["use_transformer"] = True
_config["load_path"] = "best_ckpt/best_mtp_moc_vfp.ckpt"
_config["test_only"] = True

pl.seed_everything(_config["seed"])

model = Module(_config)
model.setup("test")
model.eval()

device = "cpu"
model.to(device)

dm = Datamodule(_config)
dm.setup("test")
data_iter = dm.test_dataloader()

total_cls = np.zeros([100000, 768])
batch_size = _config["per_gpu_batchsize"]


for i, batch in enumerate(tqdm(data_iter)):
    out = model.infer(batch)
    cls_feats = out["cls_feats"].detach().numpy() # [B, hid_dim]
    total_cls[batch_size*i:batch_size*(i+1)] = cls_feats

np.save(open("cls_mtp_moc_vfp.npz","wb"), total_cls)

"""
