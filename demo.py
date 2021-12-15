import copy
import os
import torch
import pytorch_lightning as pl

from model.config import ex

from model.datamodules.datamodule import Datamodule
from model.modules.module import Module
from model.modules import objectives

from pytorch_lightning.plugins import DDPPlugin

import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings(
    "ignore", ".*Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.*"
) # when loss is huge..., skip optimize step

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = Datamodule(_config)
    model = Module(_config)
    dm.setup()
    data_iter = dm.test_dataloader()
    for batch in data_iter:
        break



    """
        dm = Datamodule(_config)
        dm.setup()
        print(dm.train_dataset[0].keys())
        data_iter = dm.train_dataloader()
        for batch in data_iter:
            print(batch.keys())
            break
        print(len(dm.train_dataset))
    """





