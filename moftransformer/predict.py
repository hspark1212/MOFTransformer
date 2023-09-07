# MOFTransformer version 2.1.1
import sys
import os
import copy
import warnings
from pathlib import Path
import re
import csv

import pytorch_lightning as pl

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.modules.module_utils import set_task
from moftransformer.utils.validation import (
    get_valid_config, get_num_devices, ConfigurationError, _IS_INTERACTIVE
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def predict(root_dataset, load_path, downstream=None, split='all', save_dir=None,
            **kwargs):
    """
     Predict MOFTransformer.

     Call signatures::
         predict(root_dataset, load_path, downstream, [split], **kwargs)

     The basic usage of the code is as follows:

     >>> predict(root_dataset, load_path, downstream)  # predict MOFTransformer from [root_dataset] with train_{downstream}.json
     >>> predict(root_dataset, load_path, downstream, split='test', save_dir='./predict') # predict MOFTransformer from trained-model path

     Dataset preperation is necessary for learning
     (url: https://hspark1212.github.io/MOFTransformer/dataset.html)

     Parameters
     __________
     :param root_dataset: A folder containing graph data, grid data, and json of MOFs that you want to train or test.
             The way to make root_dataset is at this link (https://hspark1212.github.io/MOFTransformer/dataset.html)
             The root data must be in the following format:
             root_dataset # root for generated inputs
             ├── train
             │   ├── [cif_id].graphdata # graphdata
             │   ├── [cif_id].grid # energy grid information
             │   ├── [cif_id].griddata16 # grid data
             │   ├── [cif_id].cif # primitive cif
             │   └── ...
             ├── val
             │   ├── [cif_id].graphdata # graphdata
             │   ├── [cif_id].grid # energy grid information
             │   ├── [cif_id].griddata16 # grid data
             │   ├── [cif_id].cif # primitive cif
             │   └── ...
             ├── test
             │   ├── [cif_id].graphdata # graphdata
             │   ├── [cif_id].grid # energy grid information
             │   ├── [cif_id].griddata16 # grid data
             │   ├── [cif_id].cif # primitive cif
             │   └── ...
             ├── train_{downstream}.json
             ├── val_{downstream}.json
             └── test_{downstream}.json

     :param load_path : Path for model you want to load and predict (*.ckpt).
     :param downstream: Name of user-specific task (e.g. bandgap, gasuptake, etc).
             if downstream is None, target json is 'train.json', 'val.json', and 'test.json'
     :param split : The split you want to predict on your dataset ('all', 'train', 'test', or 'val')
     :param save_dir : Path for directory you want to save *.csv file. (default : None -> path for loaded model)

     
     Other Parameters
     ________________
     loss_names: str or list, or dict, default: "regression"
         One or more of the following loss : 'regression', 'classification', 'mpt', 'moc', and 'vfp'

     n_classes: int, default: 0
         Number of classes when your loss is 'classification'

     batch_size: int, default: 1024
         desired batch size; for gradient accumulation

     per_gpu_batchsize: int, default: 8
         you should define this manually with per_gpu_batch_size

     accelerator: str, default: 'auto'
         Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto")
         as well as custom accelerator instances.

     devices: int or list, default: "auto"
         Number of devices to train on (int), which devices to train on (list or str), or "auto".
         It will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type ("cpu", "gpu", "tpu", "ipu", "auto").

     num_nodes: int, default: 1
         Number of GPU nodes for distributed training.

     num_workers: int, default: 16
         the number of cpu's core

     precision: int or str, default: 16
         MOFTransformer supports either double (64), float (32), bfloat16 (bf16), or half (16) precision training.
         Half precision, or mixed precision, is the combined use of 32 and 16 bit floating points to reduce memory footprint during model training.
         This can result in improved performance, achieving +3X speedups on modern GPUs.

     max_epochs: int, default: 20
         Stop training once this number of epochs is reached.

     seed: int, default: 0
         The random seed for pytorch_lightning.


     Normalization parameters:
     _________________________
     mean: float or None, default: None
         mean for normalizer. If None, it is automatically obtained from the train dataset.

     std: float or None, default: None
         standard deviation for normalizer. If None, it is automatically obtained from the train dataset.


     Optimzer setting parameters
     ___________________________
     optim_type: str, default: "adamw"
         Type of optimizer, which is "adamw", "adam", or "sgd" (momentum=0.9)

     learning_rate: float, default: 1e-4
         Learning rate for optimizer

     weight_decay: float, default: 1e-2
         Weight decay for optmizer

     decay_power: float, default: 1
         default polynomial decay, [cosine, constant, constant_with_warmup]

     max_steps: int, default: -1
         num_data * max_epoch // batch_size (accumulate_grad_batches)
         if -1, set max_steps automatically.

     warmup_steps : int or float, default: 0.05
         warmup steps for optimizer. If type is float, set to max_steps * warmup_steps.

     end_lr: float, default: 0

     lr_mult: float, default: 1
         multiply lr for downstream heads


     Transformer setting parameters
     ______________________________
     hid_dim = 768
     num_heads = 12
     num_layers = 12
     mlp_ratio = 4
     drop_rate = 0.1
     mpp_ratio = 0.15


     Atom-based Graph Parameters
     ___________________________
     atom_fea_len = 64
     nbr_fea_len = 64
     max_graph_len = 300 # number of maximum nodes in graph
     max_nbr_atoms = 12


     Energy-grid Parameters
     ______________________
     img_size = 30
     patch_size = 5  # length of patch
     in_chans = 1  # channels of grid image
     max_grid_len = -1  # when -1, max_image_len is set to maximum ph*pw of batch images
     draw_false_grid = False


     Visuallization Parameters
     _________________________
     visualize: bool, default: False
         return attention map (use at attetion visualization step)


     Pytorch lightning setting parameters
     ____________________________________
     resume_from = None
     val_check_interval = 1.0
     dataset_size = False  # experiments for dataset size with 100 [k] or 500 [k]

     """
    config = copy.deepcopy(_config())
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f'{key} is not in configuration.')

    config.update(kwargs)
    config['root_dataset'] = root_dataset
    config['downstream'] = downstream
    config['load_path'] = load_path
    config['test_only'] = True
    config['visualize'] = False
    config['split'] = split
    config['save_dir'] = save_dir
    
    main(config)


@ex.automain
def main(_config):
    config = copy.deepcopy(_config)

    config['test_only'] = True
    config['visualize'] = False

    os.makedirs(config["log_dir"], exist_ok=True)
    pl.seed_everything(config['seed'])

    num_device = get_num_devices(config)
    num_nodes = config['num_nodes']
    if num_nodes > 1:
        warnings.warn(f"function <predict> only support 1 devices. change num_nodes {num_nodes} -> 1")
        config['num_nodes'] = 1
    if num_device > 1:
        warnings.warn(f"function <predict> only support 1 devices. change num_devices {num_device} -> 1")
        config['devices'] = 1
    
    config = get_valid_config(config)  # valid config
    model = Module(config)
    dm = Datamodule(config)
    model.eval()

    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= '2.0.0':
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config["devices"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=1,
        log_every_n_steps=0,
        deterministic=True,
        logger=False,
    )

    # refine split
    split = config.get('split', 'all')
    if split == 'all':
        split = ['train', 'val', 'test']
    elif isinstance(split, str):
        split = re.split(r",\s?", split)

    if split == ['test']:
        dm.setup('test')
    elif 'test' not in split:
        dm.setup('fit')
    else:
        dm.setup()

    # save_dir
    save_dir = config.get('save_dir', None)
    if save_dir is None:
        save_dir = Path(config['load_path']).parent.parent
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # predict
    for s in split:
        if not s in ['train', 'test', 'val']:
            raise ValueError(f'split must be train, test, or val, not {s}')

        savefile = save_dir/f'{s}_prediction.csv'
        dataloader = getattr(dm, f'{s}_dataloader')()
        rets = trainer.predict(model, dataloader)
        write_output(rets, savefile)

    print (f'All prediction values are saved in {save_dir}')


def write_output(rets, savefile):
    keys = rets[0].keys()

    with open(savefile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for ret in rets:
            if ret.keys() != keys:
                raise ValueError(ret.keys(), keys)

            for data in zip(*ret.values()):
                wr.writerow(data)