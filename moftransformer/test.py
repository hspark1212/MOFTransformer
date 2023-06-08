# MOFTransformer version 2.1.0
import sys
import os
import copy
import warnings
import json
from pathlib import Path

import pytorch_lightning as pl

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


_IS_INTERACTIVE = hasattr(sys, "ps1")


def test(root_dataset, load_path, downstream=None, save_dir=None, **kwargs):
    """
    Test MOFTransformer from load_path.

    Call signatures::
        test(root_dataset, load_path, downstream, save_dir, **kwargs)

    The basic usage of the code is as follows:

    >>> test(root_dataset, load_path, downstream)  # test MOFTransformer from trained-model path

    Results save in 'load_path' directory.
    
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
    :param save_dir : Directory path to save the 'result.json' file. (Default: load_path)

    Other Parameters
    ________________
    loss_names: str or list, or dict, default: "regression"
        One or more of the following loss : 'regression', 'classification', 'mpt', 'moc', and 'vfp'

    n_classes: int, default: 0
        Number of classes when your loss is 'classification'

    batch_size: int, default: 32
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
            raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["root_dataset"] = root_dataset
    config["downstream"] = downstream
    config['load_path'] = load_path
    config["test_only"] = True
    config['visualize'] = False
    config['save_dir'] = save_dir

    main(config)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    _config['test_only'] = True
    _config['visualize'] = False

    pl.seed_everything(_config["seed"])

    _config = get_valid_config(_config)
    dm = Datamodule(_config)
    model = Module(_config)
    dm.setup('test')
    model.eval()

    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= '2.0.0':
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=1,
        logger=False,
        log_every_n_steps=0,
        deterministic=True,
    )

    output = trainer.test(model, datamodule=dm)

    if save_dir := _config.get('save_dir'):
        save_dir = Path(save_dir)
        if save_dir.is_dir():
            save_dir = save_dir/'result.json'
    else:
        save_dir = Path(_config['load_path'])/'../../result.json'
        save_dir = save_dir.resolve()

    with save_dir.open('w') as f:
        json.dump(output, f)

    print (f'Results are saved in {save_dir}')