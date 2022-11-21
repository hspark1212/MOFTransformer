import sys
import os
import copy
import warnings

import pytorch_lightning as pl

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.utils.validation import get_valid_config, get_num_devices, ConfigurationError

from pytorch_lightning.plugins import DDPPlugin

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


_IS_INTERACTIVE = hasattr(sys, 'ps1')


def run(data_root, downstream=None, log_dir='logs/', *, test_only=False, **kwargs):
    """
    Train or predict MOFTransformer.

    Call signatures::
        run(data_root, downstream, [test_only], **kwargs)

    The basic usage of the code is as follows:

    >>> run(data_root, downstream)  # train MOFTransformer from [data_root] with train_{downstream}.json
    >>> run(data_root, downstream, log_dir, test_only=True, load_path=model_path) # predict MOFTransformer from trained-model path

    Dataset preperation is necessary for learning
    (url: https://hspark1212.github.io/MOFTransformer/dataset.html)

    Parameters
    __________
    :param data_root: A folder containing graph data, grid data, and json of MOFs that you want to train or test.
            The root data must be in the following format:
            data_root # root for generated inputs
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

    :param downstream: Name of user-specific task (e.g. bandgap, gasuptake, etc).
            if downstream is None, target json is 'train.json', 'val.json', and 'test.json'
    :param log_dir: Directory to save log, models, and params.
    :param test_only: If True, only the test process is performed without the learning model.

    Other Parameters
    ________________
    load_path: str, default : None
        The path of the model that starts when training/testing.
        If you want to test the fine-tuned model, enter the path of the .ckpt file stored in the 'log' folder
        If you downloaded the pretrain_model, it is set to default. Else, the default is set to a scratch model.
        You can download pretrain_model as following method:
            $ moftransformer download pretrain_model

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

    max_epochs: int, default: 100
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

    config = _config()
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f'{key} is not in configuration.')

    config.update(kwargs)
    config['data_root'] = data_root
    config['downstream'] = downstream
    config['log_dir'] = log_dir
    config['test_only'] = test_only

    main(config)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    _config = get_valid_config(_config)
    dm = Datamodule(_config)
    model = Module(_config)
    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_device = get_num_devices(_config)
    print ('num_device', num_device)

    # gradient accumulation
    if num_device == 0:
        accumulate_grad_batches = _config["batch_size"] // (
                _config["per_gpu_batchsize"] * _config["num_nodes"]
        )
    else:
        accumulate_grad_batches = _config["batch_size"] // (
                _config["per_gpu_batchsize"] * num_device * _config["num_nodes"]
        )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    if _IS_INTERACTIVE:
        strategy = None
    else:
        strategy = DDPPlugin(find_unused_parameters=True)

    log_every_n_steps=10

    trainer = pl.Trainer(
        accelerator=_config['accelerator'],
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=_config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        resume_from_checkpoint=_config["resume_from"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
