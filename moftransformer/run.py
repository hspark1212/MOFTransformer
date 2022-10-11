import copy
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import _IS_INTERACTIVE

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.utils.validation import get_valid_config

from pytorch_lightning.plugins import DDPPlugin

import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def run(**kwargs):
    """
    Run MOFTransformer code
    :param kwargs: configuration for MOFTransformer
        seed = 0
        loss_names = _loss_names({"regression":1})

        # graph seeting
        # max_atom_len = 1000  # number of maximum atoms in primitive cell
        atom_fea_len = 64
        nbr_fea_len = 64
        max_graph_len = 300  # number of maximum nodes in graph
        max_nbr_atoms = 12

        # grid setting
        img_size = 30
        patch_size = 5  # length of patch
        in_chans = 1  # channels of grid image
        max_grid_len = -1  # when -1, max_image_len is set to maximum ph*pw of batch images
        draw_false_grid = False

        # transformer setting
        hid_dim = 768
        num_heads = 12
        num_layers = 12
        mlp_ratio = 4
        drop_rate = 0.1
        mpp_ratio = 0.15

        # downstream
        downstream = ""
        n_classes = 0

        # Optimizer Setting
        optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
        learning_rate = 1e-4
        weight_decay = 1e-2
        decay_power = 1  # default polynomial decay, [cosine, constant, constant_with_warmup]
        max_epochs = 100
        max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
        warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
        end_lr = 0
        lr_mult = 1  # multiply lr for downstream heads

        # PL Trainer Setting
        resume_from = None
        val_check_interval = 1.0
        test_only = False

        # below params varies with the environment
        data_root = "examples/dataset"
        log_dir = "examples/logs"
        batch_size = 1024  # desired batch size; for gradient accumulation
        per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size
        num_gpus = 1
        num_nodes = 1
        load_path = ""
        num_workers = 16  # the number of cpu's core
        precision = 16

        # experiments
        dataset_size = False  # experiments for dataset size with 100 [k] or 500 [k]

        # normalization target
        mean = None
        std = None

        # visualize
        visualize = False  # return attention map
    :return:
    """
    config = _config()
    config.update(kwargs)
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
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = _config["num_gpus"]
    if isinstance(num_gpus, list):
        num_gpus = len(num_gpus)

    # gradient accumulation
    if num_gpus == 0:
        accumulate_grad_batches = _config["batch_size"] // (
                _config["per_gpu_batchsize"] * _config["num_nodes"]
        )
    else:
        accumulate_grad_batches = _config["batch_size"] // (
                _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
        )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    if _IS_INTERACTIVE:
        strategy = None
    else:
        strategy = DDPPlugin(find_unused_parameters=True)

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=_config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
