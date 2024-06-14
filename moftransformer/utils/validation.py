# MOFTransformer version 2.2.0
import sys
import warnings
import pytorch_lightning as pl
from moftransformer.database import (
    DEFAULT_PMTRANSFORMER_PATH,
    DEFAULT_MOFTRANSFORMER_PATH,
)

if pl.__version__ >= "2.0.0":
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _AcceleratorConnector as AC,
    )
else:
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        AcceleratorConnector as AC,
    )


_IS_INTERACTIVE = hasattr(sys, "ps1")


class ConfigurationError(Exception):
    pass


def _set_loss_names(loss_name):
    if isinstance(loss_name, list):
        d = {k: 1 for k in loss_name}
    elif isinstance(loss_name, str):
        d = {loss_name: 1}
    elif isinstance(loss_name, dict):
        d = loss_name
    elif loss_name is None:
        d = {}
    else:
        raise ConfigurationError(
            f"loss_name must be list, str, or dict, not {type(loss_name)}"
        )
    return _loss_names(d)


def _loss_names(d):
    ret = {
        "ggm": 0,  # graph grid matching
        "mpp": 0,  # masked patch prediction
        "mtp": 0,  # mof topology prediction
        "vfp": 0,  # (accessible) void fraction prediction
        "moc": 0,  # metal organic classification
        "bbc": 0,  # building block classification
        "classification": 0,  # classification
        "regression": 0,  # regression
    }
    ret.update(d)
    return ret


def _set_load_path(path):
    if path == "pmtransformer":
        return DEFAULT_PMTRANSFORMER_PATH
    if path == "moftransformer":
        return DEFAULT_MOFTRANSFORMER_PATH
    elif not path:
        return ""
    elif str(path)[-4:] == "ckpt":
        return path
    else:
        raise ConfigurationError(
            f"path must be 'pmtransformer', 'moftransformer', None, or *.ckpt, not {path}"
        )


def get_num_devices(_config):
    if isinstance(devices := _config["devices"], list):
        devices = len(devices)
    elif isinstance(devices, int):
        pass
    elif devices == "auto" or devices is None:
        devices = _get_auto_device(_config)
    else:
        raise ConfigurationError(
            f'devices must be int, list, and "auto", not {devices}'
        )
    return devices


def _get_auto_device(_config):
    accelerator = AC(accelerator=_config["accelerator"]).accelerator
    devices = accelerator.auto_device_count()

    return devices


def _set_valid_batchsize(_config, devices):
    per_gpu_batchsize = _config["batch_size"] // devices

    _config["per_gpu_batchsize"] = per_gpu_batchsize
    warnings.warn(
        "'Per_gpu_batchsize' is larger than 'batch_size'.\n"
        f" Adjusted to per_gpu_batchsize to {per_gpu_batchsize}"
    )


def _check_valid_num_gpus(_config):
    devices = get_num_devices(_config)

    if devices > _config["batch_size"]:
        raise ConfigurationError(
            "Number of devices must be smaller than batch_size. "
            f'num_gpus : {devices}, batch_size : {_config["batch_size"]}'
        )

    if _IS_INTERACTIVE and devices > 1:
        _config["devices"] = 1
        warnings.warn(
            "The interactive environment (ex. jupyter notebook) does not supports multi-devices environment. "
            f"Adjusted number of devices : {devices} to 1. "
            "If you want to use multi-devices, make *.py file and run."
        )

    return devices


def get_valid_config(_config):
    # set loss_name to dictionary
    _config["loss_names"] = _set_loss_names(_config["loss_names"])

    # set load_path to directory
    _config["load_path"] = _set_load_path(_config["load_path"])

    # check_valid_num_gpus
    devices = _check_valid_num_gpus(_config)

    # Batch size must be larger than gpu_per_batch
    if _config["batch_size"] < _config["per_gpu_batchsize"] * devices:
        _set_valid_batchsize(_config, devices)

    return _config
