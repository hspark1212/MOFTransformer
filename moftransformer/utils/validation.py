import warnings
from pytorch_lightning.utilities import _IS_INTERACTIVE
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

from moftransformer.config import _loss_names


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
        raise ConfigurationError(f'loss_name must be list, str, or dict, not {type(loss_name)}')
    return _loss_names(d)


def get_num_devices(_config):
    if isinstance(devices := _config['devices'], list):
        devices = len(devices)
    elif isinstance(devices, int):
        pass
    elif devices == 'auto' or devices is None:
        accelerator = AcceleratorConnector(accelerator=_config['accelerator'])
        devices = accelerator.auto_device_count()
    else:
        raise ConfigurationError(f'devices must be int, list, and "auto", not {devices}')

    return devices


def _set_valid_batchsize(_config):
    devices = get_num_devices(_config)

    per_gpu_batchsize = _config['batch_size'] // devices

    _config['per_gpu_batchsize'] = per_gpu_batchsize
    warnings.warn("'Per_gpu_batchsize' is larger than 'batch_size'.\n"
          f" Adjusted to per_gpu_batchsize to {per_gpu_batchsize}")


def _check_valid_num_gpus(_config):
    devices = get_num_devices(_config)

    if devices > _config['batch_size']:
        raise ConfigurationError('Number of devices must be smaller than batch_size. '
                         f'num_gpus : {devices}, batch_size : {_config["batch_size"]}')

    if _IS_INTERACTIVE and devices > 1:
        raise ConfigurationError('The interactive environment (ex. jupyter notebook) does not supports multi-devices environment.'
                                 'If you want to use multi-devices, make *.py file and run.')


def get_valid_config(_config):
    # set loss_name to dictionary
    _config['loss_names'] = _set_loss_names(_config['loss_names'])

    # check_valid_num_gpus
    _check_valid_num_gpus(_config)

    # Batch size must be larger than gpu_per_batch
    if _config['batch_size'] < _config['per_gpu_batchsize']:
        _set_valid_batchsize(_config)

    return _config